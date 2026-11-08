# SPDX-License-Identifier: Apache-2.0
"""CPU checks of MP engine ownership; only NPU/native boundaries are replaced."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from uuid import UUID
import importlib
import json
import socket
import sys
import threading

# Third Party
import msgspec
import pytest
import zmq

PREFIX = "LMCACHE_HIXL_"
ADVERTISE_URL = "10.0.0.11:5555"
LISTEN_URL = "127.0.0.1:5555"


@pytest.fixture
def l1():
    # Deliberately unaligned base, and a stable prefix smaller than full size.
    return SimpleNamespace(
        ptr=0x100003, size=4096, align_bytes=65536, stable_registration_size=512
    )


@pytest.fixture
def backend(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    for name, path in [("lmcache_ascend", root), ("lmcache_ascend.v1", root / "v1")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    for suffix in ["mp_hixl_channel", "hixl_runtime"]:
        monkeypatch.delitem(sys.modules, f"lmcache_ascend.v1.{suffix}", raising=False)
    native = ModuleType("lmcache_ascend.hixl_npu_comms")

    class ThreadState(threading.local):
        device = None
        context = 0x4000

    state = ThreadState()
    native.get_current_device = lambda: state.device
    native.get_current_context = lambda: state.context
    native.set_current_context = lambda value: setattr(state, "context", value)

    class HixlError(RuntimeError):
        def __init__(self, code):
            super().__init__(f"native status {code}")
            self.code = code

    native.HixlError = HixlError
    native.HIXL_ALREADY_CONNECTED = 100
    native.HIXL_NOT_CONNECTED = 101
    native.TransferOp = SimpleNamespace(READ=0)
    native.TransferStatus = SimpleNamespace(WAITING=0, COMPLETED=1, TIMEOUT=2, FAILED=3)
    native.MEM_HOST = 1
    native.Hixl = Mock(return_value=Mock())
    native.Hixl.return_value.register_mem.return_value = 0x5678
    monkeypatch.setitem(sys.modules, native.__name__, native)
    npu = SimpleNamespace(
        set_device=Mock(side_effect=lambda d: setattr(state, "device", d))
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    for suffix in ["ENGINE_ID", "DEVICE_ID", "COMM_LISTEN_PORT", "CONNECT_TIMEOUT_MS"]:
        monkeypatch.delenv(PREFIX + suffix, raising=False)
    monkeypatch.setenv(PREFIX + "ENGINE_ID", "10.0.0.11:6000")
    module = importlib.import_module("lmcache_ascend.v1.mp_hixl_channel")
    # Exercise the real scope and lock, including lock ownership during calls.
    lock = threading.Lock()
    locks = iter([lock])
    monkeypatch.setattr(module, "Lock", lambda: next(locks, threading.Lock()))
    # Existing engine/metadata tests do not need a listening server.
    state.server_class = getattr(module, "HixlTransferChannelServer", None)
    monkeypatch.setattr(module, "HixlTransferChannelServer", Mock(), raising=False)
    yield module, native, npu, state, lock
    # Imported modules are not themselves installed by monkeypatch.
    for suffix in ["mp_hixl_channel", "hixl_runtime"]:
        sys.modules.pop(f"lmcache_ascend.v1.{suffix}", None)


def test_default_engine_lifecycle_uses_lock_and_borrowed_context(backend, l1):
    module, native, npu, state, lock = backend
    engine = native.Hixl.return_value
    events = []

    def initialize(engine_id, options):
        assert lock.locked() and state.context == 0x4000
        assert engine_id == "10.0.0.11:6000"
        assert options == {
            "LocalCommRes": '{"version":"1.3"}',
            "BufferPool": "0:0",
            "AutoConnect": "0",
            "GlobalResourceConfig": json.dumps(
                {"comm_resource_config.protocol_desc": "roce:device"}
            ),
        }
        events.append("initialize")
        state.context = 0x9000  # Native initialization may switch contexts.

    def register(ptr, size, kind):
        assert lock.locked() and state.context == 0x4000
        assert (ptr, size, kind) == (0x100003, 4096, native.MEM_HOST)
        events.append("register")
        state.context = 0x9000
        return 0x5678

    def deregister(handle):
        assert lock.locked() and state.context == 0x4000
        assert handle == 0x5678
        events.append("deregister")
        state.context = 0x9000

    def finalize():
        assert lock.locked() and state.context == 0x4000
        events.append("finalize")

    engine.initialize.side_effect = initialize
    engine.register_mem.side_effect = register
    engine.deregister_mem.side_effect = deregister
    engine.finalize.side_effect = finalize
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    npu.set_device.assert_called_once_with(0)
    assert context._config.connect_timeout_ms == 10000
    assert state.context == 0x4000 and not lock.locked()
    state.context = 0x8000  # Closing caller has a different previous context.
    context.close()
    context.close()
    engine.finalize.assert_called_once_with()
    assert state.context == 0x8000 and not lock.locked()
    assert events == ["initialize", "register", "deregister", "finalize"]


def test_explicit_configuration_reaches_engine(backend, monkeypatch, l1):
    module, native, npu, _, _ = backend
    monkeypatch.setenv(PREFIX + "ENGINE_ID", "[2001:db8::1]:6001")
    monkeypatch.setenv(PREFIX + "DEVICE_ID", "3")
    monkeypatch.setenv(PREFIX + "COMM_LISTEN_PORT", "26666")
    monkeypatch.setenv(PREFIX + "CONNECT_TIMEOUT_MS", "20000")
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    npu.set_device.assert_called_once_with(3)
    engine_id, options = native.Hixl.return_value.initialize.call_args.args
    assert engine_id == "[2001:db8::1]:6001"
    assert json.loads(options["GlobalResourceConfig"]) == {
        "comm_resource_config.protocol_desc": "roce:device",
        "comm_resource_config.listen_port": 26666,
    }
    assert context._config.connect_timeout_ms == 20000
    context.close()


@pytest.mark.parametrize(
    "suffix,value",
    [("ENGINE_ID", ""), ("COMM_LISTEN_PORT", "65536"), ("CONNECT_TIMEOUT_MS", "0")],
)
def test_invalid_configuration_has_no_runtime_side_effects(
    backend, monkeypatch, suffix, value, l1
):
    module, native, npu, _, _ = backend
    monkeypatch.setenv(PREFIX + suffix, value)
    with pytest.raises(ValueError, match=PREFIX + suffix):
        module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    npu.set_device.assert_not_called()
    native.Hixl.assert_not_called()


def test_device_failure_preserves_error_without_creating_engine(backend, l1):
    module, native, npu, _, _ = backend
    error = RuntimeError("device unavailable")
    npu.set_device.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    assert caught.value is error
    native.Hixl.assert_not_called()


def test_initialize_failure_does_not_publish_engine_or_explicitly_finalize(backend, l1):
    module, native, _, state, lock = backend
    error = RuntimeError("native Initialize failed")
    native.Hixl.return_value.initialize.side_effect = error
    context = module.HixlTransferChannelContext.__new__(
        module.HixlTransferChannelContext
    )
    with pytest.raises(RuntimeError) as caught:
        context.__init__(l1, LISTEN_URL, ADVERTISE_URL)
    assert caught.value is error
    assert context._engine is None
    native.Hixl.return_value.finalize.assert_not_called()
    assert state.context == 0x4000 and not lock.locked()


def test_cleanup_error_does_not_replace_original_failure(backend, caplog, l1):
    module, native, _, state, lock = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    native.Hixl.return_value.finalize.side_effect = RuntimeError("cleanup failed")
    error = ValueError("later construction failed")
    with pytest.raises(ValueError) as caught:
        try:
            raise error
        finally:
            context.close()
    assert caught.value is error
    assert "cleanup failed" in caplog.text
    assert state.context == 0x4000 and not lock.locked()
    assert context._engine is None


@pytest.mark.parametrize("ptr,size", [(0, 4096), ((1 << 64) - 8, 16)])
def test_invalid_l1_range_fails_before_runtime(backend, l1, ptr, size):
    module, native, npu, _, _ = backend
    l1.ptr, l1.size = ptr, size
    with pytest.raises(ValueError, match="memory range"):
        module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    npu.set_device.assert_not_called()
    native.Hixl.assert_not_called()


@pytest.mark.parametrize("failed_call", [True, False])
def test_failed_or_zero_registration_finalizes_without_invalid_deregister(
    backend, l1, failed_call
):
    module, native, _, _, _ = backend
    engine = native.Hixl.return_value
    error = RuntimeError("RegisterMem failed")
    if failed_call:
        engine.register_mem.side_effect = error
    else:
        engine.register_mem.return_value = 0
    with pytest.raises(RuntimeError) as caught:
        module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    if failed_call:
        assert caught.value is error
    engine.deregister_mem.assert_not_called()
    engine.finalize.assert_called_once_with()


def test_metadata_roundtrip_uses_registered_range_and_stable_instance(backend, l1):
    module, _, _, _, _ = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    metadata = context.metadata
    assert (
        metadata.advertise_url,
        metadata.engine_id,
        metadata.buffer_base_ptr,
        metadata.buffer_size,
    ) == (ADVERTISE_URL, "10.0.0.11:6000", 0x100003, 4096)
    assert str(UUID(metadata.instance_id)) == metadata.instance_id
    for message_type in [module.InitReq, module.InitResp]:
        wire = msgspec.msgpack.encode(message_type(metadata=metadata))
        decoded = msgspec.msgpack.decode(wire, type=module.HandshakeMsg)
        module.validate_metadata(decoded.metadata)
        assert type(decoded) is message_type
        assert decoded.metadata == metadata
    context.close()


@pytest.mark.parametrize(
    "field,value", [("buffer_size", "4096"), ("instance_id", "not-a-uuid")]
)
def test_rejects_malformed_peer_metadata(backend, l1, field, value):
    module, _, _, _, _ = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    payload = msgspec.to_builtins(module.InitReq(metadata=context.metadata))
    payload["metadata"][field] = value
    with pytest.raises((msgspec.ValidationError, ValueError)):
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(payload), type=module.HandshakeMsg
        )
        module.validate_metadata(decoded.metadata)
    context.close()


def test_metadata_failure_rolls_back_registration_and_preserves_error(
    backend, l1, monkeypatch, caplog
):
    module, native, _, state, lock = backend
    engine = native.Hixl.return_value
    error = RuntimeError("UUID generation failed")
    monkeypatch.setattr(module, "uuid4", Mock(side_effect=error))
    engine.deregister_mem.side_effect = RuntimeError("deregister failed")
    with pytest.raises(RuntimeError) as caught:
        module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    assert caught.value is error
    engine.deregister_mem.assert_called_once_with(0x5678)
    engine.finalize.assert_called_once_with()
    assert "deregister failed" in caplog.text
    assert state.context == 0x4000 and not lock.locked()


@pytest.fixture
def local_contexts(backend, monkeypatch, l1):
    module, native, _, state, _ = backend
    assert state.server_class is not None, "peer Server is not implemented"
    monkeypatch.setattr(module, "HixlTransferChannelServer", state.server_class)
    monkeypatch.setattr(module, "Lock", threading.Lock)
    contexts = []

    def create(host="127.0.0.1"):
        family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        address = "[::1]" if host == "::1" else host
        with socket.socket(family) as reservation:
            reservation.bind((host, 0))
            url = f"{address}:{reservation.getsockname()[1]}"
        engine = Mock()
        engine.register_mem.return_value = 0x5678
        native.Hixl.return_value = engine
        monkeypatch.setenv(PREFIX + "ENGINE_ID", f"{address}:{6000 + len(contexts)}")
        context = module.HixlTransferChannelContext(l1, url, url)
        contexts.append(context)
        return context, engine

    yield create
    for context in reversed(contexts):
        context.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_local_handshake_reverse_ready_and_removed_client_reuses_connection(
    backend, local_contexts, host
):
    module, _, _, state, _ = backend
    a, engine_a = local_contexts(host)
    b, engine_b = local_contexts(host)

    def connect_b(engine_id, timeout_ms):
        # B must have finished reverse admission before A receives InitResp.
        assert b.get_num_connected_clients() == 1
        assert a._engine_lock.locked() and state.context == 0x4000
        assert (engine_id, timeout_ms) == (b.metadata.engine_id, 10000)
        state.context = 0x9000

    engine_a.connect.side_effect = connect_b
    client = a.get_transfer_channel_client(b.metadata.advertise_url)
    assert client.peer_metadata == b.metadata
    reverse = b.get_transfer_channel_client(a.metadata.advertise_url)
    assert reverse.peer_metadata == a.metadata
    assert a.get_transfer_channel_client(b.metadata.advertise_url) is client
    a.remove_transfer_channel_client(b.metadata.advertise_url)
    assert client.closed and a.get_num_connected_clients() == 0
    engine_a.disconnect.assert_not_called()
    replacement = a.get_transfer_channel_client(b.metadata.advertise_url)
    assert replacement is not client and not replacement.closed
    assert replacement.connection is client.connection
    engine_a.connect.assert_called_once()
    engine_b.connect.assert_called_once_with(a.metadata.engine_id, timeout_ms=10000)
    server = a.get_transfer_channel_server()
    a.close()
    assert replacement.closed and not server._thread.is_alive()
    engine_a.disconnect.assert_called_once_with(b.metadata.engine_id, timeout_ms=1000)
    assert state.context == 0x4000


def test_duplicate_clients_share_connection_and_identity_conflicts_are_rejected(
    backend, l1
):
    module, native, _, _, _ = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    peer = msgspec.structs.replace(
        context.metadata, advertise_url="10.0.0.12:5555", engine_id="10.0.0.12:6000"
    )
    gate = threading.Barrier(2)

    def admit():
        gate.wait(timeout=5)
        connection = context._ensure_connection(peer)
        candidate = module.HixlTransferChannelClient(context, connection)
        return candidate, context.register_client(peer.advertise_url, candidate)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(admit) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]
    first = results[0][1]
    assert results[1][1] is first and not first.closed
    discarded = next(candidate for candidate, _ in results if candidate is not first)
    assert discarded.closed and discarded.connection is first.connection
    context.remove_transfer_channel_client(peer.advertise_url)
    first.connection.connected = False
    first.connection.failure = "query failed"
    first.connection.disconnect_succeeded = True
    # Removing a logical Client must not permit a restarted peer to replace
    # retained identity, even after successful fault cleanup.
    changed = msgspec.structs.replace(
        peer,
        engine_id="10.0.0.12:6001",
        instance_id="00000000-0000-0000-0000-000000000001",
    )
    with pytest.raises(ValueError, match="metadata"):
        context._ensure_connection(changed)
    native.Hixl.return_value.connect.assert_called_once()
    context.close()


def test_untracked_already_connected_is_not_admitted_or_disconnected(backend, l1):
    module, native, _, _, _ = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    engine = native.Hixl.return_value
    engine.connect.side_effect = native.HixlError(native.HIXL_ALREADY_CONNECTED)
    with pytest.raises(native.HixlError):
        context._ensure_connection(context.metadata)
    assert not context._connections
    engine.disconnect.assert_not_called()
    context.close()
    engine.disconnect.assert_not_called()


@pytest.mark.parametrize("failure", ["bind", "start"])
def test_server_start_failure_closes_socket_and_rolls_back_context(
    backend, l1, monkeypatch, failure
):
    module, native, _, state, _ = backend
    assert state.server_class is not None
    monkeypatch.setattr(module, "HixlTransferChannelServer", state.server_class)
    zmq_socket = Mock()
    error = RuntimeError("server startup failed")
    if failure == "bind":
        zmq_socket.bind.side_effect = error
    else:
        monkeypatch.setattr(
            module, "Thread", Mock(return_value=Mock(start=Mock(side_effect=error)))
        )
    monkeypatch.setattr(
        module.zmq.Context,
        "instance",
        Mock(return_value=Mock(socket=Mock(return_value=zmq_socket))),
    )
    with pytest.raises(RuntimeError) as caught:
        module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    assert caught.value is error
    zmq_socket.close.assert_called_once_with(linger=0)
    engine = native.Hixl.return_value
    engine.deregister_mem.assert_called_once_with(0x5678)
    engine.finalize.assert_called_once_with()


def test_disconnect_failure_still_deregisters_and_finalizes(backend, l1, caplog):
    module, native, _, state, _ = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    peer = context.metadata
    context._ensure_connection(peer)
    second = msgspec.structs.replace(
        peer, advertise_url="10.0.0.12:5555", engine_id="10.0.0.12:6000"
    )
    context._ensure_connection(second)
    engine = native.Hixl.return_value
    events = []

    def disconnect(engine_id, timeout_ms):
        assert context._engine_lock.locked() and state.context == 0x4000
        assert timeout_ms == 1000
        events.append(engine_id)
        state.context = 0x9000
        if engine_id == peer.engine_id:
            raise RuntimeError("disconnect failed")
        raise native.HixlError(native.HIXL_NOT_CONNECTED)

    engine.disconnect.side_effect = disconnect
    engine.deregister_mem.side_effect = lambda handle: events.append("deregister")
    engine.finalize.side_effect = lambda: events.append("finalize")
    context.close()
    context.close()
    assert events == [peer.engine_id, second.engine_id, "deregister", "finalize"]
    assert "disconnect failed" in caplog.text
    assert f"native status {native.HIXL_NOT_CONNECTED}" not in caplog.text
    assert not context._connections


def test_inbound_connect_failure_has_no_success_reply(
    backend, local_contexts, monkeypatch
):
    module, _, _, _, _ = backend
    a, engine_a = local_contexts()
    b, engine_b = local_contexts()
    engine_b.connect.side_effect = RuntimeError("reverse connect failed")
    # Load the real upstream timeout class; bypass package activation and the
    # observability service, which are unrelated to the handshake contract.
    upstream = Path(__file__).resolve().parents[3] / "LMCache" / "lmcache"
    before = set(sys.modules)
    for name, path in [("lmcache", upstream), ("lmcache.v1", upstream / "v1")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    bus = ModuleType("lmcache.v1.mp_observability.event_bus")
    bus.is_observability_enabled = lambda: False
    bus.get_event_bus = Mock()
    monkeypatch.setitem(sys.modules, bus.__name__, bus)
    try:
        errors = importlib.import_module("lmcache.v1.mp_observability.errors")
        monkeypatch.setattr(module, "_HANDSHAKE_TIMEOUT_MS", 100)
        request = zmq.Context.instance().socket(zmq.REQ)
        monkeypatch.setattr(
            a, "zmq_context", SimpleNamespace(socket=lambda kind: request)
        )
        with pytest.raises(errors.LMCacheTimeoutError):
            a.get_transfer_channel_client(b.metadata.advertise_url)
        assert request.closed
        assert a.get_num_connected_clients() == b.get_num_connected_clients() == 0
        engine_a.connect.assert_not_called()
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("lmcache."):
                sys.modules.pop(name, None)


def test_request_cleanup_preserves_handshake_error(backend, l1, caplog):
    module, _, _, _, _ = backend
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    error = ValueError("handshake send failed")
    request = Mock()
    request.send.side_effect = error
    request.close.side_effect = RuntimeError("socket close failed")
    context.zmq_context = SimpleNamespace(socket=lambda kind: request)
    try:
        with pytest.raises(ValueError) as caught:
            context.get_transfer_channel_client("10.0.0.12:5555")
        assert caught.value is error
        assert "socket close failed" in caplog.text
        assert context.get_num_connected_clients() == 0
    finally:
        context.close()


@pytest.fixture
def read_addresses(backend, l1, monkeypatch):
    module, native, _, _, _ = backend
    # Load the real public data types without activating upstream device plugins.
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache/lmcache/v1/distributed/transfer_channel/api.py"
    )
    name = "lmcache.v1.distributed.transfer_channel.api"
    spec = importlib.util.spec_from_file_location(name, path)
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, api)
    spec.loader.exec_module(api)
    native.TransferOpDesc = SimpleNamespace
    context = module.HixlTransferChannelContext(l1, LISTEN_URL, ADVERTISE_URL)
    peer = msgspec.structs.replace(
        context.metadata,
        advertise_url="10.0.0.12:5555",
        engine_id="10.0.0.12:6000",
        buffer_base_ptr=0x800005,
        buffer_size=1024,
    )
    # Address preparation needs validated metadata, not a physical connection.
    client = module.HixlTransferChannelClient(context, module._HixlConnection(peer))
    yield context, client, api.TransferChannelAddress
    context.close()


def test_local_address_conversion_preserves_offsets_and_pool_boundary(read_addresses):
    context, _, address = read_addresses
    assert context.get_transfer_channel_address([]) == []
    assert context.get_transfer_channel_address([(0, 1), (4093, 3)]) == [
        address(0, 1),
        address(4093, 3),
    ]
    with pytest.raises(ValueError, match=r"local_addresses\[0\].*range"):
        context.get_transfer_channel_address([(4093, 4)])


@pytest.mark.parametrize("offset,size", [(True, 1), (-1, 1), (0, 0)])
def test_local_address_rejects_invalid_integer_range(read_addresses, offset, size):
    context, _, _ = read_addresses
    with pytest.raises(ValueError, match=r"local_addresses\[0\]"):
        context.get_transfer_channel_address([(offset, size)])


def test_read_descriptors_preserve_pair_order_exact_length_and_raw_bases(
    backend, read_addresses
):
    _, native, _, _, _ = backend
    _, client, address = read_addresses
    local = [address(128, 7), address(0, 3)]
    remote = [address(256, 7), address(64, 3)]
    native.Hixl.return_value.reset_mock()
    descs = client._build_read_descriptors(local, remote)
    assert [(d.local_addr, d.remote_addr, d.len) for d in descs] == [
        (0x100083, 0x800105, 7),
        (0x100003, 0x800045, 3),
    ]
    assert local == [address(128, 7), address(0, 3)]
    assert remote == [address(256, 7), address(64, 3)]
    assert native.Hixl.return_value.mock_calls == []


@pytest.mark.parametrize(
    "local,remote,reason",
    [([], [], "empty"), ([(0, 8)], [], "count"), ([(0, 8)], [(0, 4)], r"pair\[0\]")],
)
def test_read_descriptors_reject_invalid_batch(read_addresses, local, remote, reason):
    _, client, address = read_addresses
    with pytest.raises(ValueError, match=reason):
        client._build_read_descriptors(
            [address(*item) for item in local], [address(*item) for item in remote]
        )


def test_later_remote_range_error_rejects_batch_without_transfer(
    backend, read_addresses
):
    _, native, _, _, _ = backend
    _, client, address = read_addresses
    native.Hixl.return_value.reset_mock()
    with pytest.raises(ValueError, match=r"remote_addresses\[1\].*1024"):
        client._build_read_descriptors(
            [address(32, 128), address(256, 128)],
            [address(0, 128), address(960, 128)],
        )
    assert native.Hixl.return_value.mock_calls == []


@pytest.fixture
def read_tasks(backend, read_addresses):
    _, native, _, _, _ = backend
    context, client, address = read_addresses
    context._connections[client.peer_metadata.engine_id] = client.connection
    context.register_client(client.peer_metadata.advertise_url, client)
    requests = count(0x9000)
    native.Hixl.return_value.transfer_async.side_effect = lambda *a: next(requests)
    return context, client, address


@pytest.mark.parametrize("terminal,success", [("COMPLETED", True), ("FAILED", False)])
def test_read_task_lifecycle_and_close(backend, read_tasks, terminal, success):
    _, native, _, state, engine_lock = backend
    context, client, address = read_tasks
    engine = native.Hixl.return_value
    local = [address(0, 7), address(16, 3)]
    remote = [address(32, 7), address(64, 3)]

    def submit(peer, op, descs):
        assert client._lock.locked() and engine_lock.locked()
        assert state.context == 0x4000
        assert peer == "10.0.0.12:6000" and op == native.TransferOp.READ
        assert [(d.local_addr, d.remote_addr, d.len) for d in descs] == [
            (0x100003, 0x800025, 7),
            (0x100013, 0x800045, 3),
        ]
        state.context = 0x8000
        return 0x9000

    states = iter(
        [native.TransferStatus.WAITING, getattr(native.TransferStatus, terminal)]
    )

    def query(req):
        assert req == 0x9000
        assert client._lock.locked() and engine_lock.locked()
        assert state.context == 0x4000
        state.context = 0x8000
        return next(states)

    engine.transfer_async.side_effect = submit
    engine.get_transfer_status.side_effect = query
    task = client.submit_read(local, remote)
    assert task != 0x9000
    remote.clear()  # The result must use the saved batch, not this mutable list.
    pending = client.query_read_status(task)
    assert not pending.finished and pending.succeeded_mask == []
    result = client.query_read_status(task)
    assert result.finished and result.succeeded_mask == [success, success]
    with pytest.raises(KeyError):
        client.query_read_status(task)
    assert engine.get_transfer_status.call_count == 2
    engine.transfer_async.side_effect = None
    engine.transfer_async.return_value = 0x9001
    next_task = client.submit_read(local, local)
    assert next_task > task
    client.close()
    with pytest.raises(KeyError):
        client.query_read_status(next_task)
    with pytest.raises(RuntimeError, match="closed"):
        client.submit_read(local, local)
    assert engine.transfer_async.call_count == 2
    engine.disconnect.assert_not_called()
    assert state.context == 0x4000


@pytest.mark.parametrize("empty_handle", [False, True])
def test_failed_submission_returns_no_task_or_query_fault(
    backend, read_tasks, empty_handle
):
    _, native, _, _, _ = backend
    _, client, address = read_tasks
    engine = native.Hixl.return_value
    error = native.HixlError(500)
    engine.transfer_async.side_effect = None if empty_handle else error
    engine.transfer_async.return_value = 0
    with pytest.raises(RuntimeError) as caught:
        client.submit_read([address(0, 8)], [address(0, 8)])
    if not empty_handle:
        assert caught.value is error
    assert not client._tasks and client.connection.connected
    engine.disconnect.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_query_fault_affects_shared_record_and_exit_handles_cleanup(
    backend, read_tasks, cleanup_fails, caplog
):
    module, native, _, state, engine_lock = backend
    context, client, address = read_tasks
    engine = native.Hixl.return_value
    sibling = module.HixlTransferChannelClient(context, client.connection)
    other_meta = msgspec.structs.replace(
        client.peer_metadata, engine_id="10.0.0.13:6000", advertise_url="10.0.0.13:5555"
    )
    other = module.HixlTransferChannelClient(
        context, context._ensure_connection(other_meta)
    )
    context.register_client(other_meta.advertise_url, other)
    pair = [address(0, 8)]
    task = client.submit_read(pair, pair)
    affected = sibling.submit_read(pair, pair)
    unaffected = other.submit_read(pair, pair)
    error = native.HixlError(500)

    def query(req):
        state.context = 0x8000
        raise error

    def disconnect(peer, timeout_ms):
        assert engine_lock.locked() and state.context == 0x4000
        assert timeout_ms == 1000
        state.context = 0x8000
        if cleanup_fails:
            raise RuntimeError("fault cleanup failed")

    engine.get_transfer_status.side_effect = query
    engine.disconnect.side_effect = disconnect
    with pytest.raises(native.HixlError) as caught:
        client.query_read_status(task)
    assert caught.value is error
    assert not client.connection.connected
    assert client.connection.failure is not None
    assert client.connection.disconnect_succeeded is not cleanup_fails
    if cleanup_fails:
        assert "fault cleanup failed" in client.connection.disconnect_error
        assert "fault cleanup failed" in caplog.text
        connects = engine.connect.call_count
        with pytest.raises(RuntimeError, match="not reusable"):
            context._ensure_connection(client.peer_metadata)
        assert engine.connect.call_count == connects
    with pytest.raises(KeyError):
        client.query_read_status(task)
    result = sibling.query_read_status(affected)
    assert result.finished and result.succeeded_mask == [False]
    with pytest.raises(KeyError):
        sibling.query_read_status(affected)
    with pytest.raises(RuntimeError):
        sibling.submit_read(pair, pair)
    assert engine.get_transfer_status.call_count == 1
    engine.disconnect.assert_called_once_with("10.0.0.12:6000", timeout_ms=1000)
    assert other.connection.connected
    engine.get_transfer_status.side_effect = None
    engine.get_transfer_status.return_value = native.TransferStatus.COMPLETED
    assert other.query_read_status(unaffected).succeeded_mask == [True]
    # Final exit gets one additional attempt only for unsuccessful fault cleanup.
    engine.disconnect.side_effect = None
    context.close()
    context.close()
    peer_calls = [
        c for c in engine.disconnect.call_args_list if c.args[0] == "10.0.0.12:6000"
    ]
    assert len(peer_calls) == (2 if cleanup_fails else 1)
    assert state.context == 0x4000


def test_acl_preparation_failure_does_not_consume_or_fault_task(
    backend, read_tasks, monkeypatch
):
    _, native, _, _, _ = backend
    _, client, address = read_tasks
    engine = native.Hixl.return_value
    task = client.submit_read([address(0, 8)], [address(0, 8)])
    with monkeypatch.context() as patch:
        patch.setattr(
            native,
            "set_current_context",
            Mock(side_effect=RuntimeError("ACL prepare failed")),
        )
        with pytest.raises(RuntimeError, match="ACL prepare failed"):
            client.query_read_status(task)
    assert task in client._tasks and client.connection.connected
    engine.get_transfer_status.assert_not_called()
    engine.disconnect.assert_not_called()
    engine.get_transfer_status.return_value = native.TransferStatus.COMPLETED
    assert client.query_read_status(task).succeeded_mask == [True]


def test_unsupported_nonwaiting_status_consumes_only_current_task(backend, read_tasks):
    _, native, _, _, _ = backend
    _, client, address = read_tasks
    pair = [address(0, 8)]
    task = client.submit_read(pair, pair)
    engine = native.Hixl.return_value
    engine.get_transfer_status.return_value = native.TransferStatus.TIMEOUT
    with pytest.raises(RuntimeError, match="Unsupported"):
        client.query_read_status(task)
    with pytest.raises(KeyError):
        client.query_read_status(task)
    assert client.connection.connected
    engine.disconnect.assert_not_called()
    engine.get_transfer_status.assert_called_once()


def test_concurrent_queries_consume_native_request_once(backend, read_tasks):
    _, native, _, _, _ = backend
    _, client, address = read_tasks
    task = client.submit_read([address(0, 8)], [address(0, 8)])
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    def query(req):
        entered.set()
        assert release.wait(5)
        return native.TransferStatus.COMPLETED

    def second_query():
        second_started.set()
        return client.query_read_status(task)

    native.Hixl.return_value.get_transfer_status.side_effect = query
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.query_read_status, task)
        try:
            assert entered.wait(5)
            second = pool.submit(second_query)
            assert second_started.wait(5)
        finally:
            release.set()
        assert first.result(timeout=5).succeeded_mask == [True]
        with pytest.raises(KeyError):
            second.result(timeout=5)
    native.Hixl.return_value.get_transfer_status.assert_called_once()


def test_descriptor_count_cannot_truncate_to_uint32(backend, read_addresses):
    _, native, _, _, _ = backend
    _, client, address = read_addresses

    class OversizedBatch(list):
        def __len__(self):
            return 1 << 32

    batch = OversizedBatch([address(0, 8)])
    with pytest.raises(ValueError, match="uint32"):
        client._build_read_descriptors(batch, batch)


def test_explicit_readmission_keeps_old_tasks_on_failed_record(
    backend, read_addresses, local_contexts
):
    module, native, _, state, _ = backend
    _, _, address = read_addresses
    a, engine_a = local_contexts()
    b, engine_b = local_contexts()
    old = a.get_transfer_channel_client(b.metadata.advertise_url)
    sibling = module.HixlTransferChannelClient(a, old.connection)
    requests = count(0x9000)
    engine_a.transfer_async.side_effect = lambda *args: next(requests)
    pair = [address(0, 8)]
    failed = old.submit_read(pair, pair)
    pending = sibling.submit_read(pair, pair)
    engine_a.get_transfer_status.side_effect = native.HixlError(500)
    with pytest.raises(native.HixlError):
        old.query_read_status(failed)
    record = old.connection
    assert record.disconnect_succeeded
    assert a.get_transfer_channel_client(b.metadata.advertise_url) is old
    assert engine_a.connect.call_count == 1
    a.remove_transfer_channel_client(b.metadata.advertise_url)

    def reconnect(engine_id, timeout_ms):
        assert a._engine_lock.locked() and state.context == 0x4000
        assert (engine_id, timeout_ms) == (b.metadata.engine_id, 10000)
        assert a._connections[engine_id] is record
        state.context = 0x9000

    engine_a.connect.side_effect = reconnect
    new = a.get_transfer_channel_client(b.metadata.advertise_url)
    assert old.closed and new is not old
    assert new.connection is not record and new.peer_metadata == old.peer_metadata
    assert a._connections[b.metadata.engine_id] is new.connection
    assert record.failure is not None and not record.connected
    assert sibling.query_read_status(pending).succeeded_mask == [False]
    with pytest.raises(RuntimeError, match="not usable"):
        sibling.submit_read(pair, pair)
    engine_a.get_transfer_status.assert_called_once_with(0x9000)
    engine_a.get_transfer_status.side_effect = None
    engine_a.get_transfer_status.return_value = native.TransferStatus.COMPLETED
    task = new.submit_read(pair, pair)
    assert new.query_read_status(task).succeeded_mask == [True]
    engine_a.get_transfer_status.assert_called_with(0x9002)
    assert engine_a.connect.call_count == 2
    engine_b.connect.assert_called_once()  # Reverse admission reuses its live record.
    a.close()
    assert engine_a.disconnect.call_count == 2  # Fault record, then new connection.
    assert state.context == 0x4000


def test_readmission_rejects_unconfirmed_disconnect(backend, read_tasks):
    _, native, _, _, _ = backend
    context, client, _ = read_tasks
    engine = native.Hixl.return_value
    engine.disconnect.side_effect = native.HixlError(native.HIXL_NOT_CONNECTED)
    with context._engine_lock:
        context._fail_connection_locked(client.connection, native.HixlError(500))
    context.remove_transfer_channel_client(client.peer_metadata.advertise_url)
    diagnostic = client.connection.disconnect_error
    assert diagnostic is not None and not client.connection.disconnect_succeeded
    with pytest.raises(RuntimeError, match="not reusable"):
        context._ensure_connection(client.peer_metadata)
    assert context._connections[client.peer_metadata.engine_id] is client.connection
    assert client.connection.disconnect_error == diagnostic
    engine.connect.assert_not_called()
    engine.disconnect.assert_called_once()  # Admission does not retry cleanup.


def test_failed_readmission_preserves_old_record(backend, read_tasks):
    _, native, _, _, _ = backend
    context, client, _ = read_tasks
    engine = native.Hixl.return_value
    with context._engine_lock:
        context._fail_connection_locked(client.connection, native.HixlError(500))
    context.remove_transfer_channel_client(client.peer_metadata.advertise_url)
    error = native.HixlError(native.HIXL_ALREADY_CONNECTED)
    engine.connect.side_effect = error
    with pytest.raises(native.HixlError) as caught:
        context._ensure_connection(client.peer_metadata)
    assert caught.value is error
    assert context._connections[client.peer_metadata.engine_id] is client.connection
    assert client.connection.failure is not None and not client.connection.connected
    assert client.connection.disconnect_succeeded
    assert context.get_num_connected_clients() == 0
    engine.connect.assert_called_once_with(
        client.peer_metadata.engine_id, timeout_ms=10000
    )
    native.Hixl.return_value.transfer_async.assert_not_called()
