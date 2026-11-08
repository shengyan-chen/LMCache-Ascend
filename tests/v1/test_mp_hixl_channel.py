# SPDX-License-Identifier: Apache-2.0
"""CPU checks of MP engine ownership; only NPU/native boundaries are replaced."""

# Standard
from concurrent.futures import ThreadPoolExecutor
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
    # Removing a logical Client must not permit a restarted peer to replace
    # retained physical connection identity, including a changed engine ID.
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
