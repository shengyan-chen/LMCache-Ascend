# SPDX-License-Identifier: Apache-2.0
"""MP HIXL lifecycle, peer admission and asynchronous READ task management."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4
import ctypes
import json
import logging
import os
import re
import socket

# Third Party
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
import msgspec
import zmq

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.distributed.transfer_channel.api import (
        TransferChannelAddress,
        TransferChannelReadResult,
    )

    # First Party
    from lmcache_ascend.hixl_npu_comms import TransferOpDesc

logger = logging.getLogger(__name__)
_ENV_PREFIX = "LMCACHE_HIXL_"
_HANDSHAKE_TIMEOUT_MS = 60000
_DISCONNECT_TIMEOUT_MS = 1000
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_UINTPTR_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


class HixlChannelMetadata(msgspec.Struct, frozen=True):
    """Immutable registered range and identity; contains no native handles."""

    advertise_url: str
    engine_id: str
    buffer_base_ptr: int
    buffer_size: int
    instance_id: str


class InitReq(msgspec.Struct, tag=True):
    metadata: HixlChannelMetadata


class InitResp(msgspec.Struct, tag=True):
    metadata: HixlChannelMetadata


HandshakeMsg = InitReq | InitResp


def _validate_memory_range(ptr: int, size: int) -> None:
    # Do not impose allocator alignment or substitute a stable prefix.
    if (
        type(ptr) is not int
        or type(size) is not int
        or not 0 < ptr <= _UINTPTR_MAX
        or not 0 < size <= _SIZE_MAX
        or size > _UINTPTR_MAX - ptr
    ):
        raise ValueError(f"Invalid Host memory range: ptr={ptr!r}, size={size!r}")


def _validate_address_range(
    offset: int, size: int, buffer_size: int, *, label: str
) -> None:
    """Check a subrange of an already validated, native-representable pool."""
    if type(offset) is not int or type(size) is not int:
        raise ValueError(f"{label}: offset and size must be integers, not bool")
    if offset < 0 or size <= 0:
        raise ValueError(
            f"{label}: require offset >= 0 and size > 0, got {offset}, {size}"
        )
    if offset > buffer_size or size > buffer_size - offset:
        raise ValueError(
            f"{label}: range offset={offset}, size={size} "
            f"exceeds buffer_size={buffer_size}"
        )


def _parse_channel_url(url: str) -> tuple[str, int]:
    """Accept Channel host:port, optionally prefixed by tcp://; do not resolve."""
    match = (
        re.fullmatch(r"(?:tcp://)?(\[[^\]]+\]|[^:/\s?#@]+):([0-9]+)", url)
        if isinstance(url, str)
        else None
    )
    try:
        if match is None:
            raise ValueError("expected host:port or tcp://host:port")
        host, port = match.groups()
        if host.startswith("["):
            socket.inet_pton(socket.AF_INET6, host[1:-1])
        number = int(port)
        if not 1 <= number <= 65535:
            raise ValueError("port must be between 1 and 65535")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid Channel address {url!r}: {exc}") from exc
    return host, number


def validate_metadata(metadata: HixlChannelMetadata) -> None:
    """Validate a typed MessagePack payload before peer use or registration.

    Typed decoding checks wire types; these checks also apply to locally built
    structs. Validation does not prove peer liveness or registration validity.
    """
    _parse_channel_url(metadata.advertise_url)
    _validate_engine_id(metadata.engine_id)
    _validate_memory_range(metadata.buffer_base_ptr, metadata.buffer_size)
    value = metadata.instance_id
    if not isinstance(value, str):
        raise ValueError("instance_id must be a canonical UUID string")
    try:
        if str(UUID(value)) != value:
            raise ValueError("expected lowercase hyphenated UUID")
    except ValueError as exc:
        raise ValueError(f"Invalid instance_id {value!r}: {exc}") from exc


@dataclass(frozen=True)
class _HixlConfig:
    engine_id: str
    device_id: int
    connect_timeout_ms: int
    comm_listen_port: int | None

    def initialize_options(self) -> dict[str, str]:
        comm: dict[str, str | int] = {
            "comm_resource_config.protocol_desc": "roce:device",
        }
        if self.comm_listen_port is not None:
            comm["comm_resource_config.listen_port"] = self.comm_listen_port
        return {
            "LocalCommRes": '{"version":"1.3"}',
            "BufferPool": "0:0",
            "AutoConnect": "0",
            "GlobalResourceConfig": json.dumps(comm),
        }


def _read_int(suffix: str, default: str, minimum: int, maximum: int) -> int:
    name = _ENV_PREFIX + suffix
    value = os.environ.get(name, default)
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{name} must be a decimal integer, got {value!r}")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value!r}"
        )
    return number


def _validate_engine_id(engine_id: str, name: str = "engine_id") -> None:
    match = (
        re.fullmatch(r"(\[[^\]]+\]|[^:]+):([0-9]+)", engine_id)
        if isinstance(engine_id, str)
        else None
    )
    try:
        if match is None:
            raise ValueError("expected IP:port or [IPv6]:port")
        host, port = match.groups()
        if host.startswith("["):
            socket.inet_pton(socket.AF_INET6, host[1:-1])
        else:
            socket.inet_pton(socket.AF_INET, host)
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid {name} {engine_id!r}: {exc}") from exc


def _read_config() -> _HixlConfig:
    """Validate configuration before importing native dependencies or using NPU."""
    name = _ENV_PREFIX + "ENGINE_ID"
    engine_id = os.environ.get(name, "")
    _validate_engine_id(engine_id, name)
    return _HixlConfig(
        engine_id=engine_id,
        device_id=_read_int("DEVICE_ID", "0", 0, _INT32_MAX),
        connect_timeout_ms=_read_int("CONNECT_TIMEOUT_MS", "10000", 1, _INT32_MAX),
        comm_listen_port=(
            _read_int("COMM_LISTEN_PORT", "", 1, 65535)
            if _ENV_PREFIX + "COMM_LISTEN_PORT" in os.environ
            else None
        ),
    )


@dataclass
class _HixlConnection:
    """Context-owned successful connection, independent of logical Clients."""

    metadata: HixlChannelMetadata
    connected: bool = True  # Admission state, not a physical liveness probe.
    failure: str | None = None
    disconnect_succeeded: bool = False
    disconnect_error: str | None = None
    close_attempted: bool = False


class HixlTransferChannelClient(TransferChannelClient):
    """Borrow a shared connection and retain its immutable peer snapshot.

    Tasks belong to this Client; a shared connection fault affects all of its
    borrowers. Closing clears local tasks without disconnecting the connection.
    """

    def __init__(
        self, context: HixlTransferChannelContext, connection: _HixlConnection
    ) -> None:
        self._context = context
        self.connection = connection
        self.peer_metadata = connection.metadata
        self._lock = Lock()
        self._closed = False
        self._task_counter = 0
        self._tasks: dict[int, tuple[int, list[TransferChannelAddress]]] = {}

    def _build_read_descriptors(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> list[TransferOpDesc]:
        """Validate a whole READ batch and describe exact remote-to-local ranges.

        Pools were validated at registration/admission, so a contained subrange
        also fits uintptr_t/size_t. No alignment, copying, native engine calls
        or ACL scope is needed here. The later submit path checks connection
        usability and retains these descriptors through the native call.
        """
        if len(local_addresses) != len(remote_addresses):
            raise ValueError("Local and remote address count must match")
        if not local_addresses:
            raise ValueError("READ address batch must not be empty")
        if len(local_addresses) > _UINT32_MAX:
            raise ValueError("READ descriptor count exceeds uint32 representation")

        # First Party
        from lmcache_ascend.hixl_npu_comms import TransferOpDesc

        local_pool = self._context.metadata
        remote_pool = self.peer_metadata
        descs = []
        for index, (local, remote) in enumerate(
            zip(local_addresses, remote_addresses, strict=True)
        ):
            _validate_address_range(
                local.offset,
                local.size,
                local_pool.buffer_size,
                label=f"local_addresses[{index}]",
            )
            _validate_address_range(
                remote.offset,
                remote.size,
                remote_pool.buffer_size,
                label=f"remote_addresses[{index}]",
            )
            if local.size != remote.size:
                raise ValueError(
                    f"pair[{index}]: local size {local.size} "
                    f"!= remote size {remote.size}"
                )
            descs.append(
                TransferOpDesc(
                    local_addr=local_pool.buffer_base_ptr + local.offset,
                    remote_addr=remote_pool.buffer_base_ptr + remote.offset,
                    len=local.size,
                )
            )
        return descs

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit one batch and return a Client-local ID after registration.

        No retry or reconnect is performed. A submission error does not prove
        that no data was accessed. Preparing the snapshot before the native call
        reduces Python work after submission, but task insertion can still fail;
        this path does not provide transactional registration or request drain.
        """
        # First Party
        from lmcache_ascend.v1.hixl_runtime import use_acl_context
        import lmcache_ascend.hixl_npu_comms as hixl_comms

        remote_snapshot = list(remote_addresses)
        descs = self._build_read_descriptors(local_addresses, remote_snapshot)
        with self._lock:
            if self._closed:
                raise RuntimeError("HIXL Client is closed")
            with self._context._engine_lock:
                self._context._check_connection_locked(self.connection)
                assert self._context._engine is not None
                with use_acl_context(self._context._acl_context):
                    req = self._context._engine.transfer_async(
                        self.peer_metadata.engine_id, hixl_comms.TransferOp.READ, descs
                    )
                if req == 0:
                    raise RuntimeError("HIXL TransferAsync returned a null request")
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (req, remote_snapshot)
            return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """Query once; consume terminal tasks and never reuse a failed native req."""
        # Third Party
        from lmcache.v1.distributed.transfer_channel.api import (
            TransferChannelReadResult,
        )

        # First Party
        from lmcache_ascend.v1.hixl_runtime import use_acl_context
        import lmcache_ascend.hixl_npu_comms as hixl_comms

        with self._lock:
            req, addresses = self._tasks[task_id]
            with self._context._engine_lock:
                if self.connection.failure is not None:
                    del self._tasks[task_id]
                    return TransferChannelReadResult(True, [False] * len(addresses))
                self._context._check_connection_locked(self.connection)
                assert self._context._engine is not None
                with use_acl_context(self._context._acl_context):
                    try:
                        status = self._context._engine.get_transfer_status(req)
                    except hixl_comms.HixlError as error:
                        # Catch only the query call, not ACL scope preparation.
                        del self._tasks[task_id]
                        self._context._fail_connection_locked(self.connection, error)
                        raise
            if status == hixl_comms.TransferStatus.WAITING:
                return TransferChannelReadResult(False, [])
            # HIXL removes the req index for every non-WAITING status, including
            # unsupported ones. Never leave that request available for requery.
            del self._tasks[task_id]
            if status == hixl_comms.TransferStatus.COMPLETED:
                return TransferChannelReadResult(True, [True] * len(addresses))
            if status == hixl_comms.TransferStatus.FAILED:
                return TransferChannelReadResult(True, [False] * len(addresses))
            raise RuntimeError(f"Unsupported HIXL transfer status: {status!r}")

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Invalidate local task IDs without cancelling or draining native reads.

        The caller must end use of this Client. Shared connection and engine
        cleanup belong to the Context; this method does not disconnect peers.
        """
        with self._lock:
            self._closed = True
            # This invalidates IDs, not native accesses. Connection/engine
            # cleanup owns remaining native resources; no per-request free.
            self._tasks.clear()


class HixlTransferChannelServer(TransferChannelServer):
    """Exchange metadata; complete reverse admission before replying success."""

    def __init__(self, context: HixlTransferChannelContext, listen_url: str) -> None:
        self._context = context
        self._running = False
        host, port = _parse_channel_url(listen_url)
        self._socket = context.zmq_context.socket(zmq.REP)
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            if host.startswith("["):
                self._socket.setsockopt(zmq.IPV6, 1)
            self._socket.bind(f"tcp://{host}:{port}")
            self._thread = Thread(
                target=self._serve, name="tc-hixl-server", daemon=True
            )
            self._running = True
            self._thread.start()
        except BaseException:
            self._running = False
            try:
                self._socket.close(linger=0)
            except Exception:
                logger.exception("Failed to close HIXL handshake socket")
            raise

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                if self._socket not in dict(poller.poll(1000)):
                    continue
                message = msgspec.msgpack.decode(self._socket.recv(), type=HandshakeMsg)
                if not isinstance(message, InitReq):
                    raise ValueError("Expected HIXL InitReq")
                connection = self._context._ensure_connection(message.metadata)
                client = HixlTransferChannelClient(self._context, connection)
                self._context.register_client(message.metadata.advertise_url, client)
                self._socket.send(
                    msgspec.msgpack.encode(InitResp(self._context.metadata))
                )
            except Exception:
                if self._running:
                    # Match the existing REP failure policy: log, no InitError
                    # or socket recreation. The caller's receive times out.
                    logger.exception("HIXL handshake request failed")

    def close(self) -> None:
        self._running = False
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=5)
        finally:
            self._socket.close(linger=0)


class HixlTransferChannelContext(TransferChannelContext):
    """Own one engine and full L1 registration; borrow memory and ACL context.

    Clients submit and query READs through this shared engine.
    The L1 owner must provide a live HIXL-compatible Host allocation and retain
    it through deregistration.
    Construction retains the selected device on the initializing thread.
    Callers must keep the NPU runtime alive and stop using the engine before
    close; the engine lock does not provide a concurrent shutdown protocol.
    No L1 allocation or borrowed ACL context is owned or destroyed here.
    """

    def __init__(
        self, l1_memory_desc: L1MemoryDesc, listen_url: str, advertise_url: str
    ) -> None:
        self._config = _read_config()
        _validate_memory_range(l1_memory_desc.ptr, l1_memory_desc.size)
        _parse_channel_url(advertise_url)
        _parse_channel_url(listen_url)
        self._l1_memory_desc = l1_memory_desc
        self._engine_lock = Lock()
        # Only a successfully initialized engine is published on this object.
        self._engine: Any | None = None
        self._mem_handle: int | None = None
        self._connections: dict[str, _HixlConnection] = {}
        self._clients_lock = Lock()
        self._clients: dict[str, HixlTransferChannelClient] = {}
        self._server: HixlTransferChannelServer | None = None
        self.zmq_context = zmq.Context.instance()

        # First Party
        from lmcache_ascend.v1.hixl_runtime import (
            prepare_acl_context,
            use_acl_context,
        )

        try:
            # First Party
            import lmcache_ascend.hixl_npu_comms as hixl_comms
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Failed to load the HIXL native extension "
                "lmcache_ascend.hixl_npu_comms; check CANN version detection "
                "during the LMCache-Ascend build, confirm the HIXL extension "
                "was built, and make its runtime libraries available"
            ) from exc

        options = self._config.initialize_options()
        self._acl_context = prepare_acl_context(self._config.device_id)
        with self._engine_lock, use_acl_context(self._acl_context):
            engine = hixl_comms.Hixl()
            try:
                engine.initialize(self._config.engine_id, options)
            except BaseException:
                # Drop our reference inside the ACL scope, even on interruption.
                # The native destructor owns its cleanup; this does not prove
                # that Initialize's internal partial resources were reclaimed.
                engine = None
                raise
            self._engine = engine
            del engine  # Let cleanup release our only owning reference in scope.

        try:
            # Initialize can switch ACL contexts: enter a fresh scope for each
            # native call, including subsequent deregistration and Finalize.
            with self._engine_lock, use_acl_context(self._acl_context):
                handle = self._engine.register_mem(
                    l1_memory_desc.ptr, l1_memory_desc.size, hixl_comms.MEM_HOST
                )
                if handle == 0:
                    raise RuntimeError("HIXL RegisterMem returned a null handle")
                self._mem_handle = handle
            self.metadata = HixlChannelMetadata(
                advertise_url=advertise_url,
                engine_id=self._config.engine_id,
                buffer_base_ptr=l1_memory_desc.ptr,
                buffer_size=l1_memory_desc.size,
                instance_id=str(uuid4()),
            )
            # All state used by incoming handshakes is ready before start().
            self._server = HixlTransferChannelServer(self, listen_url)
        except BaseException:
            self.close()
            raise

    def get_transfer_channel_address(
        self, lmcache_addresses: list[tuple[int, int]]
    ) -> list[TransferChannelAddress]:
        """Validate local L1 ranges and retain byte offsets in the public type.

        An empty list produces no addresses. This creates no allocation,
        registration or object lock and does not prove current accessibility.
        """
        # Third Party
        from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress

        addresses = []
        for index, (offset, size) in enumerate(lmcache_addresses):
            _validate_address_range(
                offset,
                size,
                self.metadata.buffer_size,
                label=f"local_addresses[{index}]",
            )
            addresses.append(TransferChannelAddress(offset=offset, size=size))
        return addresses

    def get_transfer_channel_server(self) -> HixlTransferChannelServer:
        if self._server is None:
            raise RuntimeError("HIXL metadata server is not available")
        return self._server

    def get_transfer_channel_client(
        self, peer_advertise_url: str
    ) -> HixlTransferChannelClient:
        with self._clients_lock:
            client = self._clients.get(peer_advertise_url)
            if client is not None and not client.closed:
                return client
        # No Client/engine lock spans the metadata network wait.
        client = self._connect(peer_advertise_url)
        return self.register_client(peer_advertise_url, client)

    def get_num_connected_clients(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def register_client(
        self, key: str, client: HixlTransferChannelClient
    ) -> HixlTransferChannelClient:
        """Keep a consistent live Client; close discarded objects outside the lock.

        Connection consistency is established during admission. This final
        collection update does not recheck connection currentness atomically.
        """
        conflict = False
        with self._clients_lock:
            existing = self._clients.get(key)
            if existing is client:
                return client
            if existing is None:
                self._clients[key] = client
                return client
            if existing.peer_metadata != client.peer_metadata:
                conflict = True
                discarded = client
            elif existing.closed:
                self._clients[key] = client
                discarded = existing
                existing = client
            else:
                discarded = client
        try:
            discarded.close()
        except Exception:
            logger.exception("Failed to close discarded HIXL Client for %s", key)
        if conflict:
            raise ValueError(f"Conflicting HIXL peer metadata for {key!r}")
        return existing

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        with self._clients_lock:
            client = self._clients.pop(peer_advertise_url, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.exception(
                    "Failed to close HIXL Client for %s", peer_advertise_url
                )

    def _check_connection_locked(self, record: _HixlConnection) -> None:
        """Require the same usable record; caller holds the engine lock."""
        if self._engine is None:
            raise RuntimeError("HIXL engine is closed")
        if (
            not record.connected
            or self._connections.get(record.metadata.engine_id) is not record
        ):
            raise RuntimeError("HIXL peer connection is not usable")

    def _disconnect_connection_locked(self, record: _HixlConnection) -> None:
        """Attempt owned cleanup in a fresh ACL scope; retain diagnostics.

        Caller holds the engine lock and selects the attempt (fault or final
        exit). NOT_CONNECTED is benign for this attempt, but does not certify
        historical cleanup or make this record eligible for recovery.
        """
        # First Party
        from lmcache_ascend.v1.hixl_runtime import use_acl_context
        import lmcache_ascend.hixl_npu_comms as hixl_comms

        try:
            if self._engine is None:
                raise RuntimeError("HIXL engine is closed")
            with use_acl_context(self._acl_context):
                self._engine.disconnect(
                    record.metadata.engine_id, timeout_ms=_DISCONNECT_TIMEOUT_MS
                )
            record.disconnect_succeeded = True
        except Exception as error:
            if record.disconnect_error is None:
                record.disconnect_error = f"{type(error).__name__}: {error}"
            if not (
                isinstance(error, hixl_comms.HixlError)
                and error.code == hixl_comms.HIXL_NOT_CONNECTED
            ):
                logger.exception(
                    "Failed to disconnect HIXL peer %s", record.metadata.engine_id
                )

    def _fail_connection_locked(
        self, record: _HixlConnection, error: Exception
    ) -> None:
        """Invalidate one record and attempt cleanup once, under the engine lock.

        No other Client lock is acquired. Its tasks observe this record on their
        next query. Store text, not an exception traceback retaining resources.
        """
        if record.failure is not None:
            return
        record.failure = f"{type(error).__name__}: {error}"
        record.connected = False
        if (
            self._engine is None
            or self._connections.get(record.metadata.engine_id) is not record
        ):
            record.disconnect_error = "Connection is no longer owned by this engine"
            return
        self._disconnect_connection_locked(record)

    def _ensure_connection(self, metadata: HixlChannelMetadata) -> _HixlConnection:
        """Reuse or explicitly reconnect the same peer after confirmed cleanup.

        Failed records stay with old Clients and tasks. Publish a fresh record
        only after Connect succeeds; never replace a different peer instance.
        Callers coordinate old Client removal; this is not a holder/drain check.
        """
        # First Party
        from lmcache_ascend.v1.hixl_runtime import use_acl_context

        validate_metadata(metadata)
        with self._engine_lock:
            if self._engine is None:
                raise RuntimeError("HIXL engine is closed")
            for record in self._connections.values():
                if (
                    record.metadata.engine_id == metadata.engine_id
                    or record.metadata.advertise_url == metadata.advertise_url
                ):
                    if record.metadata != metadata:
                        raise ValueError("Conflicting HIXL peer metadata")
                    if record.connected:
                        return record
                    if record.failure is None or not record.disconnect_succeeded:
                        raise RuntimeError("HIXL peer connection is not reusable")
                    # Keep the old fault/cleanup evidence if Connect fails.
                    # Never revive it: other Clients may still hold old tasks.
                    break
            record = _HixlConnection(metadata)
            with use_acl_context(self._acl_context):
                self._engine.connect(
                    metadata.engine_id, timeout_ms=self._config.connect_timeout_ms
                )
            # ALREADY_CONNECTED cannot certify a new or recovered connection;
            # known reusable records returned above without another native call.
            self._connections[metadata.engine_id] = record
            return record

    def _connect(self, server_url: str) -> HixlTransferChannelClient:
        host, port = _parse_channel_url(server_url)
        request = self.zmq_context.socket(zmq.REQ)
        try:
            request.setsockopt(zmq.LINGER, 0)
            request.setsockopt(zmq.RCVTIMEO, _HANDSHAKE_TIMEOUT_MS)
            if host.startswith("["):
                request.setsockopt(zmq.IPV6, 1)
            request.connect(f"tcp://{host}:{port}")
            request.send(msgspec.msgpack.encode(InitReq(self.metadata)))
            try:
                reply = request.recv()
            except zmq.Again as exc:
                # Third Party
                from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

                raise LMCacheTimeoutError(
                    f"Timed out after {_HANDSHAKE_TIMEOUT_MS / 1000:g}s waiting "
                    f"for a HIXL handshake reply from {server_url!r}"
                ) from exc
            response = msgspec.msgpack.decode(reply, type=HandshakeMsg)
            if not isinstance(response, InitResp):
                raise ValueError("Expected HIXL InitResp")
            connection = self._ensure_connection(response.metadata)
            return HixlTransferChannelClient(self, connection)
        finally:
            try:
                request.close(linger=0)
            except Exception:
                logger.exception("Failed to close HIXL handshake request socket")

    def close(self) -> None:
        """Stop Server/Clients, disconnect, deregister and finalize, best effort.

        Log cleanup errors without replacing a pending construction error.
        After attempting Finalize, release our engine reference in the same ACL
        scope. If scope entry fails, retain it for diagnosis; do not call native
        cleanup in an unprepared context. No automatic retry is performed.
        """
        # First Party
        from lmcache_ascend.v1.hixl_runtime import use_acl_context

        with self._clients_lock:
            server = self._server
            self._server = None
            clients = list(self._clients.values())
            self._clients.clear()
        # Never hold the engine lock while joining a thread that may need it.
        if server is not None:
            try:
                server.close()
            except Exception:
                logger.exception("Failed to close HIXL metadata server")
        for client in clients:
            try:
                client.close()
            except Exception:
                logger.exception("Failed to close HIXL Client")

        with self._engine_lock:
            if self._engine is None:
                return
            for record in self._connections.values():
                if record.disconnect_succeeded or record.close_attempted:
                    continue
                # A failed/uncertain fault cleanup gets one final exit attempt.
                # Sequential close does not replay that attempt if Finalize fails.
                record.close_attempted = True
                record.connected = False
                self._disconnect_connection_locked(record)
            if self._mem_handle is not None:
                try:
                    with use_acl_context(self._acl_context):
                        try:
                            self._engine.deregister_mem(self._mem_handle)
                        finally:
                            self._mem_handle = None
                except Exception:
                    logger.exception(
                        "Failed to deregister MP HIXL L1 range at %s",
                        self._l1_memory_desc.ptr,
                    )
            try:
                with use_acl_context(self._acl_context):
                    try:
                        self._engine.finalize()
                    finally:
                        self._mem_handle = None
                        self._engine = None
                        self._connections.clear()
            except Exception:
                logger.exception(
                    "Failed to finalize MP HIXL engine %s", self._config.engine_id
                )
