# SPDX-License-Identifier: Apache-2.0
"""Engine, full L1 registration and metadata for the MP HIXL transfer channel.

Not registered with the public factory yet: metadata services and peer clients
must be added before this implements TransferChannelContext.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4
import ctypes
import json
import logging
import os
import re
import socket

# Third Party
import msgspec

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

logger = logging.getLogger(__name__)
_ENV_PREFIX = "LMCACHE_HIXL_"
_INT32_MAX = (1 << 31) - 1
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


class HixlTransferChannelContext:
    """Own one engine and full L1 registration; borrow memory and ACL context.

    This stage prepares local metadata, but does not open a handshake server.
    It is not yet a usable public Channel. The L1 owner must provide a live,
    HIXL-compatible Host allocation and retain it through deregistration.
    Construction retains the selected device on the initializing thread.
    Callers must keep the NPU runtime alive and stop using the engine before
    close; the engine lock does not provide a concurrent shutdown protocol.
    No L1 allocation or borrowed ACL context is owned or destroyed here.
    """

    def __init__(self, l1_memory_desc: L1MemoryDesc, advertise_url: str) -> None:
        self._config = _read_config()
        _validate_memory_range(l1_memory_desc.ptr, l1_memory_desc.size)
        _parse_channel_url(advertise_url)
        self._l1_memory_desc = l1_memory_desc
        self._engine_lock = Lock()
        # Only a successfully initialized engine is published on this object.
        self._engine: Any | None = None
        self._mem_handle: int | None = None

        # First Party
        from lmcache_ascend.v1.hixl_runtime import (
            prepare_acl_context,
            use_acl_context,
        )
        import lmcache_ascend.hixl_npu_comms as hixl_comms

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
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Deregister then finalize under the engine lock, best effort.

        Log cleanup errors without replacing a pending construction error.
        After attempting Finalize, release our engine reference in the same ACL
        scope. If scope entry fails, retain it for diagnosis; do not call native
        cleanup in an unprepared context. No automatic retry is performed.
        """
        # First Party
        from lmcache_ascend.v1.hixl_runtime import use_acl_context

        with self._engine_lock:
            if self._engine is None:
                return
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
            except Exception:
                logger.exception(
                    "Failed to finalize MP HIXL engine %s", self._config.engine_id
                )
