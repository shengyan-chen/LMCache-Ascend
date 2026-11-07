# SPDX-License-Identifier: Apache-2.0
"""Engine lifecycle foundation for the MP HIXL transfer channel.

Not registered with the public factory yet: L1 registration, metadata services
and peer clients must be added before this implements TransferChannelContext.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from threading import Lock
from typing import Any
import json
import logging
import os
import re
import socket

logger = logging.getLogger(__name__)
_ENV_PREFIX = "LMCACHE_HIXL_"
_INT32_MAX = (1 << 31) - 1


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


def _read_config() -> _HixlConfig:
    """Validate configuration before importing native dependencies or using NPU."""
    name = _ENV_PREFIX + "ENGINE_ID"
    engine_id = os.environ.get(name, "")
    match = re.fullmatch(r"(\[[^\]]+\]|[^:]+):([0-9]+)", engine_id)
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
    """Own one engine and borrow the selected device's ACL context.

    This stage only prepares the engine; it is not a usable public Channel.
    Construction retains the selected device on the initializing thread.
    Callers must keep the NPU runtime alive and stop using the engine before
    close; the engine lock does not provide a concurrent shutdown protocol.
    No L1 allocation or borrowed ACL context is owned or destroyed here.
    """

    def __init__(self) -> None:
        self._config = _read_config()
        self._engine_lock = Lock()
        # Only a successfully initialized engine is published on this object.
        self._engine: Any | None = None

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

    def close(self) -> None:
        """Finalize under the engine lock and borrowed context, best effort.

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
            try:
                with use_acl_context(self._acl_context):
                    try:
                        self._engine.finalize()
                    finally:
                        self._engine = None
            except Exception:
                logger.exception(
                    "Failed to finalize MP HIXL engine %s", self._config.engine_id
                )
