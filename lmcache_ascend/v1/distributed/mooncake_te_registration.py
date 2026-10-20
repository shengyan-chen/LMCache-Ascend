# SPDX-License-Identifier: Apache-2.0
"""Opt-in ownership handoff for MP Host L1 -> traditional TE HIXL_CS.

This module has no accelerator imports at module scope. The server launcher
installs the adapter after LMCache/plugin initialization has completed.
"""

# Standard
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, Callable
import json
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)
ENV_ENABLE = "LMCACHE_ASCEND_MP_TE_UNMAP_HOST"
_active = False
_installed = False


def is_enabled() -> bool:
    """Return whether the experimental handoff was explicitly requested."""
    value = os.environ.get(ENV_ENABLE, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_ENABLE} must be 0 or 1")
    return value == "1"


def is_active() -> bool:
    """Return whether the actual MP server configuration has been validated."""
    return _active


def validate_config(config: Any) -> None:
    """Reject configurations outside the first HIXL_CS validation scope."""
    if config.supported_transfer_mode != "lmcache_driven":
        raise ValueError("Host handoff requires supported_transfer_mode=lmcache_driven")
    if config.engine_type != "default":
        raise ValueError("Host handoff currently supports engine_type=default only")
    if (
        not config.p2p_config.enabled
        or config.p2p_config.transfer_engine != "mooncake_te"
    ):
        raise ValueError("Host handoff requires enabled MP P2P with mooncake_te")
    expected = {
        "MC_TE_PROTOCOL": "ascend",
        "ASCEND_BUFFER_POOL": "0:0",
        "HCCL_INTRA_ROCE_ENABLE": "1",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError(f"Host handoff requires {name}={value}")
    resource = json.loads(os.environ.get("ASCEND_LOCAL_COMM_RES", "{}"))
    if not isinstance(resource, dict) or resource.get("version") != "1.3":
        raise ValueError('Host handoff requires ASCEND_LOCAL_COMM_RES version "1.3"')
    # These switches are presence-based in the tested TE, even with value "0".
    for name in (
        "MC_USE_TENT",
        "MC_USE_TEV1",
        "MC_FORCE_TCP",
        "MC_FORCE_HCA",
        "ASCEND_USE_ASYNC_TRANSFER",
        "ASCEND_ENABLE_USE_FABRIC_MEM",
    ):
        if name in os.environ:
            raise ValueError(f"Host handoff requires {name} to be unset")


@dataclass
class Pool:
    size: int
    device: int
    state: str = "allocated"
    # Retain the native engine on uncertain failures rather than invoking its
    # destructor while its registration state is unknown.
    owner: Any = None


class PoolRegistry:
    """Track only allocator-owned base pointers and gate their physical release."""

    def __init__(self) -> None:
        self._pools: dict[int, Pool] = {}
        self.lock = threading.RLock()

    def track(self, ptr: int, size: int, device: int) -> None:
        """Record one live allocation without changing its mapping."""
        with self.lock:
            if ptr in self._pools:
                raise RuntimeError(f"Host pool already tracked: {ptr:#x}")
            self._pools[ptr] = Pool(size, device)

    def region(self, ptr: int, size: int) -> Pool:
        """Require an allocator base address and a region within its capacity."""
        with self.lock:
            if ptr not in self._pools:
                raise ValueError(f"Untracked Host L1 base: {ptr:#x}")
            pool = self._pools[ptr]
            if size <= 0 or size > pool.size:
                raise ValueError("Host registration exceeds tracked L1 allocation")
            return pool

    def check_free(self, ptr: int) -> None:
        """Refuse to free memory with live or uncertain communication ownership."""
        with self.lock:
            pool = self._pools.get(ptr)
            if pool is not None and pool.state not in ("allocated", "released"):
                raise RuntimeError(
                    f"Refusing to free Host L1 {ptr:#x}: state={pool.state}; "
                    "stop traffic and restart the server process after diagnosis"
                )

    def forget(self, ptr: int) -> None:
        """Remove a record only after its allocation has been freed."""
        with self.lock:
            self.check_free(ptr)
            self._pools.pop(ptr, None)


POOLS = PoolRegistry()


class RegistrationAdapter:
    """Delegate TE operations while enforcing Host registration ownership.

    Failed/timed-out reads quarantine the allocation: TE failure status alone is
    not treated as proof that DMA stopped. Normal teardown requires quiescent
    peers; this adapter cannot observe inbound reads from other processes.
    """

    def __init__(
        self,
        engine: Any,
        ops: Any,
        pools: PoolRegistry,
        device_scope: Callable,
    ) -> None:
        self._engine = engine
        self._ops = ops
        self._pools = pools
        self._device_scope = device_scope
        self._ptr: int | None = None
        self._released = False
        self._pending: set[int] = set()
        self._read_failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def register_memory(self, ptr: int, size: int) -> int:
        """Unmap the LMCache-owned Host region, then register it with TE."""
        with self._pools.lock:
            pool = self._pools.region(ptr, size)
            if pool.state != "allocated" or self._ptr is not None:
                raise RuntimeError("Host L1 handoff cannot be repeated")
            pool.owner = self
            pool.state = "handoff"
            self._ptr = ptr
            started = time.monotonic()
            try:
                with self._device_scope(pool.device):
                    mapped = self._ops.get_device_ptr(ptr)
                    logger.info(
                        "MP Host handoff ptr=%#x size=%d device=%d mapping=%s",
                        ptr,
                        size,
                        pool.device,
                        mapped,
                    )
                    if mapped:
                        ret = self._ops.unregister_ptr(ptr)
                        logger.info("MP Host unregister_ptr ret=%s", ret)
                        if ret != 0:
                            raise RuntimeError(f"unregister_ptr failed: ret={ret}")
                        if self._ops.get_device_ptr(ptr):
                            raise RuntimeError("unregister_ptr left a mapping record")
                    ret = self._engine.register_memory(ptr, size)
                    if ret != 0:
                        raise RuntimeError(f"TE register_memory failed: ret={ret}")
                pool.state = "registered"
            except Exception:
                # Native calls may leave partial state. Never restore mappings
                # or free the allocation speculatively.
                pool.state = "failed"
                logger.exception("MP Host handoff failed; allocation quarantined")
                raise
            logger.info(
                "MP Host TE registration succeeded ptr=%#x size=%d seconds=%.3f",
                ptr,
                size,
                time.monotonic() - started,
            )
            return 0

    def batch_transfer_async_read(self, *args: Any) -> int:
        """Track submitted reads so close cannot unregister an active target."""
        with self._pools.lock:
            if (
                self._ptr is None
                or self._pools.region(self._ptr, 1).state != "registered"
            ):
                raise RuntimeError("Host L1 is not registered")
            if self._read_failed:
                raise RuntimeError("Host L1 has an uncertain failed transfer")
            try:
                batch = self._engine.batch_transfer_async_read(*args)
            except Exception:
                self._read_failed = True
                raise
            if batch < 0 or batch >= (1 << 63):
                self._read_failed = True
                raise RuntimeError(f"Invalid TE batch ID: {batch}")
            self._pending.add(batch)
            return batch

    def transfer_check_status(self, batch: int) -> int:
        """Only successful completion proves the tracked read finished normally."""
        with self._pools.lock:
            try:
                status = self._engine.transfer_check_status(batch)
            except Exception:
                self._read_failed = True
                raise
            if status == 1:
                self._pending.discard(batch)
            elif status != 0:
                self._read_failed = True
            return status

    def unregister_memory(self, ptr: int) -> int:
        """Release communication registration; retain memory on any uncertainty."""
        with self._pools.lock:
            if ptr != self._ptr:
                raise ValueError("Unregister address does not match the handed-off L1")
            if self._released:
                return 0
            pool = self._pools.region(ptr, 1)
            if pool.state == "released":
                return 0
            if self._pending or self._read_failed:
                raise RuntimeError("Cannot unregister L1 with in-flight/failed reads")
            if pool.state != "registered":
                raise RuntimeError(f"Cannot unregister L1 in state {pool.state}")
            try:
                with self._device_scope(pool.device):
                    ret = self._engine.unregister_memory(ptr)
                if ret != 0:
                    raise RuntimeError(f"TE unregister_memory failed: ret={ret}")
            except Exception:
                pool.state = "failed"
                raise
            pool.state = "released"
            pool.owner = None
            self._released = True
            logger.info("MP Host TE unregistered ptr=%#x", ptr)
            return 0


def install() -> None:
    """Install opt-in hooks after the LMCache/plugin import has completed.

    Only the MP module's local TransferEngine constructor is replaced. Existing
    imports of mooncake.engine.TransferEngine and inprocess channels are untouched.
    """
    global _installed, logger
    if not is_enabled() or _installed:
        return
    # Third Party
    import torch

    # First Party
    from lmcache.logging import init_logger
    from lmcache.v1.distributed.transfer_channel.impl import mooncake_te_impl
    from lmcache.v1.multiprocess import server
    import lmcache_ascend.c_ops as ops

    logger = init_logger(__name__)
    original_engine = mooncake_te_impl.TransferEngine
    original_run = server.run_cache_server
    run_signature = signature(original_run)

    def create_engine() -> Any:
        if not is_active():
            raise RuntimeError("Use the Ascend mooncake_cs_server launcher")
        return RegistrationAdapter(original_engine(), ops, POOLS, torch.npu.device)

    @wraps(original_run)
    def checked_run(*args: Any, **kwargs: Any) -> Any:
        global _active
        config = run_signature.bind(*args, **kwargs).arguments["mp_config"]
        if not is_enabled():
            raise RuntimeError("Host handoff was disabled after installation")
        validate_config(config)
        if _active:
            raise RuntimeError("Host handoff supports one MP server per process")
        _active = True
        logger.info(
            "Ascend MP TE Host handoff enabled; validated lmcache_driven HIXL_CS"
        )
        return original_run(*args, **kwargs)

    mooncake_te_impl.TransferEngine = create_engine
    server.run_cache_server = checked_run
    # The HTTP entry point may already hold a from-import reference.
    http = sys.modules.get("lmcache.v1.multiprocess.http_server")
    if http is not None and http.run_cache_server is original_run:
        http.run_cache_server = checked_run
    _installed = True
