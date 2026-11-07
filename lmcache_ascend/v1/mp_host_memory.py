# SPDX-License-Identifier: Apache-2.0
"""Configuration and allocator helpers for the opt-in MP L1 ACL Host path.

This module neither installs patches nor initializes the NPU runtime. Call
the MP validator before the server's SHM capacity fallback and L1 creation.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal
import logging
import os
import threading

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.distributed.config import (
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
    from lmcache.v1.multiprocess.config import MPServerConfig

_ALLOCATOR_ENV = "LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR"
logger = logging.getLogger(__name__)


def get_mp_host_memory_allocator() -> Literal["default", "aclrt_malloc_host"]:
    """Read the MP L1 selection; raise ValueError for an unknown value.

    An unset variable selects ``default``. Configure it before server startup;
    an allocation's eventual release must not depend on reading it again.
    """
    value = os.environ.get(_ALLOCATOR_ENV, "default")
    if value == "default":
        return "default"
    if value == "aclrt_malloc_host":
        return "aclrt_malloc_host"
    raise ValueError(
        f"{_ALLOCATOR_ENV} must be 'default' or 'aclrt_malloc_host', got {value!r}"
    )


def validate_acl_l1_config(
    config: L1MemoryManagerConfig,
    device_type: str,
    *,
    shm_name: str | None = None,
) -> None:
    """Reject unsupported ACL L1 settings without modifying the configuration.

    Args:
        config: DRAM L1 memory settings for an explicit ACL selection.
        device_type: Resolved LMCache torch device type; must be ``npu``.
        shm_name: Effective SHM name, or None to use ``config.shm_name``.
            An empty string explicitly disables SHM.

    Raises:
        ValueError: The platform or memory backing is unsupported. This checks
            configuration only, not device availability or native symbols.
    """
    if device_type != "npu":
        raise ValueError("aclrt_malloc_host requires the NPU platform")
    if config.use_lazy:
        raise ValueError("aclrt_malloc_host requires use_lazy=False")
    effective_shm_name = config.shm_name if shm_name is None else shm_name
    if effective_shm_name:
        raise ValueError("aclrt_malloc_host requires an empty shm_name")
    if config.use_hugepages:
        raise ValueError("aclrt_malloc_host does not support use_hugepages")
    if config.devdax_path:
        raise ValueError("aclrt_malloc_host does not support devdax_path")


def validate_mp_host_memory_config(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    device_type: str,
) -> None:
    """Check MP L1 selections before SHM fallback or any pool is created.

    Args:
        mp_config: MP settings, including the transfer mode and SHM override.
        storage_manager_config: L1 configurations to validate without mutation.
        device_type: Resolved LMCache torch device type.

    Raises:
        ValueError: The environment selection or an ACL L1 combination is
            unsupported. ``default`` leaves existing configuration rules alone.
    """
    if get_mp_host_memory_allocator() == "default":
        return

    configs = storage_manager_config.l1_manager_configs
    # Match the MP server's override, but do not apply its capacity fallback.
    override_shm = (
        mp_config.supported_transfer_mode != "lmcache_driven"
        and len(configs) == 1
        and mp_config.shm_name is not None
    )
    for config in configs:
        if config.gds_l1_config is not None:
            raise ValueError("aclrt_malloc_host does not support a GDS L1")
        validate_acl_l1_config(
            config.memory_config,
            device_type,
            shm_name=mp_config.shm_name if override_shm else None,
        )


def create_acl_mixed_allocator(
    config: L1MemoryManagerConfig, device_type: str
) -> MixedMemoryAllocator:
    """Build an MP L1 Mixed instance over pure ACL Host memory.

    The caller must prepare the runtime and install the Mixed close adapter
    before using this helper. No patches or runtime initialization happen here.
    Initialization mirrors the non-paging Mixed path in LMCache v0.5.7; keep
    these fields in step with that constructor when upgrading the baseline.
    """
    validate_acl_l1_config(config, device_type)
    align_bytes = config.align_bytes
    if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
        raise ValueError("align_bytes must be a positive power of two")

    # Third Party
    from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

    # First Party
    from lmcache_ascend.v1.host_memory import allocate_host_memory

    allocator = object.__new__(mixed.MixedMemoryAllocator)
    allocation = allocate_host_memory(config.size_in_bytes)
    try:
        allocator.ascend_host_allocation = allocation
        allocator.numa_mapping = None
        allocator.use_hugepages = False
        allocator.align_bytes = align_bytes
        allocator.shm_name = None
        allocator.size = allocation.size
        allocator.buffer = allocation.buffer
        allocator.pin_allocator = mixed.TensorMemoryAllocator(
            allocator.buffer, align_bytes=align_bytes
        )
        allocator.host_mem_lock = threading.Lock()
        allocator.buffer_allocator = mixed.BufferAllocator("cpu")
        return allocator
    except BaseException:
        try:
            allocation.release()
        except Exception:
            logger.exception(
                "Failed to release ACL Host allocation at %s", allocation.raw_ptr
            )
        raise


def make_mixed_close(
    original_close: Callable[[MixedMemoryAllocator], None],
) -> Callable[[MixedMemoryAllocator], None]:
    """Make a close adapter without installing it on MixedMemoryAllocator.

    Ordinary instances retain their original close. ACL instances synchronize
    local device work before releasing the pool through its saved free function.
    The caller must already have ended communication use: device synchronization
    alone does not drain HIXL or remote accesses. Close is sequential, not a
    concurrent release protocol.
    """

    def close(allocator: MixedMemoryAllocator) -> None:
        allocation = getattr(allocator, "ascend_host_allocation", None)
        if allocation is None:
            original_close(allocator)
            return
        if allocation.released:
            return

        # Third Party
        from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

        if mixed.torch_dev.is_available():
            mixed.torch_dev.synchronize()
        allocation.release()

    return close
