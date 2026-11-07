# SPDX-License-Identifier: Apache-2.0
"""Configuration policy for the opt-in MP L1 ACL host-memory path.

This module neither installs patches nor initializes the NPU runtime. Call
the MP validator before the server's SHM capacity fallback and L1 creation.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Literal
import os

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.distributed.config import (
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.multiprocess.config import MPServerConfig

_ALLOCATOR_ENV = "LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR"


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
