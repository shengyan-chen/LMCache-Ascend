# SPDX-License-Identifier: Apache-2.0
"""Run the upstream transfer benchmark with optional Ascend MP memory hooks."""

# Standard
from functools import partial
import sys


def main() -> int:
    """Install memory hooks before running the existing benchmark CLI.

    Accept the upstream benchmark arguments unchanged. Select the ACL Host
    path via LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR=aclrt_malloc_host and
    HIXL via --transfer-channel-type hixl; omit --use-lazy for Mixed memory.
    For ACL only, disable the benchmark's default SHM backing while it runs.
    Upstream owns argument parsing, transfers, verification, and cleanup.

    Returns:
        The upstream exit code: 0 on success, 1 on benchmark failure.
    """
    # Third Party
    from lmcache.tools.transfer_channel_benchmark.__main__ import main as benchmark_main

    # First Party
    from lmcache_ascend.v1.mp_host_memory import (
        get_mp_host_memory_allocator,
        install_mp_host_memory,
    )

    install_mp_host_memory()
    if get_mp_host_memory_allocator() == "default":
        return benchmark_main()

    # Third Party
    from lmcache.tools.transfer_channel_benchmark import benchmark

    # The upstream benchmark has no SHM option. Override only its constructor
    # reference, preserving the public config class and normal validation.
    config_type = benchmark.L1MemoryManagerConfig
    benchmark.L1MemoryManagerConfig = partial(config_type, shm_name="")  # type: ignore[misc, assignment]
    try:
        return benchmark_main()
    finally:
        benchmark.L1MemoryManagerConfig = config_type


if __name__ == "__main__":
    sys.exit(main())
