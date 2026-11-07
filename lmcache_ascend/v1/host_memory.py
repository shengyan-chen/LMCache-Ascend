# SPDX-License-Identifier: Apache-2.0
"""Own pure ACL Host allocations and expose non-owning CPU Tensor views."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import ctypes
import logging
import sys

if TYPE_CHECKING:
    # Third Party
    import torch

logger = logging.getLogger(__name__)


@dataclass
class HostMemoryAllocation:
    """An ACL allocation whose lifetime is independent of its Tensor views.

    Callers must end local and remote use before release. Tensor views do not
    free the allocation, and must not be used after successful release. There
    is no destructor cleanup or synchronization between concurrent releases.
    """

    raw_ptr: int
    size: int
    buffer: torch.Tensor
    _free: Callable[[int], None] = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def released(self) -> bool:
        """Whether the matching native free has returned successfully."""
        return self._released

    def release(self) -> None:
        """Release once after success; propagate failures without marking success.

        A failed free retains the record for diagnosis; it does not guarantee
        that the memory remains usable. No automatic retry is performed.
        """
        if self._released:
            return
        self._free(self.raw_ptr)
        self._released = True


def allocate_host_memory(size: int) -> HostMemoryAllocation:
    """Allocate exactly size bytes and wrap the raw address without copying.

    The caller must prepare the NPU runtime before calling. This function does
    not select a device, map/register Host memory, or adjust the base address.
    Missing pure ACL symbols fail before allocation; there is no fallback.
    If view construction fails, free the raw allocation and preserve that
    original exception even if cleanup also fails.
    """
    # Py_ssize_t bounds ctypes/Tensor views and is stricter than native size_t.
    if not 0 < size <= sys.maxsize:
        raise ValueError(f"size must be between 1 and {sys.maxsize}, got {size}")
    array_type = ctypes.c_uint8 * size

    # Third Party
    import torch

    try:
        # First Party
        from lmcache_ascend import c_ops

        alloc = c_ops.alloc_host_ptr
        free = c_ops.free_host_ptr
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            "ACL Host memory requires c_ops.alloc_host_ptr and c_ops.free_host_ptr"
        ) from exc

    raw_ptr = alloc(size)
    try:
        buffer = torch.frombuffer(array_type.from_address(raw_ptr), dtype=torch.uint8)
        return HostMemoryAllocation(raw_ptr, size, buffer, free)
    except BaseException:
        # Roll back acquired ownership even if construction is interrupted.
        try:
            free(raw_ptr)
        except Exception:
            logger.exception("Failed to release ACL Host allocation at %s", raw_ptr)
        raise
