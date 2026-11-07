# SPDX-License-Identifier: Apache-2.0
"""Borrow ACL contexts for direct HIXL calls without owning their lifetime."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def prepare_acl_context(device_id: int) -> int:
    """Select an already parsed logical device and borrow its nonempty context.

    The initialization thread keeps the selected device. The caller must keep
    the underlying runtime/context alive through all HIXL calls and cleanup.
    """
    # Third Party
    import torch

    # First Party
    import lmcache_ascend.hixl_npu_comms as hixl_comms

    torch.npu.set_device(device_id)
    actual_device = hixl_comms.get_current_device()
    if actual_device != device_id:
        raise RuntimeError(
            f"ACL device mismatch: requested {device_id}, current {actual_device}"
        )
    context = hixl_comms.get_current_context()
    if not context:
        raise RuntimeError(f"No current ACL context for device {device_id}")
    return context


@contextmanager
def use_acl_context(target_context: int) -> Iterator[None]:
    """Use a borrowed context for one call under the caller's engine lock.

    Restore a nonempty previous context even if it equals the target: HIXL may
    change it during the call. Restoration is best effort and never replaces
    the call's result or exception. With no previous context, leave the target
    installed; never reset the device or destroy a borrowed context.
    """
    # First Party
    import lmcache_ascend.hixl_npu_comms as hixl_comms

    previous = hixl_comms.get_current_context()
    hixl_comms.set_current_context(target_context)
    try:
        yield
    finally:
        if previous:
            try:
                hixl_comms.set_current_context(previous)
            except Exception:
                logger.warning("Failed to restore previous ACL context", exc_info=True)
