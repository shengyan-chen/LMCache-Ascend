# SPDX-License-Identifier: Apache-2.0
"""Validate the real ordinary MP L1 arena on one Ascend/RDMA host."""

# Standard
from pathlib import Path
import argparse
import ctypes
import hashlib
import inspect
import json
import os
import random
import sys
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Local RDMA interface IPv4")
    parser.add_argument("--device", required=True, help="Local verbs device name")
    parser.add_argument("--size-mib", type=int, default=128)
    parser.add_argument("--npu", type=int, default=0, help="Visible NPU index")
    args = parser.parse_args()
    if args.size_mib < 128 or args.size_mib % 64:
        parser.error("--size-mib must be a multiple of 64 and at least 128")
    key = "LMCACHE_ASCEND_MP_HOST_ALLOCATOR"
    os.environ.setdefault(key, "ordinary")
    if os.environ[key] != "ordinary":
        parser.error(f"this probe requires {key}=ordinary")
    if os.environ.get("MC_FORCE_TCP"):
        parser.error("unset MC_FORCE_TCP before running the RDMA probe")

    # Third Party
    from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
    from mooncake.engine import TransferEngine
    import torch
    import torch_npu  # noqa: F401

    # First Party
    import lmcache_ascend
    import lmcache_ascend.c_ops as ops

    def emit(**values):
        print(json.dumps(values, ensure_ascii=False), flush=True)

    def check(actual, expected, label):
        if actual != expected:
            raise RuntimeError(f"{label}: byte-for-byte comparison failed")
        emit(check=label, status="PASS", sha256=hashlib.sha256(actual).hexdigest())

    torch.npu.set_device(args.npu)
    torch.npu.synchronize()
    emit(
        ascend_package=str(Path(lmcache_ascend.__file__).resolve()),
        allocator_init=inspect.getfile(LazyMemoryAllocator.__init__),
        mode=os.environ[key],
    )
    size = args.size_mib * 1024 * 1024
    allocator = LazyMemoryAllocator(
        init_size=64 * 1024 * 1024, final_size=size, align_bytes=65536
    )
    if getattr(allocator, "_host_allocator_mode", None) != "ordinary":
        raise RuntimeError("ordinary allocator patch is not active")
    deadline = time.monotonic() + 60
    while allocator.get_address_manager().get_heap_size() < size:
        if time.monotonic() > deadline:
            raise TimeoutError("L1 expansion did not publish full capacity")
        time.sleep(0.05)
    buffer = allocator.get_underlying_buffer()
    ptr = buffer.data_ptr()
    if buffer.device.type != "cpu" or buffer.nbytes != size or ptr % 65536:
        raise RuntimeError("unexpected L1 memory type, size or alignment")
    if ops.get_device_ptr(ptr):
        raise RuntimeError("ordinary Host L1 unexpectedly has an NPU mapping")

    # Exercise real chunk allocation and aliasing before using raw test ranges.
    obj = allocator.allocate(torch.Size([2, 1, 256, 8192]), torch.uint8)
    if obj is None:
        raise RuntimeError("L1 chunk allocation failed")
    offset = obj.data_ptr - ptr
    if offset < 0 or offset + obj.get_size() > size or obj.data_ptr % 65536:
        raise RuntimeError("chunk lies outside L1 or is misaligned")
    ctypes.memset(obj.data_ptr, 0x4D, 64)
    if buffer[offset : offset + 64].tolist() != [0x4D] * 64:
        raise RuntimeError("chunk does not alias L1 tensor storage")
    allocator.free(obj)
    del obj
    emit(check="REAL_L1_CHUNK_ALLOCATION", status="PASS", bytes=size, ptr=hex(ptr))

    engine = TransferEngine()
    rc = engine.initialize(args.host, "P2PHANDSHAKE", "rdma", args.device)
    emit(stage="rdma_initialize", return_code=rc)
    if rc != 0:
        raise RuntimeError("RDMA initialization failed")
    rc = engine.register_memory(ptr, size)
    emit(stage="rdma_register", return_code=rc)
    if rc != 0:
        raise RuntimeError("L1 MR registration failed")

    copy_size = 4 * 1024 * 1024
    offsets = [0, LazyMemoryAllocator.PIN_CHUNK_SIZE - copy_size // 2, size - copy_size]
    for index, offset in enumerate(offsets):
        emit(stage="copy", offset=offset, bytes=copy_size)
        expected = random.Random(100 + index).randbytes(copy_size)
        source = torch.frombuffer(bytearray(expected), dtype=torch.uint8)
        device_source = source.to(f"npu:{args.npu}")
        torch.npu.synchronize()
        ctypes.memset(ptr + offset, 0xA5, copy_size)
        ops.lmcache_memcpy_async(
            ptr + offset,
            device_source.data_ptr(),
            copy_size,
            ops.TransferDirection.D2H,
            offset,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
        torch.npu.synchronize()
        check(ctypes.string_at(ptr + offset, copy_size), expected, f"D2H-{offset}")

        expected = random.Random(200 + index).randbytes(copy_size)
        ctypes.memmove(ptr + offset, expected, copy_size)
        destination = torch.zeros(
            copy_size, dtype=torch.uint8, device=f"npu:{args.npu}"
        )
        torch.npu.synchronize()
        ops.lmcache_memcpy_async(
            destination.data_ptr(),
            ptr + offset,
            copy_size,
            ops.TransferDirection.H2D,
            offset,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
        torch.npu.synchronize()
        check(destination.cpu().numpy().tobytes(), expected, f"H2D-{offset}")

    rc = engine.unregister_memory(ptr)
    emit(stage="rdma_unregister", return_code=rc)
    if rc != 0:
        raise RuntimeError("L1 MR unregister failed")
    del engine
    allocator.close()
    allocator.close()
    emit(
        status="PASS_ORDINARY_MP_L1_RDMA_NPU",
        bytes=size,
        scope="local real L1 allocation/expansion, MR, NPU copies; no remote read",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Avoid freeing buffers still referenced by failed asynchronous work.
        os._exit(1)
