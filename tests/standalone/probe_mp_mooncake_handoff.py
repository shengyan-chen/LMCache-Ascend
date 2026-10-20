# SPDX-License-Identifier: Apache-2.0
"""NPU probe: real adapter registration, MP memcpy round-trip, normal cleanup.

Run from the installed development checkout. This probe does not run cross-node
transfers or measure bandwidth. Failure exits the disposable process without
speculative cleanup of uncertain registration state.
"""

import argparse
import ctypes
import importlib.metadata
import json
import logging
import os
import sys
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--nic", required=True)
    parser.add_argument("--size-mib", type=int, default=64)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--base-port", type=int, default=15000)
    parser.add_argument("--listen-port", type=int, default=27666)
    args = parser.parse_args()
    if args.size_mib < 1 or min(args.devices) < 0:
        parser.error("size-mib must be >= 1 and devices must be nonnegative")
    if not (1024 <= args.base_port <= 65000 and 1024 <= args.listen_port <= 65535):
        parser.error("invalid port range")

    for name in (
        "MC_USE_TENT",
        "MC_USE_TEV1",
        "MC_FORCE_TCP",
        "MC_FORCE_HCA",
        "ASCEND_USE_ASYNC_TRANSFER",
        "ASCEND_ENABLE_USE_FABRIC_MEM",
    ):
        os.environ.pop(name, None)
    os.environ.update(
        {
            "MC_TE_PROTOCOL": "ascend",
            "MC_TE_DEVICE": "",
            "ASCEND_LOCAL_COMM_RES": '{"version":"1.3"}',
            "ASCEND_BUFFER_POOL": "0:0",
            "HCCL_INTRA_ROCE_ENABLE": "1",
            "HCCL_IF_IP": args.host,
            "HCCL_SOCKET_IFNAME": args.nic,
            "ASCEND_BASE_PORT": str(args.base_port),
            "ASCEND_GLOBAL_RESOURCE_CONFIG": json.dumps(
                {
                    "comm_resource_config.listen_port": args.listen_port,
                }
            ),
            "HCCL_NPU_SOCKET_PORT_RANGE": "62000-62099",
        }
    )
    logging.basicConfig(level=logging.INFO)
    stage = "imports"
    try:
        import torch
        import torch_npu  # noqa: F401
        from mooncake.engine import TransferEngine

        import lmcache_ascend.c_ops as ops
        from lmcache_ascend.v1.distributed.mooncake_te_registration import (
            PoolRegistry,
            RegistrationAdapter,
        )

        versions = {}
        for package in (
            "lmcache",
            "lmcache-ascend",
            "torch-npu",
            "mooncake-transfer-engine-npu",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        print(
            json.dumps(
                {"versions": versions, "c_ops": ops.__file__, "devices": args.devices}
            ),
            flush=True,
        )
        torch.npu.set_device(args.devices[0])
        torch.npu.init()
        stage = "allocation"
        size = args.size_mib * 1024**2
        ptr = int(ops.alloc_pinned_ptr(size, 0))
        pools = PoolRegistry()
        pools.track(ptr, size, args.devices[0])
        engine = TransferEngine()
        adapter = RegistrationAdapter(engine, ops, pools, torch.npu.device)
        stage = "te_initialize"
        ret = adapter.initialize(args.host, "P2PHANDSHAKE", "ascend", "")
        if ret != 0:
            raise RuntimeError(f"TE initialize ret={ret}")
        stage = "te_register"
        adapter.register_memory(ptr, size)
        if ops.get_device_ptr(ptr):
            raise RuntimeError("LMCache mapping must stay absent after handoff")

        # Deliberately exercise unaligned offsets and the end of the allocation.
        count = 4096
        offsets = (0, size // 2 + 17, size - count)
        for device in args.devices:
            with torch.npu.device(device):
                buf = torch.empty(count, dtype=torch.uint8, device=f"npu:{device}")
                for offset in offsets:
                    stage = f"memcpy_device_{device}_offset_{offset}"
                    pattern = (device + offset // count) % 251 + 1
                    ctypes.memset(ptr + offset, pattern, count)
                    ops.lmcache_memcpy_async(
                        buf.data_ptr(),
                        ptr + offset,
                        count,
                        ops.TransferDirection.H2D,
                        offset,
                        65536,
                    )
                    torch.npu.synchronize()
                    ctypes.memset(ptr + offset, 0, count)
                    ops.lmcache_memcpy_async(
                        ptr + offset,
                        buf.data_ptr(),
                        count,
                        ops.TransferDirection.D2H,
                        offset,
                        65536,
                    )
                    torch.npu.synchronize()
                    if (
                        ctypes.string_at(ptr + offset, count)
                        != bytes([pattern]) * count
                    ):
                        raise RuntimeError(f"Round-trip data mismatch at {stage}")
                    print(json.dumps({"stage": stage, "status": "pass"}), flush=True)
                del buf

        stage = "te_unregister"
        adapter.unregister_memory(ptr)
        stage = "free_host"
        pools.check_free(ptr)
        with torch.npu.device(args.devices[0]):
            ops.free_pinned_ptr(ptr)
        pools.forget(ptr)
        # Exercise normal TE destruction on success, unlike the original probe.
        del adapter, engine
        print(
            json.dumps(
                {
                    "status": "pass",
                    "stage": "done",
                    "size_bytes": size,
                    "normal_cleanup": True,
                }
            ),
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        print(json.dumps({"status": "failed", "stage": stage}), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
