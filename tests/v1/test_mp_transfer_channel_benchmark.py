# SPDX-License-Identifier: Apache-2.0
"""Check Ascend entry wiring with the real upstream argument parser."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import runpy
import sys

# Third Party
import pytest


def test_entry_installs_hooks_before_benchmark_and_preserves_exit_code(monkeypatch):
    before = set(sys.modules)
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    upstream = root.parents[1] / "LMCache/lmcache"
    # Bypass package activation and NPU services, retaining upstream CLI parsing.
    for name, path in [
        ("lmcache_ascend", root),
        ("lmcache_ascend.v1", root / "v1"),
        ("lmcache", upstream),
    ]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    events = []
    monkeypatch.delitem(sys.modules, "lmcache_ascend.v1.mp_host_memory", raising=False)
    policy = importlib.import_module("lmcache_ascend.v1.mp_host_memory")
    monkeypatch.setattr(
        policy, "install_mp_host_memory", lambda: events.append("installed")
    )
    monkeypatch.setenv("LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR", "aclrt_malloc_host")
    runtime = ModuleType("lmcache.tools.transfer_channel_benchmark.benchmark")

    def memory_config(*, shm_name="lmcache_l1_pool_test"):
        # Model the upstream config's nonempty SHM default at this boundary.
        return SimpleNamespace(
            shm_name=shm_name,
            use_lazy=False,
            use_hugepages=False,
            devdax_path=None,
            align_bytes=4096,
        )

    runtime.L1MemoryManagerConfig = memory_config

    def run_benchmark(args):
        assert events == ["installed"]
        assert args.role == "client"
        assert args.transfer_channel_type == "hixl"
        assert args.verify is True
        policy.validate_acl_l1_config(runtime.L1MemoryManagerConfig(), "npu")
        events.append("run")
        return False

    runtime.run_benchmark = run_benchmark
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mp_transfer_channel_benchmark",
            "--role",
            "client",
            "--transfer-channel-type",
            "hixl",
            "--verify",
        ],
    )
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module(
                "lmcache_ascend.v1.mp_transfer_channel_benchmark", run_name="__main__"
            )
        assert caught.value.code == 1
        assert events == ["installed", "run"]
        assert runtime.L1MemoryManagerConfig is memory_config
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("lmcache."):
                sys.modules.pop(name, None)
        sys.modules.pop("lmcache_ascend.v1.mp_host_memory", None)
