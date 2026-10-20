# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python; intentionally bypass tests/bootstrap.py."""

import contextlib
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmcache_ascend/v1/distributed/mooncake_te_registration.py"
)


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError("MP Mooncake registration adapter is not implemented")
    spec = importlib.util.spec_from_file_location(
        "mp_registration_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HostMemory:
    """A single live allocation with a removable NPU mapping."""

    def __init__(self):
        self.mapping = 8192
        self.unmap_ret = 0
        self.unmap_count = 0

    def get_device_ptr(self, ptr):
        return self.mapping

    def unregister_ptr(self, ptr):
        self.unmap_count += 1
        if self.unmap_ret == 0:
            self.mapping = 0
        return self.unmap_ret


class Engine:
    """Reject overlapping registrations, like the reproduced hardware failure."""

    def __init__(self, memory):
        self.memory = memory
        self.registered = False
        self.register_ret = 0
        self.unregister_ret = 0
        self.register_count = 0
        self.unregister_count = 0
        self.status = 0

    def initialize(self, *args):
        return 0

    def register_memory(self, ptr, size):
        self.register_count += 1
        if self.memory.mapping:
            return -1
        if self.register_ret == 0:
            self.registered = True
        return self.register_ret

    def unregister_memory(self, ptr):
        self.unregister_count += 1
        if self.unregister_ret == 0:
            self.registered = False
        return self.unregister_ret

    def batch_transfer_async_read(self, *args):
        if not self.registered:
            raise RuntimeError("memory not registered")
        return 17

    def transfer_check_status(self, batch_id):
        return self.status

    def get_rpc_port(self):
        return 12345


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.memory = HostMemory()
        self.engine = Engine(self.memory)
        self.pools = self.mod.PoolRegistry()
        self.pools.track(4096, 1024, 0)
        self.adapter = self.mod.RegistrationAdapter(
            self.engine, self.memory, self.pools, contextlib.nullcontext
        )

    def test_handoff_registers_and_allows_free_only_after_unregister(self):
        self.assertEqual(self.adapter.register_memory(4096, 1024), 0)
        self.assertEqual(self.memory.mapping, 0)
        self.assertTrue(self.engine.registered)
        with self.assertRaises(RuntimeError):
            self.pools.check_free(4096)
        self.adapter.unregister_memory(4096)
        self.pools.check_free(4096)
        self.assertFalse(self.engine.registered)
        self.adapter.unregister_memory(4096)
        self.assertEqual(self.engine.unregister_count, 1)

    def test_raw_host_memory_does_not_require_unmapping(self):
        self.memory.mapping = 0
        self.adapter.register_memory(4096, 1024)
        self.assertEqual(self.memory.unmap_count, 0)
        self.assertTrue(self.engine.registered)

    def test_unmap_failure_never_reaches_te_and_blocks_free(self):
        self.memory.unmap_ret = 503900
        with self.assertLogs(self.mod.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "unregister_ptr"):
                self.adapter.register_memory(4096, 1024)
        self.assertEqual(self.engine.register_count, 0)
        with self.assertRaises(RuntimeError):
            self.pools.check_free(4096)

    def test_failed_registration_is_not_reported_as_ready(self):
        self.engine.register_ret = -1
        with self.assertLogs(self.mod.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "register_memory"):
                self.adapter.register_memory(4096, 1024)
        with self.assertRaises(RuntimeError):
            self.pools.check_free(4096)
        with self.assertRaises(RuntimeError):
            self.adapter.register_memory(4096, 1024)

    def test_failed_unregister_blocks_free(self):
        self.adapter.register_memory(4096, 1024)
        self.engine.unregister_ret = -1
        with self.assertRaisesRegex(RuntimeError, "unregister_memory"):
            self.adapter.unregister_memory(4096)
        with self.assertRaises(RuntimeError):
            self.pools.check_free(4096)

    def test_unknown_or_oversized_region_never_unmaps(self):
        for ptr, size in ((4097, 512), (4096, 1025), (4096, 0)):
            with self.assertRaises((ValueError, RuntimeError)):
                self.adapter.register_memory(ptr, size)
        self.assertEqual(self.memory.unmap_count, 0)

    def test_pending_read_prevents_unregister(self):
        self.adapter.register_memory(4096, 1024)
        batch = self.adapter.batch_transfer_async_read("peer", [4096], [8192], [8])
        with self.assertRaisesRegex(RuntimeError, "in-flight"):
            self.adapter.unregister_memory(4096)
        self.assertEqual(self.engine.unregister_count, 0)
        self.engine.status = 1
        self.assertEqual(self.adapter.transfer_check_status(batch), 1)
        self.adapter.unregister_memory(4096)
        self.pools.check_free(4096)

    def test_failed_read_does_not_imply_dma_has_stopped(self):
        self.adapter.register_memory(4096, 1024)
        batch = self.adapter.batch_transfer_async_read("peer", [4096], [8192], [8])
        self.engine.status = -2
        self.assertEqual(self.adapter.transfer_check_status(batch), -2)
        with self.assertRaises(RuntimeError):
            self.adapter.unregister_memory(4096)
        with self.assertRaises(RuntimeError):
            self.pools.check_free(4096)

    def test_unrelated_methods_are_delegated(self):
        self.assertEqual(self.adapter.get_rpc_port(), 12345)

    def test_registration_and_release_use_allocation_device(self):
        visited = []

        @contextlib.contextmanager
        def scope(device):
            visited.append(device)
            yield

        pools = self.mod.PoolRegistry()
        pools.track(4096, 1024, 3)
        adapter = self.mod.RegistrationAdapter(self.engine, self.memory, pools, scope)
        adapter.register_memory(4096, 1024)
        adapter.unregister_memory(4096)
        self.assertEqual(visited, [3, 3])

    def test_pool_not_handed_to_te_can_be_freed(self):
        self.pools.check_free(4096)
        self.pools.forget(4096)
        with self.assertRaises(ValueError):
            self.pools.region(4096, 1)

    def test_unregister_remains_idempotent_after_allocator_release(self):
        self.adapter.register_memory(4096, 1024)
        self.adapter.unregister_memory(4096)
        self.pools.forget(4096)
        self.assertEqual(self.adapter.unregister_memory(4096), 0)
        self.assertEqual(self.engine.unregister_count, 1)

    def test_actual_allocator_close_refuses_registered_memory(self):
        # Execute the real nested close hook without importing torch/the plugin.
        source = MODULE_PATH.parents[2] / "__init__.py"
        tree = ast.parse(source.read_text())
        allocator_patch = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_patch_lazy_memory_allocator"
        )
        close = next(
            node
            for node in allocator_patch.body
            if isinstance(node, ast.FunctionDef) and node.name == "_ascend_close"
        )
        namespace = {"LazyMemoryAllocator": object}
        exec(
            compile(ast.Module(body=[close], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        freed = []
        ops = types.SimpleNamespace(free_pinned_ptr=freed.append)
        package = types.ModuleType("lmcache_ascend")
        package.c_ops = ops
        self.mod.POOLS = self.pools
        instance = types.SimpleNamespace(
            _ascend_pool_ptr=4096,
            _stop_expand=types.SimpleNamespace(set=lambda: None),
            _expand_thread=types.SimpleNamespace(join=lambda: None),
        )
        modules = {
            "lmcache_ascend": package,
            "lmcache_ascend.c_ops": ops,
            "lmcache_ascend.v1.distributed.mooncake_te_registration": self.mod,
        }
        self.adapter.register_memory(4096, 1024)
        with patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError):
                namespace["_ascend_close"](instance)
            self.assertEqual(freed, [])
            self.adapter.unregister_memory(4096)
            namespace["_ascend_close"](instance)
        self.assertEqual(freed, [4096])


VALID_ENV = {
    "LMCACHE_ASCEND_MP_TE_UNMAP_HOST": "1",
    "MC_TE_PROTOCOL": "ascend",
    "ASCEND_LOCAL_COMM_RES": '{"version":"1.3"}',
    "ASCEND_BUFFER_POOL": "0:0",
    "HCCL_INTRA_ROCE_ENABLE": "1",
}


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.config = types.SimpleNamespace(
            supported_transfer_mode="lmcache_driven",
            engine_type="default",
            p2p_config=types.SimpleNamespace(
                enabled=True, transfer_engine="mooncake_te"
            ),
        )

    def test_valid_configuration(self):
        with patch.dict(os.environ, VALID_ENV, clear=True):
            self.mod.validate_config(self.config)

    def test_other_modes_are_rejected_before_server_start(self):
        for mode in ("engine_driven", "auto"):
            self.config.supported_transfer_mode = mode
            with patch.dict(os.environ, VALID_ENV, clear=True):
                with self.assertRaisesRegex(ValueError, "lmcache_driven"):
                    self.mod.validate_config(self.config)

    def test_incompatible_environment_is_rejected(self):
        for name, value in (
            ("ASCEND_BUFFER_POOL", "4:8"),
            ("ASCEND_USE_ASYNC_TRANSFER", "0"),
            ("MC_USE_TENT", "0"),
            ("ASCEND_ENABLE_USE_FABRIC_MEM", "0"),
            ("ASCEND_LOCAL_COMM_RES", "{broken"),
        ):
            with patch.dict(os.environ, {**VALID_ENV, name: value}, clear=True):
                with self.assertRaises(ValueError):
                    self.mod.validate_config(self.config)

    def test_disabled_install_does_not_import_mooncake_or_lmcache(self):
        with patch.dict(os.environ, {}, clear=True):
            self.mod.install()
        self.assertFalse(self.mod.is_active())

    def test_install_wraps_only_mp_and_validates_before_running(self):
        memory = HostMemory()
        native = Engine(memory)
        mp_impl = types.ModuleType(
            "lmcache.v1.distributed.transfer_channel.impl.mooncake_te_impl"
        )
        mp_impl.TransferEngine = lambda: native
        server = types.ModuleType("lmcache.v1.multiprocess.server")
        calls = []

        def run(mp_config, **kwargs):
            calls.append(mp_config.supported_transfer_mode)
            self.mod.POOLS.track(4096, 1024, 0)
            adapter = mp_impl.TransferEngine()
            adapter.register_memory(4096, 1024)
            return adapter

        server.run_cache_server = run
        http = types.ModuleType("lmcache.v1.multiprocess.http_server")
        http.run_cache_server = run
        modules = {
            "lmcache": types.ModuleType("lmcache"),
            "lmcache.v1": types.ModuleType("lmcache.v1"),
            "lmcache.v1.distributed": types.ModuleType("lmcache.v1.distributed"),
            "lmcache.v1.distributed.transfer_channel": types.ModuleType(
                "lmcache.v1.distributed.transfer_channel"
            ),
            "lmcache.v1.distributed.transfer_channel.impl": types.ModuleType(
                "lmcache.v1.distributed.transfer_channel.impl"
            ),
            "lmcache.v1.multiprocess": types.ModuleType("lmcache.v1.multiprocess"),
            "lmcache_ascend": types.ModuleType("lmcache_ascend"),
            "lmcache_ascend.c_ops": memory,
            "torch": types.SimpleNamespace(
                npu=types.SimpleNamespace(device=contextlib.nullcontext)
            ),
            mp_impl.__name__: mp_impl,
            server.__name__: server,
            http.__name__: http,
        }
        modules["lmcache_ascend"].c_ops = memory
        modules["lmcache.logging"] = types.SimpleNamespace(
            init_logger=lambda name: self.mod.logger
        )
        modules[
            "lmcache.v1.distributed.transfer_channel.impl"
        ].mooncake_te_impl = mp_impl
        modules["lmcache.v1.multiprocess"].server = server
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, VALID_ENV, clear=True),
        ):
            self.mod.install()
            installed_factory = mp_impl.TransferEngine
            self.mod.install()
            self.assertIs(mp_impl.TransferEngine, installed_factory)
            self.config.supported_transfer_mode = "engine_driven"
            with self.assertRaises(ValueError):
                server.run_cache_server(self.config)
            self.assertEqual(calls, [])
            self.config.supported_transfer_mode = "lmcache_driven"
            adapter = http.run_cache_server(mp_config=self.config)
            self.assertTrue(native.registered)
            self.assertEqual(calls, ["lmcache_driven"])
            adapter.unregister_memory(4096)


if __name__ == "__main__":
    unittest.main()
