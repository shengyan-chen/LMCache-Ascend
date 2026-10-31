# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle tests; NPU and LMCache bootstrap are deliberately isolated.

Run with ``pytest --noconftest tests/unit/test_mp_host_allocator.py``.
The production patch is extracted to avoid importing the NPU-only package.
Buffers are real CPU tensors; native ACL calls and the upstream allocator
container are substitutes. Use tools/probe_mp_host_allocator.py on the server
to exercise the real LMCache allocator, NPU copies and RDMA registration.
"""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import ctypes
import logging
import sys

# Third Party
import pytest
import torch

ENV = "LMCACHE_ASCEND_MP_HOST_ALLOCATOR"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    allocations = []

    def native_allocate(size, flags):
        buffer = ctypes.create_string_buffer(size)
        allocations.append(buffer)
        return ctypes.addressof(buffer)

    native = ModuleType("lmcache_ascend.c_ops")
    native.alloc_pinned_ptr = Mock(side_effect=native_allocate)
    native.free_pinned_ptr = Mock()
    spec = SimpleNamespace(is_pin_supported=True)
    npu = SimpleNamespace(
        is_available=lambda: False, set_device=Mock(), current_device=lambda: 0
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    class LazyAllocator:
        PIN_CHUNK_SIZE = 65536

        def _expand_worker(self):
            self._stop_expand.wait()

        def _log_expansion_progress(self, size):
            logging.getLogger("allocator_test").info("ACL pinned expansion")

    class TensorAllocator:
        def __init__(self, tensor, align_bytes, init_address_space):
            self.buffer = tensor.view(torch.uint8)
            self.address_manager = SimpleNamespace(size=init_address_space)

    modules = {
        "lmcache": {"torch_dev": SimpleNamespace(ext=spec)},
        "lmcache.logging": {"init_logger": logging.getLogger},
        "lmcache.v1.platform": {"current_device_spec": spec},
        "lmcache.v1.memory_allocators.lazy_memory_allocator": {
            "AddressManager": SimpleNamespace(ALIGN_BYTES=65536),
            "LazyMemoryAllocator": LazyAllocator,
            "TensorMemoryAllocator": TensorAllocator,
            "align_to": lambda size, alignment: (
                (size + alignment - 1) & ~(alignment - 1)
            ),
        },
        "lmcache_ascend": {"c_ops": native},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, native.__name__, native)

    source = ROOT / "lmcache_ascend/__init__.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_lazy_memory_allocator"
    )
    namespace = {"Any": Any, "__name__": "allocator_test", "_lazy_alloc_patched": False}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    namespace["_patch_lazy_memory_allocator"]()
    instances = []

    def make(**kwargs):
        instance = LazyAllocator(**{"init_size": 65536, "final_size": 131072, **kwargs})
        instances.append(instance)
        return instance

    yield SimpleNamespace(make=make, native=native, spec=spec, npu=npu)
    for instance in instances:
        instance._stop_expand.set()
        instance._expand_thread.join()


def test_ordinary_cpu_buffer_alignment_and_alias_lifetime(runtime, monkeypatch):
    monkeypatch.setenv(ENV, "ordinary")
    allocator = runtime.make(align_bytes=65536)
    buffer = allocator._buffer
    assert buffer.device.type == "cpu"
    assert not buffer.is_pinned()
    assert buffer.nbytes == 131072
    assert buffer.data_ptr() % 65536 == 0
    view = allocator._allocator.buffer[65536:65552]
    view.fill_(73)
    assert torch.equal(buffer[65536:65552], torch.full((16,), 73, dtype=torch.uint8))
    allocator.close()
    allocator.close()
    assert not allocator._expand_thread.is_alive()
    assert view.tolist() == [73] * 16
    runtime.native.alloc_pinned_ptr.assert_not_called()
    runtime.native.free_pinned_ptr.assert_not_called()


def test_ordinary_does_not_require_npu_pinning(runtime, monkeypatch):
    monkeypatch.setenv(ENV, "ordinary")
    runtime.spec.is_pin_supported = False
    allocator = runtime.make()
    allocator._pin_memory_chunk(0, 65536)
    assert allocator._pin_record == []
    runtime.npu.set_device.assert_not_called()
    allocator.close()


@pytest.mark.parametrize("mode", ["typo", "", "ORDINARY"])
def test_invalid_mode_fails_before_allocation(runtime, monkeypatch, mode):
    monkeypatch.setenv(ENV, mode)
    with pytest.raises(ValueError, match=ENV):
        runtime.make()
    runtime.native.alloc_pinned_ptr.assert_not_called()


@pytest.mark.parametrize("alignment", [0, -1, 3])
def test_invalid_alignment_rejected(runtime, monkeypatch, alignment):
    monkeypatch.setenv(ENV, "ordinary")
    with pytest.raises(ValueError, match="align_bytes"):
        runtime.make(align_bytes=alignment)


@pytest.mark.parametrize("initial,final", [(-1, 65536), (1, 0), (131072, 65536)])
def test_invalid_pool_size_rejected(runtime, monkeypatch, initial, final):
    monkeypatch.setenv(ENV, "ordinary")
    with pytest.raises(ValueError, match="size"):
        runtime.make(init_size=initial, final_size=final)


def test_acl_is_default_and_free_is_once_despite_environment_change(
    runtime, monkeypatch
):
    allocator = runtime.make()
    runtime.native.alloc_pinned_ptr.assert_called_once_with(131072, 0)
    pointer = allocator._buffer.data_ptr()
    monkeypatch.setenv(ENV, "ordinary")
    allocator.close()
    allocator.close()
    runtime.native.free_pinned_ptr.assert_called_once_with(pointer)


def test_ordinary_close_does_not_follow_changed_environment(runtime, monkeypatch):
    monkeypatch.setenv(ENV, "ordinary")
    allocator = runtime.make()
    monkeypatch.setenv(ENV, "acl")
    allocator.close()
    runtime.native.free_pinned_ptr.assert_not_called()


def test_acl_still_requires_pin_support(runtime):
    runtime.spec.is_pin_supported = False
    with pytest.raises(RuntimeError, match="pinning"):
        runtime.make()


def test_ordinary_expansion_log_does_not_claim_pinned(runtime, monkeypatch, caplog):
    monkeypatch.setenv(ENV, "ordinary")
    with caplog.at_level(logging.INFO):
        allocator = runtime.make()
        allocator._log_expansion_progress(65536)
    assert "ordinary" in caplog.text.lower()
    assert "pinned" not in caplog.text.lower()
    allocator.close()
