# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of MP host-memory selection and read-only validation."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import ctypes
import importlib.util
import os
import sys

# Third Party
import pytest

ENV = "LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR"


@pytest.fixture
def policy() -> ModuleType:
    # Load the policy without activating the Ascend package's NPU patches.
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/mp_host_memory.py"
    spec = importlib.util.spec_from_file_location("mp_host_memory_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configs(monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, SimpleNamespace]:
    monkeypatch.setenv(ENV, "aclrt_malloc_host")
    memory = SimpleNamespace(
        size_in_bytes=4096,
        init_size_in_bytes=4096,
        align_bytes=4096,
        use_lazy=False,
        shm_name="",
        use_hugepages=False,
        devdax_path=None,
        devdax_offset_in_bytes=0,
        mlock=False,
    )
    storage = SimpleNamespace(
        l1_manager_configs=[SimpleNamespace(memory_config=memory, gds_l1_config=None)]
    )
    mp = SimpleNamespace(supported_transfer_mode="auto", shm_name=None)
    return mp, storage


def test_selection_reads_environment(policy, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert policy.get_mp_host_memory_allocator() == "default"
    monkeypatch.setenv(ENV, "default")
    assert policy.get_mp_host_memory_allocator() == "default"
    monkeypatch.setenv(ENV, "aclrt_malloc_host")
    assert policy.get_mp_host_memory_allocator() == "aclrt_malloc_host"


def test_unknown_selection_is_rejected(policy, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "aclrt_malloc")
    with pytest.raises(ValueError, match=ENV):
        policy.get_mp_host_memory_allocator()


def test_default_preserves_non_target_configuration(
    policy, configs, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp, storage = configs
    memory = storage.l1_manager_configs[0].memory_config
    memory.use_lazy = True
    monkeypatch.setenv(ENV, "default")
    policy.validate_mp_host_memory_config(mp, storage, "cpu")
    assert memory.use_lazy is True


def test_acl_accepts_nonlazy_npu_dram(policy, configs) -> None:
    mp, storage = configs
    policy.validate_mp_host_memory_config(mp, storage, "npu")


@pytest.mark.parametrize(
    "field,value", [("use_lazy", True), ("devdax_path", "/dev/dax0.0")]
)
def test_acl_rejects_incompatible_memory(policy, configs, field, value) -> None:
    mp, storage = configs
    memory = storage.l1_manager_configs[0].memory_config
    setattr(memory, field, value)
    with pytest.raises(ValueError, match=field):
        policy.validate_mp_host_memory_config(mp, storage, "npu")
    assert getattr(memory, field) == value


def test_acl_requires_npu(policy, configs) -> None:
    mp, storage = configs
    with pytest.raises(ValueError, match="NPU"):
        policy.validate_mp_host_memory_config(mp, storage, "cpu")


def test_mp_shm_override_is_checked_without_mutation(policy, configs) -> None:
    mp, storage = configs
    memory = storage.l1_manager_configs[0].memory_config
    mp.shm_name = "lmcache_l1_pool_override"
    with pytest.raises(ValueError, match="shm_name"):
        policy.validate_mp_host_memory_config(mp, storage, "npu")
    assert memory.shm_name == ""
    assert mp.shm_name == "lmcache_l1_pool_override"

    memory.shm_name = "lmcache_l1_pool_configured"
    mp.shm_name = ""
    policy.validate_mp_host_memory_config(mp, storage, "npu")
    assert memory.shm_name == "lmcache_l1_pool_configured"


def test_lmcache_driven_keeps_l1_shm_selection(policy, configs) -> None:
    mp, storage = configs
    mp.supported_transfer_mode = "lmcache_driven"
    mp.shm_name = ""
    storage.l1_manager_configs[0].memory_config.shm_name = "lmcache_l1_pool_configured"
    with pytest.raises(ValueError, match="shm_name"):
        policy.validate_mp_host_memory_config(mp, storage, "npu")


@pytest.fixture
def host_memory(monkeypatch):
    # Only the ACL boundary is substituted; views use real CPU torch storage.
    pytest.importorskip("torch")
    package = ModuleType("lmcache_ascend")
    native = ModuleType("lmcache_ascend.c_ops")
    backing = {}

    def allocate(size):
        storage = (ctypes.c_uint8 * (size + 1))()
        # Deliberately not align_bytes-aligned: the adapter must keep this address.
        ptr = ctypes.addressof(storage) + 1
        backing[ptr] = storage
        return ptr

    native.alloc_host_ptr = Mock(side_effect=allocate)
    native.free_host_ptr = Mock(side_effect=backing.pop)
    package.c_ops = native
    monkeypatch.setitem(sys.modules, "lmcache_ascend", package)
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", native)
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/host_memory.py"
    name = "lmcache_ascend.v1.host_memory"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, native, backing


@pytest.fixture
def mixed_module(monkeypatch):
    # Use the frozen LMCache checkout; do not run the NPU bootstrap or fetch it.
    if "LMCACHEPATH" in os.environ:
        monkeypatch.syspath_prepend(os.environ["LMCACHEPATH"])
    lmcache = pytest.importorskip("lmcache")
    if not hasattr(lmcache, "device_ops"):
        # Native CPU pin/unpin operations are unavailable on the local macOS host.
        # None should be used by the ACL path under test.
        monkeypatch.setattr(lmcache, "device_ops", SimpleNamespace(), raising=False)
    return importlib.import_module(
        "lmcache.v1.memory_allocators.mixed_memory_allocator"
    )


def test_raw_view_owns_matching_release(host_memory, monkeypatch):
    module, native, backing = host_memory
    allocation = module.allocate_host_memory(123)
    ptr = allocation.raw_ptr
    native.alloc_host_ptr.assert_called_once_with(123)
    assert allocation.size == allocation.buffer.numel() == 123
    assert allocation.buffer.data_ptr() == ptr
    allocation.buffer[0] = 19
    assert ctypes.c_uint8.from_address(ptr).value == 19

    saved_free = native.free_host_ptr
    monkeypatch.setattr(native, "free_host_ptr", Mock(side_effect=AssertionError))
    monkeypatch.setenv(ENV, "default")
    allocation.release()
    allocation.release()
    saved_free.assert_called_once_with(ptr)
    assert allocation.released and not backing


def test_unrepresentable_size_rejected_before_allocation(host_memory):
    module, native, _ = host_memory
    with pytest.raises(ValueError, match="size"):
        module.allocate_host_memory(sys.maxsize + 1)
    native.alloc_host_ptr.assert_not_called()


def test_missing_free_capability_rejected_before_allocation(host_memory, monkeypatch):
    module, native, _ = host_memory
    monkeypatch.delattr(native, "free_host_ptr")
    with pytest.raises(RuntimeError, match="free_host_ptr"):
        module.allocate_host_memory(123)
    native.alloc_host_ptr.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_view_failure_rolls_back_without_masking_error(
    host_memory, monkeypatch, caplog, cleanup_fails
):
    module, native, backing = host_memory
    torch = pytest.importorskip("torch")
    error = RuntimeError("view failed")
    monkeypatch.setattr(torch, "frombuffer", Mock(side_effect=error))
    if cleanup_fails:
        native.free_host_ptr.side_effect = RuntimeError("free failed")
    with pytest.raises(RuntimeError) as caught:
        module.allocate_host_memory(123)
    assert caught.value is error
    native.free_host_ptr.assert_called_once()
    if cleanup_fails:
        assert "free failed" in caplog.text
        assert str(next(iter(backing))) in caplog.text
    else:
        assert not backing


def test_release_failure_keeps_record_unreleased(host_memory):
    module, native, backing = host_memory
    allocation = module.allocate_host_memory(123)
    native.free_host_ptr.side_effect = RuntimeError("free failed")
    with pytest.raises(RuntimeError, match="free failed"):
        allocation.release()
    assert not allocation.released
    assert allocation.raw_ptr in backing
    native.free_host_ptr.assert_called_once_with(allocation.raw_ptr)


def test_mixed_reuses_pool_and_closes_through_record(
    policy, configs, host_memory, mixed_module, monkeypatch
):
    _, native, backing = host_memory
    torch = pytest.importorskip("torch")
    memory = configs[1].l1_manager_configs[0].memory_config
    memory.align_bytes = 64
    allocator = policy.create_acl_mixed_allocator(memory, "npu")
    assert isinstance(allocator, mixed_module.MixedMemoryAllocator)
    ptr = allocator.buffer.data_ptr()
    first = allocator.allocate(torch.Size([17]), torch.uint8)
    second = allocator.allocate(torch.Size([17]), torch.uint8)
    assert first.tensor.data_ptr() == ptr
    assert second.tensor.data_ptr() == ptr + 64
    allocator.free(first)
    allocator.free(second)
    native.free_host_ptr.assert_not_called()

    device = SimpleNamespace(is_available=lambda: True, synchronize=Mock())
    monkeypatch.setattr(mixed_module, "torch_dev", device)
    original_close = Mock(side_effect=AssertionError("ordinary close used for ACL"))
    close = policy.make_mixed_close(original_close)
    close(allocator)
    close(allocator)
    assert not backing and allocator.ascend_host_allocation.released
    device.synchronize.assert_called_once()
    native.free_host_ptr.assert_called_once_with(ptr)


def test_mixed_construction_failure_releases_pool(
    policy, configs, host_memory, mixed_module, monkeypatch
):
    _, native, backing = host_memory
    error = RuntimeError("allocator construction failed")
    monkeypatch.setattr(mixed_module, "TensorMemoryAllocator", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        policy.create_acl_mixed_allocator(
            configs[1].l1_manager_configs[0].memory_config, "npu"
        )
    assert caught.value is error
    assert not backing
    native.free_host_ptr.assert_called_once()


def test_close_sync_failure_keeps_pool(
    policy, configs, host_memory, mixed_module, monkeypatch
):
    _, native, backing = host_memory
    allocator = policy.create_acl_mixed_allocator(
        configs[1].l1_manager_configs[0].memory_config, "npu"
    )
    device = SimpleNamespace(
        is_available=lambda: True,
        synchronize=Mock(side_effect=RuntimeError("sync failed")),
    )
    monkeypatch.setattr(mixed_module, "torch_dev", device)
    with pytest.raises(RuntimeError, match="sync failed"):
        policy.make_mixed_close(Mock())(allocator)
    assert not allocator.ascend_host_allocation.released and backing
    native.free_host_ptr.assert_not_called()


def test_ordinary_mixed_close_delegates(policy):
    allocator = object()
    original_close = Mock()
    policy.make_mixed_close(original_close)(allocator)
    original_close.assert_called_once_with(allocator)
