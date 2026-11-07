# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of MP host-memory selection and read-only validation."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util

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
