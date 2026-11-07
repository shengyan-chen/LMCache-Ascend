# SPDX-License-Identifier: Apache-2.0
"""Test the real Python scope using the compiled ACL boundary substitute."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys

# Third Party
import pytest


@pytest.fixture
def runtime(binding, monkeypatch):
    native, control = binding
    package = ModuleType("lmcache_ascend")
    package.hixl_npu_comms = native
    monkeypatch.setitem(sys.modules, "lmcache_ascend", package)
    monkeypatch.setitem(sys.modules, "lmcache_ascend.hixl_npu_comms", native)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(set_device=control.set_device)),
    )
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/hixl_runtime.py"
    spec = importlib.util.spec_from_file_location("hixl_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, native, control


def test_prepare_selects_and_verifies_device_then_borrows_context(runtime):
    module, native, control = runtime
    assert module.prepare_acl_context(3) == 0x4000
    assert control.calls() == [
        ("aclrtGetDevice", True),
        ("aclrtGetCurrentContext", True),
    ]
    assert native.get_current_device() == 3


def test_prepare_rejects_device_mismatch_without_fallback(runtime, monkeypatch):
    module, native, control = runtime
    monkeypatch.setattr(sys.modules["torch"].npu, "set_device", lambda _: None)
    with pytest.raises(RuntimeError, match="device"):
        module.prepare_acl_context(3)
    assert control.calls() == [("aclrtGetDevice", True)]


def test_prepare_requires_valid_context(runtime):
    module, native, control = runtime
    control.set_context(0)
    with pytest.raises(RuntimeError):
        module.prepare_acl_context(0)


def test_prepare_does_not_hide_torch_device_failure(runtime, monkeypatch):
    module, native, control = runtime
    error = RuntimeError("device unavailable")

    def fail(_):
        raise error

    monkeypatch.setattr(sys.modules["torch"].npu, "set_device", fail)
    with pytest.raises(RuntimeError) as caught:
        module.prepare_acl_context(3)
    assert caught.value is error
    assert control.calls() == []


@pytest.mark.parametrize("previous", [0, 0x4000, 0x8000])
def test_scope_restores_nonempty_previous_even_when_equal_to_target(runtime, previous):
    module, native, control = runtime
    control.set_context(previous)
    with module.use_acl_context(0x8000):
        assert control.context() == 0x8000
        if previous:
            # HIXL may change the current context during the call.
            control.set_context(0x9000)
    assert control.context() == (previous or 0x8000)
    assert control.context_switches() == ([0x8000, previous] if previous else [0x8000])


@pytest.mark.parametrize("stage", ["get", "set"])
def test_scope_entry_failure_never_runs_body(runtime, stage):
    module, native, control = runtime
    if stage == "get":
        control.set_acl_error("aclrtGetCurrentContext", 123456)
    else:
        control.fail_set_context(0x8000, 123456)
    with pytest.raises(RuntimeError):
        with module.use_acl_context(0x8000):
            pytest.fail("body must not run after context preparation fails")
    assert control.context_switches() == ([] if stage == "get" else [0x8000])


def test_scope_preserves_original_hixl_exception(runtime, caplog):
    module, native, control = runtime
    control.fail_set_context(0x4000, 123456)
    control.set_result(103902)
    with pytest.raises(native.HixlError) as caught:
        with module.use_acl_context(0x8000):
            native.Hixl().get_transfer_status(0x5678)
    assert caught.value.code == 103902
    assert caught.value.operation == "GetTransferStatus"
    assert control.context_switches() == [0x8000, 0x4000]
    assert "aclrtSetCurrentContext" in caplog.text and "123456" in caplog.text


def test_restore_failure_keeps_successful_request_available(runtime, caplog):
    module, native, control = runtime
    control.fail_set_context(0x4000, 123456)

    def submit():
        with module.use_acl_context(0x8000):
            return native.Hixl().transfer_async("remote", native.READ, [])

    assert submit() == 0x5678
    assert control.context_switches() == [0x8000, 0x4000]
    assert "aclrtSetCurrentContext" in caplog.text and "123456" in caplog.text
