# SPDX-License-Identifier: Apache-2.0
"""CPU checks of MP engine ownership; only NPU/native boundaries are replaced."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import importlib
import json
import sys
import threading

# Third Party
import pytest

PREFIX = "LMCACHE_HIXL_"


@pytest.fixture
def backend(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    for name, path in [("lmcache_ascend", root), ("lmcache_ascend.v1", root / "v1")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    for suffix in ["mp_hixl_channel", "hixl_runtime"]:
        monkeypatch.delitem(sys.modules, f"lmcache_ascend.v1.{suffix}", raising=False)
    native = ModuleType("lmcache_ascend.hixl_npu_comms")
    state = SimpleNamespace(device=None, context=0x4000)
    native.get_current_device = lambda: state.device
    native.get_current_context = lambda: state.context
    native.set_current_context = lambda value: setattr(state, "context", value)
    native.Hixl = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, native.__name__, native)
    npu = SimpleNamespace(
        set_device=Mock(side_effect=lambda d: setattr(state, "device", d))
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    for suffix in ["ENGINE_ID", "DEVICE_ID", "COMM_LISTEN_PORT", "CONNECT_TIMEOUT_MS"]:
        monkeypatch.delenv(PREFIX + suffix, raising=False)
    monkeypatch.setenv(PREFIX + "ENGINE_ID", "10.0.0.11:6000")
    module = importlib.import_module("lmcache_ascend.v1.mp_hixl_channel")
    # Exercise the real scope and lock, including lock ownership during calls.
    lock = threading.Lock()
    monkeypatch.setattr(module, "Lock", lambda: lock)
    yield module, native, npu, state, lock
    # Imported modules are not themselves installed by monkeypatch.
    for suffix in ["mp_hixl_channel", "hixl_runtime"]:
        sys.modules.pop(f"lmcache_ascend.v1.{suffix}", None)


def test_default_engine_lifecycle_uses_lock_and_borrowed_context(backend):
    module, native, npu, state, lock = backend
    engine = native.Hixl.return_value

    def initialize(engine_id, options):
        assert lock.locked() and state.context == 0x4000
        assert engine_id == "10.0.0.11:6000"
        assert options == {
            "LocalCommRes": '{"version":"1.3"}',
            "BufferPool": "0:0",
            "AutoConnect": "0",
            "GlobalResourceConfig": json.dumps(
                {"comm_resource_config.protocol_desc": "roce:device"}
            ),
        }
        state.context = 0x9000  # Native initialization may switch contexts.

    def finalize():
        assert lock.locked() and state.context == 0x4000

    engine.initialize.side_effect = initialize
    engine.finalize.side_effect = finalize
    context = module.HixlTransferChannelContext()
    npu.set_device.assert_called_once_with(0)
    assert context._config.connect_timeout_ms == 10000
    assert state.context == 0x4000 and not lock.locked()
    state.context = 0x8000  # Closing caller has a different previous context.
    context.close()
    context.close()
    engine.finalize.assert_called_once_with()
    assert state.context == 0x8000 and not lock.locked()


def test_explicit_configuration_reaches_engine(backend, monkeypatch):
    module, native, npu, _, _ = backend
    monkeypatch.setenv(PREFIX + "ENGINE_ID", "[2001:db8::1]:6001")
    monkeypatch.setenv(PREFIX + "DEVICE_ID", "3")
    monkeypatch.setenv(PREFIX + "COMM_LISTEN_PORT", "26666")
    monkeypatch.setenv(PREFIX + "CONNECT_TIMEOUT_MS", "20000")
    context = module.HixlTransferChannelContext()
    npu.set_device.assert_called_once_with(3)
    engine_id, options = native.Hixl.return_value.initialize.call_args.args
    assert engine_id == "[2001:db8::1]:6001"
    assert json.loads(options["GlobalResourceConfig"]) == {
        "comm_resource_config.protocol_desc": "roce:device",
        "comm_resource_config.listen_port": 26666,
    }
    assert context._config.connect_timeout_ms == 20000
    context.close()


@pytest.mark.parametrize(
    "suffix,value",
    [("ENGINE_ID", ""), ("COMM_LISTEN_PORT", "65536"), ("CONNECT_TIMEOUT_MS", "0")],
)
def test_invalid_configuration_has_no_runtime_side_effects(
    backend, monkeypatch, suffix, value
):
    module, native, npu, _, _ = backend
    monkeypatch.setenv(PREFIX + suffix, value)
    with pytest.raises(ValueError, match=PREFIX + suffix):
        module.HixlTransferChannelContext()
    npu.set_device.assert_not_called()
    native.Hixl.assert_not_called()


def test_device_failure_preserves_error_without_creating_engine(backend):
    module, native, npu, _, _ = backend
    error = RuntimeError("device unavailable")
    npu.set_device.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        module.HixlTransferChannelContext()
    assert caught.value is error
    native.Hixl.assert_not_called()


def test_initialize_failure_does_not_publish_engine_or_explicitly_finalize(backend):
    module, native, _, state, lock = backend
    error = RuntimeError("native Initialize failed")
    native.Hixl.return_value.initialize.side_effect = error
    context = module.HixlTransferChannelContext.__new__(
        module.HixlTransferChannelContext
    )
    with pytest.raises(RuntimeError) as caught:
        context.__init__()
    assert caught.value is error
    assert context._engine is None
    native.Hixl.return_value.finalize.assert_not_called()
    assert state.context == 0x4000 and not lock.locked()


def test_cleanup_error_does_not_replace_original_failure(backend, caplog):
    module, native, _, state, lock = backend
    context = module.HixlTransferChannelContext()
    native.Hixl.return_value.finalize.side_effect = RuntimeError("cleanup failed")
    error = ValueError("later construction failed")
    with pytest.raises(ValueError) as caught:
        try:
            raise error
        finally:
            context.close()
    assert caught.value is error
    assert "cleanup failed" in caplog.text
    assert state.context == 0x4000 and not lock.locked()
    assert context._engine is None
