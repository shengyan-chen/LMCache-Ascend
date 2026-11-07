# SPDX-License-Identifier: Apache-2.0
"""Exercise the module entry without starting NPU services or opening sockets."""

# Standard
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
import runpy
import sys

# Third Party
import pytest


@pytest.fixture
def entry_modules(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    # Bypass only package activation, which needs the target NPU installation.
    for name, path in [("lmcache_ascend", root), ("lmcache_ascend.v1", root / "v1")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    policy = ModuleType("lmcache_ascend.v1.mp_host_memory")
    policy.install_mp_host_memory = Mock()
    cli = ModuleType("lmcache.cli.main")
    cli.main = Mock()
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    monkeypatch.setitem(sys.modules, cli.__name__, cli)
    monkeypatch.delitem(sys.modules, "lmcache_ascend.v1.mp_server", raising=False)
    return policy, cli


def test_module_entry_installs_before_forwarding_server_arguments(
    entry_modules, monkeypatch
):
    policy, cli = entry_modules
    events = []
    policy.install_mp_host_memory.side_effect = lambda: events.append("installed")

    def dispatch(argv):
        assert events == ["installed"]
        assert argv == ["server", "--no-l1-use-lazy", "--shm-name", ""]
        events.append("dispatched")

    cli.main.side_effect = dispatch
    monkeypatch.setattr(
        sys, "argv", ["mp_server", "--no-l1-use-lazy", "--shm-name", ""]
    )
    runpy.run_module("lmcache_ascend.v1.mp_server", run_name="__main__")
    assert events == ["installed", "dispatched"]


def test_install_failure_stops_cli_dispatch(entry_modules, monkeypatch):
    policy, cli = entry_modules
    error = RuntimeError("installation failed")
    policy.install_mp_host_memory.side_effect = error
    monkeypatch.setattr(sys, "argv", ["mp_server"])
    with pytest.raises(RuntimeError) as caught:
        runpy.run_module("lmcache_ascend.v1.mp_server", run_name="__main__")
    assert caught.value is error
    cli.main.assert_not_called()
