# SPDX-License-Identifier: Apache-2.0
"""CPU tests of production C++ bindings with substituted external libraries."""

# Standard
from pathlib import Path
import importlib.util
import os
import shlex
import subprocess
import sys
import sysconfig

# Third Party
import pybind11
import pytest

ROOT = Path(__file__).resolve().parents[2]
STUBS = Path(__file__).with_name("stubs")


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_extension(output_dir, name, sources, includes=()):
    output = output_dir / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    command = shlex.split(os.environ.get("CXX", "c++")) + [
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O0",
        "-g",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_path("include"),
        "-I" + str(STUBS),
        "-I" + str(ROOT / "csrc"),
    ]
    command += ["-I" + str(path) for path in includes]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    command += [str(path) for path in sources] + ["-o", str(output)]
    subprocess.run(command, check=True)
    return output


@pytest.fixture(scope="session")
def hixl_extension(tmp_path_factory):
    include = Path(os.environ["HIXL_INCLUDE_DIR"])
    output = build_extension(
        tmp_path_factory.mktemp("hixl-binding"),
        "hixl_npu_comms",
        [ROOT / "csrc/hixl/bindings.cpp", STUBS / "hixl.cpp"],
        [include],
    )
    return (
        load_extension("hixl_npu_comms", output),
        load_extension("_hixl_test_control", output),
    )


@pytest.fixture
def binding(hixl_extension):
    module, control = hixl_extension
    control.reset()
    return module, control
