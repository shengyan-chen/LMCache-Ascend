# Native boundary tests without CANN

These tests compile production C++ against real HIXL public headers and
test-only substitutes for GE, HIXL, ACL and runtime libraries. They verify
binding behavior and GIL boundaries, not CANN ABI compatibility, actual
device/context behavior, memory registration or data transfer.

Use Python 3.10–3.12, a C++17 compiler, `pytest` and `pybind11`. From the
repository root:

```sh
HIXL_INCLUDE_DIR=/path/to/HIXL/include \
LMCACHEPATH=/path/to/LMCache \
python -m pytest --confcutdir=tests/native tests/native -q
```

`--confcutdir` intentionally excludes the NPU-oriented parent conftest and
its bootstrap; no checkout is cloned, fetched or changed. HIXL headers
must match the intended target version (v9.1.0 for this implementation).
Set `CXX` if the default compiler is unavailable. Build artifacts live in
pytest's temporary directory. The substitutes are never used by the
production CMake build.

Still required in the target environment: build/import the installed
`hixl_npu_comms` and `c_ops` with the actual CANN/HIXL libraries, then run
the A3 single-device and subsequent MP integration regressions.
