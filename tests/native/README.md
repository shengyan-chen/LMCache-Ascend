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

Without `HIXL_INCLUDE_DIR`, HIXL tests skip and Host allocation tests remain
available. Without `pybind11`, native builds skip instead of failing test
collection. An explicitly configured bad include path or a compiler error
still fails the tests.

The allocation tests compile the real `mem_alloc.cpp` behind a small test
wrapper, including the old mapped allocation paths. They check ACL calls,
arguments, failure handling and GIL state. They do not compile the full
PyTorch-dependent `c_ops` module. The NUMA shim only enables linking on macOS;
these tests do not validate NUMA allocation. Context scope tests run the real
Python helper against the compiled ACL substitute; PyTorch device selection
is simulated.

Still required in the target environment: build/import the installed
`hixl_npu_comms` and `c_ops` with the actual CANN/HIXL libraries, then run
the A3 single-device and subsequent MP integration regressions.
