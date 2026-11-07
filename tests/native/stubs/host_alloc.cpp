// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "managed_mem.h"
#include "mem_alloc.h"

namespace py = pybind11;
namespace {
constexpr uintptr_t host_ptr = 0x12340000;
aclError alloc_error = 0;
aclError free_error = 0;
std::vector<std::tuple<std::string, uintptr_t, bool>> calls;
void record(const char *name, uintptr_t argument = 0) {
  calls.emplace_back(name, argument, PyGILState_Check() != 0);
}
} // namespace

aclError aclrtMallocHost(void **ptr, std::size_t size) {
  record("aclrtMallocHost", size);
  *ptr = alloc_error ? nullptr : reinterpret_cast<void *>(host_ptr);
  return alloc_error;
}
aclError aclrtFreeHost(void *ptr) {
  record("aclrtFreeHost", reinterpret_cast<uintptr_t>(ptr));
  return free_error;
}
const char *aclrtGetSocName() {
  record("aclrtGetSocName");
  return "Ascend910";
}
aclError aclrtGetLastError(aclrtLastErrLevel) { return 0; }
void *register_ptr(void *ptr, size_t) {
  record("register_ptr", reinterpret_cast<uintptr_t>(ptr));
  return ptr;
}
int unregister_ptr(void *ptr) {
  record("unregister_ptr", reinterpret_cast<uintptr_t>(ptr));
  return 0;
}

// This small test wrapper calls the real mem_alloc.cpp implementations. It
// does not compile c_ops' unrelated PyTorch/Ascend kernels or prove its
// linkage.
PYBIND11_MODULE(_host_alloc_test, m) {
  m.def("alloc_host_ptr", &alloc_host_ptr, py::arg("size"));
  m.def("free_host_ptr", &free_host_ptr, py::arg("ptr"));
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("reset", [] {
    alloc_error = free_error = 0;
    calls.clear();
  });
  m.def("fail_alloc", [](aclError code) { alloc_error = code; });
  m.def("fail_free", [](aclError code) { free_error = code; });
  m.def("calls", [] { return calls; });
}
