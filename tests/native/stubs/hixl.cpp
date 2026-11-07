// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "hcomm_devva.h"
#include "hixl/hixl.h"

namespace py = pybind11;
namespace {
hixl::Status result = 0;
hixl::TransferStatus transfer_status = hixl::TransferStatus::COMPLETED;
bool throw_cpp_error = false;
std::vector<std::pair<std::string, bool>> calls;
int current_device = 0;
uintptr_t current_context = 0x4000;
std::map<std::string, aclError> acl_errors;
std::map<uintptr_t, aclError> set_context_errors;
std::vector<uintptr_t> context_switches;

hixl::Status record(const char *operation) {
  calls.emplace_back(operation, PyGILState_Check() != 0);
  if (throw_cpp_error) {
    throw std::runtime_error("external C++ failure");
  }
  return result;
}
} // namespace

namespace hixl {
class Hixl::HixlImpl {};
Hixl::Hixl() = default;
Hixl::~Hixl() = default;
Status Hixl::Initialize(const AscendString &,
                        const std::map<AscendString, AscendString> &) {
  return record("Initialize");
}
void Hixl::Finalize() {}
Status Hixl::RegisterMem(const MemDesc &, MemType, MemHandle &handle) {
  handle = reinterpret_cast<void *>(0x1234);
  return record("RegisterMem");
}
Status Hixl::DeregisterMem(MemHandle) { return record("DeregisterMem"); }
Status Hixl::Connect(const AscendString &, int32_t) {
  return record("Connect");
}
Status Hixl::Disconnect(const AscendString &, int32_t) {
  return record("Disconnect");
}
Status Hixl::TransferSync(const AscendString &, TransferOp,
                          const std::vector<TransferOpDesc> &, int32_t) {
  return record("TransferSync");
}
Status Hixl::TransferAsync(const AscendString &, TransferOp,
                           const std::vector<TransferOpDesc> &,
                           const TransferArgs &, TransferReq &req) {
  req = reinterpret_cast<void *>(0x5678);
  return record("TransferAsync");
}
Status Hixl::GetTransferStatus(const TransferReq &, TransferStatus &status) {
  status = transfer_status;
  return record("GetTransferStatus");
}
} // namespace hixl

extern "C" int hcomm_get_dev_va(int, void *, uint64_t, void **) { return 1; }

aclError aclrtGetDevice(int *device) {
  calls.emplace_back("aclrtGetDevice", PyGILState_Check() != 0);
  *device = current_device;
  return acl_errors["aclrtGetDevice"];
}
aclError aclrtGetCurrentContext(aclrtContext *context) {
  calls.emplace_back("aclrtGetCurrentContext", PyGILState_Check() != 0);
  *context = reinterpret_cast<aclrtContext>(current_context);
  return acl_errors["aclrtGetCurrentContext"];
}
aclError aclrtSetCurrentContext(aclrtContext context) {
  calls.emplace_back("aclrtSetCurrentContext", PyGILState_Check() != 0);
  auto ptr = reinterpret_cast<uintptr_t>(context);
  context_switches.push_back(ptr);
  auto error = set_context_errors[ptr];
  if (error == ACL_SUCCESS) {
    current_context = ptr;
  }
  return error;
}

PYBIND11_MODULE(_hixl_test_control, m) {
  m.def("reset", [] {
    result = 0;
    transfer_status = hixl::TransferStatus::COMPLETED;
    throw_cpp_error = false;
    calls.clear();
    current_device = 0;
    current_context = 0x4000;
    acl_errors.clear();
    set_context_errors.clear();
    context_switches.clear();
  });
  m.def("set_result", [](uint32_t value) { result = value; });
  m.def("set_transfer_status",
        [](hixl::TransferStatus value) { transfer_status = value; });
  m.def("throw_cpp_error", [] { throw_cpp_error = true; });
  m.def("calls", [] { return calls; });
  m.def("set_device", [](int value) { current_device = value; });
  m.def("set_context", [](uintptr_t value) { current_context = value; });
  m.def("context", [] { return current_context; });
  m.def("set_acl_error", [](const std::string &operation, aclError code) {
    acl_errors[operation] = code;
  });
  m.def("fail_set_context",
        [](uintptr_t ptr, aclError code) { set_context_errors[ptr] = code; });
  m.def("context_switches", [] { return context_switches; });
}
