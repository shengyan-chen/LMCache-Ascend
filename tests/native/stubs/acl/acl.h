// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
using aclError = int;
constexpr aclError ACL_SUCCESS = 0;
constexpr aclError ACL_ERROR_RT_CONTEXT_NULL = 107002;
using aclrtContext = void *;
aclError aclrtGetDevice(int *device);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtMallocHost(void **ptr, std::size_t size);
aclError aclrtFreeHost(void *ptr);
const char *aclrtGetSocName();
enum class aclrtLastErrLevel { ACL_RT_THREAD_LEVEL };
aclError aclrtGetLastError(aclrtLastErrLevel level);
