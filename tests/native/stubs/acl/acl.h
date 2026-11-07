// SPDX-License-Identifier: Apache-2.0
#pragma once
using aclError = int;
constexpr aclError ACL_SUCCESS = 0;
constexpr aclError ACL_ERROR_RT_CONTEXT_NULL = 107002;
using aclrtContext = void *;
aclError aclrtGetDevice(int *device);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtSetCurrentContext(aclrtContext context);
