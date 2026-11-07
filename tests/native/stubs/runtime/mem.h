// SPDX-License-Identifier: Apache-2.0
#pragma once
constexpr int RT_MEMORY_TYPE_HOST = 0;
constexpr int RT_MEMORY_TYPE_USER = 1;
struct rtPointerAttributes_t {
  int memoryType;
};
inline int rtPointerGetAttributes(rtPointerAttributes_t *attributes, void *) {
  attributes->memoryType = RT_MEMORY_TYPE_HOST;
  return 0;
}
