// SPDX-License-Identifier: Apache-2.0
#pragma once
// Only permits linking the unexercised NUMA functions on non-Linux hosts.
constexpr int MPOL_BIND = 2;
constexpr unsigned int MPOL_MF_STRICT = 1;
constexpr unsigned int MPOL_MF_MOVE = 2;
inline long mbind(void *, unsigned long, int, const unsigned long *,
                  unsigned long, unsigned int) {
  return -1;
}
