# Ordinary Host L1 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in ordinary Host L1 allocator without changing ACL defaults.

**Architecture:** Extend the existing LazyMemoryAllocator patch in Python.
Retain the upstream TensorMemoryAllocator and address management. Ship the
separate LMCache registration check as an applicable patch with instructions.

**Tech Stack:** Python, PyTorch CPU, pytest; optional Ascend/Mooncake validation.

**Spec:** `docs/superpowers/specs/2026-10-31-ordinary-host-l1.md`

## Global Constraints

- Default `acl`; explicit `ordinary`; unknown values fail at construction.
- No NPU pin/free for ordinary buffers; preserve 64 KiB base alignment.
- Keep current compiled extensions and HIXL_CS checkout unchanged.
- Do not claim NPU/RDMA validation on the Mac.

## Review Focus

- Environment changes after allocation must not change deallocation ownership.
- Alignment/size errors must fail before allocation or background thread start.
- Close must be idempotent and preserve storage referenced by live views.
- Ordinary expansion must not invoke NPU registration or report pinned memory.
- Registration failure must stop server startup before exposing the P2P service.

## Tasks

- [x] Add CPU tests in `tests/unit/test_mp_host_allocator.py`; exercise ordinary
   allocation, alignment, aliasing, invalid config, expansion, and both close
   modes. Run against current code and confirm expected failures.
- [x] Extend `lmcache_ascend/__init__.py` allocator patch; run CPU tests and lint.
- [x] Add `tools/probe_mp_host_allocator.py`, a separate LMCache patch under
   `patches/`, and `docs/mp_ordinary_host_l1.md` with installation, rollout and
   rollback commands. Update local 910B launcher to enable ordinary mode.
- [ ] Review the complete diff, run available tests, record hardware/full-suite
   limitations, commit with the user's existing identity and push only after
   confirming the personal repository's public/private destination.

## Verification record

- New lifecycle tests before implementation: 14 failed, 1 passed as expected.
- After implementation: 15 CPU tests passed; Ruff, isort and diff checks passed.
- Existing deployment tests: 10 passed; shell syntax and dry-run passed.
- LMCache patch: `git apply --check` passed against the adjacent source; isolated
  context execution confirmed that `-202` raises before the P2P server starts,
  while `0` continues initialization.
- Full repository pytest cannot load conftest on this Mac because this source
  checkout has no generated `lmcache_ascend._version`; the installed Ascend
  runtime/native extensions are also unavailable here. No full-suite or
  hardware pass is claimed.
- Independent review found no blocking new defect. Tests substitute upstream
  containers/native ACL functions; real capacity expansion and MR/NPU copies
  remain server probe work. Ordinary close retains storage while owners live.
- User explicitly confirmed pushing the new branch to the public personal fork
  `shengyan-chen/LMCache-Ascend`. No community PR or default-branch update.
