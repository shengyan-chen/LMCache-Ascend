# Ordinary Host L1 for Ascend MP and Mooncake RDMA

This branch adds an opt-in Host allocator for **MP `lmcache_driven`**. On the
tested 910B hosts, ordinary Host buffers could be registered by the HNS RDMA
driver, while raw `aclrtMallocHost` buffers failed registration with `EINVAL`.
The option changes the L1 allocation; it does not fix the ACL/driver interaction.

## Behavior

```bash
# Existing behavior; also the default when unset.
export LMCACHE_ASCEND_MP_HOST_ALLOCATOR=acl

# New opt-in behavior, set before starting the 910B MP server.
export LMCACHE_ASCEND_MP_HOST_ALLOCATOR=ordinary
```

Unknown values, including an empty value, fail when the allocator is created.
Changing the variable does not migrate an existing pool; restart the MP server.

| | `acl` | `ordinary` |
|---|---|---|
| Allocation | Existing `alloc_pinned_ptr` | Aligned CPU tensor, `pin_memory=False` |
| NPU Host registration | Existing allocation-time registration | None |
| Per-chunk NPU pin | Skipped | Skipped intentionally |
| Release | Existing ACL helper | Tensor storage ownership |
| RDMA registration | Mooncake still registers the whole L1 | Same |

Ordinary mode keeps the L1 base address stable, preserves the requested base
alignment (use `--l1-align-bytes 65536`), and retains the existing chunk/address
manager. Capacity expansion is logged as ordinary Host capacity, not as NPU
pinned memory. The MR may lock ordinary pages for RDMA; this does **not** imply
that CANN treats them as asynchronous-copy pinned memory.

The native MP path gathers KV into NPU temporary buffers and calls
`aclrtMemcpyAsync` to/from the Host arena. The NPU kernels do not directly consume
the ordinary Host L1 pointer in this path. MP imports the serving engine's NPU
KV through device IPC; this option does not add a shared Host arena for workers.

Do not enable this option for engine-driven transfer, in-process connectors,
or HIXL_CS without separate validation. Keep the A100 allocator unchanged.
The existing whole-batch transfer-completion workaround is still required for
affected Mooncake versions; this allocator change does not replace it.

## Update an existing editable installation

On the 910B container, use the source directory reported by
`python3 -c 'import lmcache_ascend; print(lmcache_ascend.__file__)'`.
For the new environment it is expected to be `/home/csy/hetero_pd_xfer/LMCache-Ascend`.

```bash
cd /home/csy/hetero_pd_xfer/LMCache-Ascend
git status --short
# Preserve any local changes before switching. These commands do not force a checkout.
git fetch https://github.com/shengyan-chen/LMCache-Ascend.git \
  codex/ascend-mp-ordinary-host-l1
git switch -c codex/ascend-mp-ordinary-host-l1 FETCH_HEAD
```

If the local branch already exists, inspect it and fast-forward it instead of
running `switch -c` again. The implementation changes Python only, so an editable
installation of this checkout does not require rebuilding `c_ops` or Mooncake.
Confirm the loaded package path before proceeding. A different installed copy
will not pick up the checkout's changes.

## Make LMCache registration failures fatal

LMCache's transfer context currently ignores the result of `register_memory`.
The accompanying patch belongs to the **LMCache repository**, not this package.
Apply it separately on both nodes before starting services:

```bash
export ASCEND_SRC=/home/csy/hetero_pd_xfer/LMCache-Ascend
export LMCACHE_SRC=/home/csy/hetero_pd_xfer/LMCache
git -C "$LMCACHE_SRC" apply --check \
  "$ASCEND_SRC/patches/lmcache-mooncake-check-registration.patch"
git -C "$LMCACHE_SRC" apply \
  "$ASCEND_SRC/patches/lmcache-mooncake-check-registration.patch"
```

On A100, use its LMCache source directory (currently
`/mnt/hw/csy/hetero_pd_xfer/sources/LMCache`) and copy the patch file there.
Do not install LMCache-Ascend on A100. If the patch check fails, inspect the
existing function; do not overwrite a differently patched source file.
Successful application makes nonzero registration results stop initialization
before the P2P server is created. It does not modify completion polling.

## Validate the actual L1 allocator on 910B

The probe requires the updated editable package, torch_npu, numpy, and the
ordinary Mooncake RDMA build. It uses logical NPU 0 by default. Use the local
RDMA interface IP and verbs device name, not an NPU HCCL IP.

```bash
cd /home/csy/hetero_pd_xfer/LMCache-Ascend
unset MC_FORCE_TCP MC_USE_TENT MC_USE_TEV1 MC_SLICE_TIMEOUT
export MC_FORCE_HCA=1 MC_GID_INDEX=3 MC_MTU=1024
export LMCACHE_ASCEND_MP_HOST_ALLOCATOR=ordinary
# Set these to the values verified on this host.
export RDMA_IP=YOUR_LOCAL_RDMA_IPV4
export RDMA_DEVICE=YOUR_VERBS_DEVICE
set -o pipefail
timeout --kill-after=5s 180s python3 -u tools/probe_mp_host_allocator.py \
  --host "$RDMA_IP" --device "$RDMA_DEVICE" --size-mib 128 \
  2>&1 | tee /tmp/ordinary-mp-l1-probe.log
```

Expected final marker: `PASS_ORDINARY_MP_L1_RDMA_NPU`.
This validates real lazy expansion, a real L1 chunk, base alignment, whole-pool
MR registration, and full-byte NPU copies at the pool beginning, across a
64 MiB boundary, and at the end. It is a local functional probe, not a remote
READ test or a bandwidth benchmark. It has not been run on hardware as part of
the Mac-side implementation.

## Start the existing P2P deployment

In the 910B start script's `lmcache` action, before starting Python, add:

```bash
export LMCACHE_ASCEND_MP_HOST_ALLOCATOR=ordinary
```

Keep `--supported-transfer-mode lmcache_driven`, `--l1-align-bytes 65536`,
`MC_TE_PROTOCOL=rdma`, and the verified device/GID configuration. Existing chunk
size, tensor parallelism, model, and L1 capacity settings stay applicable.
The actual configured pool size must successfully register; a 128 MiB probe
does not guarantee that a 100 GiB registration fits the host's resources.

Restart the MP server and reconnect/restart its vLLM workers using the normal
deployment sequence. Confirm `MP Host L1 allocated: mode=ordinary` in its log.
First validate local and remote KV reuse and exact output correctness. Then run
the existing per-concurrency benchmark and metrics collection.

Ordinary Host copies may have different blocking/overlap behavior from ACL Host
copies. Record the allocator mode alongside performance results. Do not infer
network bandwidth from end-to-end request latency alone.

## Rollback and tests

Set the mode to `acl` (or remove the export) and restart the services to restore
the allocation behavior. This does not resolve the known ACL MR failure on the
affected HNS setup. Never free the arena before draining transfers and releasing
its MR. Ordinary tensor storage is retained while tensor views still own it.

CPU lifecycle tests (real CPU tensors, substituted native/upstream containers):

```bash
python3 -m pytest --noconftest tests/unit/test_mp_host_allocator.py -q
```

The repository's complete test suite additionally needs its LMCache bootstrap,
Ascend runtime and native extensions. CPU tests do not substitute for the
hardware probe or end-to-end accuracy verification.
