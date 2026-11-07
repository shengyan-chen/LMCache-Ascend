# MP Mode (Multiprocess LMCache) on Ascend NPU

MP mode is the multiprocess deployment architecture of LMCache: the KV cache is owned by a standalone `lmcache server` process, and vLLM attaches to it through the `LMCacheMPConnector`. The server owns L1/L2 storage, eviction, and lookup; the engine talks to it over the MP protocol (ZMQ control plane + shared-memory / device-handle data plane). Compared with running LMCache inside the vLLM process, this keeps cache work (store, retrieve, chunking, eviction) out of the engine loop and allows one cache to be shared by several engines.

The data plane has two transfer modes, selected with the `lmcache.mp.mp_transfer_mode` key in `kv_connector_extra_config`:

| | `lmcache_driven` | `engine_driven` |
| :--- | :--- | :--- |
| Data path | The LMCache server pulls/pushes KV through device IPC handles | The vLLM workers gather/scatter the KV copies themselves, zero-copy from a named shared-memory pool |
| KV cache formats | Standard GQA/MHA, MLA/DSA multi-plane layouts (incl. DeepSeek-V4 `(latent, scale)`), and hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8; [qwen3_5.md](qwen3_5.md)) | Single KV-cache format models only (standard GQA/MHA, e.g. Qwen3); not Mamba/GDN or MLA/DSA |
| Server requirements | `--chunk-size 4096` for DeepSeek-V4 (see [lmcache_driven.md](lmcache_driven.md)) | `--shm-name <name> --no-l1-use-lazy` are mandatory (see [engine_driven.md](engine_driven.md)) |
| Deployment guide | [lmcache_driven.md](lmcache_driven.md) | [engine_driven.md](engine_driven.md) |

On NPU, leaving `lmcache.mp.mp_transfer_mode` unset (or `auto`) resolves to `lmcache_driven`, so DeepSeek-V4 needs no extra setting; to use `engine_driven` you must set it explicitly.

## Opt-in pure ACL Host memory for MP L1

The experimental pure ACL path uses `aclrtMallocHost` / `aclrtFreeHost`
without the Host mapping performed by the existing pinned allocation path.
It keeps the upstream Mixed allocator's object management and uses the raw
allocation address, without adding base-address alignment or padding.

Use the explicit Ascend entry point to install the MP memory hooks before
starting the upstream server:

```bash
LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR=aclrt_malloc_host \
python -m lmcache_ascend.v1.mp_server \
    --supported-transfer-mode lmcache_driven \
    --no-l1-use-lazy --shm-name "" --l1-size-gb 1
```

Pass the same options as `lmcache server`, without the `server` subcommand.
The entry point delegates argument parsing, HTTP/MP services and shutdown to
the upstream CLI. It validates the effective MP L1 configuration before SHM
capacity fallback, prepares the NPU runtime before ACL allocation, and installs
the matching Mixed close adapter. Configure the environment before startup.

This path requires NPU, non-Lazy DRAM L1, and no SHM, hugepages or devdax/GDS
backing. It does not support NUMA allocation. Allocation/initialization failures
are reported without falling back to the existing pinned path. With the variable
unset or set to `default`, the entry point leaves upstream functions unchanged.
Setting the variable alone does **not** install the hooks when using the original
`lmcache server` or `python -m lmcache.v1.multiprocess.server` commands.

This is the memory adaptation only; it does not enable a direct HIXL transfer
channel. CPU tests cover the wiring and memory lifecycle with native boundaries
substituted. Target CANN loading, NPU cold start, local transfers and HIXL
registration/transfer coexistence still require A3 validation.

## Supported models and limitations

**Supported in MP mode (both transfer modes, where the format applies):**

- Standard dense GQA/MHA models (for example Qwen3).
- Multi-plane MLA/DSA packed KV layouts (for example DeepSeek-V4) when using **`lmcache_driven`**.
- Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8, architecture `Qwen3_5ForConditionalGeneration`) when using **`lmcache_driven`**. The flags that match the LMCache recipe, and the Ascend differences, are in [qwen3_5.md](qwen3_5.md).

In the table above, “hybrid KV cache management formats” means **MLA/DSA multi-plane attention KV**, not SSM/Mamba hybrids.

**Not supported in MP mode (current release):**

- Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8) in **`engine_driven`**. Use `lmcache_driven`.
- `--mamba-cache-mode all`. These models require `align`.

## Documents

- [lmcache_driven.md](lmcache_driven.md) — the `lmcache_driven` transfer mode full deployment guide: environment preparation, container startup, source installation, server/vLLM startup, and verification (DeepSeek-V4 example).
- [engine_driven.md](engine_driven.md) — deploy and run the `engine_driven` transfer mode (Qwen3 example).
- [qwen3_5.md](qwen3_5.md) — hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8) on Ascend: same flags as the [LMCache recipe](https://docs.lmcache.ai/recipes/qwen3_5.html), plus the Ascend differences.

## Relationship to the LMCache dev branch

The MP architecture (the `lmcache server` CLI, the `LMCacheMPConnector`, and the transfer-mode layer) is developed on the LMCache `dev` branch — the main line — and the matching Ascend backends (NPU platform routing, block-transfer ops, hccl/hixl bindings) live in the LMCache-Ascend repository. Install both from the official repositories only — see the installation section of [lmcache_driven.md](lmcache_driven.md) for the exact procedure. The classic in-process connector (`LMCacheAscendConnectorV1Dynamic`) is a separate integration that uses its own LMCache release line; its installation must not be mixed with the MP-mode checkout in one Python environment.
