# Ordinary Host L1 for Ascend MP

Implement the design approved in the conversation on 2026-10-31. The existing
`v0.5.6rc4` branch remains the base; the HIXL_CS fork is not changed.

Add `LMCACHE_ASCEND_MP_HOST_ALLOCATOR=acl|ordinary`, defaulting to `acl`.
Ordinary mode uses an aligned CPU tensor with `pin_memory=False` and skips
NPU host registration. Keep tensor ownership, chunk allocation, capacity
expansion and the native staging-copy interface. Close must stop expansion,
be idempotent, and never pass ordinary memory to ACL free/unregister.

Prefer Python-owned storage over new C++ malloc/free bindings so the existing
native extension remains usable. Preserve the existing ACL allocation and
release functions. This is an opt-in prototype for MP `lmcache_driven` with
ordinary Mooncake RDMA; it does not establish support for other transfer modes.

Supply CPU-side lifecycle tests, an optional NPU/RDMA allocator probe, and
server rollout instructions. Include a separately applicable LMCache patch
checking Mooncake's registration return code, since that file belongs to the
other repository. Existing whole-batch completion handling remains required.

The current Mac has no NPU/RDMA devices. Report hardware validation as pending;
the prior 4 MiB experiments do not validate the final L1 implementation.
