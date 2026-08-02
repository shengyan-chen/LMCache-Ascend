# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for AscendPDBackend and its sender/receiver mixins.

Unit tests use mocks (no NPU required).  Integration tests require NPU
hardware and are gated behind ``@pytest.mark.skipif``.
"""

# Standard
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import MagicMock, patch
import threading
import time

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.pd_backend import AllocRequest
import msgspec
import pytest
import torch

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.storage_backend.pd.messages import (
    AscendAllocResponse,
    AscendPDMsg,
    PullDoneSignal,
    PullReadyDoneAck,
    PullReadyNotif,
)

logger = init_logger(__name__)


def _make_key(key_id: str = "test_key") -> CacheEngineKey:
    return CacheEngineKey("test_model", 2, 0, hash(key_id), torch.bfloat16, None)


DEFAULT_SHAPE = torch.Size([2, 2, 256, 512])
DEFAULT_DTYPE = torch.bfloat16


def _make_metadata(
    kv_shape: tuple[int, int, int, int, int],
    use_mla: bool,
    kv_layer_groups_manager: KVLayerGroupsManager | None = None,
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role="worker",
        chunk_size=256,
        kv_layer_groups_manager=(
            kv_layer_groups_manager
            if kv_layer_groups_manager is not None
            else KVLayerGroupsManager()
        ),
    )


def _make_mock_mem_obj(
    shape: torch.Size = DEFAULT_SHAPE,
    dtype: torch.dtype = DEFAULT_DTYPE,
    address: int = 0,
) -> MagicMock:
    mock = MagicMock(spec=MemoryObj)
    mock.tensor = MagicMock()
    mock.data_ptr = 0xDEAD
    mock.meta = MagicMock(spec=MemoryObjMetadata)
    mock.meta.address = address
    mock.meta.shape = shape
    mock.meta.dtype = dtype
    mock.meta.fmt = MemoryFormat.KV_2LTD
    mock.ref_count_down = MagicMock()
    mock.ref_count_up = MagicMock()
    mock.unpin = MagicMock()
    mock.get_ref_count = MagicMock(return_value=1)
    return mock


def _make_consumed_proxy() -> ProxyMemoryObj:
    """Create a ProxyMemoryObj that is already consumed."""
    proxy = ProxyMemoryObj(
        backing_obj=None,
        transfer_channel=MagicMock(),
        target_peer_url="fake_url",
        remote_buffer_uuid="fake_uuid",
        remote_mem_index=0,
        transfer_context=MagicMock(),
        chunk_index=0,
        shapes=[DEFAULT_SHAPE],
        dtypes=[DEFAULT_DTYPE],
        fmt=MemoryFormat.KV_2LTD,
    )
    proxy.mark_consumed()
    return proxy


def _make_pd_backend_stub(
    role: str = "receiver",
    buffer_device: str = "npu:0",
    use_cpu_offload: bool = False,
    pull_mode: bool = False,
    delay_pull: bool = False,
    chunk_size: int = 256,
    kv_shape: Tuple[int, ...] = DEFAULT_SHAPE,
    kv_dtype: torch.dtype = DEFAULT_DTYPE,
):
    """Create a mock object with the minimal attributes needed by PD backend methods."""
    # First Party
    from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

    backend = MagicMock()
    backend.data = {}
    backend._pd_entries = {}
    backend._pd_request_keys = {}
    backend._pd_handoff_deadlines = {}
    backend._pd_handoff_lease_ttl = 300.0
    backend.data_lock = threading.Lock()
    backend.tp_rank = 0
    backend.pd_config = MagicMock()
    backend.pd_config.role = role
    backend.pd_config.buffer_device = buffer_device
    backend.use_cpu_offload = use_cpu_offload
    backend.pull_mode = pull_mode
    backend.delay_pull = delay_pull
    backend.running = True
    backend.transfer_channel = MagicMock()
    backend.memory_allocator = MagicMock()
    backend.full_chunk_size = chunk_size
    backend._fmt = MemoryFormat.KV_2LTD
    backend._kv_shapes = [DEFAULT_SHAPE]
    backend._kv_dtypes = [kv_dtype]

    # Wire internal delegation methods to their real implementations so tests
    # that call e.g. AscendPDBackend.contains(backend, ...) actually exercise
    # the eviction / partition logic instead of hitting auto-mocked no-ops.
    backend._lookup = lambda key, pin=False: AscendPDBackend._lookup(
        backend, key, pin=pin
    )
    backend._contains_and_pin = lambda key: AscendPDBackend._contains_and_pin(
        backend, key
    )
    backend._partition_keys = lambda keys: AscendPDBackend._partition_keys(
        backend, keys
    )
    backend._ensure_request_lease_locked = (
        lambda key, entry, request_id: AscendPDBackend._ensure_request_lease_locked(
            backend, key, entry, request_id
        )
    )
    backend._detach_request_lease_locked = (
        lambda key, entry, request_id: AscendPDBackend._detach_request_lease_locked(
            backend, key, entry, request_id
        )
    )
    backend._release_context_leases = (
        lambda contexts, request_id: AscendPDBackend._release_context_leases(
            contexts, request_id
        )
    )
    backend._refresh_handoff_deadline_locked = lambda lease_id: (
        AscendPDBackend._refresh_handoff_deadline_locked(backend, lease_id)
    )
    backend._delete_pd_entry_locked = (
        lambda key, entry, release_obj: AscendPDBackend._delete_pd_entry_locked(
            backend, key, entry, release_obj
        )
    )
    backend._partition_keys_with_handoff = lambda keys, handoff_id: (
        AscendPDBackend._partition_keys_with_handoff(backend, keys, handoff_id)
    )
    backend.put_with_handoff_lease = lambda key, obj, handoff_id: (
        AscendPDBackend.put_with_handoff_lease(backend, key, obj, handoff_id)
    )
    backend.promote_handoff_lease = lambda handoff_id, request_id: (
        AscendPDBackend.promote_handoff_lease(backend, handoff_id, request_id)
    )
    backend.release_request_lease = (
        lambda request_id, expected_handoff_deadline=None: (
            AscendPDBackend.release_request_lease(
                backend,
                request_id,
                expected_handoff_deadline=expected_handoff_deadline,
            )
        )
    )
    backend.release_handoff_lease = lambda handoff_id: (
        AscendPDBackend.release_handoff_lease(backend, handoff_id)
    )
    backend.release_expired_handoff_leases = lambda: (
        AscendPDBackend.release_expired_handoff_leases(backend)
    )

    return backend


class TestAscendPDBackend:
    """Mock-based unit tests for AscendPDBackend logic."""

    def test_runtime_dsa_layout_overrides_legacy_metadata(self):
        """The registered DSA tuple, not legacy 576, defines the PD page."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        layer_groups = KVLayerGroupsManager()
        kv_caches = {
            f"model.layers.{layer_idx}.self_attn.attn": (
                torch.empty((1, 1, 1, 512), dtype=torch.bfloat16),
                torch.empty((1, 1, 1, 64), dtype=torch.bfloat16),
                torch.empty((1, 1, 1, 128), dtype=torch.bfloat16),
            )
            for layer_idx in range(78)
        }
        layer_groups.build_kv_layer_groups(kv_caches)
        metadata = _make_metadata(
            kv_shape=(78, 1, 256, 1, 576),
            use_mla=True,
            kv_layer_groups_manager=layer_groups,
        )

        fmt, shapes, dtypes, page_size_bytes = (
            AscendPDBackend._resolve_page_layout(
                SimpleNamespace(chunk_size=256), metadata
            )
        )

        assert metadata.kv_shape[-1] == 576
        assert layer_groups.num_groups == 1
        assert layer_groups.kv_layer_groups[0].hidden_dim_size == 704
        assert fmt is MemoryFormat.KV_MLA_FMT
        assert shapes == [torch.Size([1, 78, 256, 704])]
        assert dtypes == [torch.bfloat16]
        assert page_size_bytes == 28_114_944

    def test_mha_legacy_layout_fallback_preserves_page_size(self):
        """A standard MHA model remains compatible without runtime groups."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        metadata = _make_metadata(
            kv_shape=(32, 2, 256, 8, 128),
            use_mla=False,
        )

        fmt, shapes, dtypes, page_size_bytes = (
            AscendPDBackend._resolve_page_layout(
                SimpleNamespace(chunk_size=256), metadata
            )
        )

        assert fmt is MemoryFormat.KV_2LTD
        assert shapes == [torch.Size([2, 32, 256, 1024])]
        assert dtypes == [torch.bfloat16]
        assert page_size_bytes == 33_554_432

    def test_allocator_uses_resolved_page_layout(self):
        """Both NPU and CPU allocators receive the canonical DSA page."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        page_size_bytes = 28_114_944
        shape = torch.Size([1, 78, 256, 704])
        backend = object.__new__(AscendPDBackend)
        backend.pd_config = SimpleNamespace(buffer_device="npu:0")
        backend.use_cpu_offload = True
        backend._fmt = MemoryFormat.KV_MLA_FMT
        backend._kv_shapes = [shape]
        backend._kv_dtypes = [torch.bfloat16]
        backend._page_size_bytes = page_size_bytes

        config = SimpleNamespace(
            pd_buffer_size=page_size_bytes * 2 + 1,
            pd_cpu_buffer_size=page_size_bytes * 3 + 1,
        )
        metadata = SimpleNamespace(worker_id=0)
        allocator = MagicMock()

        with (
            patch(
                "lmcache_ascend.v1.storage_backend.pd.backend."
                "PagedCpuGpuMemoryAllocator",
                return_value=allocator,
            ),
            patch(
                "lmcache_ascend.v1.storage_backend.pd.backend.get_correct_device",
                return_value="npu:0",
            ),
            patch(
                "lmcache_ascend.v1.storage_backend.pd.backend.torch.npu.set_device"
            ),
        ):
            result = AscendPDBackend.initialize_allocator(
                backend, config, metadata
            )

        assert result is allocator
        allocator.init_gpu_memory_allocator.assert_called_once_with(
            page_size_bytes * 3,
            [shape],
            [torch.bfloat16],
            MemoryFormat.KV_MLA_FMT,
            "npu:0",
        )
        allocator.init_cpu_memory_allocator.assert_called_once_with(
            page_size_bytes * 4,
            [shape],
            [torch.bfloat16],
            MemoryFormat.KV_MLA_FMT,
        )

    def test_registered_buffers_reject_mismatched_page_size(self):
        """HCCL buffers cannot be registered with different page sizes."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = object.__new__(AscendPDBackend)
        backend._page_size_bytes = 28_114_944

        backend._validate_registered_buffers([28_114_944, 28_114_944])
        with pytest.raises(RuntimeError, match="resolved page size"):
            backend._validate_registered_buffers([28_114_944, 23_003_136])

    def test_pd_message_types(self):
        """All Ascend PD message types roundtrip through msgspec."""
        msgs = [
            AllocRequest(
                keys=["k1", "k2"],
                fmt=MemoryFormat.KV_2LTD.value,
                shape=list(DEFAULT_SHAPE),
                dtype="bfloat16",
                last_chunk_toks=256,
            ),
            AscendAllocResponse(
                already_sent_indexes=[0],
                remote_indexes=[1, 2],
                remote_buffer_uuids=["uuid-a", "uuid-b"],
                alloc_failed=False,
            ),
            PullReadyNotif(
                pull_id="pull_1",
                handoff_id="handoff_1",
                keys=["k1"],
                sender_buffer_uuids=["suuid-1"],
                sender_mem_indexes=[0],
                sender_id="sender_1",
                sender_done_url="tcp://sender:9999",
                fmt=MemoryFormat.KV_2LTD.value,
                shape=list(DEFAULT_SHAPE),
                dtype="bfloat16",
                last_chunk_toks=256,
            ),
            PullReadyDoneAck(
                already_sent_indexes=[],
                alloc_failed=False,
            ),
            PullDoneSignal(pull_id="pull_1"),
        ]
        for msg in msgs:
            encoded = msgspec.msgpack.encode(msg)
            decoded = msgspec.msgpack.decode(encoded, type=AscendPDMsg)
            assert type(decoded) is type(msg)

    def test_allocate_receiver_uses_gpu(self):
        """Receiver allocates on GPU (NPU)."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub(
            role="receiver",
            buffer_device="npu:0",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=DEFAULT_DTYPE,
            chunk_size=256,
            pull_mode=False,
            delay_pull=False,
            use_cpu_offload=False,
        )
        backend.memory_allocator.allocate = MagicMock(return_value="gpu_obj")

        result = AscendPDBackend.allocate(
            backend,
            DEFAULT_SHAPE,
            DEFAULT_DTYPE,
            MemoryFormat.KV_2LTD,
        )

        backend.memory_allocator.allocate.assert_called_once()
        call_kwargs = backend.memory_allocator.allocate.call_args
        assert call_kwargs.kwargs.get("allocator_type") == "gpu"
        assert result == "gpu_obj"

    def test_allocate_sender_with_offload_uses_cpu(self):
        """Sender with cpu_offload allocates on CPU."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub(
            role="sender",
            buffer_device="npu:0",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=DEFAULT_DTYPE,
            chunk_size=256,
            pull_mode=False,
            delay_pull=False,
            use_cpu_offload=True,
        )
        backend.memory_allocator.allocate = MagicMock(return_value="cpu_obj")

        result = AscendPDBackend.allocate(
            backend,
            DEFAULT_SHAPE,
            DEFAULT_DTYPE,
            MemoryFormat.KV_2LTD,
        )

        call_kwargs = backend.memory_allocator.allocate.call_args
        assert call_kwargs.kwargs.get("allocator_type") == "cpu"
        assert result == "cpu_obj"

    def test_contains_evicts_consumed_proxy(self):
        """Consumed ProxyMemoryObj is evicted from data on contains()."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("consumed_key")
        AscendPDBackend.put(backend, key, _make_consumed_proxy())

        result = AscendPDBackend.contains(backend, key, pin=False)

        assert result is False
        assert key not in backend.data

    def test_contains_normal_obj_returns_true(self):
        """Regular MemoryObj is found by contains()."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("normal_key")
        AscendPDBackend.put(backend, key, _make_mock_mem_obj())

        result = AscendPDBackend.contains(backend, key, pin=False)
        assert result is True

    def test_contains_missing_key(self):
        """Missing key returns False."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("missing")

        result = AscendPDBackend.contains(backend, key, pin=False)
        assert result is False

    def test_contains_pin_calls_ref_count_up(self):
        """Pinning a key calls ref_count_up on the object."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("pin_key")
        mock_obj = _make_mock_mem_obj()
        AscendPDBackend.put(backend, key, mock_obj)

        result = AscendPDBackend.contains(backend, key, pin=True)

        assert result is True
        mock_obj.ref_count_up.assert_called_once()

    def test_partition_keys(self):
        """Keys are partitioned into already-sent and new indexes."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key0 = _make_key("k0")
        key1 = _make_key("k1")
        key2 = _make_key("k2")

        mock_obj0 = _make_mock_mem_obj()
        AscendPDBackend.put(backend, key0, mock_obj0)

        str_keys = [key0.to_string(), key1.to_string(), key2.to_string()]

        already_sent_idx, already_sent_objs, new_idx = AscendPDBackend._partition_keys(
            backend, str_keys
        )

        assert already_sent_idx == [0]
        assert len(already_sent_objs) == 1
        assert already_sent_objs[0] is mock_obj0
        assert new_idx == [1, 2]
        mock_obj0.ref_count_up.assert_called_once()

    def test_push_mode_allocate_and_put(self):
        """Push-mode allocate_and_put returns UUID-based refs."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        mock_obj = _make_mock_mem_obj()
        backend.allocate = MagicMock(return_value=mock_obj)
        backend.put = MagicMock()
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-alloc"],
            [42],
        )

        alloc_req = AllocRequest(
            keys=[_make_key("k1").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert isinstance(resp, AscendAllocResponse)
        assert resp.alloc_failed is False
        assert resp.remote_buffer_uuids == ["uuid-alloc"]
        assert resp.remote_indexes == [42]
        assert resp.already_sent_indexes == []
        backend.put.assert_called_once()

    def test_push_mode_alloc_failure(self):
        """Push-mode allocation failure returns alloc_failed=True."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.allocate = MagicMock(return_value=None)
        backend.put = MagicMock()

        alloc_req = AllocRequest(
            keys=[_make_key("k1").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=None,
        ):
            resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert isinstance(resp, AscendAllocResponse)
        assert resp.alloc_failed is True
        backend.put.assert_not_called()

    def test_pull_eager_flow(self):
        """Pull-eager: allocates, reads from sender, returns ack + callback."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        mock_obj = _make_mock_mem_obj()
        backend.allocate = MagicMock(return_value=mock_obj)
        backend.transfer_channel.batched_read = MagicMock(return_value=1)
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_eager_1",
            handoff_id="handoff_eager_1",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-1"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=mock_obj,
        ):
            ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_eager(
                backend, msg, "sender_1"
            )

        assert isinstance(ack, PullReadyDoneAck)
        assert ack.alloc_failed is False
        assert ack.already_sent_indexes == []
        backend.transfer_channel.batched_read.assert_called_once()
        key = _make_key("k1")
        assert backend._pd_entries[key].base_obj is mock_obj
        assert "__lmcache_pd_handoff__:handoff_eager_1" in (
            backend._pd_entries[key].owners
        )

        # Post-ack callback sends Done signal
        assert post_ack_fn is not None
        post_ack_fn()
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_1", "pull_eager_1"
        )

    def test_pull_eager_alloc_failure(self):
        """Pull-eager with alloc failure returns alloc_failed=True."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.allocate = MagicMock(return_value=None)
        backend.put = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_fail",
            handoff_id="handoff_fail",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-1"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=None,
        ):
            ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_eager(
                backend, msg, "sender_1"
            )

        assert ack.alloc_failed is True
        assert post_ack_fn is None

    def test_pull_delay_flow(self):
        """Pull-delay creates ProxyMemoryObj instances in data store."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu:0",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=True,
            use_cpu_offload=True,
        )
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_delay_1",
            handoff_id="handoff_delay_1",
            keys=[_make_key("k1").to_string(), _make_key("k2").to_string()],
            sender_buffer_uuids=["suuid-0", "suuid-1"],
            sender_mem_indexes=[0, 1],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_delay(
            backend, msg, "sender_1"
        )

        assert isinstance(ack, PullReadyDoneAck)
        assert ack.alloc_failed is False
        assert post_ack_fn is None
        assert len(backend._pd_entries) == 2
        for entry in backend._pd_entries.values():
            assert isinstance(entry.base_obj, ProxyMemoryObj)
            assert "__lmcache_pd_handoff__:handoff_delay_1" in entry.owners

    def test_pull_delay_transfer_context_done_callback_is_idempotent(self):
        """Delay-pull transfer context sends done signal at most once."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=True,
            use_cpu_offload=True,
        )
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_delay_done_once",
            handoff_id="handoff_delay_done_once",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-0"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_delay(
            backend, msg, "sender_1"
        )
        assert isinstance(ack, PullReadyDoneAck)
        assert post_ack_fn is None
        assert len(backend._pd_entries) == 1
        proxy_obj = next(iter(backend._pd_entries.values())).base_obj
        assert isinstance(proxy_obj, ProxyMemoryObj)

        transfer_ctx = proxy_obj.transfer_context
        backend.promote_handoff_lease(
            "handoff_delay_done_once", "decode_request_1"
        )
        backend.release_request_lease("decode_request_1")
        transfer_ctx.send_done_now()
        transfer_ctx.send_done_now()
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_1", "pull_delay_done_once"
        )

    def test_pull_delay_handoff_closes_pullready_to_lookup_race(self):
        """An old owner cannot send Done between shared hit ack and lookup."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu:0",
            pull_mode=True,
        )
        backend._send_pull_done_to_sender = MagicMock()
        key = _make_key("shared")

        first = PullReadyNotif(
            pull_id="pull_old",
            handoff_id="handoff_old",
            keys=[key.to_string()],
            sender_buffer_uuids=["old-uuid"],
            sender_mem_indexes=[0],
            sender_id="sender_old",
            sender_done_url="tcp://sender-old:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )
        first_ack, _ = AscendPDReceiverMixin._handle_pull_delay(
            backend, first, "sender_old"
        )
        assert first_ack.already_sent_indexes == []
        assert backend.promote_handoff_lease("handoff_old", "request_old") == 1

        second = PullReadyNotif(
            pull_id="pull_new",
            handoff_id="handoff_new",
            keys=[key.to_string()],
            sender_buffer_uuids=["new-uuid"],
            sender_mem_indexes=[1],
            sender_id="sender_new",
            sender_done_url="tcp://sender-new:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )
        second_ack, _ = AscendPDReceiverMixin._handle_pull_delay(
            backend, second, "sender_new"
        )
        assert second_ack.already_sent_indexes == [0]

        # This is the old failing interleaving: request_old releases after
        # PullReady ack but before request_new lookup.  The synthetic handoff
        # owner keeps both the entry and producer context alive.
        backend.release_request_lease("request_old")
        assert key in backend._pd_entries
        backend._send_pull_done_to_sender.assert_not_called()

        assert backend.promote_handoff_lease("handoff_new", "request_new") == 1
        assert backend._pd_entries[key].owners == {"request_new"}
        backend.release_request_lease("request_new")

        assert key not in backend._pd_entries
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_old", "pull_old"
        )

    def test_handoff_control_config_is_removed_from_cache_namespace(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.handoff import (
            split_pd_handoff_request_config,
        )

        original = {
            "lmcache.pd_handoff_id": "handoff-123",
            "lmcache.tag.tenant": "tenant-a",
        }
        handoff_id, sanitized = split_pd_handoff_request_config(original)

        assert handoff_id == "handoff-123"
        assert sanitized == {"lmcache.tag.tenant": "tenant-a"}
        assert "lmcache.pd_handoff_id" in original

    def test_pull_sender_propagates_logical_handoff_id(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = _make_pd_backend_stub(role="sender", pull_mode=True)
        backend._wait_for_backpressure = MagicMock()
        backend._ensure_peer_connection = MagicMock()
        backend._sender_done_url = "tcp://sender:9999"
        backend._pull_pending = {}
        backend._early_pull_done = set()
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_pinned_count = 0
        backend._pull_pending_ttl = 360.0
        backend.proxy_side_channel = MagicMock()

        side_channel = MagicMock()
        side_channel.recv.return_value = msgspec.msgpack.encode(
            PullReadyDoneAck(already_sent_indexes=[0])
        )
        backend.mem_alloc_sockets = {"decoder_7710": side_channel}
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["sender-uuid"],
            [7],
        )

        transfer_spec = SimpleNamespace(
            req_id="logical-handoff-123",
            receiver_host="decoder_",
            receiver_init_port=[7710],
            receiver_alloc_port=[7810],
            is_last_prefill=False,
        )
        mem_obj = _make_mock_mem_obj()

        AscendPDSenderMixin._batched_submit_put_task_pull(
            backend,
            [_make_key("sender-propagation")],
            [mem_obj],
            transfer_spec,
        )

        encoded = side_channel.send.call_args.args[0]
        decoded = msgspec.msgpack.decode(encoded, type=AscendPDMsg)
        assert isinstance(decoded, PullReadyNotif)
        assert decoded.handoff_id == "logical-handoff-123"

    def test_proxy_submit_resolve_batch_fallback_uses_sync_batched_read(self):
        """No submit_batched_read: fallback uses synchronous batched_read."""

        class _NoSubmitChannel:
            def __init__(self):
                self.batched_read = MagicMock(return_value=1)

        transfer_channel = _NoSubmitChannel()

        proxy = ProxyMemoryObj(
            backing_obj=None,
            transfer_channel=transfer_channel,
            target_peer_url="sender_1",
            remote_buffer_uuid="suuid-0",
            remote_mem_index=0,
            transfer_context=MagicMock(_loop=None),
            chunk_index=0,
            shapes=[DEFAULT_SHAPE],
            dtypes=[DEFAULT_DTYPE],
            fmt=MemoryFormat.KV_2LTD,
        )
        backing_obj = _make_mock_mem_obj()
        proxy.set_backing_obj(backing_obj)

        event = ProxyMemoryObj.submit_resolve_batch([proxy])

        assert event is None
        assert proxy.resolved is True
        transfer_channel.batched_read.assert_called_once()

    def test_proxy_submit_resolve_batch_uses_submit_when_supported(self):
        """submit_batched_read path returns event and marks proxies resolved."""
        transfer_channel = MagicMock()
        expected_event = MagicMock()
        transfer_channel.submit_batched_read = MagicMock(return_value=expected_event)
        transfer_channel.batched_read = MagicMock()

        proxy = ProxyMemoryObj(
            backing_obj=None,
            transfer_channel=transfer_channel,
            target_peer_url="sender_1",
            remote_buffer_uuid="suuid-0",
            remote_mem_index=0,
            transfer_context=MagicMock(_loop=None),
            chunk_index=0,
            shapes=[DEFAULT_SHAPE],
            dtypes=[DEFAULT_DTYPE],
            fmt=MemoryFormat.KV_2LTD,
        )
        backing_obj = _make_mock_mem_obj()
        proxy.set_backing_obj(backing_obj)

        event = ProxyMemoryObj.submit_resolve_batch([proxy])

        assert event is expected_event
        assert proxy.resolved is True
        transfer_channel.submit_batched_read.assert_called_once()
        transfer_channel.batched_read.assert_not_called()

    def test_circuit_breaker_skips_backed_off_peer(self):
        """When peer is backed off, put task is skipped."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = _make_pd_backend_stub(role="sender")
        backend._peer_alloc_backoff = {
            "receiver_1234": time.monotonic() + 60,
        }
        backend._peer_alloc_backoff_lock = threading.Lock()
        backend.tp_rank = 0
        backend.proxy_side_channel = MagicMock()
        backend._ensure_peer_connection = MagicMock()
        backend._remote_allocate = MagicMock()

        transfer_spec = MagicMock()
        transfer_spec.receiver_init_port = [1234]
        transfer_spec.receiver_host = "receiver_"
        transfer_spec.is_last_prefill = True
        transfer_spec.req_id = "req_1"

        mock_objs = [_make_mock_mem_obj()]

        AscendPDSenderMixin.batched_submit_put_task(
            backend, [_make_key("k1")], mock_objs, transfer_spec
        )

        # Should NOT have called _ensure_peer_connection or _remote_allocate
        backend._ensure_peer_connection.assert_not_called()
        backend._remote_allocate.assert_not_called()
        # Should still send proxy notification for last prefill
        backend.proxy_side_channel.send.assert_called_once()

    def test_handle_pull_done_releases_resources(self):
        """_handle_pull_done releases pinned MemObjs."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        mock_obj = _make_mock_mem_obj()
        backend._pull_pending = {"pull_1": (time.monotonic(), [mock_obj])}
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_pinned_count = 1
        backend._early_pull_done = set()

        AscendPDSenderMixin._handle_pull_done(backend, "pull_1")

        assert "pull_1" not in backend._pull_pending
        mock_obj.ref_count_down.assert_called_once()
        assert backend._pull_pending_pinned_count == 0

    def test_handle_pull_done_early_signal(self):
        """Early Done signal is buffered for later processing."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        backend._pull_pending = {}
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_pinned_count = 0
        backend._early_pull_done = set()

        AscendPDSenderMixin._handle_pull_done(backend, "pull_early")

        assert "pull_early" in backend._early_pull_done

    def test_backpressure_blocks_when_above_hwm(self):
        """_wait_for_backpressure blocks until count drops below HWM."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_hwm = 5
        # Start above HWM, then release in background
        backend._pull_pending_pinned_count = 10

        released = threading.Event()

        def release_after_delay():
            time.sleep(0.05)
            with backend._pull_pending_lock:
                backend._pull_pending_pinned_count = 0
            released.set()

        t = threading.Thread(target=release_after_delay, daemon=True)
        t.start()

        # This should block until count drops
        AscendPDSenderMixin._wait_for_backpressure(backend, 2)

        assert released.is_set()
        t.join(timeout=2)

    def test_sweep_expired_pull_pending(self):
        """Expired entries are released by the sweep."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_ttl = 0.001

        mock_obj = _make_mock_mem_obj()
        # Entry pinned well in the past
        backend._pull_pending = {
            "expired_pull": (0.0, [mock_obj]),
        }
        backend._pull_pending_pinned_count = 1

        time.sleep(0.01)
        AscendPDSenderMixin._sweep_expired_pull_pending(backend)

        assert "expired_pull" not in backend._pull_pending
        mock_obj.ref_count_down.assert_called_once()
        assert backend._pull_pending_pinned_count == 0

    def test_allocate_and_put_with_already_sent(self):
        """Already-sent keys are identified and not re-allocated."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()

        key0 = _make_key("existing")
        existing_obj = _make_mock_mem_obj()
        backend.data[key0] = existing_obj

        new_obj = _make_mock_mem_obj(address=1)
        backend.allocate = MagicMock(return_value=new_obj)
        backend.put = MagicMock()
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-new"],
            [1],
        )

        alloc_req = AllocRequest(
            keys=[key0.to_string(), _make_key("new_key").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=new_obj,
        ):
            resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert resp.already_sent_indexes == [0]
        assert len(resp.remote_buffer_uuids) == 1
        assert resp.alloc_failed is False
        # Only the new key was put
        backend.put.assert_called_once()
        # Already-sent obj was unpinned
        existing_obj.ref_count_down.assert_called()
