# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest


def _import_and_patch_vllm_connector():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    lmcache_ascend = pytest.importorskip("lmcache_ascend")
    lmcache_ascend._patch_vllm_v1_adapter()
    return LMCacheConnectorV1


def _make_adapter(adapter_mod, *, store_async, kv_role, lmcache_engine):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = store_async
    adapter.kv_role = kv_role
    adapter._manager = SimpleNamespace(lmcache_engine=lmcache_engine)
    return adapter


def test_lmcache_connector_delegates_preemptions_after_ascend_patch():
    """Ascend patches the outer vLLM connector to delegate preemptions."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    preempted_req_ids = {"req-1", "req-2"}
    connector.handle_preemptions(preempted_req_ids)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        preempted_req_ids
    )


def test_lmcache_connector_preemption_patch_handles_no_inner_impl():
    """The Ascend patch should tolerate inner implementations without a hook."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = object()

    connector.handle_preemptions({"req-1"})


def test_ascend_adapter_drains_pending_stores_for_async_producer():
    """Async non-consumer workers must drain pending stores before reuse."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = {"req-1"}
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    preempted_req_ids = {"req-1", "req-2"}
    adapter.handle_preemptions(preempted_req_ids)

    lmcache_engine.wait_for_pending_stores.assert_called_once_with(preempted_req_ids)


@pytest.mark.parametrize(
    ("store_async", "kv_role", "has_engine"),
    [
        (False, "kv_both", True),
        (True, "kv_consumer", True),
        (True, "kv_both", False),
    ],
)
def test_ascend_adapter_skips_preemption_drain_when_not_required(
    store_async, kv_role, has_engine
):
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock() if has_engine else None
    adapter = _make_adapter(
        adapter_mod,
        store_async=store_async,
        kv_role=kv_role,
        lmcache_engine=lmcache_engine,
    )

    adapter.handle_preemptions({"req-1"})

    if has_engine:
        lmcache_engine.wait_for_pending_stores.assert_not_called()


@pytest.mark.parametrize("preempted", [None, set(), {"req-1"}])
@pytest.mark.parametrize("store_async", [False, True])
@pytest.mark.parametrize("kv_role", ["kv_both", "kv_consumer"])
def test_scheduler_metadata_drives_worker_preemption_cleanup(
    preempted, store_async, kv_role
):
    """vLLM 0.23 metadata must preserve IDs and reach the real worker hook."""
    # Standard
    import pickle

    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    scheduler = _make_adapter(
        adapter_mod, store_async=store_async, kv_role=kv_role, lmcache_engine=None
    )
    scheduler.force_skip_save = False
    output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        preempted_req_ids=None if preempted is None else set(preempted),
    )
    metadata = scheduler.build_connector_meta(output)
    assert metadata.preempted_req_ids == (preempted or set())
    if output.preempted_req_ids is not None:
        output.preempted_req_ids.add("later")
        assert "later" not in metadata.preempted_req_ids
    metadata = pickle.loads(pickle.dumps(metadata))

    engine = MagicMock()
    engine.wait_for_pending_stores.return_value = set()
    worker = _make_adapter(
        adapter_mod, store_async=store_async, kv_role=kv_role, lmcache_engine=engine
    )
    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = worker
    connector.handle_preemptions(metadata)

    if preempted:
        engine.lookup_unpin.assert_called_once_with("req-1")
    else:
        engine.lookup_unpin.assert_not_called()
    if preempted and store_async and kv_role != "kv_consumer":
        engine.wait_for_pending_stores.assert_called_once_with({"req-1"})
    else:
        engine.wait_for_pending_stores.assert_not_called()


def test_preemption_metadata_keeps_upstream_request_payload(monkeypatch):
    """Adding preemption IDs must not discard load/store request metadata."""
    _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    requests = [SimpleNamespace(req_id="req-to-load")]
    upstream_metadata = adapter_mod.LMCacheConnectorMetadata(requests=requests)
    monkeypatch.setattr(
        adapter_mod.LMCacheConnectorV1Impl,
        "build_connector_meta",
        lambda self, output: upstream_metadata,
    )
    scheduler = _make_adapter(
        adapter_mod, store_async=True, kv_role="kv_both", lmcache_engine=None
    )
    # Older schedulers may not provide a preempted_req_ids attribute.
    metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert isinstance(metadata, adapter_mod.LMCacheConnectorMetadata)
    assert metadata.requests is requests
    assert metadata.preempted_req_ids == set()
