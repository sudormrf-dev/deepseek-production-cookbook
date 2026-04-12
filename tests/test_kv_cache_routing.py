"""Tests for kv_cache_routing.py."""

from __future__ import annotations

import pytest

from patterns.kv_cache_routing import (
    KVCacheRouter,
    RequestProfile,
    RoutingDecision,
    RoutingStrategy,
    estimate_kv_size_gb,
)


class TestRequestProfile:
    def test_prefix_hash_is_hex(self):
        req = RequestProfile(request_id="r1", prefix="system prompt here")
        assert len(req.prefix_hash) == 16
        int(req.prefix_hash, 16)  # should not raise

    def test_same_prefix_same_hash(self):
        req1 = RequestProfile(request_id="a", prefix="same prefix")
        req2 = RequestProfile(request_id="b", prefix="same prefix")
        assert req1.prefix_hash == req2.prefix_hash

    def test_different_prefix_different_hash(self):
        req1 = RequestProfile(request_id="a", prefix="prefix A")
        req2 = RequestProfile(request_id="b", prefix="prefix B")
        assert req1.prefix_hash != req2.prefix_hash


class TestKVCacheRouter:
    def _router(self, strategy: RoutingStrategy = RoutingStrategy.PREFIX_HASH) -> KVCacheRouter:
        return KVCacheRouter(
            replicas=["gpu0", "gpu1", "gpu2", "gpu3"],
            strategy=strategy,
        )

    def test_route_returns_decision(self):
        router = self._router()
        req = RequestProfile(request_id="r1", prefix="test prefix")
        decision = router.route(req)
        assert isinstance(decision, RoutingDecision)
        assert decision.target_replica in router.replica_ids

    def test_same_prefix_routes_consistently(self):
        router = self._router()
        req1 = RequestProfile(request_id="r1", prefix="shared prefix")
        req2 = RequestProfile(request_id="r2", prefix="shared prefix")
        d1 = router.route(req1)
        d2 = router.route(req2)
        assert d1.target_replica == d2.target_replica

    def test_cache_hit_after_registration(self):
        router = self._router()
        req = RequestProfile(request_id="r1", prefix="system: you are helpful")
        decision1 = router.route(req)
        # Simulate the first request completing and registering cache
        router.register_cache_entry(decision1.target_replica, req.prefix_hash, 0.1)
        # Second identical request should hit the cache
        decision2 = router.route(RequestProfile(request_id="r2", prefix=req.prefix))
        assert decision2.cache_hit is True
        assert decision2.target_replica == decision1.target_replica

    def test_round_robin_distributes(self):
        router = self._router(RoutingStrategy.ROUND_ROBIN)
        targets = set()
        for i in range(8):
            req = RequestProfile(request_id=f"r{i}", prefix=f"unique {i}")
            d = router.route(req)
            targets.add(d.target_replica)
        assert len(targets) >= 2

    def test_least_loaded_strategy(self):
        router = self._router(RoutingStrategy.LEAST_LOADED)
        req = RequestProfile(request_id="r1", prefix="test")
        d = router.route(req)
        assert d.target_replica in router.replica_ids

    def test_unhealthy_replica_skipped(self):
        router = self._router()
        router.set_replica_health("gpu0", False)
        router.set_replica_health("gpu1", False)
        router.set_replica_health("gpu2", False)
        req = RequestProfile(request_id="r1", prefix="test")
        d = router.route(req)
        assert d.target_replica == "gpu3"

    def test_no_replicas_raises(self):
        with pytest.raises((ValueError, IndexError)):
            KVCacheRouter(replicas=[])

    def test_evict_cache_entry(self):
        router = self._router()
        req = RequestProfile(request_id="r1", prefix="evict me")
        d = router.route(req)
        router.register_cache_entry(d.target_replica, req.prefix_hash, 1.0)
        router.evict_cache_entry(d.target_replica, req.prefix_hash, 1.0)
        # After eviction, should not hit cache
        d2 = router.route(RequestProfile(request_id="r2", prefix=req.prefix))
        assert d2.cache_hit is False

    def test_ttft_reduction_on_hit(self):
        router = self._router()
        req = RequestProfile(request_id="r1", prefix="long system prompt" * 20, prompt_len=512)
        d = router.route(req)
        router.register_cache_entry(d.target_replica, req.prefix_hash, 0.5)
        d2 = router.route(RequestProfile(request_id="r2", prefix=req.prefix, prompt_len=512))
        assert d2.estimated_ttft_reduction_pct > 0


class TestEstimateKvSizeGb:
    def test_scales_with_tokens(self):
        small = estimate_kv_size_gb(1024)
        large = estimate_kv_size_gb(4096)
        assert large > small * 3

    def test_fp8_smaller_than_bf16(self):
        fp8 = estimate_kv_size_gb(4096, dtype_bytes=1.0)
        bf16 = estimate_kv_size_gb(4096, dtype_bytes=2.0)
        assert fp8 < bf16

    def test_returns_positive(self):
        assert estimate_kv_size_gb(1000) > 0
