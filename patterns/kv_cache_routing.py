"""KV-cache-aware routing for optimal DeepSeek latency.

When running multiple DeepSeek replicas, naively round-robin routing
wastes the KV cache: requests for the same system prompt go to different
replicas and each pays the full prefill cost again.

KV-cache-aware routing hashes the request's prefix to always route to the
replica that already cached it, cutting TTFT by 50-90% on repeated prefixes.

Pattern:
    Request → hash prefix → lookup cache table → route to warm replica
    If no warm replica → route to least-loaded → mark new cache entry

This implements both a simulated routing table and the hash-based selection
logic. In production, use vLLM's built-in prefix hashing or SGLang's
RadixAttention, which do this transparently.

Usage::

    router = KVCacheRouter(replicas=["gpu0", "gpu1", "gpu2", "gpu3"])
    decision = router.route(RequestProfile(
        request_id="req1",
        prefix="You are a helpful assistant.",
        prompt_len=2048,
    ))
    print(decision.target_replica, decision.cache_hit_ratio)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class RoutingStrategy(str, Enum):
    """How to select a target replica for a request."""

    PREFIX_HASH = "prefix_hash"  # Hash prefix → sticky replica (best KV reuse)
    ROUND_ROBIN = "round_robin"  # Ignore cache, distribute evenly
    LEAST_LOADED = "least_loaded"  # Route to replica with fewest active requests
    RANDOM = "random"  # Random (baseline, worst for KV reuse)


@dataclass
class RequestProfile:
    """Describes an incoming inference request for routing decisions.

    Attributes:
        request_id: Unique request identifier.
        prefix: The cacheable prefix (system prompt + few-shot examples).
        prompt_len: Total prompt token count.
        expected_output_len: Expected output token count.
        priority: Request priority (higher = more important).
    """

    request_id: str
    prefix: str
    prompt_len: int = 512
    expected_output_len: int = 256
    priority: int = 0

    @property
    def prefix_hash(self) -> str:
        """SHA-256 hash of the first 1024 chars of the prefix."""
        return hashlib.sha256(self.prefix[:1024].encode()).hexdigest()[:16]

    @property
    def estimated_kv_tokens(self) -> int:
        """Tokens that will be cached in KV (approximately the prefix)."""
        return self.prompt_len


@dataclass
class ReplicaState:
    """Runtime state of a single inference replica.

    Attributes:
        replica_id: Unique identifier for this GPU/worker.
        active_requests: Number of currently active requests.
        cached_prefix_hashes: Set of prefix hashes currently in KV cache.
        kv_cache_used_gb: Current KV cache VRAM usage.
        kv_cache_total_gb: Total KV cache VRAM available.
        healthy: Whether this replica is accepting requests.
    """

    replica_id: str
    active_requests: int = 0
    cached_prefix_hashes: set[str] = field(default_factory=set)
    kv_cache_used_gb: float = 0.0
    kv_cache_total_gb: float = 20.0
    healthy: bool = True

    @property
    def kv_utilization(self) -> float:
        """Fraction of KV cache currently in use."""
        if self.kv_cache_total_gb == 0:
            return 1.0
        return self.kv_cache_used_gb / self.kv_cache_total_gb

    @property
    def is_overloaded(self) -> bool:
        """True if KV cache is nearly full or request backlog is large."""
        return self.kv_utilization > 0.90 or self.active_requests > 32


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        request_id: Forwarded from the incoming request.
        target_replica: ID of the replica to send this request to.
        strategy_used: Which strategy produced this decision.
        cache_hit: Whether the target already has this prefix cached.
        cache_hit_ratio: Fraction of prompt tokens that are cached.
        estimated_ttft_reduction_pct: Estimated TTFT improvement vs cold cache.
        reason: Human-readable explanation.
    """

    request_id: str
    target_replica: str
    strategy_used: RoutingStrategy
    cache_hit: bool
    cache_hit_ratio: float
    estimated_ttft_reduction_pct: float
    reason: str


class KVCacheRouter:
    """Routes inference requests to replicas that already hold the KV cache.

    Maintains a lightweight in-process routing table mapping prefix hashes
    to replica IDs. In a multi-process deployment, replace with a shared
    Redis hash or consistent-hashing ring.

    Args:
        replicas: List of replica IDs (e.g. hostnames or GPU indices).
        strategy: Routing strategy to use.
        kv_cache_gb_per_replica: KV cache capacity per replica.
    """

    def __init__(
        self,
        replicas: list[str],
        strategy: RoutingStrategy = RoutingStrategy.PREFIX_HASH,
        kv_cache_gb_per_replica: float = 20.0,
    ) -> None:
        if not replicas:
            msg = "At least one replica required"
            raise ValueError(msg)
        self._strategy = strategy
        self._states: dict[str, ReplicaState] = {
            r: ReplicaState(replica_id=r, kv_cache_total_gb=kv_cache_gb_per_replica)
            for r in replicas
        }
        self._round_robin_idx: int = 0

    @property
    def replica_ids(self) -> list[str]:
        """Sorted list of replica IDs."""
        return sorted(self._states)

    def route(self, request: RequestProfile) -> RoutingDecision:
        """Route a request to the optimal replica.

        Args:
            request: Incoming request profile.

        Returns:
            A :class:`RoutingDecision` with the selected replica.
        """
        healthy = [r for r in self._states.values() if r.healthy and not r.is_overloaded]
        if not healthy:
            # Failsafe: route to least-loaded even if overloaded
            healthy = [min(self._states.values(), key=lambda r: r.active_requests)]

        if self._strategy == RoutingStrategy.PREFIX_HASH:
            return self._route_prefix_hash(request, healthy)
        if self._strategy == RoutingStrategy.LEAST_LOADED:
            return self._route_least_loaded(request, healthy)
        if self._strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(request, healthy)
        # RANDOM fallback
        import random

        replica = random.choice(healthy)  # noqa: S311  # nosec B311
        return self._make_decision(request, replica, RoutingStrategy.RANDOM, "random selection")

    def register_cache_entry(self, replica_id: str, prefix_hash: str, size_gb: float) -> None:
        """Record that a replica has cached a prefix.

        Call this after a request completes to keep the routing table warm.

        Args:
            replica_id: Replica that cached the prefix.
            prefix_hash: Hash from :attr:`RequestProfile.prefix_hash`.
            size_gb: KV cache size consumed by this entry.
        """
        if replica_id in self._states:
            self._states[replica_id].cached_prefix_hashes.add(prefix_hash)
            self._states[replica_id].kv_cache_used_gb += size_gb

    def evict_cache_entry(self, replica_id: str, prefix_hash: str, size_gb: float) -> None:
        """Remove a cache entry (called on LRU eviction).

        Args:
            replica_id: Replica evicting the prefix.
            prefix_hash: Hash of the evicted prefix.
            size_gb: KV cache bytes freed.
        """
        if replica_id in self._states:
            self._states[replica_id].cached_prefix_hashes.discard(prefix_hash)
            self._states[replica_id].kv_cache_used_gb = max(
                0.0, self._states[replica_id].kv_cache_used_gb - size_gb
            )

    def set_replica_health(self, replica_id: str, healthy: bool) -> None:
        """Mark a replica healthy or unhealthy.

        Args:
            replica_id: Target replica.
            healthy: New health state.
        """
        if replica_id in self._states:
            self._states[replica_id].healthy = healthy

    def _route_prefix_hash(
        self, request: RequestProfile, candidates: list[ReplicaState]
    ) -> RoutingDecision:
        phash = request.prefix_hash

        # First choice: replica that already has this prefix cached
        warm = [r for r in candidates if phash in r.cached_prefix_hashes]
        if warm:
            replica = min(warm, key=lambda r: r.active_requests)
            return self._make_decision(
                request, replica, RoutingStrategy.PREFIX_HASH, "KV cache hit", cache_hit=True
            )

        # Second choice: consistent hash ring — deterministic assignment
        # so the same prefix always goes to the same replica (warms up quickly)
        ring_idx = int(phash, 16) % len(candidates)
        replica = candidates[ring_idx]
        return self._make_decision(
            request, replica, RoutingStrategy.PREFIX_HASH, "consistent hash assignment"
        )

    def _route_least_loaded(
        self, request: RequestProfile, candidates: list[ReplicaState]
    ) -> RoutingDecision:
        replica = min(candidates, key=lambda r: r.active_requests + r.kv_utilization * 10)
        phash = request.prefix_hash
        cache_hit = phash in replica.cached_prefix_hashes
        return self._make_decision(
            request, replica, RoutingStrategy.LEAST_LOADED, "least loaded", cache_hit=cache_hit
        )

    def _route_round_robin(
        self, request: RequestProfile, candidates: list[ReplicaState]
    ) -> RoutingDecision:
        replica = candidates[self._round_robin_idx % len(candidates)]
        self._round_robin_idx += 1
        phash = request.prefix_hash
        cache_hit = phash in replica.cached_prefix_hashes
        return self._make_decision(
            request, replica, RoutingStrategy.ROUND_ROBIN, "round robin", cache_hit=cache_hit
        )

    def _make_decision(
        self,
        request: RequestProfile,
        replica: ReplicaState,
        strategy: RoutingStrategy,
        reason: str,
        cache_hit: bool = False,
    ) -> RoutingDecision:
        # Estimate KV cache hit ratio (prefix tokens / total tokens)
        if cache_hit:
            hit_ratio = min(1.0, request.estimated_kv_tokens / max(1, request.prompt_len))
        else:
            hit_ratio = 0.0

        # Published: 50-90% TTFT reduction on full cache hit
        ttft_reduction = hit_ratio * 80.0

        return RoutingDecision(
            request_id=request.request_id,
            target_replica=replica.replica_id,
            strategy_used=strategy,
            cache_hit=cache_hit,
            cache_hit_ratio=hit_ratio,
            estimated_ttft_reduction_pct=ttft_reduction,
            reason=reason,
        )


def estimate_kv_size_gb(
    prompt_tokens: int,
    num_layers: int = 61,
    num_kv_heads: int = 128,
    head_dim: int = 128,
    dtype_bytes: float = 1.0,  # FP8 default for KV cache
) -> float:
    """Estimate KV cache size for a single request prefix.

    Args:
        prompt_tokens: Number of prefix tokens to cache.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads (GQA).
        head_dim: Head dimension.
        dtype_bytes: Bytes per element in the KV cache.

    Returns:
        KV cache size in gigabytes.
    """
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    return prompt_tokens * bytes_per_token / (1024**3)
