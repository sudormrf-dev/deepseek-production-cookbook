"""DeepSeek production deployment patterns.

Covers MoE expert parallelism, FP8/FP4 quantization, multi-engine
benchmarking, KV-cache-aware routing, and distilled local deployment.
"""

from .inference_engines import (
    EngineConfig,
    EngineType,
    InferenceEngine,
    benchmark_engines,
    select_optimal_engine,
)
from .kv_cache_routing import (
    KVCacheRouter,
    RequestProfile,
    RoutingDecision,
    RoutingStrategy,
    estimate_kv_size_gb,
)
from .moe_parallelism import (
    ExpertParallelismConfig,
    GPUTier,
    MoEDeploymentPlan,
    ParallelismStrategy,
    plan_moe_deployment,
)
from .quantization_mixed import (
    QuantizationConfig,
    QuantizationMethod,
    QuantizationType,
    estimate_memory_gb,
    select_quantization,
)

__all__ = [
    "EngineConfig",
    "EngineType",
    "ExpertParallelismConfig",
    "GPUTier",
    "InferenceEngine",
    "KVCacheRouter",
    "MoEDeploymentPlan",
    "ParallelismStrategy",
    "QuantizationConfig",
    "QuantizationMethod",
    "QuantizationType",
    "RequestProfile",
    "RoutingDecision",
    "RoutingStrategy",
    "benchmark_engines",
    "estimate_kv_size_gb",
    "estimate_memory_gb",
    "plan_moe_deployment",
    "select_optimal_engine",
    "select_quantization",
]
