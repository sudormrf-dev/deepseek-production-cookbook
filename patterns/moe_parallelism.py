"""DeepSeek MoE expert parallelism — production deployment configurations.

DeepSeek-V3/R1 uses a Mixture-of-Experts architecture with 256 routed experts
(each token activates 8). This module covers how to distribute those experts
across GPU clusters ranging from a single RTX 5080 (distilled) to 4xH100 nodes.

Pattern overview:
    - Expert Parallelism (EP): each GPU hosts a shard of experts
    - Tensor Parallelism (TP): splits individual weight matrices across GPUs
    - Pipeline Parallelism (PP): stages transformer blocks across nodes
    - Data Parallelism (DP): independent replicas for throughput scaling

Usage::

    plan = plan_moe_deployment(
        num_experts=256,
        gpu_tier=GPUTier.H100_80GB,
        num_gpus=8,
    )
    print(plan.recommended_strategy)
    print(plan.vllm_launch_cmd)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class GPUTier(str, Enum):
    """GPU hardware tiers for deployment planning."""

    RTX_4090 = "rtx_4090"  # 24 GB VRAM, 82.6 TFLOPS BF16
    RTX_5080 = "rtx_5080"  # 16 GB VRAM, ~100 TFLOPS BF16 (est.)
    RTX_5090 = "rtx_5090"  # 32 GB VRAM, ~200 TFLOPS BF16 (est.)
    A100_40GB = "a100_40gb"  # 40 GB HBM2e, 312 TFLOPS BF16
    A100_80GB = "a100_80gb"  # 80 GB HBM2e, 312 TFLOPS BF16
    H100_80GB = "h100_80gb"  # 80 GB HBM3, 989 TFLOPS BF16
    H200_141GB = "h200_141gb"  # 141 GB HBM3e, 989 TFLOPS BF16
    B200_192GB = "b200_192gb"  # 192 GB HBM3e, ~4500 TFLOPS FP4


# VRAM in GB per GPU tier
_VRAM_GB: dict[GPUTier, float] = {
    GPUTier.RTX_4090: 24.0,
    GPUTier.RTX_5080: 16.0,
    GPUTier.RTX_5090: 32.0,
    GPUTier.A100_40GB: 40.0,
    GPUTier.A100_80GB: 80.0,
    GPUTier.H100_80GB: 80.0,
    GPUTier.H200_141GB: 141.0,
    GPUTier.B200_192GB: 192.0,
}

# Whether NVLink / NVSwitch high-bandwidth fabric is available
_HAS_NVLINK: dict[GPUTier, bool] = {
    GPUTier.RTX_4090: False,
    GPUTier.RTX_5080: False,
    GPUTier.RTX_5090: False,
    GPUTier.A100_40GB: True,
    GPUTier.A100_80GB: True,
    GPUTier.H100_80GB: True,
    GPUTier.H200_141GB: True,
    GPUTier.B200_192GB: True,
}


class ParallelismStrategy(str, Enum):
    """How to distribute the model across GPUs."""

    SINGLE_GPU = "single_gpu"  # Fits on one GPU (distilled models)
    TENSOR_PARALLEL = "tensor_parallel"  # TP across GPUs, same node
    EXPERT_PARALLEL = "expert_parallel"  # EP: each GPU hosts N experts
    PIPELINE_PARALLEL = "pipeline_parallel"  # PP across nodes
    HYBRID_EP_TP = "hybrid_ep_tp"  # EP within nodes + TP within GPU pairs


@dataclass
class ExpertParallelismConfig:
    """Expert parallelism layout for a DeepSeek MoE model.

    Attributes:
        num_experts: Total routed experts in the model (256 for DeepSeek-V3).
        experts_per_token: Experts activated per token (8 for DeepSeek-V3).
        num_gpus: Total GPU count.
        experts_per_gpu: How many expert weight shards each GPU holds.
        tensor_parallel_size: TP degree within each node.
        pipeline_parallel_size: PP stages across nodes.
        data_parallel_size: DP replicas.
    """

    num_experts: int
    experts_per_token: int
    num_gpus: int
    experts_per_gpu: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int

    @property
    def expert_parallel_size(self) -> int:
        """Expert parallel degree = GPUs used for expert distribution."""
        return self.num_gpus // (self.tensor_parallel_size * self.pipeline_parallel_size)

    @property
    def utilization_pct(self) -> float:
        """Fraction of experts that are active per forward pass (per DP replica)."""
        return self.experts_per_token / self.num_experts * 100


@dataclass
class MoEDeploymentPlan:
    """Complete deployment plan for a DeepSeek MoE model.

    Attributes:
        gpu_tier: GPU hardware tier.
        num_gpus: Number of GPUs.
        strategy: Recommended parallelism strategy.
        ep_config: Expert parallelism layout.
        model_vram_gb: Estimated VRAM for model weights.
        kv_cache_vram_gb: Estimated VRAM for KV cache.
        fits_in_memory: Whether the plan fits within available VRAM.
        vllm_launch_cmd: Suggested vLLM launch command.
        sglang_launch_cmd: Suggested SGLang launch command.
        notes: Human-readable guidance notes.
    """

    gpu_tier: GPUTier
    num_gpus: int
    strategy: ParallelismStrategy
    ep_config: ExpertParallelismConfig
    model_vram_gb: float
    kv_cache_vram_gb: float
    fits_in_memory: bool
    vllm_launch_cmd: str
    sglang_launch_cmd: str
    notes: list[str] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        """Total VRAM across all GPUs."""
        return _VRAM_GB[self.gpu_tier] * self.num_gpus

    @property
    def peak_vram_per_gpu_gb(self) -> float:
        """Peak VRAM usage per GPU (model + KV cache)."""
        return (self.model_vram_gb + self.kv_cache_vram_gb) / self.num_gpus


# Model parameter counts (billions) for DeepSeek variants
_MODEL_PARAMS_B: dict[str, float] = {
    "deepseek-v3": 671.0,  # 37B active per forward pass
    "deepseek-r1": 671.0,
    "deepseek-v3-0324": 671.0,
    "deepseek-r1-distill-llama-70b": 70.0,
    "deepseek-r1-distill-llama-8b": 8.0,
    "deepseek-r1-distill-qwen-32b": 32.0,
    "deepseek-r1-distill-qwen-14b": 14.0,
    "deepseek-r1-distill-qwen-7b": 7.0,
    "deepseek-r1-distill-qwen-1.5b": 1.5,
}


def estimate_model_vram_gb(
    model: str,
    dtype_bytes: int = 2,  # BF16 default
    overhead_factor: float = 1.15,  # Activations + optimizer states
) -> float:
    """Estimate VRAM needed for model weights.

    Args:
        model: Model name key from the known catalogue.
        dtype_bytes: Bytes per parameter (2=BF16/FP16, 1=INT8/FP8, 0.5=FP4/INT4).
        overhead_factor: Multiplier for activations, fragmentation, etc.

    Returns:
        Estimated VRAM in gigabytes.
    """
    params_b = _MODEL_PARAMS_B.get(model, 671.0)
    raw_gb = params_b * 1e9 * dtype_bytes / (1024**3)
    return raw_gb * overhead_factor


def plan_moe_deployment(
    num_experts: int = 256,
    experts_per_token: int = 8,
    gpu_tier: GPUTier = GPUTier.H100_80GB,
    num_gpus: int = 8,
    model: str = "deepseek-v3",
    dtype_bytes: int = 2,
    kv_cache_fraction: float = 0.20,
) -> MoEDeploymentPlan:
    """Generate an expert parallelism deployment plan.

    Determines optimal TP/EP/PP degrees, estimates VRAM, and emits
    launch commands for vLLM and SGLang.

    Args:
        num_experts: Total routed expert count in the model.
        experts_per_token: How many experts each token routes to.
        gpu_tier: GPU hardware tier.
        num_gpus: Available GPU count.
        model: Model identifier for VRAM estimation.
        dtype_bytes: Parameter dtype size (2=BF16, 1=FP8, 0.5=FP4).
        kv_cache_fraction: Fraction of VRAM reserved for KV cache.

    Returns:
        A :class:`MoEDeploymentPlan` with parallelism config and launch commands.
    """
    vram_per_gpu = _VRAM_GB[gpu_tier]
    total_vram = vram_per_gpu * num_gpus
    model_vram = estimate_model_vram_gb(model, dtype_bytes)
    kv_vram = total_vram * kv_cache_fraction
    has_nvlink = _HAS_NVLINK[gpu_tier]

    notes: list[str] = []

    # --- Choose parallelism strategy ---
    if model_vram <= vram_per_gpu * 0.85:
        strategy = ParallelismStrategy.SINGLE_GPU
        tp, pp, dp = 1, 1, num_gpus
        ep_size = 1
    elif has_nvlink and num_gpus <= 8:
        # NVLink enables high-bandwidth TP within a single DGX node
        strategy = ParallelismStrategy.TENSOR_PARALLEL
        tp = min(num_gpus, _nearest_power_of_two(math.ceil(model_vram / vram_per_gpu)))
        pp = 1
        dp = max(1, num_gpus // tp)
        ep_size = num_gpus // (tp * pp)
    elif num_experts >= 64 and num_gpus >= 4:
        # Expert parallelism shines when there are many experts and enough GPUs
        strategy = ParallelismStrategy.EXPERT_PARALLEL
        ep_size = min(num_gpus, num_experts)
        tp = 1
        pp = max(1, num_gpus // ep_size)
        dp = 1
    else:
        strategy = ParallelismStrategy.PIPELINE_PARALLEL
        pp = min(num_gpus, 4)
        tp = 1
        ep_size = num_gpus // pp
        dp = 1

    experts_per_gpu = max(1, num_experts // ep_size)

    ep_config = ExpertParallelismConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        num_gpus=num_gpus,
        experts_per_gpu=experts_per_gpu,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
    )

    fits = (model_vram + kv_vram) <= total_vram * 0.95

    if not fits:
        notes.append(
            f"Model requires ~{model_vram:.0f} GB but only {total_vram:.0f} GB available. "
            "Consider FP8 quantization (dtype_bytes=1) or more GPUs."
        )
    if not has_nvlink and tp > 1:
        notes.append(
            "PCIe bandwidth limits TP efficiency. Consider EP over TP for consumer GPUs."
        )
    if gpu_tier in {GPUTier.RTX_5080, GPUTier.RTX_4090}:
        notes.append(
            "Consumer GPU detected. Use distilled models (7B-70B) for single-GPU deployment."
        )

    vllm_cmd = _build_vllm_cmd(model, tp, num_gpus, dtype_bytes, strategy)
    sglang_cmd = _build_sglang_cmd(model, tp, num_gpus, dtype_bytes)

    return MoEDeploymentPlan(
        gpu_tier=gpu_tier,
        num_gpus=num_gpus,
        strategy=strategy,
        ep_config=ep_config,
        model_vram_gb=model_vram,
        kv_cache_vram_gb=kv_vram,
        fits_in_memory=fits,
        vllm_launch_cmd=vllm_cmd,
        sglang_launch_cmd=sglang_cmd,
        notes=notes,
    )


def _nearest_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_DTYPE_MAP: dict[int, str] = {
    2: "bfloat16",
    1: "float8_e4m3fn",
}


def _build_vllm_cmd(
    model: str,
    tp: int,
    num_gpus: int,
    dtype_bytes: int,
    strategy: ParallelismStrategy,
) -> str:
    dtype = _DTYPE_MAP.get(dtype_bytes, "bfloat16")
    ep_flag = ""
    if strategy == ParallelismStrategy.EXPERT_PARALLEL:
        ep_size = num_gpus // tp
        ep_flag = f" --enable-expert-parallel --expert-parallel-size {ep_size}"
    return (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model} "
        f"--tensor-parallel-size {tp} "
        f"--dtype {dtype} "
        f"--max-model-len 32768 "
        f"--gpu-memory-utilization 0.90"
        f"{ep_flag}"
    )


def _build_sglang_cmd(model: str, tp: int, num_gpus: int, dtype_bytes: int) -> str:
    dtype = _DTYPE_MAP.get(dtype_bytes, "bfloat16")
    return (
        f"python -m sglang.launch_server "
        f"--model-path {model} "
        f"--tp {tp} "
        f"--dtype {dtype} "
        f"--context-length 32768 "
        f"--enable-flashinfer"
    )
