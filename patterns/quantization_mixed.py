"""Mixed-precision quantization for DeepSeek models.

DeepSeek-V3 was natively trained with FP8 (E4M3) weights and BF16 activations.
This module documents how to select, configure, and estimate memory for
different quantization strategies across the weight → KV-cache → activation stack.

Quantization hierarchy (best quality → smallest size):
    BF16 → FP8 (E4M3) → INT8 → FP4 (E2M1) → INT4 → GGUF Q2_K

Usage::

    cfg = select_quantization(
        target_vram_gb=16.0,
        model_params_b=70.0,
        priority="quality",
    )
    print(cfg.method, cfg.estimated_gb)

    # For GGUF / llama.cpp workflow
    script = cfg.llama_cpp_quantize_cmd(model_path="deepseek-r1-70b")
    print(script)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantizationType(str, Enum):
    """Numeric format for quantized weights."""

    BF16 = "bf16"  # Native training dtype, best quality
    FP16 = "fp16"  # Slightly lower dynamic range than BF16
    FP8_E4M3 = "fp8_e4m3"  # DeepSeek native; H100/H200 hardware support
    FP8_E5M2 = "fp8_e5m2"  # Higher dynamic range, slightly lower precision
    INT8 = "int8"  # Classical post-training quantization
    FP4_E2M1 = "fp4_e2m1"  # B200 hardware native; extreme compression
    INT4 = "int4"  # Common GPTQ/AWQ target
    GGUF_Q8_0 = "gguf_q8_0"  # llama.cpp 8-bit block quantization
    GGUF_Q6_K = "gguf_q6_k"  # llama.cpp 6-bit k-quantization
    GGUF_Q5_K_M = "gguf_q5_k_m"  # llama.cpp 5-bit mixed k-quant
    GGUF_Q4_K_M = "gguf_q4_k_m"  # llama.cpp 4-bit mixed k-quant (most popular)
    GGUF_Q3_K_M = "gguf_q3_k_m"  # llama.cpp 3-bit, quality starts degrading
    GGUF_Q2_K = "gguf_q2_k"  # llama.cpp 2-bit, significant quality loss


class QuantizationMethod(str, Enum):
    """Algorithm used for quantization."""

    RTN = "rtn"  # Round-to-Nearest — fast, low quality
    GPTQ = "gptq"  # Layer-wise second-order optimization
    AWQ = "awq"  # Activation-aware Weight Quantization
    GGUF = "gguf"  # llama.cpp k-quantization
    FP8_STATIC = "fp8_static"  # Static FP8 per-tensor calibration
    FP8_DYNAMIC = "fp8_dynamic"  # Dynamic FP8 per-token scale
    BITSANDBYTES = "bitsandbytes"  # NF4/INT8 via BnB — easy but slow inference
    MARLIN = "marlin"  # INT4/FP8 GPTQ kernel for NVIDIA (fast)


# Bytes per parameter for each dtype
_BYTES_PER_PARAM: dict[QuantizationType, float] = {
    QuantizationType.BF16: 2.0,
    QuantizationType.FP16: 2.0,
    QuantizationType.FP8_E4M3: 1.0,
    QuantizationType.FP8_E5M2: 1.0,
    QuantizationType.INT8: 1.0,
    QuantizationType.FP4_E2M1: 0.5,
    QuantizationType.INT4: 0.5,
    QuantizationType.GGUF_Q8_0: 1.0,
    QuantizationType.GGUF_Q6_K: 0.75,
    QuantizationType.GGUF_Q5_K_M: 0.625,
    QuantizationType.GGUF_Q4_K_M: 0.5,
    QuantizationType.GGUF_Q3_K_M: 0.375,
    QuantizationType.GGUF_Q2_K: 0.25,
}

# Perplexity degradation (PPL delta vs BF16, approximate, WikiText-2)
_QUALITY_DELTA: dict[QuantizationType, float] = {
    QuantizationType.BF16: 0.0,
    QuantizationType.FP16: 0.01,
    QuantizationType.FP8_E4M3: 0.05,
    QuantizationType.FP8_E5M2: 0.08,
    QuantizationType.INT8: 0.10,
    QuantizationType.FP4_E2M1: 0.30,
    QuantizationType.INT4: 0.40,
    QuantizationType.GGUF_Q8_0: 0.02,
    QuantizationType.GGUF_Q6_K: 0.05,
    QuantizationType.GGUF_Q5_K_M: 0.12,
    QuantizationType.GGUF_Q4_K_M: 0.25,
    QuantizationType.GGUF_Q3_K_M: 0.60,
    QuantizationType.GGUF_Q2_K: 1.80,
}

# Hardware that natively supports the format in kernels
_HARDWARE_SUPPORT: dict[QuantizationType, list[str]] = {
    QuantizationType.BF16: ["ampere", "hopper", "blackwell"],
    QuantizationType.FP16: ["ampere", "hopper", "blackwell"],
    QuantizationType.FP8_E4M3: ["hopper", "blackwell"],
    QuantizationType.FP8_E5M2: ["hopper", "blackwell"],
    QuantizationType.INT8: ["ampere", "hopper", "blackwell"],
    QuantizationType.FP4_E2M1: ["blackwell"],
    QuantizationType.INT4: ["ampere", "hopper", "blackwell"],
    QuantizationType.GGUF_Q8_0: ["cpu", "ampere", "hopper"],
    QuantizationType.GGUF_Q6_K: ["cpu", "ampere", "hopper"],
    QuantizationType.GGUF_Q5_K_M: ["cpu", "ampere", "hopper"],
    QuantizationType.GGUF_Q4_K_M: ["cpu", "ampere", "hopper"],
    QuantizationType.GGUF_Q3_K_M: ["cpu", "ampere", "hopper"],
    QuantizationType.GGUF_Q2_K: ["cpu"],
}


@dataclass
class QuantizationConfig:
    """Resolved quantization configuration for a DeepSeek model.

    Attributes:
        weight_type: Dtype for model weights.
        kv_cache_type: Dtype for key-value cache tensors.
        activation_type: Dtype for activations (usually BF16 on H-series).
        method: Quantization algorithm.
        model_params_b: Parameter count in billions.
        estimated_gb: Estimated VRAM for weights at this config.
        quality_delta: Approximate perplexity increase vs BF16 baseline.
        notes: Deployment notes and caveats.
    """

    weight_type: QuantizationType
    kv_cache_type: QuantizationType
    activation_type: QuantizationType
    method: QuantizationMethod
    model_params_b: float
    estimated_gb: float
    quality_delta: float
    notes: list[str]

    def llama_cpp_quantize_cmd(self, model_path: str) -> str:
        """Generate a llama.cpp quantize command for GGUF formats.

        Args:
            model_path: Path to the original model directory or HF repo.

        Returns:
            Shell command string, or empty string for non-GGUF types.
        """
        gguf_map = {
            QuantizationType.GGUF_Q8_0: "Q8_0",
            QuantizationType.GGUF_Q6_K: "Q6_K",
            QuantizationType.GGUF_Q5_K_M: "Q5_K_M",
            QuantizationType.GGUF_Q4_K_M: "Q4_K_M",
            QuantizationType.GGUF_Q3_K_M: "Q3_K_M",
            QuantizationType.GGUF_Q2_K: "Q2_K",
        }
        gguf_type = gguf_map.get(self.weight_type)
        if gguf_type is None:
            return ""
        safe_name = model_path.replace("/", "_").replace("-", "_")
        return (
            f"# Convert to GGUF first:\n"
            f"python3 convert_hf_to_gguf.py {model_path} "
            f"--outfile {safe_name}.gguf --outtype f16\n\n"
            f"# Quantize:\n"
            f"./llama-quantize {safe_name}.gguf "
            f"{safe_name}_{gguf_type}.gguf {gguf_type}"
        )

    def vllm_kwargs(self) -> dict[str, str]:
        """Produce vLLM engine kwargs dict for this quantization config."""
        dtype_map = {
            QuantizationType.BF16: "bfloat16",
            QuantizationType.FP16: "float16",
            QuantizationType.FP8_E4M3: "float8_e4m3fn",
            QuantizationType.FP8_E5M2: "float8_e5m2",
            QuantizationType.INT8: "int8",
            QuantizationType.INT4: "int4",
        }
        kwargs: dict[str, str] = {}
        if dt := dtype_map.get(self.weight_type):
            kwargs["dtype"] = dt
        if self.method in {QuantizationMethod.GPTQ, QuantizationMethod.MARLIN}:
            kwargs["quantization"] = "gptq_marlin"
        elif self.method == QuantizationMethod.AWQ:
            kwargs["quantization"] = "awq_marlin"
        elif self.method in {QuantizationMethod.FP8_STATIC, QuantizationMethod.FP8_DYNAMIC}:
            kwargs["quantization"] = "fp8"
        kv_map = {
            QuantizationType.FP8_E4M3: "fp8",
            QuantizationType.INT8: "int8",
        }
        if kv_dt := kv_map.get(self.kv_cache_type):
            kwargs["kv_cache_dtype"] = kv_dt
        return kwargs


def estimate_memory_gb(
    params_b: float,
    weight_type: QuantizationType,
    kv_cache_type: QuantizationType = QuantizationType.BF16,
    context_len: int = 32768,
    num_layers: int = 61,  # DeepSeek-V3 transformer layers
    num_kv_heads: int = 128,
    head_dim: int = 128,
    overhead_factor: float = 1.12,
) -> tuple[float, float]:
    """Estimate VRAM for weights and KV cache separately.

    Args:
        params_b: Model parameter count in billions.
        weight_type: Quantization type for model weights.
        kv_cache_type: Quantization type for KV cache tensors.
        context_len: Maximum context length tokens.
        num_layers: Transformer layer count.
        num_kv_heads: KV head count (uses GQA for DeepSeek — 128).
        head_dim: Attention head dimension.
        overhead_factor: Multiplier for activations, fragmentation.

    Returns:
        ``(weight_gb, kv_cache_gb)`` tuple.
    """
    weight_bytes = params_b * 1e9 * _BYTES_PER_PARAM[weight_type]
    weight_gb = weight_bytes * overhead_factor / (1024**3)

    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * _BYTES_PER_PARAM[kv_cache_type]
    kv_gb = context_len * kv_bytes_per_token / (1024**3)

    return weight_gb, kv_gb


def select_quantization(
    target_vram_gb: float,
    model_params_b: float = 671.0,
    priority: str = "quality",
    hardware: str = "hopper",
) -> QuantizationConfig:
    """Select the best quantization config that fits in target VRAM.

    Tries configs from highest quality to lowest until one fits.

    Args:
        target_vram_gb: Available VRAM in gigabytes.
        model_params_b: Model size in billions of parameters.
        priority: ``"quality"`` (minimize PPL delta) or ``"speed"``
            (prefer hardware-native formats).
        hardware: GPU architecture name for hardware-support filtering.

    Returns:
        A :class:`QuantizationConfig` that fits the VRAM budget.

    Raises:
        ValueError: If no quantization fits even at minimum precision.
    """
    # Ordered from highest quality to lowest
    candidates: list[tuple[QuantizationType, QuantizationMethod]] = [
        (QuantizationType.BF16, QuantizationMethod.RTN),
        (QuantizationType.FP8_E4M3, QuantizationMethod.FP8_DYNAMIC),
        (QuantizationType.INT8, QuantizationMethod.GPTQ),
        (QuantizationType.GGUF_Q6_K, QuantizationMethod.GGUF),
        (QuantizationType.INT4, QuantizationMethod.AWQ),
        (QuantizationType.GGUF_Q4_K_M, QuantizationMethod.GGUF),
        (QuantizationType.FP4_E2M1, QuantizationMethod.RTN),
        (QuantizationType.GGUF_Q3_K_M, QuantizationMethod.GGUF),
        (QuantizationType.GGUF_Q2_K, QuantizationMethod.GGUF),
    ]

    if priority == "speed":
        # Prefer hardware-native formats: reorder to put FP8 before INT8
        candidates = sorted(
            candidates,
            key=lambda c: (
                hardware not in _HARDWARE_SUPPORT.get(c[0], []),
                _QUALITY_DELTA.get(c[0], 999),
            ),
        )

    for weight_type, method in candidates:
        # Skip formats with no hardware support unless GGUF (CPU fallback ok)
        if hardware not in _HARDWARE_SUPPORT.get(weight_type, []) and "gguf" not in weight_type:
            continue

        # Use FP8 KV cache when weights are FP8+, else BF16
        kv_type = (
            QuantizationType.FP8_E4M3
            if weight_type in {QuantizationType.FP8_E4M3, QuantizationType.FP8_E5M2}
            else QuantizationType.BF16
        )

        weight_gb, _kv_gb = estimate_memory_gb(model_params_b, weight_type, kv_type)
        # Check weight-only fit: KV cache is runtime-configurable and should be
        # managed separately (context length can be reduced to fit).
        total_gb = weight_gb

        if total_gb <= target_vram_gb * 0.85:
            notes = _build_notes(weight_type, method, hardware, model_params_b)
            return QuantizationConfig(
                weight_type=weight_type,
                kv_cache_type=kv_type,
                activation_type=QuantizationType.BF16,
                method=method,
                model_params_b=model_params_b,
                estimated_gb=total_gb,
                quality_delta=_QUALITY_DELTA.get(weight_type, 0.0),
                notes=notes,
            )

    msg = (
        f"No quantization fits {model_params_b:.0f}B model in {target_vram_gb:.0f} GB. "
        "Consider a distilled model (7B-70B) or additional GPUs."
    )
    raise ValueError(msg)


def _build_notes(
    weight_type: QuantizationType,
    method: QuantizationMethod,
    hardware: str,
    params_b: float,
) -> list[str]:
    notes: list[str] = []
    if weight_type == QuantizationType.FP8_E4M3:
        notes.append(
            "FP8 E4M3 is DeepSeek's native training dtype. "
            "H100/H200 run this at full hardware speed."
        )
    if weight_type == QuantizationType.FP4_E2M1:
        notes.append("FP4 requires Blackwell (B200/B100). Falls back to INT4 on Hopper.")
    if method in {QuantizationMethod.GPTQ, QuantizationMethod.AWQ}:
        notes.append(
            f"{method.value.upper()} requires calibration with ~128 representative samples. "
            "Use a domain-relevant dataset for best results."
        )
    if "gguf" in weight_type:
        notes.append(
            "GGUF runs on CPU with llama.cpp. For GPU inference, "
            "prefer FP8/INT4 via vLLM or SGLang."
        )
    if hardware not in _HARDWARE_SUPPORT.get(weight_type, []):
        notes.append(f"WARNING: {weight_type} not natively supported on {hardware}. May be slow.")
    if params_b >= 200:
        notes.append(
            "Full model (671B) requires multi-GPU. "
            "Consider distilled variants (7B-70B) for single-GPU."
        )
    return notes
