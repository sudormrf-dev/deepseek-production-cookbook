"""Deploying DeepSeek distilled models on consumer GPUs (RTX 5080, 16 GB).

DeepSeek-R1 comes in distilled variants based on Qwen and Llama architectures
(1.5B to 70B). These run on consumer GPUs — this module documents the exact
configurations for each model size tier on 16-24 GB VRAM GPUs.

Key constraints for RTX 5080 (16 GB VRAM):
    - 7B Q4_K_M GGUF: ~4.5 GB, fast, good quality
    - 14B Q4_K_M GGUF: ~8.5 GB, best quality/size tradeoff
    - 32B Q4_K_M GGUF: ~19 GB — requires CPU offload or Q2 quant
    - 70B: too large for single 16 GB GPU, needs Q2 + CPU offload

Usage::

    cfg = get_consumer_config(model_size_b=14, vram_gb=16)
    print(cfg.recommended_quant, cfg.ollama_model_tag)
    print(cfg.ollama_run_cmd)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsumerDeploymentConfig:
    """Deployment configuration for a distilled DeepSeek model on consumer GPU.

    Attributes:
        model_name: Full model name.
        model_size_b: Parameter count in billions.
        vram_gb: Target GPU VRAM in gigabytes.
        recommended_quant: Recommended GGUF quantization type.
        estimated_vram_gb: Estimated VRAM with this quantization.
        gpu_layers: Number of layers to offload to GPU (-1 = all).
        context_len: Safe context length at this quant/VRAM combination.
        ollama_model_tag: Ollama registry pull tag.
        threads: Recommended CPU thread count for mixed GPU/CPU inference.
        notes: Deployment guidance notes.
    """

    model_name: str
    model_size_b: float
    vram_gb: float
    recommended_quant: str
    estimated_vram_gb: float
    gpu_layers: int
    context_len: int
    ollama_model_tag: str
    threads: int
    notes: list[str]

    @property
    def fits_fully_on_gpu(self) -> bool:
        """True if the model fits entirely in VRAM without CPU offload."""
        return self.estimated_vram_gb <= self.vram_gb * 0.90

    @property
    def ollama_run_cmd(self) -> str:
        """Ollama one-liner to pull and run the model."""
        num_ctx = self.context_len
        return f"OLLAMA_NUM_CTX={num_ctx} ollama run {self.ollama_model_tag}"

    @property
    def llama_cpp_cmd(self) -> str:
        """llama.cpp server command with optimal flags."""
        model_tag = self.model_name.replace("-", "_").replace("/", "_")
        return (
            f"./llama-server \\\n"
            f"  --model {model_tag}_{self.recommended_quant}.gguf \\\n"
            f"  --n-gpu-layers {self.gpu_layers} \\\n"
            f"  --ctx-size {self.context_len} \\\n"
            f"  --threads {self.threads} \\\n"
            f"  --batch-size 512 \\\n"
            f"  --ubatch-size 512 \\\n"
            f"  --flash-attn \\\n"
            f"  --host 0.0.0.0 --port 8080"
        )

    @property
    def vllm_cmd(self) -> str:
        """vLLM command (for GPTQ/AWQ quants, not GGUF)."""
        return (
            f"python -m vllm.entrypoints.openai.api_server \\\n"
            f"  --model deepseek-ai/{self.model_name} \\\n"
            f"  --quantization awq_marlin \\\n"
            f"  --max-model-len {self.context_len} \\\n"
            f"  --gpu-memory-utilization 0.92"
        )


# RTX 5080 (16 GB) and RTX 4090 (24 GB) configs
_CONSUMER_CONFIGS: dict[tuple[float, float], ConsumerDeploymentConfig] = {
    # (model_size_b, vram_gb) → config
    (1.5, 16): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-1.5B",
        model_size_b=1.5,
        vram_gb=16,
        recommended_quant="Q8_0",
        estimated_vram_gb=1.8,
        gpu_layers=-1,
        context_len=65536,
        ollama_model_tag="deepseek-r1:1.5b",
        threads=8,
        notes=[
            "Fits in VRAM with 8-bit, leaves room for 65k context.",
            "Good for quick experiments; quality noticeably lower than 7B+.",
        ],
    ),
    (7, 16): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-7B",
        model_size_b=7.0,
        vram_gb=16,
        recommended_quant="Q4_K_M",
        estimated_vram_gb=4.8,
        gpu_layers=-1,
        context_len=32768,
        ollama_model_tag="deepseek-r1:7b",
        threads=8,
        notes=[
            "Q4_K_M offers best quality/size tradeoff for 7B.",
            "Leaves ~11 GB for KV cache → 32k context comfortably.",
        ],
    ),
    (14, 16): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-14B",
        model_size_b=14.0,
        vram_gb=16,
        recommended_quant="Q4_K_M",
        estimated_vram_gb=9.0,
        gpu_layers=-1,
        context_len=16384,
        ollama_model_tag="deepseek-r1:14b",
        threads=12,
        notes=[
            "Sweet spot for RTX 5080 — best reasoning quality in 16 GB.",
            "Reduced context (16k) due to VRAM split.",
            "For 32k context, use Q3_K_M (~7.5 GB).",
        ],
    ),
    (32, 16): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-32B",
        model_size_b=32.0,
        vram_gb=16,
        recommended_quant="Q2_K",
        estimated_vram_gb=12.0,
        gpu_layers=60,  # Partial GPU offload; rest on CPU RAM
        context_len=8192,
        ollama_model_tag="deepseek-r1:32b",
        threads=16,
        notes=[
            "Q2_K is aggressive — significant quality degradation at 32B.",
            "Prefer 14B Q4_K_M for better quality on 16 GB GPU.",
            "For 32B with good quality: use 2x GPUs or 32 GB+ VRAM.",
        ],
    ),
    (7, 24): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-7B",
        model_size_b=7.0,
        vram_gb=24,
        recommended_quant="Q8_0",
        estimated_vram_gb=8.5,
        gpu_layers=-1,
        context_len=65536,
        ollama_model_tag="deepseek-r1:7b-q8_0",
        threads=8,
        notes=[
            "24 GB allows Q8_0 for near-lossless quality.",
            "65k context fully in VRAM with Q8_0.",
        ],
    ),
    (14, 24): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-14B",
        model_size_b=14.0,
        vram_gb=24,
        recommended_quant="Q6_K",
        estimated_vram_gb=12.0,
        gpu_layers=-1,
        context_len=32768,
        ollama_model_tag="deepseek-r1:14b-q6_k",
        threads=12,
        notes=[
            "24 GB allows Q6_K — excellent quality with 32k context.",
            "RTX 4090 sweet spot configuration.",
        ],
    ),
    (32, 24): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Qwen-32B",
        model_size_b=32.0,
        vram_gb=24,
        recommended_quant="Q4_K_M",
        estimated_vram_gb=20.0,
        gpu_layers=-1,
        context_len=8192,
        ollama_model_tag="deepseek-r1:32b",
        threads=16,
        notes=[
            "Fits in 24 GB with Q4_K_M — tight but workable.",
            "Reduce context to 4096 if VRAM is insufficient.",
            "Flash attention reduces VRAM overhead significantly.",
        ],
    ),
    (70, 24): ConsumerDeploymentConfig(
        model_name="DeepSeek-R1-Distill-Llama-70B",
        model_size_b=70.0,
        vram_gb=24,
        recommended_quant="Q2_K",
        estimated_vram_gb=24.0,
        gpu_layers=40,  # Partial offload; needs 64+ GB system RAM
        context_len=4096,
        ollama_model_tag="deepseek-r1:70b",
        threads=24,
        notes=[
            "70B on 24 GB requires aggressive quantization + CPU offload.",
            "Requires 64+ GB system RAM for CPU layers.",
            "Very slow generation (~3-5 tok/s) due to PCIe bottleneck.",
            "Strongly prefer 14B Q6_K for better quality at this VRAM.",
        ],
    ),
}


def get_consumer_config(
    model_size_b: float,
    vram_gb: float = 16,
) -> ConsumerDeploymentConfig:
    """Get the optimal deployment config for a consumer GPU.

    Snaps to the nearest model size and VRAM tier from known configs.

    Args:
        model_size_b: Target model size in billions of parameters.
        vram_gb: Available GPU VRAM in gigabytes.

    Returns:
        A :class:`ConsumerDeploymentConfig` for the requested setup.

    Raises:
        ValueError: If no config matches the given parameters.
    """
    # Snap VRAM to known tiers
    if vram_gb < 12:
        msg = f"VRAM {vram_gb:.0f} GB is below minimum 12 GB for consumer deployment."
        raise ValueError(msg)
    vram_tier = 24 if vram_gb >= 20 else 16

    # Snap model size to nearest supported tier
    supported_sizes = [1.5, 7.0, 14.0, 32.0, 70.0]
    nearest = min(supported_sizes, key=lambda s: abs(s - model_size_b))

    key = (nearest, float(vram_tier))
    if key not in _CONSUMER_CONFIGS:
        msg = f"No config for {model_size_b}B on {vram_gb} GB GPU."
        raise ValueError(msg)
    return _CONSUMER_CONFIGS[key]


def list_consumer_configs(vram_gb: float = 16) -> list[ConsumerDeploymentConfig]:
    """List all available configs for a given VRAM tier, sorted by model size.

    Args:
        vram_gb: GPU VRAM in gigabytes.

    Returns:
        All matching :class:`ConsumerDeploymentConfig` objects.
    """
    vram_tier = 24 if vram_gb >= 20 else 16
    configs = [v for (_, tier), v in _CONSUMER_CONFIGS.items() if tier == vram_tier]
    return sorted(configs, key=lambda c: c.model_size_b)


def recommend_model(
    vram_gb: float = 16,
    priority: str = "quality",
) -> ConsumerDeploymentConfig:
    """Recommend the best model for consumer GPU based on priority.

    Args:
        vram_gb: Available GPU VRAM.
        priority: ``"quality"`` selects the largest model that fits fully;
            ``"speed"`` selects the smallest model; ``"balanced"`` picks the
            middle option.

    Returns:
        Recommended :class:`ConsumerDeploymentConfig`.
    """
    configs = list_consumer_configs(vram_gb)
    if not configs:
        msg = f"No configs found for {vram_gb} GB VRAM"
        raise ValueError(msg)

    fitting = [c for c in configs if c.fits_fully_on_gpu]
    if not fitting:
        fitting = configs  # fallback: take all if nothing fits perfectly

    if priority == "quality":
        return fitting[-1]  # largest fitting model
    if priority == "speed":
        return fitting[0]  # smallest = fastest
    # balanced: middle option
    return fitting[len(fitting) // 2]
