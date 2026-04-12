"""Inference engine comparison: vLLM vs SGLang vs TensorRT-LLM for DeepSeek MoE.

Each engine has different tradeoffs for MoE models. This module provides
a structured comparison with configuration helpers and a simulated
benchmark framework for validating your specific hardware setup.

Key differences for DeepSeek MoE:
    - vLLM: Best ecosystem, easy to deploy, good for diverse workloads
    - SGLang: Best TTFT via RadixAttention + speculative decoding
    - TensorRT-LLM: Best throughput on NVIDIA hardware, complex setup

Usage::

    config = EngineConfig(
        engine=EngineType.VLLM,
        model="deepseek-ai/DeepSeek-V3",
        tensor_parallel_size=8,
        dtype="bfloat16",
    )
    engine = InferenceEngine(config)
    report = benchmark_engines([EngineType.VLLM, EngineType.SGLANG], model="deepseek-r1-distill-qwen-7b")
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum


class EngineType(str, Enum):
    """Supported inference engines."""

    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt_llm"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"


@dataclass
class EngineConfig:
    """Configuration for a single inference engine deployment.

    Attributes:
        engine: Which inference engine to use.
        model: Model path or HuggingFace repo ID.
        tensor_parallel_size: TP degree (must match GPU count).
        dtype: Weight dtype string for the engine.
        max_model_len: Maximum sequence length.
        gpu_memory_utilization: Fraction of VRAM for KV cache (vLLM).
        enable_expert_parallel: Enable EP for MoE (vLLM >=0.6.4).
        enable_flashinfer: Use FlashInfer kernel backend (SGLang).
        chunked_prefill: Enable chunked prefill for TTFT reduction.
        speculative_decoding: Enable spec decoding (SGLang).
        extra_args: Engine-specific additional arguments.
    """

    engine: EngineType
    model: str
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    enable_expert_parallel: bool = False
    enable_flashinfer: bool = True
    chunked_prefill: bool = True
    speculative_decoding: bool = False
    extra_args: dict[str, str] = field(default_factory=dict)

    def to_launch_cmd(self) -> str:
        """Generate the shell launch command for this engine configuration."""
        if self.engine == EngineType.VLLM:
            return self._vllm_cmd()
        if self.engine == EngineType.SGLANG:
            return self._sglang_cmd()
        if self.engine == EngineType.TENSORRT_LLM:
            return self._trtllm_cmd()
        if self.engine == EngineType.OLLAMA:
            return self._ollama_cmd()
        return self._llama_cpp_cmd()

    def _vllm_cmd(self) -> str:
        parts = [
            "python -m vllm.entrypoints.openai.api_server",
            f"--model {self.model}",
            f"--tensor-parallel-size {self.tensor_parallel_size}",
            f"--dtype {self.dtype}",
            f"--max-model-len {self.max_model_len}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
        ]
        if self.enable_expert_parallel:
            parts.append("--enable-expert-parallel")
        if self.chunked_prefill:
            parts.append("--enable-chunked-prefill")
        for k, v in self.extra_args.items():
            parts.append(f"--{k} {v}")
        return " \\\n  ".join(parts)

    def _sglang_cmd(self) -> str:
        parts = [
            "python -m sglang.launch_server",
            f"--model-path {self.model}",
            f"--tp {self.tensor_parallel_size}",
            f"--dtype {self.dtype}",
            f"--context-length {self.max_model_len}",
        ]
        if self.enable_flashinfer:
            parts.append("--enable-flashinfer")
        if self.chunked_prefill:
            parts.append("--chunked-prefill-size 4096")
        if self.speculative_decoding:
            parts.append("--speculative-algorithm EAGLE")
        for k, v in self.extra_args.items():
            parts.append(f"--{k} {v}")
        return " \\\n  ".join(parts)

    def _trtllm_cmd(self) -> str:
        # TRT-LLM requires a two-phase build → run workflow
        safe_name = self.model.replace("/", "--")
        return (
            f"# Phase 1: Build TRT-LLM engine (run once)\n"
            f"python convert_checkpoint.py \\\n"
            f"  --model_dir {self.model} \\\n"
            f"  --output_dir ./checkpoints/{safe_name} \\\n"
            f"  --dtype {self.dtype} \\\n"
            f"  --tp_size {self.tensor_parallel_size}\n\n"
            f"trtllm-build \\\n"
            f"  --checkpoint_dir ./checkpoints/{safe_name} \\\n"
            f"  --output_dir ./engines/{safe_name} \\\n"
            f"  --max_seq_len {self.max_model_len} \\\n"
            f"  --workers {self.tensor_parallel_size}\n\n"
            f"# Phase 2: Serve\n"
            f"python -m tensorrt_llm.serve \\\n"
            f"  --engine_dir ./engines/{safe_name} \\\n"
            f"  --host 0.0.0.0 --port 8000"
        )

    def _ollama_cmd(self) -> str:
        model_tag = self.model.split("/")[-1].lower()
        return (
            f"# Pull and serve via Ollama\n"
            f"ollama pull {model_tag}\n"
            f"ollama serve\n\n"
            f"# Or run directly:\n"
            f"ollama run {model_tag}"
        )

    def _llama_cpp_cmd(self) -> str:
        n_gpu_layers = -1  # offload all
        return (
            f"./llama-server \\\n"
            f"  --model {self.model}.gguf \\\n"
            f"  --n-gpu-layers {n_gpu_layers} \\\n"
            f"  --ctx-size {self.max_model_len} \\\n"
            f"  --threads 8 \\\n"
            f"  --host 0.0.0.0 --port 8080"
        )


@dataclass
class BenchmarkResult:
    """Benchmark metrics for one engine configuration.

    Attributes:
        engine: Engine that produced these results.
        model: Model benchmarked.
        ttft_ms: Time-to-first-token latency in milliseconds.
        tpot_ms: Time-per-output-token in milliseconds.
        throughput_tok_s: Output tokens per second.
        memory_gb: Observed peak VRAM in gigabytes.
        concurrency: Number of concurrent requests tested.
        success_rate: Fraction of requests that completed successfully.
    """

    engine: EngineType
    model: str
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float
    memory_gb: float
    concurrency: int
    success_rate: float = 1.0

    @property
    def e2e_latency_ms_at_200_tokens(self) -> float:
        """Estimated end-to-end latency for a 200-token response."""
        return self.ttft_ms + 200 * self.tpot_ms


class InferenceEngine:
    """Wrapper for managing and querying an inference engine.

    In production, replace _simulate_* methods with actual HTTP calls
    to the running engine's OpenAI-compatible API.

    Args:
        config: Engine configuration.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._cfg = config

    @property
    def config(self) -> EngineConfig:
        """Engine configuration."""
        return self._cfg

    def launch_command(self) -> str:
        """Return the shell command to launch this engine."""
        return self._cfg.to_launch_cmd()

    def simulate_benchmark(
        self,
        num_requests: int = 50,
        input_len: int = 512,
        output_len: int = 256,
        concurrency: int = 8,
    ) -> BenchmarkResult:
        """Simulate benchmark results based on published benchmarks.

        These numbers approximate published results from vLLM/SGLang papers
        and blog posts. Run a real benchmark for your hardware.

        Args:
            num_requests: Total requests in the benchmark.
            input_len: Input token count per request.
            output_len: Output token count per request.
            concurrency: Parallel request count.

        Returns:
            Simulated :class:`BenchmarkResult`.
        """
        base = _ENGINE_BASELINES.get(self._cfg.engine, _ENGINE_BASELINES[EngineType.VLLM])

        # Scale TTFT with input length (longer context = slower prefill)
        ttft = base["ttft_ms"] * math.log10(max(10, input_len) / 100 + 1)

        # Scale throughput inversely with concurrency (batching helps but has limits)
        tpot = base["tpot_ms"] * (1 + 0.1 * math.log(max(1, concurrency)))

        tp_size = self._cfg.tensor_parallel_size
        throughput = (concurrency * 1000) / tpot * tp_size

        # Jitter for realism
        samples = [ttft * (1 + 0.02 * i) for i in range(num_requests)]
        ttft_p50 = statistics.median(samples)

        return BenchmarkResult(
            engine=self._cfg.engine,
            model=self._cfg.model,
            ttft_ms=ttft_p50,
            tpot_ms=tpot,
            throughput_tok_s=throughput,
            memory_gb=base["memory_gb"],
            concurrency=concurrency,
            success_rate=0.999,
        )


# Published approximate baselines (H100 SXM5, DeepSeek-R1 70B, BF16, TP=4)
_ENGINE_BASELINES: dict[EngineType, dict[str, float]] = {
    EngineType.VLLM: {"ttft_ms": 350.0, "tpot_ms": 18.0, "memory_gb": 145.0},
    EngineType.SGLANG: {"ttft_ms": 180.0, "tpot_ms": 14.0, "memory_gb": 142.0},
    EngineType.TENSORRT_LLM: {"ttft_ms": 140.0, "tpot_ms": 11.0, "memory_gb": 140.0},
    EngineType.OLLAMA: {"ttft_ms": 2500.0, "tpot_ms": 45.0, "memory_gb": 6.0},
    EngineType.LLAMA_CPP: {"ttft_ms": 3000.0, "tpot_ms": 60.0, "memory_gb": 5.5},
}


def benchmark_engines(
    engines: list[EngineType],
    model: str = "deepseek-r1-distill-qwen-7b",
    concurrency: int = 8,
    input_len: int = 512,
    output_len: int = 256,
) -> list[BenchmarkResult]:
    """Run simulated benchmarks across multiple engines.

    Args:
        engines: List of engine types to benchmark.
        model: Model to benchmark against.
        concurrency: Parallel request count.
        input_len: Input tokens per request.
        output_len: Output tokens per request.

    Returns:
        List of :class:`BenchmarkResult`, sorted by TTFT ascending.
    """
    results: list[BenchmarkResult] = []
    for engine_type in engines:
        cfg = EngineConfig(engine=engine_type, model=model)
        engine = InferenceEngine(cfg)
        result = engine.simulate_benchmark(
            input_len=input_len,
            output_len=output_len,
            concurrency=concurrency,
        )
        results.append(result)
    return sorted(results, key=lambda r: r.ttft_ms)


def select_optimal_engine(
    workload: str = "interactive",
    hardware: str = "hopper",
    require_openai_api: bool = True,
) -> EngineType:
    """Select the best engine for a given workload and hardware.

    Args:
        workload: ``"interactive"`` (low TTFT), ``"batch"`` (high throughput),
            or ``"local"`` (single consumer GPU).
        hardware: GPU architecture: ``"hopper"``, ``"ampere"``, or ``"consumer"``.
        require_openai_api: Whether OpenAI-compatible API is required.

    Returns:
        The recommended :class:`EngineType`.
    """
    if hardware == "consumer" or workload == "local":
        return EngineType.OLLAMA

    if workload == "interactive":
        # SGLang's RadixAttention gives best TTFT for chat/interactive
        return EngineType.SGLANG

    if workload == "batch":
        # TRT-LLM wins on throughput for offline batch jobs on NVIDIA
        if hardware == "hopper":
            return EngineType.TENSORRT_LLM
        # vLLM has better AMD/non-NVIDIA support
        return EngineType.VLLM

    return EngineType.VLLM
