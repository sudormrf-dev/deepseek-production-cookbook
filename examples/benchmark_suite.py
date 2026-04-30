"""Benchmark suite for comparing DeepSeek inference engines.

Run simulated benchmarks across vLLM, SGLang, and TensorRT-LLM,
then print a comparison table.

Run::

    python examples/benchmark_suite.py
"""

from __future__ import annotations

from patterns.inference_engines import EngineType, benchmark_engines


def main() -> None:
    engines = [EngineType.VLLM, EngineType.SGLANG, EngineType.TENSORRT_LLM]

    print("\nBenchmark: DeepSeek-R1-Distill-Qwen-70B, TP=4, H100x4")
    print(f"{'=' * 70}")
    results = benchmark_engines(
        engines,
        model="deepseek-r1-distill-qwen-70b",
        concurrency=16,
        input_len=2048,
        output_len=512,
    )

    print(
        f"{'Engine':<20} {'TTFT (ms)':<12} {'TPOT (ms)':<12} "
        f"{'Tok/s':<12} {'E2E@200T (ms)':<15} {'Mem (GB)':<10}"
    )
    print("-" * 81)
    for r in results:
        print(
            f"{r.engine.value:<20} {r.ttft_ms:<12.0f} {r.tpot_ms:<12.1f} "
            f"{r.throughput_tok_s:<12.0f} {r.e2e_latency_ms_at_200_tokens:<15.0f} "
            f"{r.memory_gb:<10.0f}"
        )

    winner = results[0]
    print(f"\nLowest TTFT: {winner.engine.value} ({winner.ttft_ms:.0f} ms)")

    print("\n\nBenchmark: Interactive chat (concurrency=1, 256-token output)")
    print(f"{'=' * 70}")
    interactive = benchmark_engines(
        engines,
        model="deepseek-r1-distill-qwen-7b",
        concurrency=1,
        input_len=512,
        output_len=256,
    )
    for r in interactive:
        print(f"  {r.engine.value}: TTFT={r.ttft_ms:.0f}ms, TPOT={r.tpot_ms:.1f}ms")


if __name__ == "__main__":
    main()
