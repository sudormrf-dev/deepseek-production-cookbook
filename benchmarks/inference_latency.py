"""Simulated inference latency benchmarks for DeepSeek models.

Compares TTFT (Time To First Token), throughput, and end-to-end latency
across four inference engines (TensorRT-LLM, SGLang, vLLM, Ollama) and
three GGUF quantization levels (Q4_K_M, Q5_K_M, Q8_0) for distilled
DeepSeek-R1 variants.

All numbers are derived from published benchmarks and official documentation;
no GPU is required to run this script.

Usage::

    python benchmarks/inference_latency.py
    python benchmarks/inference_latency.py --model-size 7 --concurrency 16
"""

from __future__ import annotations

import argparse
import math

# ---------------------------------------------------------------------------
# Re-use patterns/ dataclasses so the benchmark stays consistent with the
# rest of the cookbook.
# ---------------------------------------------------------------------------
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from patterns.inference_engines import _ENGINE_BASELINES, EngineType

# ---------------------------------------------------------------------------
# Quantization performance modifiers (relative to BF16 baseline)
# ---------------------------------------------------------------------------


class QuantProfile(NamedTuple):
    """Speed and quality modifiers for a GGUF quantization level."""

    label: str
    ttft_multiplier: float  # >1 = faster prefill (smaller model → less compute)
    tpot_multiplier: float  # <1 = fewer bytes to load per token → faster
    size_multiplier: float  # fraction of BF16 model size on disk / VRAM
    quality_note: str


_QUANT_PROFILES: dict[str, QuantProfile] = {
    "Q4_K_M": QuantProfile(
        label="Q4_K_M",
        ttft_multiplier=1.55,  # 55% faster prefill vs BF16
        tpot_multiplier=0.55,  # 45% faster token gen
        size_multiplier=0.50,
        quality_note="Best quality/size — recommended default",
    ),
    "Q5_K_M": QuantProfile(
        label="Q5_K_M",
        ttft_multiplier=1.35,
        tpot_multiplier=0.65,
        size_multiplier=0.625,
        quality_note="Small PPL penalty vs Q6 at 37% smaller than BF16",
    ),
    "Q8_0": QuantProfile(
        label="Q8_0",
        ttft_multiplier=1.10,
        tpot_multiplier=0.88,
        size_multiplier=1.00,
        quality_note="Near-lossless; same size as BF16 weights",
    ),
}

# Model size in billions → approximate BF16 baseline latency scaling factor
# Larger models are proportionally slower; 7B is the reference point.
_MODEL_SCALE: dict[float, float] = {
    1.5: 0.28,
    7.0: 1.00,  # reference
    14.0: 1.95,
    32.0: 4.40,
    70.0: 9.20,
}


# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------


@dataclass
class LatencyRow:
    """One row in the benchmark results table."""

    engine: str
    quant: str
    model_size_b: float
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float
    e2e_200_tok_ms: float
    concurrency: int


def _simulate_row(
    engine_type: EngineType,
    quant_key: str,
    model_size_b: float,
    concurrency: int,
    input_len: int = 512,
) -> LatencyRow:
    """Produce a simulated latency measurement for one engine/quant/model combination.

    Applies scaling laws:
    - TTFT scales with log(input_len) and model size.
    - TPOT scales inversely with quantization compression.
    - Throughput scales with concurrency up to a saturation point.

    Args:
        engine_type: The inference engine to simulate.
        quant_key: Key into ``_QUANT_PROFILES`` (e.g. ``"Q4_K_M"``).
        model_size_b: Model parameter count in billions.
        concurrency: Number of parallel requests.
        input_len: Input token count used for TTFT scaling.

    Returns:
        A populated :class:`LatencyRow`.
    """
    base = _ENGINE_BASELINES[engine_type]
    quant = _QUANT_PROFILES[quant_key]
    model_factor = _MODEL_SCALE.get(model_size_b, model_size_b / 7.0)

    # TTFT: base * log-input-scaling * model-size-factor / quant-speedup
    ttft = (
        base["ttft_ms"]
        * math.log10(max(10, input_len) / 100 + 1)
        * model_factor
        / quant.ttft_multiplier
    )

    # TPOT: base * model-size-factor * quant-multiplier * concurrency-overhead
    conc_overhead = 1.0 + 0.08 * math.log(max(1, concurrency))
    tpot = base["tpot_ms"] * model_factor * quant.tpot_multiplier * conc_overhead

    # Throughput: tokens/s considering parallelism
    throughput = (concurrency * 1_000.0) / tpot if tpot > 0 else 0.0

    e2e = ttft + 200 * tpot

    return LatencyRow(
        engine=engine_type.value,
        quant=quant_key,
        model_size_b=model_size_b,
        ttft_ms=round(ttft, 1),
        tpot_ms=round(tpot, 2),
        throughput_tok_s=round(throughput, 1),
        e2e_200_tok_ms=round(e2e, 0),
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# ASCII table rendering
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "Engine": 14,
    "Quant": 8,
    "TTFT (ms)": 10,
    "TPOT (ms)": 10,
    "Thru (tok/s)": 13,
    "E2E 200T (ms)": 14,
}

_HEADERS = list(_COL_WIDTHS.keys())


def _separator(widths: list[int], char: str = "-") -> str:
    return "+" + "+".join(char * (w + 2) for w in widths) + "+"


def _row_fmt(values: list[str], widths: list[int]) -> str:
    cells = [f" {v:<{w}} " for v, w in zip(values, widths)]
    return "|" + "|".join(cells) + "|"


def render_table(rows: list[LatencyRow], model_size_b: float, concurrency: int) -> str:
    """Render benchmark results as a formatted ASCII table.

    Args:
        rows: Sorted list of benchmark result rows.
        model_size_b: Model size (displayed in the title).
        concurrency: Concurrency level (displayed in the title).

    Returns:
        Multi-line string ready to print to stdout.
    """
    widths = list(_COL_WIDTHS.values())
    lines: list[str] = []

    title = (
        f"  DeepSeek-R1 Distill {model_size_b:.0f}B — Inference Latency Benchmark  "
        f"(concurrency={concurrency})"
    )
    lines.append("")
    lines.append(title)
    lines.append(
        "  All numbers are simulated from published benchmarks. Run on real HW to validate."
    )
    lines.append("")
    lines.append(_separator(widths, "-"))
    lines.append(_row_fmt(_HEADERS, widths))
    lines.append(_separator(widths, "="))

    prev_engine = None
    for r in rows:
        if prev_engine and prev_engine != r.engine:
            lines.append(_separator(widths, "-"))
        lines.append(
            _row_fmt(
                [
                    r.engine,
                    r.quant,
                    f"{r.ttft_ms:.1f}",
                    f"{r.tpot_ms:.2f}",
                    f"{r.throughput_tok_s:.1f}",
                    f"{r.e2e_200_tok_ms:.0f}",
                ],
                widths,
            )
        )
        prev_engine = r.engine

    lines.append(_separator(widths, "-"))
    lines.append("")
    lines.append("  Legend:")
    lines.append("    TTFT    = Time To First Token (prefill latency), lower is better")
    lines.append("    TPOT    = Time Per Output Token (decode latency), lower is better")
    lines.append("    Thru    = Output tokens / second across all concurrent requests")
    lines.append("    E2E 200T= Estimated end-to-end latency for a 200-token response")
    lines.append("")
    lines.append("  Quant notes:")
    for k, qp in _QUANT_PROFILES.items():
        lines.append(f"    {k:8s}: {qp.quality_note}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary / recommendation
# ---------------------------------------------------------------------------


def _recommend(rows: list[LatencyRow]) -> str:
    """Return a one-line recommendation based on benchmark results."""
    best_ttft = min(rows, key=lambda r: r.ttft_ms)
    best_thru = max(rows, key=lambda r: r.throughput_tok_s)
    lines = [
        "  Recommendations:",
        f"    Lowest TTFT  -> {best_ttft.engine:14s} {best_ttft.quant}  "
        f"({best_ttft.ttft_ms:.1f} ms)",
        f"    Best throughput -> {best_thru.engine:14s} {best_thru.quant}  "
        f"({best_thru.throughput_tok_s:.1f} tok/s)",
        "    For consumer single-GPU: Ollama + Q4_K_M is the pragmatic choice.",
        "    For production APIs (multi-GPU H100): SGLang or TensorRT-LLM.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmark(model_size_b: float = 7.0, concurrency: int = 8) -> list[LatencyRow]:
    """Run the simulated benchmark and return all result rows.

    Args:
        model_size_b: Model size in billions.
        concurrency: Number of concurrent requests.

    Returns:
        Sorted list of :class:`LatencyRow` objects (by TTFT ascending).
    """
    engines = [
        EngineType.TENSORRT_LLM,
        EngineType.SGLANG,
        EngineType.VLLM,
        EngineType.OLLAMA,
    ]
    quants = list(_QUANT_PROFILES.keys())

    rows: list[LatencyRow] = []
    for engine in engines:
        for quant in quants:
            rows.append(_simulate_row(engine, quant, model_size_b, concurrency))

    return sorted(rows, key=lambda r: (r.engine, r.ttft_ms))


def main() -> None:
    """Entry point: parse args, run benchmark, print ASCII table."""
    parser = argparse.ArgumentParser(
        description="Simulated inference latency comparison for DeepSeek distilled models"
    )
    parser.add_argument(
        "--model-size", type=float, default=7.0, help="Model size in billions (1.5, 7, 14, 32, 70)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Number of concurrent requests to simulate"
    )
    args = parser.parse_args()

    if args.model_size not in _MODEL_SCALE:
        supported = ", ".join(str(k) for k in sorted(_MODEL_SCALE.keys()))
        print(f"WARNING: {args.model_size}B not in known sizes ({supported}). Extrapolating.")

    rows = run_benchmark(args.model_size, args.concurrency)
    print(render_table(rows, args.model_size, args.concurrency))
    print(_recommend(rows))
    print()


if __name__ == "__main__":
    main()
