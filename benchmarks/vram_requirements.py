"""VRAM requirements calculator for DeepSeek model deployment.

Given a target VRAM budget, calculates weight memory and KV-cache overhead for
each model size / quantization combination and recommends the best configuration.
Output is both a human-readable Markdown table and a machine-readable dict.

Covers VRAM tiers that practitioners commonly encounter:
    8 GB  — RTX 3070/4060 Ti
    16 GB — RTX 4080 / 5080 / Pro 7900 XTX
    24 GB — RTX 4090 / A5000
    48 GB — RTX 6000 Ada / A6000
    80 GB — A100-80G / H100 SXM5 (single card)

Usage::

    python benchmarks/vram_requirements.py
    python benchmarks/vram_requirements.py --vram 24 --context 32768
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from patterns.distilled_local import _CONSUMER_CONFIGS  # noqa: PLC2701
from patterns.quantization_mixed import (
    QuantizationType,
    _BYTES_PER_PARAM,
    _QUALITY_DELTA,
    estimate_memory_gb,
)


# ---------------------------------------------------------------------------
# Model catalogue — (size_b, arch) → layer/head metadata for KV estimation
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Architecture details needed for VRAM estimation."""

    name: str
    params_b: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    is_moe: bool = False

    def weight_gb(self, quant: QuantizationType) -> float:
        """Estimated VRAM for weights at *quant*, including 12% overhead."""
        raw = self.params_b * 1e9 * _BYTES_PER_PARAM[quant]
        return raw * 1.12 / (1024**3)

    def kv_cache_gb(self, quant: QuantizationType, context_len: int) -> float:
        """Estimated VRAM for KV cache at *context_len* tokens."""
        _, kv = estimate_memory_gb(
            self.params_b,
            quant,
            quant,
            context_len=context_len,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            overhead_factor=1.0,
        )
        return kv

    def total_gb(self, quant: QuantizationType, context_len: int) -> float:
        """Combined weight + KV cache VRAM estimate."""
        return self.weight_gb(quant) + self.kv_cache_gb(quant, context_len)


_MODELS: list[ModelSpec] = [
    ModelSpec("DeepSeek-R1-Distill-1.5B", 1.5, num_layers=28, num_kv_heads=2, head_dim=64),
    ModelSpec("DeepSeek-R1-Distill-7B", 7.0, num_layers=32, num_kv_heads=8, head_dim=128),
    ModelSpec("DeepSeek-R1-Distill-14B", 14.0, num_layers=40, num_kv_heads=8, head_dim=128),
    ModelSpec("DeepSeek-R1-Distill-32B", 32.0, num_layers=64, num_kv_heads=8, head_dim=128),
    ModelSpec("DeepSeek-R1-Distill-70B", 70.0, num_layers=80, num_kv_heads=8, head_dim=128),
    ModelSpec(
        "DeepSeek-V3-671B", 671.0, num_layers=61, num_kv_heads=128, head_dim=128, is_moe=True
    ),
]

# GGUF quantizations in quality order (best → smallest)
_QUANT_ORDER: list[QuantizationType] = [
    QuantizationType.GGUF_Q8_0,
    QuantizationType.GGUF_Q6_K,
    QuantizationType.GGUF_Q5_K_M,
    QuantizationType.GGUF_Q4_K_M,
    QuantizationType.GGUF_Q3_K_M,
    QuantizationType.GGUF_Q2_K,
]

_VRAM_TIERS: list[int] = [8, 16, 24, 48, 80]


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


@dataclass
class VRAMRecommendation:
    """Best model+quantization for a given VRAM budget."""

    vram_gb: int
    model: ModelSpec
    quant: QuantizationType
    weight_gb: float
    kv_gb: float
    total_gb: float
    context_len: int
    quality_delta: float
    fits: bool
    notes: list[str]

    @property
    def ollama_tag(self) -> Optional[str]:
        """Ollama pull tag from the pre-built config table, if available."""
        for (size_b, tier), cfg in _CONSUMER_CONFIGS.items():
            vram_tier = 24 if self.vram_gb >= 20 else 16
            if abs(size_b - self.model.params_b) < 0.1 and tier == vram_tier:
                return cfg.ollama_model_tag
        return None


def recommend_for_vram(
    vram_gb: int,
    context_len: int = 32768,
) -> VRAMRecommendation:
    """Find the highest-quality model/quantization that fits in *vram_gb*.

    Iterates from largest model → smallest, and from best quantization →
    most compressed, stopping at the first combination that fits within
    90% of the available VRAM.

    Args:
        vram_gb: Available GPU VRAM in gigabytes.
        context_len: Target context length in tokens.

    Returns:
        A :class:`VRAMRecommendation` with the best matching configuration.
    """
    budget = vram_gb * 0.90  # leave 10% for activations / overhead

    for model in reversed(_MODELS):
        for quant in _QUANT_ORDER:
            w_gb = model.weight_gb(quant)
            kv_gb = model.kv_cache_gb(quant, context_len)
            total = w_gb + kv_gb
            if total <= budget:
                notes = _build_rec_notes(model, quant, vram_gb, context_len)
                return VRAMRecommendation(
                    vram_gb=vram_gb,
                    model=model,
                    quant=quant,
                    weight_gb=round(w_gb, 2),
                    kv_gb=round(kv_gb, 2),
                    total_gb=round(total, 2),
                    context_len=context_len,
                    quality_delta=_QUALITY_DELTA.get(quant, 0.0),
                    fits=True,
                    notes=notes,
                )

    # Nothing fits: return smallest possible
    m = _MODELS[0]
    q = QuantizationType.GGUF_Q2_K
    return VRAMRecommendation(
        vram_gb=vram_gb,
        model=m,
        quant=q,
        weight_gb=round(m.weight_gb(q), 2),
        kv_gb=0.0,
        total_gb=round(m.weight_gb(q), 2),
        context_len=min(context_len, 4096),
        quality_delta=_QUALITY_DELTA.get(q, 0.0),
        fits=False,
        notes=["VRAM too small for any recommended config — reduce context or upgrade GPU."],
    )


def _build_rec_notes(
    model: ModelSpec,
    quant: QuantizationType,
    vram_gb: int,
    context_len: int,
) -> list[str]:
    notes: list[str] = []
    if model.is_moe:
        notes.append(
            "MoE model: only active-expert weights load per token — effective VRAM usage lower."
        )
    if quant in {QuantizationType.GGUF_Q2_K, QuantizationType.GGUF_Q3_K_M}:
        notes.append(
            "Aggressive quantization: noticeable quality degradation. Consider smaller model size."
        )
    if context_len > 32768:
        notes.append(
            f"KV cache at {context_len:,} tokens is large; reduce context_len to save VRAM."
        )
    if vram_gb <= 8:
        notes.append("8 GB VRAM is tight; only 1.5B or small 7B configs are viable.")
    return notes


# ---------------------------------------------------------------------------
# Full VRAM matrix for all model sizes × quantizations
# ---------------------------------------------------------------------------


@dataclass
class MatrixRow:
    """One row in the VRAM requirements matrix."""

    model_name: str
    params_b: float
    quant: str
    weight_gb: float
    kv_gb_32k: float
    total_gb_32k: float
    quality_delta: float


def build_vram_matrix(context_len: int = 32768) -> list[MatrixRow]:
    """Generate the full VRAM matrix for all models and GGUF quantizations.

    Args:
        context_len: Context length used for KV cache calculation.

    Returns:
        List of :class:`MatrixRow` objects.
    """
    rows: list[MatrixRow] = []
    for model in _MODELS:
        for quant in _QUANT_ORDER:
            w = model.weight_gb(quant)
            kv = model.kv_cache_gb(quant, context_len)
            rows.append(
                MatrixRow(
                    model_name=model.name,
                    params_b=model.params_b,
                    quant=quant.value,
                    weight_gb=round(w, 2),
                    kv_gb_32k=round(kv, 2),
                    total_gb_32k=round(w + kv, 2),
                    quality_delta=_QUALITY_DELTA.get(quant, 0.0),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------


def render_markdown_matrix(rows: list[MatrixRow], context_len: int) -> str:
    """Render the VRAM matrix as a Markdown table.

    Args:
        rows: Matrix rows from :func:`build_vram_matrix`.
        context_len: Context length (shown in header).

    Returns:
        Markdown-formatted table string.
    """
    lines: list[str] = []
    lines.append(f"## VRAM Requirements Matrix (context={context_len:,} tokens)\n")
    lines.append(
        "| Model | Params | Quant | Weights (GB) | KV Cache (GB) | Total (GB) | PPL Delta |"
    )
    lines.append("|-------|--------|-------|-------------|--------------|-----------|-----------|")
    prev_model = None
    for r in rows:
        " " if r.model_name == prev_model else "**" + r.model_name.split("-")[-1] + "**"
        display_name = "" if r.model_name == prev_model else r.model_name.split("Distill-")[-1]
        lines.append(
            f"| {display_name} | {r.params_b:.1f}B | {r.quant} "
            f"| {r.weight_gb:.2f} | {r.kv_gb_32k:.2f} | {r.total_gb_32k:.2f} "
            f"| +{r.quality_delta:.2f} |"
        )
        prev_model = r.model_name
    return "\n".join(lines)


def render_recommendations(context_len: int) -> str:
    """Render per-tier recommendations as an ASCII summary block.

    Args:
        context_len: Context length used for VRAM estimation.

    Returns:
        Formatted string with one recommendation per VRAM tier.
    """
    lines: list[str] = []
    lines.append("")
    lines.append(f"  Hardware Recommendations (context={context_len:,} tokens)")
    lines.append("  " + "=" * 72)
    header = f"  {'VRAM':>6}  {'Model':35}  {'Quant':9}  {'Total GB':>8}  {'PPL+':>6}"
    lines.append(header)
    lines.append("  " + "-" * 72)

    for vram in _VRAM_TIERS:
        rec = recommend_for_vram(vram, context_len)
        status = "OK" if rec.fits else "TIGHT"
        lines.append(
            f"  {vram:>4} GB  {rec.model.name:35}  {rec.quant.value:9}  "
            f"{rec.total_gb:>7.1f}  {rec.quality_delta:>5.2f}  [{status}]"
        )
        for note in rec.notes:
            lines.append(f"           NOTE: {note}")

    lines.append("  " + "-" * 72)
    lines.append("")
    lines.append("  GPU reference:")
    lines.append("    8 GB  — RTX 3070 Ti / RTX 4060 Ti")
    lines.append("   16 GB  — RTX 4080 / RTX 5080 / RX 7900 XTX")
    lines.append("   24 GB  — RTX 4090 / A5000 / RX 7900 XTX (24 GB)")
    lines.append("   48 GB  — RTX 6000 Ada / A6000")
    lines.append("   80 GB  — A100-80G SXM / H100 SXM5")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, print Markdown matrix + recommendation table."""
    parser = argparse.ArgumentParser(description="Calculate VRAM requirements for DeepSeek models")
    parser.add_argument(
        "--vram",
        type=int,
        default=0,
        help="Show recommendation for a specific VRAM size (GB). 0 = all tiers.",
    )
    parser.add_argument(
        "--context", type=int, default=32768, help="Context length in tokens (default: 32768)"
    )
    args = parser.parse_args()

    if args.vram > 0:
        rec = recommend_for_vram(args.vram, args.context)
        print(f"\n  Best config for {args.vram} GB VRAM, context={args.context:,} tokens:")
        print(f"    Model : {rec.model.name}")
        print(f"    Quant : {rec.quant.value}")
        print(f"    Weight VRAM : {rec.weight_gb:.2f} GB")
        print(f"    KV Cache    : {rec.kv_gb:.2f} GB")
        print(f"    Total       : {rec.total_gb:.2f} GB")
        print(f"    PPL delta   : +{rec.quality_delta:.2f}")
        if rec.ollama_tag:
            print(f"    Ollama tag  : {rec.ollama_tag}")
        for note in rec.notes:
            print(f"    NOTE: {note}")
        print()
    else:
        # Print Markdown matrix to stdout (can be piped to a .md file)
        matrix = build_vram_matrix(args.context)
        print(render_markdown_matrix(matrix, args.context))
        print()
        print(render_recommendations(args.context))


if __name__ == "__main__":
    main()
