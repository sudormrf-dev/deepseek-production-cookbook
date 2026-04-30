"""Single GPU deployment guide for DeepSeek distilled models.

Step-by-step guide for deploying DeepSeek-R1 distilled variants on a single
consumer GPU (RTX 5080 16 GB, RTX 4090 24 GB, etc.) using Ollama, llama.cpp,
or vLLM.  Also validates that the chosen configuration is sane before printing
the commands.

Run::

    # Recommended 14B model on a 16 GB GPU (auto-detect best config):
    python examples/single_gpu_deployment.py --vram 16

    # Specify a concrete size and priority:
    python examples/single_gpu_deployment.py --vram 24 --size 32 --priority quality

    # Show all available commands without running anything:
    python examples/single_gpu_deployment.py --vram 16 --size 7 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from patterns.distilled_local import ConsumerDeploymentConfig, get_consumer_config, recommend_model

_SEPARATOR = "=" * 70


def _print_section(title: str) -> None:
    """Print a clearly delimited section header."""
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def _validate_config(cfg: ConsumerDeploymentConfig) -> list[str]:
    """Run sanity checks on a config and return a list of warnings.

    Args:
        cfg: Deployment configuration to validate.

    Returns:
        List of warning strings; empty list means all checks passed.
    """
    warnings: list[str] = []
    if cfg.estimated_vram_gb > cfg.vram_gb:
        warnings.append(
            f"Estimated VRAM {cfg.estimated_vram_gb:.1f} GB exceeds available "
            f"{cfg.vram_gb:.0f} GB — CPU offload required."
        )
    if cfg.context_len < 4096:
        warnings.append("Context length < 4096 tokens; chain-of-thought quality may suffer.")
    if cfg.gpu_layers != -1:
        warnings.append(
            f"Only {cfg.gpu_layers} layers on GPU; remaining layers run on CPU RAM. "
            "Ensure you have 32+ GB system RAM."
        )
    return warnings


def _print_step_by_step_guide(cfg: ConsumerDeploymentConfig) -> None:
    """Print a numbered step-by-step deployment guide for Ollama.

    Args:
        cfg: Deployment configuration to guide the user through.
    """
    _print_section("Step-by-step guide: Deploy with Ollama (easiest)")

    print("""
  Prerequisites
  -------------
  Install Ollama (Linux / macOS):
    curl -fsSL https://ollama.com/install.sh | sh

  Verify the Ollama daemon is running:
    ollama serve &         # starts in background if not already running
    ollama list            # should print an empty table on first run
""")

    print(f"  Step 1 — Pull the model ({cfg.ollama_model_tag})")
    print(f"  {'─' * 60}")
    print(f"    ollama pull {cfg.ollama_model_tag}")
    print()
    print(
        "    This downloads the GGUF file (~{:.1f} GB). Grab a coffee.".format(
            cfg.estimated_vram_gb
        )
    )

    print()
    print("  Step 2 — Run with optimal context window")
    print(f"  {'─' * 60}")
    print(f"    {cfg.ollama_run_cmd}")
    print()
    print(f"    Context set to {cfg.context_len:,} tokens for this VRAM tier.")
    print("    For interactive chat (REPL), just omit OLLAMA_NUM_CTX and let Ollama default.")

    print()
    print("  Step 3 — Validate the deployment")
    print(f"  {'─' * 60}")
    print(
        """    # Quick sanity test (should reply in < 10 seconds):
    curl http://localhost:11434/api/generate \\
      -d '{"model": \""""
        + cfg.ollama_model_tag
        + """\", "prompt": "What is 2+2?", "stream": false}'

    # OpenAI-compatible chat endpoint (works with LangChain / LlamaIndex):
    curl http://localhost:11434/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": \""""
        + cfg.ollama_model_tag
        + """\",
        "messages": [{"role": "user", "content": "Explain FP8 quantization."}]
      }'"""
    )

    print()
    print("  Step 4 — (Optional) Expose as an OpenAI-compatible API server")
    print(f"  {'─' * 60}")
    print("    # Ollama already serves an OpenAI-compatible endpoint at :11434.")
    print("    # Point any OpenAI SDK client to http://localhost:11434/v1")
    print()
    print("    from openai import OpenAI")
    print("    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')")
    print(f'    resp = client.chat.completions.create(model="{cfg.ollama_model_tag}",')
    print('      messages=[{"role": "user", "content": "Hello"}])')
    print("    print(resp.choices[0].message.content)")


def _print_llama_cpp_guide(cfg: ConsumerDeploymentConfig) -> None:
    """Print the llama.cpp deployment guide for maximum control.

    Args:
        cfg: Deployment configuration.
    """
    _print_section("Alternative: llama.cpp server (maximum control / lowest overhead)")
    print()
    print("  Build llama.cpp with CUDA support:")
    print("    git clone https://github.com/ggerganov/llama.cpp")
    print("    cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j8")
    print()
    print("  Download the GGUF:")
    print(
        f"    huggingface-cli download bartowski/{cfg.model_name}-GGUF "
        f"--include '*{cfg.recommended_quant}*' --local-dir ./models"
    )
    print()
    print("  Launch server:")
    print(f"    {cfg.llama_cpp_cmd}")
    print()
    print("  The server exposes an OpenAI-compatible API at http://0.0.0.0:8080/v1")


def _print_vllm_guide(cfg: ConsumerDeploymentConfig) -> None:
    """Print the vLLM deployment guide for production APIs.

    Args:
        cfg: Deployment configuration.
    """
    _print_section("Production API: vLLM (AWQ quantization, OpenAI-compatible)")
    print()
    print("  Install (requires CUDA 12.1+, Python 3.10+):")
    print("    pip install vllm")
    print()
    print("  Launch:")
    print(f"    {cfg.vllm_cmd}")
    print()
    print("  vLLM exposes an OpenAI-compatible endpoint at http://0.0.0.0:8000/v1")
    print("  Use --quantization awq_marlin for the fastest INT4 inference on Ampere/Hopper GPUs.")


def main() -> None:
    """Parse CLI arguments and print the deployment guide."""
    parser = argparse.ArgumentParser(
        description="Single-GPU DeepSeek deployment helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/single_gpu_deployment.py --vram 16          # auto-select best model
  python examples/single_gpu_deployment.py --vram 16 --size 7 # force 7B
  python examples/single_gpu_deployment.py --vram 24 --priority quality
""",
    )
    parser.add_argument("--vram", type=float, default=16, help="GPU VRAM in GB (default: 16)")
    parser.add_argument(
        "--size", type=float, default=0, help="Model size in billions (0 = auto-select)"
    )
    parser.add_argument(
        "--priority",
        default="quality",
        choices=["quality", "speed", "balanced"],
        help="Selection priority when auto-choosing model (default: quality)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without interactive pauses"
    )
    args = parser.parse_args()

    # Resolve config
    try:
        if args.size > 0:
            cfg = get_consumer_config(args.size, args.vram)
        else:
            cfg = recommend_model(args.vram, args.priority)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Header
    _print_section(f"DeepSeek Single-GPU Deployment  —  {cfg.vram_gb:.0f} GB VRAM")
    print(f"\n  Model        : {cfg.model_name}")
    print(f"  Quantization : {cfg.recommended_quant}")
    print(f"  Est. VRAM    : {cfg.estimated_vram_gb:.1f} GB / {cfg.vram_gb:.0f} GB available")
    print(f"  Context len  : {cfg.context_len:,} tokens")
    print(
        f"  Fits on GPU  : {'Yes — full GPU inference' if cfg.fits_fully_on_gpu else 'No — CPU offload required'}"
    )

    warnings = _validate_config(cfg)
    if warnings:
        print()
        for w in warnings:
            print(f"  WARNING: {w}")

    print()
    print("  Deployment notes:")
    for note in cfg.notes:
        print(f"    • {note}")

    # Step-by-step guides
    _print_step_by_step_guide(cfg)
    _print_llama_cpp_guide(cfg)
    _print_vllm_guide(cfg)

    _print_section("Summary")
    print(f"\n  Quickest start:  {cfg.ollama_run_cmd}")
    print(f"  Ollama model tag: {cfg.ollama_model_tag}")
    print()


if __name__ == "__main__":
    main()
