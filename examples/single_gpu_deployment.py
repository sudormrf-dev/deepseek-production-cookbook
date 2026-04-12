"""Single GPU deployment example for DeepSeek distilled models.

Demonstrates deploying DeepSeek-R1 distilled variants on an RTX 5080
(16 GB) or RTX 4090 (24 GB) using the optimal configuration.

Run::

    python examples/single_gpu_deployment.py --vram 16 --size 14
"""

from __future__ import annotations

import argparse

from patterns.distilled_local import get_consumer_config, recommend_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU DeepSeek deployment helper")
    parser.add_argument("--vram", type=float, default=16, help="GPU VRAM in GB")
    parser.add_argument("--size", type=float, default=0, help="Model size in B (0 = auto)")
    parser.add_argument("--priority", default="quality", choices=["quality", "speed", "balanced"])
    args = parser.parse_args()

    if args.size > 0:
        cfg = get_consumer_config(args.size, args.vram)
    else:
        cfg = recommend_model(args.vram, args.priority)

    print(f"\n{'='*60}")
    print(f"Model: {cfg.model_name}")
    print(f"Quantization: {cfg.recommended_quant}")
    print(f"Estimated VRAM: {cfg.estimated_vram_gb:.1f} GB / {cfg.vram_gb:.0f} GB available")
    print(f"Context length: {cfg.context_len:,} tokens")
    print(f"Fits fully on GPU: {cfg.fits_fully_on_gpu}")
    print()
    for note in cfg.notes:
        print(f"  NOTE: {note}")

    print(f"\n{'='*60}")
    print("OLLAMA (easiest):")
    print(f"  {cfg.ollama_run_cmd}")

    print(f"\n{'='*60}")
    print("llama.cpp (most control):")
    print(f"  {cfg.llama_cpp_cmd}")

    print(f"\n{'='*60}")
    print("vLLM (production API):")
    print(f"  {cfg.vllm_cmd}")


if __name__ == "__main__":
    main()
