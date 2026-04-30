"""Multi-node cluster deployment example for DeepSeek-V3/R1.

Demonstrates generating deployment plans for 4xH100 and 8xH100 clusters
with expert parallelism and tensor parallelism configurations.

Run::

    python examples/multi_node_cluster.py
"""

from __future__ import annotations

from patterns.moe_parallelism import GPUTier, MoEDeploymentPlan, plan_moe_deployment


def main() -> None:
    # (label, gpu_tier, num_gpus, dtype_bytes)
    configs: list[tuple[str, GPUTier, int, int]] = [
        ("4xH100 (FP8 MoE)", GPUTier.H100_80GB, 4, 1),
        ("8xH100 (FP8 MoE)", GPUTier.H100_80GB, 8, 1),
        ("8xH200 (BF16 MoE)", GPUTier.H200_141GB, 8, 2),
        ("4xB200 (FP4 MoE)", GPUTier.B200_192GB, 4, 1),
    ]

    for label, gpu_tier, num_gpus, dtype_bytes in configs:
        plan: MoEDeploymentPlan = plan_moe_deployment(
            num_experts=256,
            experts_per_token=8,
            gpu_tier=gpu_tier,
            num_gpus=num_gpus,
            dtype_bytes=dtype_bytes,
        )
        print(f"\n{'=' * 60}")
        print(f"Config: {label}")
        print(f"Strategy: {plan.strategy.value}")
        print(f"Total VRAM: {plan.total_vram_gb:.0f} GB")
        print(f"Model VRAM: {plan.model_vram_gb:.0f} GB")
        print(f"KV Cache: {plan.kv_cache_vram_gb:.0f} GB")
        print(f"Fits in memory: {plan.fits_in_memory}")
        print(f"Experts/GPU: {plan.ep_config.experts_per_gpu}")
        print(f"TP degree: {plan.ep_config.tensor_parallel_size}")
        if plan.notes:
            for note in plan.notes:
                print(f"  NOTE: {note}")
        print("\nvLLM:")
        print(f"  {plan.vllm_launch_cmd}")
        print("\nSGLang:")
        print(f"  {plan.sglang_launch_cmd}")


if __name__ == "__main__":
    main()
