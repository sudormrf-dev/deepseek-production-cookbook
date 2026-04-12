"""Tests for moe_parallelism.py."""

from __future__ import annotations

from patterns.moe_parallelism import (
    GPUTier,
    MoEDeploymentPlan,
    ParallelismStrategy,
    estimate_model_vram_gb,
    plan_moe_deployment,
)


class TestEstimateModelVram:
    def test_bf16_large_model(self):
        gb = estimate_model_vram_gb("deepseek-v3", dtype_bytes=2)
        # 671B * 2 bytes / 1024^3 * 1.15 ≈ 1434 GB
        assert gb > 1000

    def test_fp8_reduces_memory(self):
        bf16_gb = estimate_model_vram_gb("deepseek-v3", dtype_bytes=2)
        fp8_gb = estimate_model_vram_gb("deepseek-v3", dtype_bytes=1)
        assert fp8_gb < bf16_gb * 0.6

    def test_distilled_model_small(self):
        gb = estimate_model_vram_gb("deepseek-r1-distill-qwen-7b", dtype_bytes=2)
        # 7B * 2 bytes / 1024^3 * 1.15 ≈ 15 GB
        assert gb < 20

    def test_unknown_model_defaults_to_671b(self):
        gb = estimate_model_vram_gb("unknown-model")
        assert gb > 500


class TestPlanMoeDeployment:
    def test_returns_deployment_plan(self):
        plan = plan_moe_deployment(gpu_tier=GPUTier.H100_80GB, num_gpus=8)
        assert isinstance(plan, MoEDeploymentPlan)

    def test_single_gpu_for_distilled(self):
        plan = plan_moe_deployment(
            model="deepseek-r1-distill-qwen-7b",
            gpu_tier=GPUTier.H100_80GB,
            num_gpus=8,
            dtype_bytes=2,
        )
        assert plan.strategy == ParallelismStrategy.SINGLE_GPU

    def test_expert_parallel_for_full_model(self):
        plan = plan_moe_deployment(
            num_experts=256,
            gpu_tier=GPUTier.H100_80GB,
            num_gpus=8,
            model="deepseek-v3",
            dtype_bytes=1,
        )
        assert plan.strategy in {
            ParallelismStrategy.EXPERT_PARALLEL,
            ParallelismStrategy.TENSOR_PARALLEL,
        }

    def test_vllm_cmd_generated(self):
        plan = plan_moe_deployment(gpu_tier=GPUTier.H100_80GB, num_gpus=4)
        assert "vllm" in plan.vllm_launch_cmd

    def test_sglang_cmd_generated(self):
        plan = plan_moe_deployment(gpu_tier=GPUTier.H100_80GB, num_gpus=4)
        assert "sglang" in plan.sglang_launch_cmd

    def test_consumer_gpu_notes(self):
        plan = plan_moe_deployment(
            model="deepseek-r1-distill-qwen-7b",
            gpu_tier=GPUTier.RTX_5080,
            num_gpus=1,
            dtype_bytes=2,
        )
        assert any("consumer" in n.lower() or "distilled" in n.lower() for n in plan.notes)

    def test_ep_config_experts_per_gpu(self):
        plan = plan_moe_deployment(
            num_experts=256,
            gpu_tier=GPUTier.H100_80GB,
            num_gpus=8,
        )
        assert plan.ep_config.experts_per_gpu >= 1

    def test_total_vram_property(self):
        plan = plan_moe_deployment(gpu_tier=GPUTier.H100_80GB, num_gpus=8)
        assert plan.total_vram_gb == 640.0  # 80 * 8


class TestMoEDeploymentPlanProperties:
    def test_utilization_pct(self):
        plan = plan_moe_deployment(
            num_experts=256, experts_per_token=8, gpu_tier=GPUTier.H100_80GB, num_gpus=8
        )
        assert 2 <= plan.ep_config.utilization_pct <= 5  # 8/256 = 3.125%

    def test_expert_parallel_size(self):
        plan = plan_moe_deployment(
            num_experts=256, gpu_tier=GPUTier.A100_80GB, num_gpus=4, model="deepseek-v3"
        )
        assert plan.ep_config.expert_parallel_size >= 1
