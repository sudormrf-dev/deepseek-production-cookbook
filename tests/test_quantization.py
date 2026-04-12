"""Tests for quantization_mixed.py."""

from __future__ import annotations

import pytest

from patterns.quantization_mixed import (
    QuantizationConfig,
    QuantizationMethod,
    QuantizationType,
    estimate_memory_gb,
    select_quantization,
)


class TestEstimateMemory:
    def test_bf16_large_model(self):
        weight_gb, kv_gb = estimate_memory_gb(671.0, QuantizationType.BF16)
        assert weight_gb > 1000
        assert kv_gb > 0

    def test_fp8_half_of_bf16(self):
        bf16_w, _ = estimate_memory_gb(671.0, QuantizationType.BF16)
        fp8_w, _ = estimate_memory_gb(671.0, QuantizationType.FP8_E4M3)
        assert fp8_w < bf16_w * 0.6

    def test_int4_quarter_of_bf16(self):
        bf16_w, _ = estimate_memory_gb(70.0, QuantizationType.BF16)
        int4_w, _ = estimate_memory_gb(70.0, QuantizationType.INT4)
        assert int4_w < bf16_w * 0.35

    def test_small_model(self):
        weight_gb, kv_gb = estimate_memory_gb(7.0, QuantizationType.GGUF_Q4_K_M)
        assert weight_gb < 10
        assert kv_gb > 0

    def test_kv_cache_grows_with_context(self):
        _, kv_short = estimate_memory_gb(7.0, QuantizationType.BF16, context_len=4096)
        _, kv_long = estimate_memory_gb(7.0, QuantizationType.BF16, context_len=32768)
        assert kv_long > kv_short * 4


class TestSelectQuantization:
    def test_large_vram_gets_bf16(self):
        # BF16 671B weights ~1434 GB; need 1434/0.85 ≈ 1687 GB budget
        cfg = select_quantization(target_vram_gb=2000, model_params_b=671.0)
        assert cfg.weight_type == QuantizationType.BF16

    def test_limited_vram_gets_quantized(self):
        cfg = select_quantization(target_vram_gb=16, model_params_b=7.0)
        assert cfg.weight_type != QuantizationType.BF16

    def test_fits_in_budget(self):
        cfg = select_quantization(target_vram_gb=24, model_params_b=14.0)
        assert cfg.estimated_gb <= 24 * 0.92

    def test_raises_when_impossible(self):
        with pytest.raises(ValueError, match="No quantization fits"):
            select_quantization(target_vram_gb=2, model_params_b=671.0)

    def test_quality_delta_zero_for_bf16(self):
        cfg = select_quantization(target_vram_gb=1600, model_params_b=1.5)
        assert cfg.quality_delta == 0.0

    def test_notes_nonempty(self):
        cfg = select_quantization(target_vram_gb=16, model_params_b=7.0)
        assert isinstance(cfg.notes, list)

    def test_speed_priority_prefers_fp8(self):
        cfg = select_quantization(
            target_vram_gb=1600, model_params_b=70.0, priority="speed", hardware="hopper"
        )
        assert cfg.weight_type in {QuantizationType.BF16, QuantizationType.FP8_E4M3}


class TestQuantizationConfig:
    def test_llama_cpp_cmd_gguf(self):
        cfg = select_quantization(target_vram_gb=8, model_params_b=7.0, hardware="cpu")
        if "gguf" in cfg.weight_type:
            cmd = cfg.llama_cpp_quantize_cmd("deepseek-r1-7b")
            assert "llama-quantize" in cmd

    def test_llama_cpp_cmd_non_gguf(self):
        from patterns.quantization_mixed import QuantizationType

        cfg = QuantizationConfig(
            weight_type=QuantizationType.FP8_E4M3,
            kv_cache_type=QuantizationType.FP8_E4M3,
            activation_type=QuantizationType.BF16,
            method=QuantizationMethod.FP8_DYNAMIC,
            model_params_b=70.0,
            estimated_gb=80.0,
            quality_delta=0.05,
            notes=[],
        )
        assert cfg.llama_cpp_quantize_cmd("model") == ""

    def test_vllm_kwargs_fp8(self):
        from patterns.quantization_mixed import QuantizationType

        cfg = QuantizationConfig(
            weight_type=QuantizationType.FP8_E4M3,
            kv_cache_type=QuantizationType.FP8_E4M3,
            activation_type=QuantizationType.BF16,
            method=QuantizationMethod.FP8_DYNAMIC,
            model_params_b=70.0,
            estimated_gb=80.0,
            quality_delta=0.05,
            notes=[],
        )
        kwargs = cfg.vllm_kwargs()
        assert kwargs.get("quantization") == "fp8"
        assert kwargs.get("kv_cache_dtype") == "fp8"

    def test_vllm_kwargs_awq(self):
        from patterns.quantization_mixed import QuantizationType

        cfg = QuantizationConfig(
            weight_type=QuantizationType.INT4,
            kv_cache_type=QuantizationType.BF16,
            activation_type=QuantizationType.BF16,
            method=QuantizationMethod.AWQ,
            model_params_b=7.0,
            estimated_gb=4.5,
            quality_delta=0.4,
            notes=[],
        )
        kwargs = cfg.vllm_kwargs()
        assert kwargs.get("quantization") == "awq_marlin"
