"""Tests for distilled_local.py."""

from __future__ import annotations

import pytest

from patterns.distilled_local import (
    ConsumerDeploymentConfig,
    get_consumer_config,
    list_consumer_configs,
    recommend_model,
)


class TestGetConsumerConfig:
    def test_7b_16gb(self):
        cfg = get_consumer_config(model_size_b=7, vram_gb=16)
        assert isinstance(cfg, ConsumerDeploymentConfig)
        assert cfg.model_size_b == 7.0
        assert cfg.vram_gb == 16

    def test_14b_16gb(self):
        cfg = get_consumer_config(model_size_b=14, vram_gb=16)
        assert cfg.model_size_b == 14.0

    def test_snaps_to_nearest_size(self):
        cfg = get_consumer_config(model_size_b=8, vram_gb=16)
        assert cfg.model_size_b == 7.0

    def test_snaps_to_vram_tier(self):
        # 22 GB >= 20 threshold snaps to the 24 GB tier
        cfg = get_consumer_config(model_size_b=7, vram_gb=22)
        assert cfg.vram_gb == 24

    def test_invalid_size_raises(self):
        # vram_gb=8 is below the minimum 12 GB threshold
        with pytest.raises(ValueError):
            get_consumer_config(model_size_b=7, vram_gb=8)


class TestListConsumerConfigs:
    def test_returns_list(self):
        configs = list_consumer_configs(vram_gb=16)
        assert len(configs) >= 3

    def test_sorted_by_size(self):
        configs = list_consumer_configs(vram_gb=16)
        sizes = [c.model_size_b for c in configs]
        assert sizes == sorted(sizes)

    def test_all_match_vram_tier(self):
        configs = list_consumer_configs(vram_gb=24)
        assert all(c.vram_gb == 24 for c in configs)

    def test_different_tiers(self):
        configs_16 = list_consumer_configs(vram_gb=16)
        configs_24 = list_consumer_configs(vram_gb=24)
        assert configs_16 != configs_24


class TestConsumerConfig:
    def test_ollama_run_cmd(self):
        cfg = get_consumer_config(7, 16)
        cmd = cfg.ollama_run_cmd
        assert "ollama run" in cmd
        assert str(cfg.context_len) in cmd

    def test_llama_cpp_cmd(self):
        cfg = get_consumer_config(7, 16)
        cmd = cfg.llama_cpp_cmd
        assert "llama-server" in cmd
        assert "--n-gpu-layers" in cmd

    def test_vllm_cmd(self):
        cfg = get_consumer_config(7, 16)
        cmd = cfg.vllm_cmd
        assert "vllm" in cmd

    def test_fits_fully_7b_16gb(self):
        cfg = get_consumer_config(7, 16)
        assert cfg.fits_fully_on_gpu is True

    def test_notes_nonempty(self):
        cfg = get_consumer_config(7, 16)
        assert len(cfg.notes) > 0

    def test_32b_on_16gb_has_warning(self):
        cfg = get_consumer_config(32, 16)
        combined = " ".join(cfg.notes).lower()
        assert "quality" in combined or "aggressive" in combined


class TestRecommendModel:
    def test_quality_returns_largest(self):
        cfg = recommend_model(vram_gb=16, priority="quality")
        all_configs = list_consumer_configs(vram_gb=16)
        fitting = [c for c in all_configs if c.fits_fully_on_gpu]
        assert cfg == fitting[-1]

    def test_speed_returns_smallest(self):
        cfg = recommend_model(vram_gb=16, priority="speed")
        all_configs = list_consumer_configs(vram_gb=16)
        fitting = [c for c in all_configs if c.fits_fully_on_gpu]
        assert cfg == fitting[0]

    def test_balanced_returns_middle(self):
        cfg = recommend_model(vram_gb=16, priority="balanced")
        all_configs = list_consumer_configs(vram_gb=16)
        fitting = [c for c in all_configs if c.fits_fully_on_gpu]
        expected = fitting[len(fitting) // 2]
        assert cfg == expected
