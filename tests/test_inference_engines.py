"""Tests for inference_engines.py."""

from __future__ import annotations

from patterns.inference_engines import (
    BenchmarkResult,
    EngineConfig,
    EngineType,
    InferenceEngine,
    benchmark_engines,
    select_optimal_engine,
)


class TestEngineConfig:
    def test_vllm_cmd_contains_model(self):
        cfg = EngineConfig(
            engine=EngineType.VLLM, model="deepseek-ai/DeepSeek-V3", tensor_parallel_size=8
        )
        cmd = cfg.to_launch_cmd()
        assert "deepseek-ai/DeepSeek-V3" in cmd
        assert "vllm" in cmd

    def test_sglang_cmd_contains_tp(self):
        cfg = EngineConfig(engine=EngineType.SGLANG, model="deepseek-r1", tensor_parallel_size=4)
        cmd = cfg.to_launch_cmd()
        assert "--tp 4" in cmd

    def test_trtllm_cmd_two_phase(self):
        cfg = EngineConfig(engine=EngineType.TENSORRT_LLM, model="deepseek-r1-70b")
        cmd = cfg.to_launch_cmd()
        assert "convert_checkpoint" in cmd
        assert "trtllm-build" in cmd

    def test_ollama_cmd(self):
        cfg = EngineConfig(engine=EngineType.OLLAMA, model="deepseek-ai/DeepSeek-R1-7B")
        cmd = cfg.to_launch_cmd()
        assert "ollama" in cmd

    def test_llama_cpp_cmd(self):
        cfg = EngineConfig(engine=EngineType.LLAMA_CPP, model="my_model")
        cmd = cfg.to_launch_cmd()
        assert "llama-server" in cmd

    def test_expert_parallel_flag(self):
        cfg = EngineConfig(
            engine=EngineType.VLLM,
            model="deepseek-v3",
            enable_expert_parallel=True,
        )
        cmd = cfg.to_launch_cmd()
        assert "expert-parallel" in cmd

    def test_extra_args_included(self):
        cfg = EngineConfig(
            engine=EngineType.VLLM,
            model="test",
            extra_args={"max-num-seqs": "256"},
        )
        cmd = cfg.to_launch_cmd()
        assert "max-num-seqs" in cmd


class TestInferenceEngine:
    def test_benchmark_returns_result(self):
        cfg = EngineConfig(engine=EngineType.VLLM, model="deepseek-r1-7b")
        engine = InferenceEngine(cfg)
        result = engine.simulate_benchmark()
        assert isinstance(result, BenchmarkResult)
        assert result.ttft_ms > 0
        assert result.throughput_tok_s > 0

    def test_sglang_faster_ttft_than_vllm(self):
        vllm_cfg = EngineConfig(engine=EngineType.VLLM, model="model")
        sglang_cfg = EngineConfig(engine=EngineType.SGLANG, model="model")
        vllm_result = InferenceEngine(vllm_cfg).simulate_benchmark()
        sglang_result = InferenceEngine(sglang_cfg).simulate_benchmark()
        assert sglang_result.ttft_ms < vllm_result.ttft_ms

    def test_e2e_latency_property(self):
        cfg = EngineConfig(engine=EngineType.VLLM, model="model")
        result = InferenceEngine(cfg).simulate_benchmark()
        expected = result.ttft_ms + 200 * result.tpot_ms
        assert abs(result.e2e_latency_ms_at_200_tokens - expected) < 0.01

    def test_launch_command_delegated(self):
        cfg = EngineConfig(engine=EngineType.SGLANG, model="mymodel")
        engine = InferenceEngine(cfg)
        assert "sglang" in engine.launch_command()


class TestBenchmarkEngines:
    def test_returns_sorted_by_ttft(self):
        results = benchmark_engines([EngineType.VLLM, EngineType.SGLANG, EngineType.TENSORRT_LLM])
        ttfts = [r.ttft_ms for r in results]
        assert ttfts == sorted(ttfts)

    def test_all_engines_in_results(self):
        engines = [EngineType.VLLM, EngineType.OLLAMA]
        results = benchmark_engines(engines)
        result_engines = {r.engine for r in results}
        assert result_engines == set(engines)

    def test_single_engine(self):
        results = benchmark_engines([EngineType.VLLM])
        assert len(results) == 1


class TestSelectOptimalEngine:
    def test_consumer_gets_ollama(self):
        engine = select_optimal_engine(hardware="consumer")
        assert engine == EngineType.OLLAMA

    def test_interactive_gets_sglang(self):
        engine = select_optimal_engine(workload="interactive", hardware="hopper")
        assert engine == EngineType.SGLANG

    def test_batch_hopper_gets_trtllm(self):
        engine = select_optimal_engine(workload="batch", hardware="hopper")
        assert engine == EngineType.TENSORRT_LLM

    def test_batch_non_hopper_gets_vllm(self):
        engine = select_optimal_engine(workload="batch", hardware="ampere")
        assert engine == EngineType.VLLM
