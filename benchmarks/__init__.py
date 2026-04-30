"""Benchmarks package for DeepSeek production cookbook.

Provides simulation-based latency and VRAM estimation tools that run
entirely without GPU hardware, useful for capacity planning and
hardware selection before committing to an infrastructure choice.

Modules:
    inference_latency: Compare TTFT and throughput across engines and quantizations.
    vram_requirements: Calculate VRAM needs and recommend hardware configs.
"""
