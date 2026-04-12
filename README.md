# deepseek-production-cookbook

Production deployment patterns for DeepSeek V3/R1 — MoE expert parallelism, FP8/FP4 quantization, vLLM/SGLang/TensorRT-LLM configuration, KV-cache-aware routing, and distilled models on consumer GPUs.

## Why this exists

DeepSeek's 671B MoE architecture has very different deployment characteristics from dense models. The official docs cover the basics, but production teams repeatedly hit the same problems: wrong parallelism strategy for their GPU count, suboptimal quantization choies, no KV cache reuse across replicas, and distilled models that run out of VRAM with default settings. This cookbook documents what actually works.

## Patterns

| File | What it covers |
|------|---------------|
| `patterns/01_moe_parallelism.py` | Expert parallelism configs for H100/B200 clusters and single GPUs |
| `patterns/02_quantization_mixed.py` | BF16 → FP8 → INT4 → GGUF selection with VRAM estimates |
| `patterns/03_inference_engines.py` | vLLM vs SGLang vs TensorRT-LLM: when to use which |
| `patterns/04_kv_cache_routing.py` | Prefix-aware routing for 50-90% TTFT reduction on repeated prompts |
| `patterns/05_distilled_local.py` | RTX 5080/4090 deployment configs for 1.5B–70B distilled variants |

## Quick start

```bash
pip install deepseek-production-cookbook

# Find optimal config for your GPU
python examples/single_gpu_deployment.py --vram 16 --size 14

# Generate multi-node cluster launch commands
python examples/multi_node_cluster.py

# Compare engine benchmarks
python examples/benchmark_suite.py
```

## Key decisions

### Which engine?

```
Interactive chat (low TTFT) → SGLang (RadixAttention gives 2× better TTFT)
Batch/offline processing    → TensorRT-LLM (highest throughput on H100)
Ecosystem compatibility     → vLLM (easiest, best community support)
Consumer GPU / local        → Ollama (easiest) or llama.cpp (most control)
```

### Which quantization for DeepSeek-V3 (671B)?

| Hardware | Recommended | VRAM Required | Quality Delta |
|----------|-------------|---------------|---------------|
| 8× H100 80GB | FP8 E4M3 (native) | ~720 GB | 0.05 PPL |
| 4× H100 80GB | FP8 + EP=4 | ~360 GB | 0.05 PPL |
| 8× A100 80GB | INT8 GPTQ | ~360 GB | 0.10 PPL |
| Multi-node CPU | GGUF Q4_K_M | ~360 GB RAM | 0.25 PPL |

### RTX 5080 (16 GB) sweet spot

**14B Q4_K_M** is the best quality/VRAM tradeoff:
- ~9 GB model + KV cache
- 16k context comfortably
- Near-GPT-4-class reasoning

```bash
OLLAMA_NUM_CTX=16384 ollama run deepseek-r1:14b
```

## KV cache routing

When running multiple replicas, naive round-robin wastes KV cache. Route by prefix hash:

```python
from patterns.kv_cache_routing import KVCacheRouter, RequestProfile, RoutingStrategy

router = KVCacheRouter(
    replicas=["gpu0", "gpu1", "gpu2", "gpu3"],
    strategy=RoutingStrategy.PREFIX_HASH,
)

req = RequestProfile(
    request_id="req1",
    prefix="You are a helpful assistant. Context: ...",
    prompt_len=2048,
)
decision = router.route(req)
# → Always routes same prefix to same replica = 50-90% TTFT reduction
```

## Docker deployment

```bash
cp .env.example .env
# Add HF_TOKEN=hf_...

docker compose up deepseek-vllm
# API available at http://localhost:8000/v1
```

## Kubernetes

```bash
kubectl create namespace inference
kubectl create secret generic hf-credentials \
  --from-literal=token=$HF_TOKEN -n inference
kubectl apply -f configs/kubernetes/deployment.yaml
```

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Benchmarks

These are approximate published results. Run `examples/benchmark_suite.py` for simulated comparisons, then benchmark on your actual hardware with real traffic patterns.

| Engine | TTFT (H100×4, 70B) | Throughput | Setup complexity |
|--------|-------------------|------------|-----------------|
| SGLang | ~180 ms | High | Medium |
| vLLM | ~350 ms | High | Low |
| TRT-LLM | ~140 ms | Highest | High |
| Ollama | ~2500 ms | Low | Trivial |

## License

MIT
