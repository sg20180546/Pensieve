# Pensieve: Stateful LLM Serving with KV Cache Management

A simplified implementation of the Pensieve system from the EuroSys 2025 paper, demonstrating stateful LLM serving with multi-tier KV cache management for multi-turn conversations.

Implemented by Sungjin Byeon(sg20180546)

## Overview

**Problem**: Stateless LLM serving systems reprocess entire conversation history with each new request, causing redundant computation that grows with conversation length.

**Solution**: Pensieve caches KV embeddings across requests in a two-tier GPU-CPU cache, eliminating redundant computation and achieving **1.5-3.0× throughput improvement** over stateless systems like vLLM.

## Key Features

- **Two-Tier Cache**: GPU (hot, fast) + CPU (warm, large) storage tiers
- **Retention Value Eviction**: Intelligent eviction preferring old sessions and leading tokens (cheaper to recompute)
- **Chunk-Level Management**: 32-token granularity for fine-grained cache control
- **Multi-Turn Support**: Efficient reuse of cached KV across conversation turns
- **vLLM Baseline**: Compare Pensieve against stateless baseline

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with Pensieve (stateful) mode
python main.py --dataset sharegt   --num-concurrent-users 3 --model  meta-llama/Meta-Llama-3-8B-Instruct --gpu-cache 20 --cpu-cache 128 --min-turns 6 --max-turns 8  --max-new-tokens 1024 --request-interval 2


# Interactive multi-turn conversation
python main.py --mode pensieve --model  meta-llama/Meta-Llama-3-8B-Instruct --interactive
```

### Command-Line Options

```
--mode {pensieve,vllm,compare}   Inference mode (default: pensieve)
--model MODEL_NAME               HuggingFace model (default: gpt2)
--gpu-cache GB                   GPU cache size in GB (default: 40)
--cpu-cache GB                   CPU cache size in GB (default: 100)
--device DEVICE                  GPU device (default: cuda:0)
--interactive                    Run interactive multi-turn mode
--max-new-tokens N               Max new tokens per turn (default: 32)
```


## Key Components

### 1. KVChunk (32-token cache chunks)
```python
chunk = KVChunk(
    session_id="session_1",
    chunk_id=0,
    layer_kv_tensors={0: (k_tensor, v_tensor), ...},
    context_length=0,  # Tokens BEFORE this chunk
    location=CacheLocation.GPU
)
```

### 2. Two-Tier Cache
```python
cache = TwoTierCache(gpu_capacity_gb=40, cpu_capacity_gb=100)
cache.store_chunk(chunk, location=CacheLocation.GPU)
cache.swap_chunk_to_cpu(chunk_key)  # Move GPU -> CPU
```

### 3. Retention Value Eviction
```python
policy = RetentionValuePolicy()
retention_value = policy.calculate_retention_value(chunk)
# V = Cost(context_length) / time_inactive
# Lower value = evict first
# Leading tokens (small context_length) evicted first!
```

### 4. Server with Mode Selection
```python
# Pensieve mode (stateful)
server = create_server(mode="pensieve")
response = server.process_request("session_1", "Hello")

# vLLM baseline mode (stateless)
baseline = create_server(mode="vllm")
response = baseline.process_request("session_1", "Hello")
```

## Design Decisions

### Simplifications from Paper

| Feature | Paper | Prototype |
|---------|-------|-----------|
| Attention Kernel | Custom Cutlass | PyTorch `scaled_dot_product_attention` |
| Scheduling | Optimized | Simple FCFS |
| Memory Pool | Fragment handling | Dict-based |
| Profiling | Online adaptive | Offline only |

These simplifications preserve core concepts while reducing implementation complexity.

### Key Insights from Paper

1. **Leading tokens are cheap to recompute** (attention cost is O(context_length))
2. **Two-tier caching** effectively uses GPU speed + CPU capacity
3. **Token-level eviction** (vs conversation-level) enables fine-grained control
4. **Unified batching** (prefill + generation). While Cache reducing computation, Let's dispatch other sessions prefill/genereation request together. This project does not include this feature.
5. **Multi-Token Attention** Fast Attention Kernel module that overcomes single execution attention kernel tfor non-contigiuous QKV on GPU memory. This project does not include this feature.

## Performance

Expected improvements (from paper, OPT-13B):

- **Throughput**: 1.36× vs vLLM
- **Prefill speedup**: 1.0× (turn 1) → 1.2× (turn 5) → 1.5× (turn 10)
- **Cache hit rate**: 70% GPU, 20% CPU, 10% miss

## Architecture

```
┌─────────────────────────────┐
│    Request/Conversation     │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│        Scheduler            │  ← Batching, Cache Management
├─────────────────────────────┤
│   Batch Scheduler (FCFS)    │
│   Cache Manager             │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│         Worker              │
├─────────────────────────────┤
│  GPU LLM Model              │
│  Custom KV Cache            │
│  ├─ GPU Tier (40GB)        │
│  └─ CPU Tier (100GB)       │
└─────────────────────────────┘
```

## Inference Modes

### Pensieve Mode (Stateful)
- Caches KV across requests
- Reuses cached KV for same session
- Lower latency for multi-turn conversations
- Memory usage: GPU + CPU cache

### vLLM Baseline Mode (Stateless)
- Simulates vLLM baseline behavior
- Reprocesses entire history each turn
- Higher latency (especially for long histories)
- Enables direct performance comparison

## Statistics

Both modes track:
- Total requests processed
- Total tokens generated
- Average prefill time
- Cache hit rates (GPU, CPU, miss)
- Memory usage

Access with `server.get_statistics_str()`:

```
=== Pensieve Server Statistics ===
Inference Mode: Pensieve (Stateful)
Total Requests: 10
Active Sessions: 2

Cache Statistics:
GPU: 5.23/40.00 GB | CPU: 2.15/100.00 GB
GPU Hit Rate: 70.0%
CPU Hit Rate: 20.0%
Miss Rate: 10.0%
```

## References

- **Paper**: "Stateful Large Language Model Serving with Pensieve" (Lingfan Yu, Jinkun Lin, Jinyang Li, EuroSys '25)
- **DOI**: https://doi.org/10.1145/3689031.3696086

## Notes

- This is an **educational prototype** demonstrating the paper's core concepts, not a production system
- For maximum performance in production, use actual vLLM or TensorRT-LLM
- As this project does not touch cuda level optimization, mesurement of TTFT or prefill/generation time separation is infeasible.
- The codebase prioritizes clarity over performance
