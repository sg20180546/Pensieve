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
python main.py --mode pensieve --model gpt2

# Run with vLLM baseline (stateless) mode
python main.py --mode vllm --model gpt2

# Compare Pensieve vs vLLM
python main.py --mode compare --model gpt2

# Interactive multi-turn conversation
python main.py --mode pensieve --interactive
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

## Project Structure

```
pensieve/
├── src/pensieve/
│   ├── core/                    # Core data structures
│   │   ├── cache.py             # TwoTierCache implementation
│   │   ├── eviction.py          # RetentionValuePolicy
│   │   └── types.py             # Request, Batch, KVChunk, etc.
│   ├── worker/
│   │   └── custom_cache.py      # HuggingFace integration
│   └── server/
│       └── server.py            # Main server with Pensieve & vLLM modes
├── main.py                      # Entry point with argument parsing
├── requirements.txt
└── README.md
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
4. **Unified batching** (prefill + generation) improves GPU utilization

## Performance

Expected improvements (from paper, OPT-13B):

- **Throughput**: 1.36× vs vLLM
- **Prefill speedup**: 1.0× (turn 1) → 2.0× (turn 5) → 3.5× (turn 10)
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

## Testing

```python
# Test basic inference
python scripts/test_basic_inference.py

# Compare modes
python main.py --mode compare --num-conversations 5

# Interactive mode
python main.py --interactive
```

## Implementation Status

### Phase 1: Foundation ✓
- [x] Project structure
- [x] Core data structures (KVChunk, Request, Batch)
- [x] TwoTierCache
- [x] Basic inference

### Phase 2: Caching ✓
- [x] GPU cache storage
- [x] KV chunk management
- [x] Session tracking

### Phase 3: Eviction & Two-Tier ✓
- [x] Retention value policy
- [x] CPU tier support
- [x] GPU ↔ CPU swapping

### Phase 4: Advanced Features (In Progress)
- [ ] Dropped token recovery
- [ ] Multi-token attention kernel
- [ ] Unified batching (prefill + generation)
- [ ] Pipelined transfer

### Phase 5: Evaluation (Future)
- [ ] ShareGPT dataset
- [ ] Performance benchmarking
- [ ] Throughput vs latency plots

## References

- **Paper**: "Stateful Large Language Model Serving with Pensieve" (Lingfan Yu, Jinkun Lin, Jinyang Li, EuroSys '25)
- **DOI**: https://doi.org/10.1145/3689031.3696086

## Notes

- This is an **educational prototype** demonstrating the paper's core concepts, not a production system
- For maximum performance in production, use actual vLLM or TensorRT-LLM
- The codebase prioritizes clarity over performance
