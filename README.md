# Pensieve: Stateful LLM Serving with KV Cache Management

A simplified implementation of the Pensieve system from the EuroSys 2025 paper, demonstrating stateful LLM serving with multi-tier KV cache management for multi-turn conversations.

Implemented by Sungjin Byeon(sg20180546)

## Overview

**Problem**: Stateless LLM serving systems reprocess entire conversation history with each new request, causing redundant computation that grows with conversation length.

**Solution**: Pensieve caches KV embeddings across requests in a two-tier GPU-CPU cache, eliminating redundant computation and achieving **1.5-3.0× throughput improvement** over stateless systems like vLLM.

**Model**: Implemented for **Meta-Llama-3-8B-Instruct** using HuggingFace Transformers.

**Hardware**: Single-GPU implementation (multi-GPU support not implemented)

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
python main.py --mode pensieve --dataset sharegt --num-concurrent-users 1

# Run with vLLM baseline (stateless) mode
python main.py --mode vllm --dataset sharegt --num-concurrent-users 1

# Run with multiple concurrent users
python main.py --mode pensieve --dataset sharegt --num-concurrent-users 5 \
    --gpu-cache 64 --cpu-cache 128 --max-turns 10 --max-new-tokens 128
```

### Command-Line Options

```
--mode {pensieve,vllm}           Inference mode (default: pensieve)
--model MODEL_NAME               HuggingFace model (default: meta-llama/Meta-Llama-3-8B-Instruct)
--gpu-cache GB                   GPU cache size in GB (default: 40)
--cpu-cache GB                   CPU cache size in GB (default: 100)
--device DEVICE                  GPU device (default: cuda:0)
--dataset DATASET                Dataset to use (sharegt, etc.)
--num-concurrent-users N         Number of concurrent users (default: 1)
--max-new-tokens N               Max new tokens per turn (default: 128)
--max-turns N                    Max turns per conversation (default: 5)
--min-turns N                    Min turns per conversation (default: 1)
```

## Project Structure

```
pensieve/
├── src/pensieve/
│   ├── core/                    # Core cache management
│   │   ├── cache.py             # TwoTierCache (GPU + CPU + DROPPED tiers)
│   │   ├── eviction.py          # RetentionValuePolicy
│   │   ├── types.py             # KVChunk, CacheLocation, CacheStatistics
│   │   └── logger.py            # Logging utilities
│   ├── scheduler/               # Batch scheduling and cache planning
│   │   ├── batch_scheduler.py   # BatchScheduler with cache planning
│   │   ├── request_queue.py     # Request queue management
│   │   └── request.py           # Request/Batch types
│   ├── worker/                  # Inference worker
│   │   ├── worker.py            # Worker with cache execution
│   │   ├── custom_cache.py      # HuggingFace custom cache integration
│   │   └── batch_recovery.py    # Dropped token recovery
│   └── server/
│       └── server.py            # PensieveServer with pensieve/vllm modes
├── main.py                      # Entry point with argument parsing
├── requirements.txt
├── README.md
└── CLAUDE.md                    # Project instructions for Claude Code
```

## Key Components

### 1. KVChunk (32-token cache chunks)
Represents 32 tokens of KV cache across all transformer layers:
- **Keys**: `session_id:chunk:id:layer:idx` (unique per chunk)
- **Content**: K and V tensors for all 80 layers of Llama-3-8B
- **Metadata**: context_length (tokens before this chunk), location, access time
- **Lifecycle**: GPU → CPU → DROPPED (with recovery)

### 2. Three-Tier Cache Architecture
```python
cache = TwoTierCache(
    gpu_capacity_gb=40,      # Hot tier (fast access)
    cpu_capacity_gb=100,     # Warm tier (slower, larger)
    num_layers=80            # Llama-3-8B
)
```
- **GPU Tier**: Fast access, limited capacity
- **CPU Tier**: Slower access, larger capacity
- **DROPPED Tier**: Metadata only (tensors released), needs recomputation to recover

### 3. Retention Value Eviction Policy
Cost-based eviction optimized for minimizing recomputation:
```
Retention Value V = Cost(context_length) / time_inactive
- Lower V → evict first (cheap to recompute)
- Leading tokens (small context_length) → evicted first
- Inactive sessions → evicted before active ones
```

### 4. Batch Scheduler & Cache Planning
Coordinates multiple concurrent users with intelligent cache management:
- **BatchScheduler**: Groups requests into batches, plans cache operations
- **CachePlan**: Specifies chunks to evict, demote, and load for each batch
- **ExecutionPhase**: Swaps chunks GPU ↔ CPU with retry logic for concurrent access

### 5. Worker with Recovery
Executes inference with automatic recovery of dropped chunks:
- **execute_batch**: Runs forward pass with KV cache
- **BatchedRecoveryManager**: Recomputes dropped prefixes on-demand
- **Thread-safe Cache**: Uses RLock for concurrent multi-user access

## Design Decisions

### Simplifications from Paper

| Feature | Paper | Implementation |
|---------|-------|-----------------|
| Multi-GPU Tensor Parallelization | Per-worker independent caches | Single GPU only |
| Unified Batching | Custom CUDA kernels for prefill+generation | Not implemented (requires CUDA modifications) |
| Attention Kernel | Custom Cutlass | PyTorch `scaled_dot_product_attention` |
| Scheduling | Optimized SRPT (Shortest Remaining Processing Time) | FCFS (First-Come First-Served) |
| Profiling | Online adaptive cost modeling | Offline fixed cost model |
| Memory Management | Sophisticated fragment handling | Dict-based simple allocation |

These simplifications preserve the paper's core concepts while reducing implementation complexity. The single-GPU constraint means multi-GPU tensor parallelization from the paper is not applicable.

### Key Insights from Paper Implemented

1. **Leading tokens are cheap to recompute** (attention cost is O(context_length))
2. **Two-tier caching** effectively uses GPU speed + CPU capacity
3. **Token-level eviction** (vs conversation-level) enables fine-grained control
4. **Retention value policy** for intelligent eviction decisions

### Notes on Simplifications

- **Unified batching** (prefill + generation) requires custom CUDA kernel modifications and is not implemented in this prototype

## Performance

Expected improvements (from paper, OPT-13B):

- **Throughput**: 1.36× vs vLLM
- **Prefill speedup**: 1.0× (turn 1) → 2.0× (turn 5) → 3.5× (turn 10)
- **Cache hit rate**: 70% GPU, 20% CPU, 10% miss

## Architecture

```
┌──────────────────────────────────────┐
│   Multiple Concurrent Requests       │
│   (Multi-turn conversations)         │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│       Batch Scheduler                 │  ← FCFS Batching
├──────────────────────────────────────┤
│ • Groups requests into batches       │
│ • Plans cache operations per batch   │
│ • Coordinates eviction/demotion      │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│    Worker (Thread-Safe)               │  ← RLock protected
├──────────────────────────────────────┤
│ ┌──────────────────────────────────┐ │
│ │   TwoTierCache (GPU+CPU+DROPPED) │ │
│ ├──────────────────────────────────┤ │
│ │  GPU Tier    (hot, fast, ~40GB)  │ │
│ │  CPU Tier    (warm, large, ~100GB)
│ │  DROPPED Tier (metadata, recover)│ │
│ └──────────────────────────────────┘ │
│                                      │
│ ┌──────────────────────────────────┐ │
│ │  Llama-3-8B-Instruct Model       │ │
│ │  with Custom KV Cache Integration│ │
│ └──────────────────────────────────┘ │
│                                      │
│ ┌──────────────────────────────────┐ │
│ │  Batched Recovery Manager        │ │
│ │  (Recompute dropped tokens)      │ │
│ └──────────────────────────────────┘ │
└──────────────────────────────────────┘
```

### Concurrent Execution with Thread Safety

Multiple users submit requests concurrently:
1. **Request Batching**: BatchScheduler groups requests (FCFS order)
2. **Cache Planning**: Computes eviction/load operations for the batch
3. **Cache Execution**: Executes swaps with automatic retry on GPU full
4. **Model Inference**: Runs forward pass with custom KV cache
5. **Recovery**: Recomputes dropped chunks on-demand (batched)

All cache operations protected by `threading.RLock` to ensure consistency.

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

```bash
# Single-user Pensieve mode (stateful caching)
python main.py --mode pensieve --dataset sharegt \
    --num-concurrent-users 1 --max-turns 5 --max-new-tokens 128

# Single-user vLLM baseline (stateless, recomputes all)
python main.py --mode vllm --dataset sharegt \
    --num-concurrent-users 1 --max-turns 5 --max-new-tokens 128

# Multi-user stress test (5 concurrent users)
python main.py --mode pensieve --dataset sharegt \
    --num-concurrent-users 5 --max-turns 10 --max-new-tokens 128 \
    --gpu-cache 64 --cpu-cache 128

# Multi-user stress test (10 concurrent users)
python main.py --mode pensieve --dataset sharegt \
    --num-concurrent-users 10 --max-turns 20 --max-new-tokens 128 \
    --gpu-cache 64 --cpu-cache 256
```

## Implementation Status

### Phase 1: Foundation ✓
- [x] Project structure (core, scheduler, worker, server)
- [x] Core data structures (KVChunk, CacheLocation, CacheStatistics)
- [x] TwoTierCache with three tiers (GPU, CPU, DROPPED)
- [x] Basic HuggingFace integration with custom cache

### Phase 2: Caching & Memory Management ✓
- [x] GPU cache storage with capacity limits
- [x] KV chunk management (32-token granularity)
- [x] CPU tier with larger capacity
- [x] Session and chunk tracking
- [x] DROPPED tier for evicted chunks (metadata-only)

### Phase 3: Eviction & Two-Tier Swapping ✓
- [x] Retention value eviction policy
- [x] GPU ↔ CPU swapping with state transitions
- [x] Layer-wise cost modeling for recomputation
- [x] Token-relative cost weighting (session token length awareness)
- [x] Dictionary-based hierarchical eviction (layer → position)
- [x] Explicit tensor memory release for DROPPED chunks

### Phase 4: Concurrent Multi-User Support ✓
- [x] Dropped token recovery (recomputation)
- [x] Batched recovery manager for efficient recovery
- [x] Batch scheduler with FCFS ordering
- [x] Cache planning per batch (evict/load/recover decisions)
- [x] Thread-safe cache with `threading.RLock`
- [x] Concurrent user support (num-concurrent-users parameter)
- [x] Retry logic with GPU-full handling
- [x] Multi-turn conversation support
- [ ] Unified batching (requires custom CUDA kernels - not applicable for single GPU)

### Phase 5: Evaluation & Benchmarking
- [x] Multi-concurrent user support
- [x] ShareGPT dataset integration
- [x] Statistics tracking (hit rates, memory usage)
- [x] vLLM baseline mode for comparison
- [ ] Detailed performance profiling
- [ ] Comprehensive throughput vs latency analysis

### Recent Major Changes

**Significant refactoring from initial prototype:**

1. **Cache Architecture**: Added DROPPED tier, explicit tensor memory release, proper state transitions
2. **Concurrency**: Switched to RLock, minimized lock hold time, prevented deadlocks
3. **Cost Modeling**: Added layer-wise costs, token-relative weighting, hierarchical eviction
4. **Batch Management**: CachePlan generation, accumulated GPU tracking, retry logic
5. **Recovery**: BatchedRecoveryManager for efficient batch-level recomputation

## References

- **Paper**: "Stateful Large Language Model Serving with Pensieve" (Lingfan Yu, Jinkun Lin, Jinyang Li, EuroSys '25)
- **DOI**: https://doi.org/10.1145/3689031.3696086

## Notes

- **Single-GPU Only**: This implementation is limited to single-GPU systems. Multi-GPU tensor parallelization from the paper is not implemented.
- **Educational Prototype**: Demonstrates the paper's core concepts (two-tier caching, retention value eviction, concurrent recovery) without production-grade optimizations
- **No Unified Batching**: The paper's unified batching optimization (combining prefill+generation in custom CUDA kernels) is not implemented, as it requires low-level CUDA modifications
- **Performance Focus**: Prioritizes correctness and clarity over absolute performance. No CUDA-level kernel optimizations
- **Baseline Comparison**: Includes vLLM baseline mode for direct performance comparison on single GPU
- **Production Use**: For production deployment, use actual vLLM, TensorRT-LLM, or similar optimized systems

### Known Limitations

1. **Single GPU**: No multi-GPU support (future work would require per-worker cache managers as described in paper)
2. **No Unified Batching**: Prefill and generation phases are separate (requires custom CUDA kernels)
3. **FCFS Scheduling**: Simple first-come-first-served, not SRPT (paper uses Shortest Remaining Processing Time)
4. **Offline Cost Modeling**: Recomputation costs are fixed, not online adaptive
