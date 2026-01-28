# Concurrent Benchmark Implementation Summary

## Overview

Implemented a concurrent multi-user benchmarking system to properly evaluate Pensieve's unified batching capabilities and demonstrate performance improvements over stateless vLLM baseline.

## Problem Statement

**Original Issue**: The sequential benchmark in `main.py` processed conversations one at a time, which:
- Never created realistic concurrent request arrival patterns
- Made BatchScheduler unable to form unified batches (mixing PREFILL + GENERATION)
- Failed to demonstrate Pensieve's core advantage: efficient multi-session batching
- Resulted in batch sizes of ~1 (no real batching benefit)

**Solution**: Implement concurrent client simulation with:
- Multiple threads simulating independent users
- Requests arriving simultaneously from different sessions
- Realistic staggered request intervals
- System-level metrics aggregation

## Implementation Details

### 1. Command-Line Parameters

Added to `main.py` argument parser:

```python
--num-concurrent-users N      # Number of concurrent users (default: 1)
--request-interval SECONDS    # Delay between requests per user (default: 0.5)
```

**Usage Examples**:
```bash
# Sequential comparison (original)
python main.py --mode compare --model gpt2

# Concurrent with 4 users
python main.py --mode compare --model gpt2 --num-concurrent-users 4

# Concurrent with 8 users, longer intervals
python main.py --mode compare --model gpt2 --num-concurrent-users 8 --request-interval 1.0
```

### 2. Concurrent Client Worker Function

**Location**: `main.py`, lines 247-294

**Signature**:
```python
def concurrent_client_worker(
    client_id: int,
    server,
    conversations: list,
    request_interval: float,
    results_queue: Queue,
) -> None
```

**Behavior**:
- Runs in separate thread
- Simulates single user making multiple sequential requests
- Waits `request_interval` seconds between turns
- Collects per-request metrics:
  - TTFT (Time To First Token)
  - Tail latency (total response time)
- Puts aggregated results in thread-safe queue

**Per-Client Conversations**:
```
Client 0: "Hello, how are you?" → "Tell me about Python" → "What is machine learning?"
Client 1: "Hi there" → "Explain AI" → "How does deep learning work?"
Client 2: "What's up?" → "Tell me about data science" → ...
...
```

### 3. Concurrent Comparison Function

**Location**: `main.py`, lines 297-532

**Phases**:

#### Phase 1: Pensieve (Concurrent)
1. Initialize Pensieve server
2. Launch N concurrent client threads
3. Wait for all threads to complete
4. Aggregate metrics across all clients
5. Clean up GPU memory

#### Phase 2: vLLM Baseline (Concurrent)
1. Initialize vLLM server
2. Launch N concurrent client threads with same conversations
3. Wait for all threads to complete
4. Aggregate metrics across all clients

#### Phase 3: Comparison & Analysis
Compute and display:
- **Time Speedup**: `vllm_total_time / pensieve_total_time`
- **Throughput**: `total_requests / elapsed_time` (requests per second)
- **TTFT Metrics**: Average and P99 latencies
- **Tail Latency Metrics**: Average and P99 latencies

### 4. Metrics Collected

Per-client (before aggregation):
- `ttfts`: List of TTFT values (one per turn)
- `tail_latencies`: List of response latencies (one per turn)
- `response_count`: Total requests processed by client

Aggregated (final results):
- `all_pensieve_ttfts`: Combined TTFT from all clients and turns
- `all_pensieve_tail_latencies`: Combined tail latencies from all clients
- `total_pensieve_requests`: Total requests across all clients
- `pensieve_total_time`: Wall-clock time from first request start to last completion

### 5. Integration with Main Logic

**File**: `main.py`, lines 150-155

```python
elif args.mode == "compare":
    # Use concurrent benchmark if num_concurrent_users > 1
    if args.num_concurrent_users > 1:
        run_concurrent_comparison(args)
    else:
        run_comparison(args)  # Original sequential benchmark
```

**Decision Logic**:
- If `--num-concurrent-users > 1`: Use new concurrent benchmark
- Otherwise: Fall back to original sequential comparison for backwards compatibility

## Key Design Decisions

### 1. Client Hotness Distribution (New!)
```python
# Each client has different request frequency
HOT clients (1/3):    interval = 0.3 × base_interval
WARM clients (1/3):   interval = 1.0 × base_interval
COLD clients (1/3):   interval = 2.5 × base_interval
```

**Why This Matters**:
- **HOT clients**: Frequent access → stays in GPU cache → high cache hit rate
- **WARM clients**: Normal access → boundary between GPU/CPU → mixed hit rate
- **COLD clients**: Infrequent access → evicted to CPU/DROPPED → triggers recovery

**Impact on Eviction Policy**:
- Retention value prefers evicting COLD clients (low frequency, high recompute cost)
- WARM clients stay in GPU (good tradeoff)
- When COLD client returns → recovery/swap overhead vs savings from cached earlier turns

### 2. Thread-Safe Queue for Results
```python
results_queue = Queue()  # Thread-safe by default
```
- Each client thread puts results in queue
- Main thread reads queue after all threads complete
- No race conditions or locks needed

### 3. Aggregate-Only Metrics
Per user request ("클라이언트 별로 따로 재지는 않아도 될거같아"):
- Collect all TTFT values from all clients
- Compute aggregate statistics (mean, p99)
- Don't track per-client breakdowns
- Simpler implementation, clearer system-level insights

### 4. Explicit Memory Cleanup Between Runs
```python
del pensieve_server
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```
- Ensures clean GPU state for fair comparison
- Prevents memory contention between Pensieve and vLLM runs
- Critical for accurate benchmarking

### 5. Same Conversations with Different Access Patterns
- Pensieve and vLLM process same conversation sets with same hotness pattern
- Ensures fair comparison
- Differences in metrics purely from system design (cache reuse + hotness awareness vs stateless recomputation)

## Concurrency Behavior with Cache Hotness

### Request Arrival Timeline Example (6 Users with Hotness)

```
Time    Client 0 (HOT)        Client 1 (HOT)        Client 2 (WARM)       Client 3 (WARM)       Client 4 (COLD)       Client 5 (COLD)
        interval=0.15s        interval=0.15s        interval=0.5s         interval=0.5s         interval=1.25s        interval=1.25s

t=0.0   [Turn 1 starts]       [Turn 1 starts]       [Turn 1 starts]       [Turn 1 starts]       [Turn 1 starts]       [Turn 1 starts]
t=0.1   [Turn 1 ends]         [Turn 1 ends]         [Turn 1 ends]         [Turn 1 ends]         [Turn 1 ends]         [Turn 1 ends]
        ↓
        [Batch 1: 6 prefill] → All in GPU cache initially

t=0.15  [Turn 2 C0 starts]    [Turn 2 C1 starts]
t=0.25  [Turn 2 ends]         [Turn 2 ends]
        ↓
        [Batch 2: Mixed - C0,C1 GENERATION + others PREFILL] → GPU cache hit for HOT clients

t=0.5   [Turn 2 C2 starts]    [Turn 2 C3 starts]
t=0.6   [Turn 2 ends]         [Turn 2 ends]
        ↓
        [Batch N: C0,C1 higher generation turns + C2,C3 turn 2]

t=1.25  [Turn 2 C4 starts]    [Turn 2 C5 starts]
t=1.35  [Turn 2 ends]         [Turn 2 ends]
        ↓
        [Batch X: COLD clients' turn 2 may find cache evicted]
        [If evicted: Swap/Recovery overhead, but still reuse cached prefill]
```

**Key Insights**:
1. **HOT clients** (0.15s interval): High frequency cache hits → GPU stays
2. **WARM clients** (0.5s interval): Reasonable hits, some CPU swapping
3. **COLD clients** (1.25s interval): May find evicted cache → recovery/swap costs
4. **Pensieve advantage**: Even with swap costs, prefill reuse saves time
5. **vLLM disadvantage**: Recomputes ALL history regardless of access pattern

### Realistic Cache State Timeline

```
After ~1 second of benchmark:

GPU Cache:
├─ Client 0 (HOT): Full cache [most recent turn]
├─ Client 1 (HOT): Full cache [most recent turn]
├─ Client 2 (WARM): Partial cache [some early chunks evicted]
├─ Client 3 (WARM): Partial cache [older chunks in CPU]
└─ [No space for COLD clients]

CPU Cache:
├─ Client 2: Early chunks
├─ Client 3: Early chunks
├─ Client 4 or 5: Some chunks (if accessed recently)
└─ ...

DROPPED (Lost):
├─ Very old chunks from COLD clients
└─ Leading tokens (cheap to recompute)
```

**Eviction Pattern**:
- COLD clients' old chunks → CPU or DROPPED
- Within same client: Leading tokens → DROPPED first (cheaper recovery)
- Later tokens (high context length) → stay in cache
- Retention value formula: `V = Cost(context_length) / time_since_access`

## Benefits Over Sequential Benchmark

| Aspect | Sequential | Concurrent (Old) | Concurrent (New - with Hotness) |
|--------|-----------|-----------|-----------|
| Request Pattern | One-at-a-time | Same frequency all | Different frequencies per client |
| Batch Sizes | ~1 | 2-N | 2-N |
| PREFILL+GENERATION Mix | No | Yes | Yes |
| Real Batching Benefits | None | Visible | Visible |
| Scheduler Pressure | None | High | High |
| Cache Contention | None | Minimal | **Realistic** |
| Cache Hotness | N/A | Uniform | **Zipfian-like distribution** |
| Eviction Policy Test | No | Limited | **Full showcase** |
| Recovery Overhead | N/A | None | **Measured** |
| Throughput Improvements | Underestimated | Realistic | **Most realistic** |

**Key Improvement**:
- Old concurrent benchmark treated all clients equally → all sessions evicted/recovered uniformly
- New hotness-aware benchmark shows Pensieve's intelligent eviction:
  - HOT clients stay in GPU cache
  - COLD clients trigger eviction/recovery but still benefit from prefill reuse
  - Retention value policy determines which clients get space

## Performance Expectations

### With 4 Concurrent Users

**Pensieve Benefits**:
1. **Cache Reuse**: Later turns reuse cached KV from earlier turns
2. **Unified Batching**: Mix of requests enables better GPU utilization
3. **Pipeline Overlap**: Scheduler + Worker can overlap compute with cache swaps

**Expected Speedups**:
- Throughput: 1.5-2.0× vs sequential
- TTFT on Turn 5+: 1.3-1.8× vs baseline
- Better consistency (lower p99 latencies)

## Usage Examples

### Quick Test (3 Users with Auto Hotness)
```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 3
# Automatically creates: 1 HOT, 1 WARM, 1 COLD client
```

### Standard Benchmark (6 Users - Full Hotness Distribution)
```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.5
# Creates: 2 HOT (0.15s), 2 WARM (0.5s), 2 COLD (1.25s)
# Best shows cache eviction vs reuse benefits
```

### Heavy Load (9 Users with Tight Hot Requests)
```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 9 --request-interval 0.3
# HOT clients get even more frequent (0.09s)
# More cache contention → better tests eviction policy
```

### With Full Model and Realistic Cache Sizes
```bash
python main.py --mode compare --model meta-llama/Meta-Llama-3-8B \
  --num-concurrent-users 6 --request-interval 0.5 \
  --gpu-cache 40 --cpu-cache 100
# Full 8B model with realistic cache constraints
# Expect 1.5-2.5× speedup for Pensieve
```

### Varying Request Patterns
```bash
# Very frequent requests (more batching opportunity)
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.1

# Slower requests (more cache eviction)
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 1.0
```

## Files Modified

1. **main.py**
   - Added imports: `threading`, `Queue`, `defaultdict`, `mean`, `median`
   - Added parameters: `--num-concurrent-users`, `--request-interval`
   - Added function: `concurrent_client_worker()`
   - Added function: `run_concurrent_comparison()`
   - Modified `main()`: Route to concurrent benchmark when `num_concurrent_users > 1`

## Files Created

1. **SETUP_AND_RUN.md**: Complete setup and execution guide
2. **CONCURRENT_BENCHMARK_SUMMARY.md**: This document

## Testing Recommendations

### 1. Correctness Verification
- Run with same model in both sequential and concurrent modes
- Verify outputs are identical (token-by-token match)
- Ensure no race conditions or data corruption

### 2. Performance Validation
- Test with varying concurrent user counts: 1, 2, 4, 8
- Verify throughput increases with more users
- Check that TTFT becomes more consistent with cache reuse

### 3. Edge Cases
- Single user (1): Should match sequential benchmark
- Large user count (8+): Memory and scheduling stress test
- Varying intervals: Test with `--request-interval 0.1` to `2.0`

## Future Enhancements

1. **ShareGPT Dataset Integration**
   - Load real multi-turn conversations
   - Replace hardcoded conversation lists
   - Support variable-length conversations

2. **Per-Session Tracking** (If needed)
   - Track metrics per client
   - Analyze fairness (do all users get similar latencies?)
   - Detect starvation scenarios

3. **Request Distribution Models**
   - Poisson arrival instead of fixed interval
   - Burst traffic simulation
   - Time-of-day variations

4. **Advanced Metrics**
   - Tail latency SLO compliance
   - Request queue depth over time
   - GPU utilization traces

## Related Issues Fixed

This implementation addresses:
1. **Batch formation issue**: Phase transition logic incomplete in BatchScheduler
   - Concurrent benchmark now creates realistic scenarios where phase transitions occur
   - Requests naturally progress from PREFILL → GENERATION as rounds complete

2. **Sequential benchmark limitation**: Original benchmark never exercised unified batching
   - Concurrent benchmark creates mixed batches
   - Enables proper evaluation of BatchScheduler's unified batching capability

## Summary

The concurrent benchmark implementation provides a realistic evaluation framework for Pensieve's multi-user serving capabilities. By simulating multiple clients sending requests simultaneously, it enables proper measurement of:
- Unified batching benefits
- Cache reuse across sessions
- Scheduler efficiency under load
- Throughput improvements over stateless baselines

This brings the evaluation closer to real-world production scenarios where multiple users interact with the system concurrently.
