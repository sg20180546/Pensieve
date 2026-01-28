# Hotness-Aware Concurrent Benchmark

## Overview

The concurrent benchmark now automatically creates **realistic client hotness distribution** to properly evaluate Pensieve's cache eviction and recovery mechanisms.

## Problem This Solves

**Original Issue**:
- Sequential benchmark: Single user → no eviction, no realistic cache pressure
- Old concurrent benchmark: All clients had same request frequency → uniform hotness
- Result: Eviction policy advantages not visible

**Solution**:
- New concurrent benchmark: Automatic 3-tier hotness distribution
- HOT, WARM, and COLD clients created automatically
- Shows cache eviction vs reuse benefits realistically

## How It Works

### Automatic Hotness Distribution

When running with `--num-concurrent-users N`:

```python
HOT clients (N/3):    interval = 0.3 × base_interval
WARM clients (N/3):   interval = 1.0 × base_interval
COLD clients (N/3):   interval = 2.5 × base_interval
```

**Example with 6 users and `--request-interval 0.5`**:

```
Client 0 (HOT):   requests every 0.15s  → GPU cache
Client 1 (HOT):   requests every 0.15s  → GPU cache
Client 2 (WARM):  requests every 0.50s  → GPU/CPU mixed
Client 3 (WARM):  requests every 0.50s  → GPU/CPU mixed
Client 4 (COLD):  requests every 1.25s  → Eviction/Recovery
Client 5 (COLD):  requests every 1.25s  → Eviction/Recovery
```

### Cache State Evolution

**Timeline (simplified)**:

```
t=0-1s:   GPU cache fills with recent history from all clients

t=1-2s:   HOT clients keep requesting → stay in GPU
          WARM clients partially evicted to CPU
          COLD clients make 2nd request → may find cache evicted

t=2-3s:   Clear 3-tier separation visible:
          - GPU: Full history from HOT (high cache hit)
          - CPU: Partial history from WARM (medium cache hit)
          - DROPPED: Old chunks from COLD (recovery needed)

t=3+:     Steady state where eviction policy balances competing demands
```

## Key Insights from Results

### 1. Performance Distribution (Expected)

| Client Type | Expected Speedup | Why |
|---|---|---|
| HOT | 1.8-2.5× | High cache hit rate, minimal eviction |
| WARM | 1.3-1.8× | Mixed hits/misses, some CPU swap |
| COLD | 1.1-1.5× | Recovery overhead, but still beats recompute |

### 2. What Retention Value Policy Achieves

```
Cost = Cost(context_length) / time_since_access

HOT clients:
  - Low time_since_access → High retention value → Kept in GPU

COLD clients:
  - High time_since_access → Low retention value → Evicted

Within session:
  - Leading tokens (low context_length) → Evicted first → Lower recovery cost
  - Later tokens (high context_length) → Kept longer → Avoid high recovery cost
```

### 3. Unified Batching Benefits

With concurrent requests:
```
Batch composition over time:

t=0.2s: [HOT_prefill, HOT_prefill, WARM_prefill, WARM_prefill, COLD_prefill, COLD_prefill]
        → 6 PREFILL requests

t=0.35s: [HOT_gen_turn2, HOT_gen_turn2, WARM_prefill, WARM_prefill, COLD_prefill, COLD_prefill]
         → Unified batch (2 GEN + 4 PREFILL)

t=0.5s: [HOT_gen, HOT_gen, WARM_gen, WARM_gen, COLD_prefill, COLD_prefill]
        → Unified batch (4 GEN + 2 PREFILL)
```

This is where Pensieve's BatchScheduler (unlike simple vLLM) shows benefits:
- Better GPU utilization by mixing PREFILL and GENERATION
- Adaptive batching based on request arrivals

## Comparing Different Scenarios

### Scenario 1: Very Fast Requests (0.1s interval)

```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.1
```

**Expected**:
- Less eviction (requests arrive faster than eviction happens)
- Batching benefit dominant
- Speedup: ~1.3-1.8× (batching > cache reuse)

### Scenario 2: Medium Requests (0.5s interval) - RECOMMENDED

```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.5
```

**Expected**:
- Clear 3-tier cache state visible
- Both batching AND cache reuse benefits
- Eviction policy clearly demonstrates intelligence
- Speedup: ~1.8-2.5× (batching + cache reuse)

### Scenario 3: Slow Requests (1.0s interval)

```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 1.0
```

**Expected**:
- Heavy eviction and recovery
- Recovery overhead visible (high P99 latency for COLD clients)
- Still faster than vLLM (recovery cost < recompute cost)
- Speedup: ~1.5-2.2× (recovery overhead reduces benefit)

## Metrics Interpretation

### Throughput (Most Important)

- Higher throughput = better batching + cache reuse
- vLLM: Uniform, no cache benefit
- Pensieve: Progressive improvement as cache warms up

### TTFT (Time To First Token)

```
Avg TTFT: ~= (HOT_ttft + WARM_ttft + COLD_ttft) / 3

Expected breakdown:
  - HOT TTFT: 10-20ms (cache hits)
  - WARM TTFT: 20-30ms (mixed)
  - COLD TTFT: 30-50ms (recovery/swap overhead)
```

### P99 TTFT vs Avg TTFT

```
If P99_ttft >> Avg_ttft:
  → Suggests COLD clients experiencing heavy recovery
  → Normal - shows system is evicting appropriately

If P99_ttft ≈ Avg_ttft:
  → Uniform performance
  → May indicate all clients have cache (no eviction needed)
```

### Tail Latency

- Pensieve: Decreases with turns (cache benefit)
- vLLM: Increases with turns (more history to recompute)

## Running the Full Evaluation

### Quick Test (2-3 minutes)

```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 3 --request-interval 0.5
```

### Standard Evaluation (5-10 minutes)

```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.5
```

### Full Evaluation with Llama (10-20 minutes)

```bash
python main.py --mode compare --model meta-llama/Meta-Llama-3-8B \
  --num-concurrent-users 6 --request-interval 0.5 \
  --gpu-cache 40 --cpu-cache 100
```

## What You Should Observe

### If Everything Works Correctly

1. **Output shows hotness distribution**:
   ```
   Client Hotness Distribution:
     • HOT clients (1/3): Request every 0.15s → GPU cache hit
     • WARM clients (1/3): Request every 0.50s → GPU/CPU mixed
     • COLD clients (1/3): Request every 1.25s → Eviction/Recovery
   ```

2. **Pensieve faster than vLLM**:
   - Time speedup > 1.0
   - Throughput speedup > 1.0
   - TTFT speedup > 1.0

3. **Metrics show 3-tier distribution**:
   - Avg TTFT ≤ P99 TTFT (some clients slower)
   - Speedup increases with turns (cache warming effect)

4. **Performance interpretation shown**:
   ```
   Hotness-Based Analysis:
     • HOT clients (1/3): Request every 0.15s → GPU cache hit
     • WARM clients (1/3): Request every 0.50s → GPU/CPU mixed
     • COLD clients (1/3): Request every 1.25s → Eviction/Recovery
   ```

### If Something's Wrong

1. **vLLM faster than Pensieve**:
   - Check: GPU cache is large enough (--gpu-cache 40)
   - Check: Model isn't too large for batch
   - Likely: Scheduling/swap overhead > benefit (try fewer users)

2. **P99 TTFT much larger than Avg**:
   - Expected: Shows recovery overhead
   - Abnormal: If difference > 2×, check system stability

3. **Uniform performance (no 3-tier effect)**:
   - Likely: Not enough requests to trigger eviction
   - Try: `--num-concurrent-users 9` or `--request-interval 0.3`

## Related Paper Sections

This benchmark demonstrates concepts from:
- **Paper §3.1**: Multi-turn conversation with KV cache reuse
- **Paper §4.2**: Unified iteration-level batching
- **Paper §4.3**: Two-tier GPU/CPU cache with retention value
- **Paper §4.3.1**: Leading-token eviction preference
- **Paper §4.3.4**: Dropped token recovery

## Future Enhancements

1. **Per-client metrics**: Track performance per hotness level
2. **Arrival distribution**: Poisson/Zipfian instead of fixed intervals
3. **ShareGPT dataset**: Real multi-turn conversations with realistic hotness
4. **Detailed timeline**: Visualize cache state evolution over time
5. **Memory profiling**: Track GPU/CPU usage throughout benchmark

## Summary

The hotness-aware concurrent benchmark properly demonstrates Pensieve's key innovations:

1. **Smart eviction**: Retention value intelligently evicts infrequent clients
2. **Cache reuse**: Even with recovery overhead, saves time vs recomputation
3. **Unified batching**: Mixes PREFILL+GENERATION for better utilization
4. **Realistic scenario**: Multiple clients with different access patterns

Run it with:
```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.5
```

Expected result: **1.5-2.5× speedup** for Pensieve over vLLM.
