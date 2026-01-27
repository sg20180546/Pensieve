# Layer-Wise Chunking & Intelligent Cost Model

## Overview

This document explains the improved eviction policy that addresses two key insights:

1. **Layer-Wise Pipelining**: Different layers have different costs due to pipelined execution
2. **Session-Relative Positioning**: A chunk's eviction priority depends on its position relative to session length

---

## Problem Statement

### Original Implementation Issues

The initial cost model was oversimplified:

```python
# Old (incorrect)
cost = alpha * context_length + beta  # Single formula for all layers
retention_value = cost / time_inactive  # Doesn't account for layer depth
```

**Problems:**
1. ❌ All 40 layers treated equally (Layer 0 is on critical path!)
2. ❌ Session-relative position ignored (Chunk 0 vs Chunk 99 in 100-token session treated the same)
3. ❌ Leading tokens get same eviction priority as trailing tokens
4. ❌ Doesn't exploit pipelining assumption from paper (§4.3.2)

---

## Solution: Checkerboard Architecture

### Data Structure

Instead of storing all layers in one chunk:

```
BEFORE (monolithic):
  KVChunk(session="s1", chunk_id=0, layer_kv_tensors={0:(K0,V0), 1:(K1,V1), ..., 39:(K39,V39)})
  → One object per position

AFTER (layer-wise):
  KVChunk(session="s1", chunk_id=0, layer_idx=0, key_tensor=K0, value_tensor=V0)
  KVChunk(session="s1", chunk_id=0, layer_idx=1, key_tensor=K1, value_tensor=V1)
  ...
  KVChunk(session="s1", chunk_id=0, layer_idx=39, key_tensor=K39, value_tensor=V39)
  → 40 objects per position (one per layer)
```

**Visualization (Checkerboard):**

```
         Layer0  Layer1  Layer2  ... Layer39
Sess1:    ■       ■       ■             ■
Sess2:    ▓       ▓       ▓             ▓    (CPU)
Sess3:    ░       ░       ░             ░    (DROPPED)

■ = GPU (hot cache)
▓ = CPU (warm cache)
░ = DROPPED (evicted, needs recovery)
```

**Benefits:**
- ✅ Can evict individual layer chunks
- ✅ Enables layer-specific cost modeling
- ✅ Exploits pipelined execution
- ✅ Fine-grained memory control

---

## Cost Model with Layer & Position Weights

### Formula

```
retention_value = Cost(layer, position, context_length) / time_inactive

where:
  Cost = layer_weight × position_weight × base_cost

  base_cost = (alpha × context_length + beta) + const_non_attention

  layer_weight = (num_layers - layer_idx) / num_layers

  position_weight = (chunk_id + 1) / session_total_chunks
```

### Layer Weight Explanation

**Motivation**: Layer-wise pipelining from paper (§4.3.2)

```
Timeline of execution:

Layer 0:  [████ Compute]
Layer 1:        [████ Compute]    ← Pipelined (overlaps with Layer 0)
Layer 2:              [████ Compute]
...
Layer 39:                          [████ Compute]

Critical path = Layer 0 latency (others are hidden by pipelining)
```

**Formula:**
```
layer_weight(layer_idx) = (num_layers - layer_idx) / num_layers

Layer 0:  weight = 40/40 = 1.00  (100% of cost on critical path)
Layer 19: weight = 21/40 = 0.525  (52.5% cost)
Layer 39: weight =  1/40 = 0.025  (2.5% cost, fully pipelined)
```

**Implication:**
- Evicting Layer 0 chunk is **40× more costly** than Layer 39 chunk
- Evict from back layers first (Layer39 → Layer0)

### Position Weight Explanation

**Motivation**: Session-relative efficiency of recomputation

```
Session with 100 tokens split into 4 chunks:

Chunk 0: 0-31 tokens     (context_length=0)   Position_weight = 1/100 = 0.01
Chunk 1: 32-63 tokens    (context_length=32)  Position_weight = 2/100 = 0.02
Chunk 2: 64-95 tokens    (context_length=64)  Position_weight = 3/100 = 0.03
Chunk 3: 96-99 tokens    (context_length=96)  Position_weight = 4/100 = 0.04
```

**Why leading chunks have lower cost:**
- Chunk 0: To recompute, run attention with 0 prior tokens → **fast**
- Chunk 3: To recompute, run attention with 96 prior tokens → **slow**
- Attention is O(context_length) → leading tokens exponentially cheaper

**Formula:**
```
position_weight(chunk_id, session_length) = (chunk_id + 1) / session_total_chunks

Range: 0.01 to 1.0
Impact: 100× difference between leading and trailing chunks
```

**Implication:**
- Evict leading chunks first (Chunk0 → Chunk_N)
- Preserve trailing chunks (more expensive to recompute)

### Combined Effect: "Checkerboard" Eviction Order

**Eviction priority (lowest → highest):**

```
1. [Least expensive] Layer 39, Chunk 0 (back layer, leading token)
2. Layer 39, Chunk 1
3. ...
4. Layer 39, Chunk N
5. Layer 38, Chunk 0
6. ...
7. [Most expensive] Layer 0, Chunk N (critical path, trailing token)
```

**Visual eviction order:**

```
Iteration 1: Evict Layer39,Chunk0  ░■■■
Iteration 2: Evict Layer39,Chunk1  ░░■■
Iteration 3: Evict Layer39,Chunk2  ░░░■
Iteration 4: Evict Layer39,Chunk3  ░░░░
Iteration 5: Evict Layer38,Chunk0  ■░░░
...                               (▼ eviction proceeds backward)
Last: Evict Layer0,Chunk3         ░░░░
```

---

## LRU Component

The time-based component (LRU) is preserved:

```
retention_value = cost / time_inactive
```

**How it works:**
- Chunks accessed **recently** have small `time_inactive` → high retention_value → preserved
- Chunks accessed **long ago** have large `time_inactive` → low retention_value → evicted first
- This ensures active sessions are preserved over idle ones

**Combined with cost:**
- Within similar recency, evict by cost (layer & position)
- Between different recency, evict the older session

---

## Implementation Details

### KVChunk Fields (New)

```python
@dataclass
class KVChunk:
    session_id: str                  # "session_1"
    chunk_id: int                    # Position in session (0, 1, 2, ...)
    layer_idx: int                   # Which layer (0 to 39)
    key_tensor: torch.Tensor         # K for this layer
    value_tensor: torch.Tensor       # V for this layer
    context_length: int              # Tokens before this chunk (same for all layers)
    session_total_chunks: int        # Total chunks in session (for position weight)
    num_layers: int                  # Total layers in model (for layer weight)
    last_accessed: float             # For LRU component
    location: CacheLocation          # GPU/CPU/DROPPED
```

### Cost Calculation

```python
def calculate_cost(self, chunk: KVChunk) -> float:
    # Base cost (attention + non-attention)
    cost_attention = alpha * context_length + beta
    cost_non_attention = const_non_attention
    base_cost = cost_attention + cost_non_attention

    # Layer weight (pipelining)
    layer_weight = (num_layers - layer_idx) / num_layers

    # Position weight (session-relative)
    position_weight = (chunk_id + 1) / session_total_chunks

    # Combined
    return layer_weight * position_weight * base_cost
```

### Retention Value (with LRU)

```python
def calculate_retention_value(self, chunk: KVChunk) -> float:
    cost = calculate_cost(chunk)
    time_inactive = time.time() - last_accessed
    return cost / time_inactive
```

---

## Storage & Retrieval

### Storing KV from Forward Pass

When model finishes generating a chunk, we store all layers:

```python
# After model forward pass, we have:
layer_kv_dict = {0: (K0, V0), 1: (K1, V1), ..., 39: (K39, V39)}

# Store as 40 separate chunks
cache.store_chunks_for_position(
    session_id="s1",
    chunk_id=0,
    layer_kv_dict=layer_kv_dict,
    context_length=0,
    session_total_chunks=4,  # We'll have 4 chunks total
    num_layers=40,
)
```

### Retrieving KV for Attention

When running attention for a layer, gather all position's layers:

```python
# CustomCache.__getitem__(layer_idx=5)
# Returns: K and V tensors concatenated from all positions

all_keys = []
for position in [0, 1, 2, ...]:
    chunk = cache.get_chunk(session_id="s1", chunk_id=position, layer_idx=5)
    all_keys.append(chunk.key_tensor)

return torch.cat(all_keys, dim=1)  # Full context for layer 5
```

---

## Example Scenario

### Setup

```
Session: "user_1"
Model: 2 layers (for simplicity)
Conversation: 64 tokens total (2 chunks of 32)
Available GPU: 1 chunk

Chunk 0: tokens 0-31   (context_length=0)
Chunk 1: tokens 32-63  (context_length=32)
```

### KVChunks Created

After first inference:
- KVChunk(s1, chunk_0, layer_0, context_length=0, session_total_chunks=2, num_layers=2)
- KVChunk(s1, chunk_0, layer_1, context_length=0, session_total_chunks=2, num_layers=2)
- KVChunk(s1, chunk_1, layer_0, context_length=32, session_total_chunks=2, num_layers=2)
- KVChunk(s1, chunk_1, layer_1, context_length=32, session_total_chunks=2, num_layers=2)

All 4 chunks fit in GPU.

### Memory Pressure: Need to Evict 1 Chunk

Calculate costs:

```
(s1, chunk_0, layer_0):
  base_cost = 0.001 * 0 + 0.01 + 0.005 = 0.015
  layer_weight = (2 - 0) / 2 = 1.0 (critical path)
  position_weight = (0 + 1) / 2 = 0.5 (leading)
  cost = 1.0 * 0.5 * 0.015 = 0.0075

(s1, chunk_0, layer_1):
  base_cost = 0.015
  layer_weight = (2 - 1) / 2 = 0.5 (pipelined)
  position_weight = 0.5 (leading)
  cost = 0.5 * 0.5 * 0.015 = 0.00375  ← LOWEST COST, EVICT FIRST

(s1, chunk_1, layer_0):
  base_cost = 0.001 * 32 + 0.01 + 0.005 = 0.047
  layer_weight = 1.0
  position_weight = (1 + 1) / 2 = 1.0 (trailing)
  cost = 1.0 * 1.0 * 0.047 = 0.047

(s1, chunk_1, layer_1):
  base_cost = 0.047
  layer_weight = 0.5
  position_weight = 1.0
  cost = 0.5 * 1.0 * 0.047 = 0.0235  ← SECOND LOWEST
```

**Eviction order:** Layer1,Chunk0 → Layer1,Chunk1 → Layer0,Chunk1 → Layer0,Chunk0

**Interpretation:**
- ✅ Evict back layer (Layer1) before front layer (Layer0)
- ✅ Evict leading chunks before trailing chunks
- ✅ Preserve Layer0 (critical path) as long as possible
- ✅ Preserve trailing chunks (expensive to recompute)

---

## Integration with HuggingFace

### Before (All Layers in One Chunk)

```python
# KVChunk contains all layers
chunk.layer_kv_tensors = {0: (K0, V0), 1: (K1, V1), ..., 39: (K39, V39)}

# CustomCache.__getitem__(layer_idx) just extracts one layer
def __getitem__(self, layer_idx):
    return chunk.layer_kv_tensors[layer_idx]
```

### After (One Layer per Chunk)

```python
# Each KVChunk is single layer
chunk = KVChunk(session="s1", chunk_id=0, layer_idx=5)
chunk.key_tensor = K5
chunk.value_tensor = V5

# CustomCache.__getitem__(layer_idx=5) gathers from multiple positions
def __getitem__(self, layer_idx):
    all_keys = []
    for chunk in cache.chunks_for_layer(layer_idx):
        all_keys.append(chunk.key_tensor)
    return torch.cat(all_keys, dim=1)
```

**HuggingFace Integration:**

```python
# During model.forward():
for layer_idx in range(num_layers):
    past_key_values = cache[layer_idx]  # Calls our __getitem__
    # → Returns concatenated KV from all positions for this layer
    output = transformer_layer(output, past_key_values)
```

---

## Performance Implications

### Eviction Order Impact

With cost model:
- **Leading tokens evicted first** → cheap to recompute when needed (§4.3.4)
- **Back layers evicted first** → critical path (Layer0) preserved
- **Older sessions evicted first** → LRU behavior preserved

### Memory Efficiency

```
Before: 1 chunk per position × 1 session
After:  40 chunks per position × multiple sessions (but layer-wise control)
```

**Trade-off:**
- ❌ More objects to manage (40×)
- ✅ Fine-grained eviction control
- ✅ Can partially preserve sessions (keep Layer0, drop Layer39)

---

## Testing the Design

### Scenario 1: Single Session, Memory Pressure

```
3 sessions, GPU space for 2 chunks total
Expected eviction order:
1. Oldest session's Layer39,Chunk0 (lowest cost)
2. Oldest session's Layer39,Chunk1
3. ... (back layer before front layer)
```

### Scenario 2: Multi-Turn Within Session

```
Session: 5 turns (5 chunks)
GPU space: 3 chunks
Expected: Keep most recent chunks, evict leading chunks of old turns
```

### Scenario 3: Verify Layer Priority

```
Session: 2 chunks, Layer0 vs Layer39
Under memory pressure, Layer39 chunks evicted first
Layer0 preserved longer
```

---

## Next Steps

1. **Test cost calculations** - Verify formula produces correct ordering
2. **Test eviction ordering** - Run with multiple sessions, verify priority
3. **Measure performance** - Compare with flat cost model
4. **Phase 4 Integration** - Implement token recovery when chunks dropped
5. **Benchmarking** - Evaluate on real workloads (ShareGPT)

---

## References

- Paper: "Stateful Large Language Model Serving with Pensieve" (§4.3)
- Section 4.3.1: Retention value policy
- Section 4.3.2: Layer-wise pipelining
- Section 4.3.4: Dropped token recovery
