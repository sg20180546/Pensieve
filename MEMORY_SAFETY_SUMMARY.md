# Memory Safety Fix Summary

## User's Critical Question
User asked: **"음 근데 이거는 recovery할때부를게아니라, 최초 생성시에도 불러야하지않아?"**

Translation: *"But wait, this shouldn't just be called during recovery, shouldn't it also be called at initial creation?"*

The user correctly identified that the cleanup logic in `store_chunk()` must apply to **BOTH** code paths:
1. **Recovery path**: `token_recovery.py:_store_recovered_chunks()`
2. **Initial generation path**: `worker.py:_store_new_kv_chunks()`

## The Problem

The original `store_chunk()` only checked the target cache (GPU or CPU) for duplicates:

```python
# OLD CODE - BUGGY!
if chunk_key in target_cache:  # Only checks GPU or CPU, not both!
    # cleanup...
```

This created a **cross-tier memory leak**:

### Problematic Scenario

```
Timeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=1: Initial generation creates chunk_0
     store_chunk(chunk_0, location=GPU)
     ↓
     GPU cache: {chunk_0}
     CPU cache: {}

t=2: GPU fills up, chunk_0 evicted to CPU
     swap_chunk_to_cpu(chunk_0)
     ↓
     GPU cache: {}              ← REMOVED from GPU
     CPU cache: {chunk_0}       ← ADDED to CPU

t=3: Recovery recreates chunk_0
     store_chunk(chunk_0_new, location=GPU)
     ↓
     Check: is chunk_key in gpu_cache? → FALSE
     (It's in CPU now, not GPU!)
     ↓
     GPU cache: {chunk_0_new}   ← NEW version added
     CPU cache: {chunk_0}       ← OLD version still here!

     ❌ MEMORY LEAK: chunk_0 exists in TWO places!
```

This violates memory invariants:
- Memory counters become inaccurate
- Queries might find stale versions
- Cache efficiency degrades
- Memory pressure management breaks

## The Solution

Updated `store_chunk()` to check **ALL caches** before storing:

```python
# NEW CODE - FIXED!
for cache_dict, cache_location in [
    (self.gpu_cache, CacheLocation.GPU),
    (self.cpu_cache, CacheLocation.CPU),
    (self.dropped_chunks, CacheLocation.DROPPED),
]:
    if chunk_key in cache_dict:
        old_chunk = cache_dict[chunk_key]
        freed_bytes = old_chunk.size_bytes

        # Update memory tracking
        if cache_location == CacheLocation.GPU:
            self.gpu_used_bytes -= freed_bytes
        elif cache_location == CacheLocation.CPU:
            self.cpu_used_bytes -= freed_bytes

        # Remove old version from wherever it is
        del cache_dict[chunk_key]
```

Now the scenario works correctly:

```
t=3: Recovery recreates chunk_0 (FIXED)
     store_chunk(chunk_0_new, location=GPU)
     ↓
     Check ALL caches:
     ✓ Check gpu_cache: not found
     ✓ Check cpu_cache: FOUND! chunk_0
     ✓ Free memory: cpu_used_bytes -= chunk_0.size
     ✓ Remove: del cpu_cache[chunk_0]
     ↓
     GPU cache: {chunk_0_new}   ← NEW version
     CPU cache: {}              ← OLD version cleaned up

     ✅ CORRECT: chunk_0 exists in exactly ONE place!
```

## Why Both Paths Needed This Fix

The fix handles **both** code paths because they both call `store_chunk()`:

### Path 1: Recovery (token_recovery.py)
```python
# Line ~340
def _store_recovered_chunks(self, ...):
    for chunk in recovered_chunks:
        self.cache.store_chunk(chunk, location=CacheLocation.GPU)
        # ↑ Might replace chunk that was evicted to CPU
```

### Path 2: Initial Generation (worker.py)
```python
# Line 534
def _store_new_kv_chunks(self, batch, past_key_values):
    for layer_idx, (k, v) in enumerate(past_key_values):
        chunk = KVChunk(...)
        self.cache.store_chunk(chunk, location=CacheLocation.GPU)
        # ↑ Might replace stale chunk from previous generation
```

Both paths need the multi-tier cleanup because chunks can exist in any tier due to previous evictions.

## Guarantees After Fix

### 1. No Cross-Tier Duplicates
```
Invariant: ∀ chunk_key, count in {GPU, CPU, DROPPED} ≤ 1
```
A chunk exists in **at most ONE** cache tier at any time.

### 2. Memory Tracking is Accurate
```
gpu_used_bytes = Σ(size of chunks in gpu_cache)
cpu_used_bytes = Σ(size of chunks in cpu_cache)
```
Memory counters always reflect actual stored chunks.

### 3. session_chunks is Consistent
```
∀ chunk_key in session_chunks[session_id],
  chunk_key exists in some cache (GPU/CPU/DROPPED)
```
Tracking list never has duplicates or stale entries.

### 4. Both Recovery and Generation Work Safely
```
Recovery path: Old versions properly cleaned up
Initial gen path: New chunks don't create duplicates
Combined paths: No conflicts or memory leaks
```

## Changes Made

**File**: `/Users/sj/pensieve/src/pensieve/core/cache.py`

**Function**: `store_chunk()` (lines 106-193)

**Changes**:
- Lines 143-167: Added multi-tier cleanup loop
- Updated docstring to clarify both paths are safe
- Added explanatory comments

## Testing

New test file: `/Users/sj/pensieve/scripts/test_memory_safety.py`

Tests verify:
1. ✅ Cross-tier cleanup during recovery
2. ✅ Same-tier replacement works
3. ✅ Mixed recovery + generation is safe
4. ✅ session_chunks tracking stays consistent

Run tests:
```bash
python scripts/test_memory_safety.py
```

## Code Flow After Fix

### Recovery Path
```
token_recovery.py:recover_batch()
├─ For each recovered chunk
└─ store_chunk(recovered_chunk, GPU)
   ├─ Check gpu_cache: found/not found
   ├─ Check cpu_cache: found/not found
   ├─ Check dropped_chunks: found/not found
   ├─ Clean up old version (free memory, remove from cache)
   └─ Store new version
```

### Initial Generation Path
```
worker.py:_store_new_kv_chunks()
├─ For each generated chunk
└─ store_chunk(new_chunk, GPU)
   ├─ Check gpu_cache: found/not found
   ├─ Check cpu_cache: found/not found
   ├─ Check dropped_chunks: found/not found
   ├─ Clean up old version (shouldn't exist normally)
   └─ Store new version
```

Both paths use the **same safe logic**.

## Impact on System Correctness

This fix ensures:
- ✅ Recovery works correctly (no memory leaks)
- ✅ Generation works correctly (no duplicates)
- ✅ Cache coherency maintained across tiers
- ✅ Memory budgets respected
- ✅ Eviction policy functions correctly
- ✅ Multi-turn conversations work safely

## Related Files

- **Implementation**: `src/pensieve/core/cache.py` (store_chunk method)
- **Recovery caller**: `src/pensieve/recovery/token_recovery.py` (_store_recovered_chunks)
- **Generation caller**: `src/pensieve/worker/worker.py` (_store_new_kv_chunks)
- **Tests**: `scripts/test_memory_safety.py`
- **Documentation**: This file + MEMORY_SAFETY_FIX.md

## Summary

The user's question identified a critical gap: the cleanup logic needed to apply to **both** recovery and generation paths. The fix ensures that `store_chunk()` checks all cache tiers and prevents cross-tier duplicates, making the system safe and correct for both scenarios.

**Key insight**: Since both paths call the same `store_chunk()` function, the memory safety logic only needs to be implemented once, at that central point.
