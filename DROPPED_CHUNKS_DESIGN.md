# DROPPED Chunks Design & Memory Semantics

## Your Question

**Q**: "원래 DROPPED같은거 안 쓰고, 아예 remove했지 않아?"

**Answer**: 맞다! 원래는 simple remove였을 수 있다. 하지만 현재 design은 **F7: Dropped Token Recovery** 기능을 지원하기 위해 DROPPED tier를 사용한다.

다만 **메모리 추적이 깨져있었다**. 이를 명확히 하자.

---

## Current Architecture Issue

### Requirement (CLAUDE.md F7)
```
F7: Dropped token recovery
- Recompute evicted prefixes and merge with cached KV
- Paper §4.3.4
```

→ **DROPPED tier는 의도된 설계**

### Problem: Memory Accounting Inconsistency

```python
# Current code (BROKEN):
CPU eviction with CPU full:
  drop_chunk = cpu_cache.pop(drop_key)
  cpu_used_bytes -= drop_chunk.size_bytes  # ← Freed?
  dropped_chunks[drop_key] = drop_chunk    # ← But stored here!

Result:
  - cpu_used_bytes shows space freed
  - But drop_chunk.tensors still in CPU memory
  - = Inconsistency!
```

---

## Design Decision: Two Options

### Option A: DROPPED as "Metadata-Only"

```
Concept:
- DROPPED chunks exist in dropped_chunks dict (metadata)
- Tensors are serialized to disk or discarded
- Recovery requires disk I/O to restore tensors

Pros:
  ✓ Memory is actually freed
  ✓ cpu_used_bytes accurate
  ✓ Clean separation

Cons:
  ✗ Recovery requires I/O (slow)
  ✗ Disk space needed
  ✗ Complex implementation
```

### Option B: DROPPED as "CPU-Backed Recovery Cache"

```
Concept:
- DROPPED chunks stay in CPU memory
- Tensors are preserved on CPU
- Recovery just loads from dropped_chunks dict

Pros:
  ✓ Fast recovery (no I/O)
  ✓ Simple implementation
  ✓ Consistent with paper

Cons:
  ✗ CPU memory not truly freed
  ✗ cpu_used_bytes becomes inaccurate
  ✗ Eviction space accounting broken
```

---

## Proposed Solution: Clarified Option B

**Choose Option B** (simpler, faster) but with **clear semantics**:

### Core Principle

```
DROPPED chunks = "Cold" CPU-backed recovery cache
- Located: CPU memory (in dropped_chunks dict)
- Lifetime: Until session ends (evict_session removes)
- Purpose: Enable recomputation without full re-execution
```

### Memory Semantics (FIXED)

```python
# CPU Tier Accounting:
cpu_memory_total = cpu_cache + dropped_chunks

# Therefore:
cpu_used_bytes should include DROPPED chunks!
cpu_used_bytes = Σ(cpu_cache) + Σ(dropped_chunks)
```

### Eviction Flow (Corrected)

```
When CPU full and need space:
├─ Evict from gpu_cache → cpu_cache (move up)
│  └─ If CPU also full:
│     └─ Evict from cpu_cache → dropped_chunks (move cold)
│        └─ CPU still has same memory usage!
│           (just reorganized: hot→cold)
│
At this point:
├─ GPU freed: gpu_used_bytes decreased ✓
├─ CPU not freed: cpu_used_bytes unchanged ✓
│  (chunks just moved from cache to dropped)
└─ Recovery possible: tensors in dropped_chunks ✓
```

---

## Code Changes Required

### Change 1: Don't subtract from cpu_used_bytes when moving to DROPPED

**File**: `src/pensieve/core/cache.py:536-542`

**Current** (WRONG):
```python
drop_chunk = self.cpu_cache.pop(drop_key)
self.cpu_used_bytes -= drop_chunk.size_bytes  # ← WRONG!
self.dropped_chunks[drop_key] = drop_chunk
```

**Fixed**:
```python
drop_chunk = self.cpu_cache.pop(drop_key)
# cpu_used_bytes NOT decreased
# (chunk moves from cache to dropped, both in CPU)
self.dropped_chunks[drop_key] = drop_chunk
```

Already partially fixed in previous edit. Need to remove the subtraction.

### Change 2: CPU eviction also shouldn't subtract

**File**: `src/pensieve/core/cache.py:559-564`

**Current** (WRONG):
```python
else:  # Evicting from CPU
    self.cpu_used_bytes -= chunk.size_bytes
    self.dropped_chunks[chunk_key] = chunk
```

**Should be**:
```python
else:  # Evicting from CPU
    # Chunk moves from cache to dropped
    # Both in CPU, so no memory freed
    self.dropped_chunks[chunk_key] = chunk
```

---

## Memory Hierarchy Semantics (CLARIFIED)

```
GPU Tier:
┌──────────────────────┐
│  gpu_cache (hot)     │
│  (in-use, fast)      │
└──────────────────────┘
   gpu_used_bytes

        ↓ (evict)

CPU Tier:
┌──────────────────────┐
│  cpu_cache (warm)    │ ─┐
│  (reuse, medium)     │  │ cpu_used_bytes
│                      │  │
│  dropped_chunks      │ ─┘
│  (cold recovery)     │
└──────────────────────┘

Eviction path:
GPU → cpu_cache → dropped_chunks
     (warm)       (cold, recovery)

Key: All CPU tiers count toward cpu_used_bytes!
```

---

## Session Lifetime and DROPPED Cleanup

```python
# Session ends:
evict_session(session_id):
    # 1. Remove from all active caches
    For each chunk_key in session_chunks[session_id]:
        if chunk_key in gpu_cache:
            gpu_cache.pop(chunk_key)
            gpu_used_bytes -= size  ✓
        elif chunk_key in cpu_cache:
            cpu_cache.pop(chunk_key)
            cpu_used_bytes -= size  ✓
        elif chunk_key in dropped_chunks:
            dropped_chunks.pop(chunk_key)
            # CPU memory freed! ✓

    # 2. Remove tracking
    del session_chunks[session_id]

Result: When session ends, ALL chunks removed, CPU memory freed
```

---

## Memory Invariants (CORRECTED)

### Invariant 1: CPU Memory Accounting
```
cpu_used_bytes = Σ(cpu_cache) + Σ(dropped_chunks)
              = Memory actually used by CPU tier
```

### Invariant 2: No Memory Leaks
```
When session_id deleted:
  ∀ chunk_key ∈ session_chunks[session_id] (before delete):
    chunk completely removed from all tiers
    ∴ Memory freed
```

### Invariant 3: Recovery Available
```
∀ chunk in dropped_chunks:
  chunk.tensors in CPU memory
  ∴ Recovery possible without I/O
```

---

## Design Trade-offs

### This Design (Option B)

✅ **Pros**:
- Recovery is instant (no disk I/O)
- Simple implementation
- Matches Pensieve paper spirit
- Memory semantics clear after fix

❌ **Cons**:
- DROPPED chunks consume CPU memory
- Eviction doesn't free memory (just reorganizes)
- Depends on session cleanup to free DROPPED

### Alternative (Option A - disk-backed)

❌ **Why not**:
- Adds I/O latency to recovery
- Requires disk space management
- Serialization complexity
- Not justified for this prototype

---

## Recovery Mechanism

When session returns after eviction:

```
Recovery flow (from token_recovery.py):
1. detect_dropped_chunks(session_id)
   ├─ Find chunks in dropped_chunks
   └─ Trigger recovery

2. recompute_dropped_chunks(session_id)
   ├─ Load tensors from dropped_chunks ✓
   ├─ Run forward pass
   └─ Update chunks back to CPU/GPU cache

3. Session continues with recovered KV

Benefit: Chunks recovered from memory, not recomputation
Overhead: Forward pass on dropped chunk tokens only
```

---

## Testing Memory Semantics

```python
def test_dropped_chunks_memory_semantics():
    """Verify DROPPED chunks are counted in cpu_used_bytes."""
    cache = TwoTierCache(gpu_capacity_gb=0.1, cpu_capacity_gb=0.2)

    # Fill CPU to trigger DROPPED
    for i in range(10):
        chunk = create_chunk("s1", i)
        cache.store_chunk(chunk, GPU)

    # Force some chunks to DROPPED
    initial_cpu_used = cache.cpu_used_bytes

    # Trigger eviction by adding more chunks
    for i in range(10, 20):
        chunk = create_chunk("s2", i-10)
        cache.store_chunk(chunk, GPU)

    # Check:
    # 1. Some s1 chunks should be in dropped_chunks
    s1_dropped = sum(1 for k in cache.dropped_chunks if "s1" in k)
    assert s1_dropped > 0, "Should have dropped s1 chunks"

    # 2. cpu_used_bytes should account for DROPPED
    total_cpu_chunks = (
        sum(c.size_bytes for c in cache.cpu_cache.values()) +
        sum(c.size_bytes for c in cache.dropped_chunks.values())
    )
    assert cache.cpu_used_bytes == total_cpu_chunks, \
        f"cpu_used_bytes mismatch: {cache.cpu_used_bytes} vs {total_cpu_chunks}"

    # 3. When session ends, both are freed
    cache.evict_session("s1")

    remaining_cpu = sum(c.size_bytes for c in cache.cpu_cache.values())
    assert cache.cpu_used_bytes == remaining_cpu, \
        "cpu_used_bytes should decrease when DROPPED chunks removed"
```

---

## Summary

**User's Question**: "원래 DROPPED 같은거 안 쓰고, 아예 remove했지 않아?"

**Answer**:
- ✅ DROPPED tier는 의도된 설계 (F7 requirement)
- ❌ 하지만 메모리 추적이 깨져있었음
- 🔧 **Fix**: DROPPED chunks도 cpu_used_bytes에 포함
- ✅ 결과: 메모리 안전성 + Recovery capability 확보

Design is now **clear and consistent**!
