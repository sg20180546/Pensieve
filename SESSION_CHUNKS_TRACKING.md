# session_chunks Tracking Design

## Your Question Analyzed

**Q**: "GPU에서 CPU로 안 옮기고 바로 DROPPED로 날릴 때는 어떻게 되는거야?"

**Answer**: 좋은 지적! 이건 더 큰 설계 문제였다.

---

## Problem Found: Orphaned Chunks

### Scenario: GPU → DROPPED Direct Move

```
Before (BUGGY):
┌─────────────────────────────────────────┐
│ GPU eviction with full CPU               │
├─────────────────────────────────────────┤
│ 1. GPU에서 pop: chunk_key            │
│ 2. session_chunks에서 제거: ← ERROR!   │
│    └─ chunk은 DROPPED tier로 갈 예정  │
│ 3. GPU→DROPPED move                     │
│    └─ session_chunks에는 없음!         │
│                                          │
│ Result: chunk은 dropped_chunks에 있지만 │
│         session_chunks에는 없음         │
│         = ORPHANED CHUNK!              │
└─────────────────────────────────────────┘
```

### Why This Matters

```python
# get_session_chunks() 예시
def get_session_chunks(self, session_id: str) -> List[KVChunk]:
    chunks = []
    if session_id in self.session_chunks:  # ← 여기 확인
        for chunk_key in self.session_chunks[session_id]:
            # 각 tier에서 검색
            if chunk_key in self.gpu_cache:
                chunk = self.gpu_cache[chunk_key]
            elif chunk_key in self.cpu_cache:
                chunk = self.cpu_cache[chunk_key]
            elif chunk_key in self.dropped_chunks:
                chunk = self.dropped_chunks[chunk_key]
            chunks.append(chunk)
    return chunks
```

**If chunk is in dropped_chunks but NOT in session_chunks**:
→ 이 메서드에서 찾을 수 없음 = 고아 청크!

---

## Solution: session_chunks Design

### Key Insight: Chunks Move Between Tiers

```
                   ┌─────────────────────────┐
                   │  session_chunks[sid]    │
                   │  = all chunks ever      │
                   │  stored for this session│
                   └─────────────────────────┘
                              ▲
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
            gpu_cache    cpu_cache    dropped_chunks
              (hot)       (warm)         (cold)

  Chunks move between tiers:
  gpu_cache → cpu_cache → dropped_chunks
```

**Critical**: session_chunks는 chunks가 어느 tier에 있든 추적해야 한다.

### Design Rules

**Rule 1: When to ADD to session_chunks**
- `store_chunk()`: 새 chunk 저장할 때만 (line 165-166)

**Rule 2: When to REMOVE from session_chunks**
- **ONLY in evict_session()** (세션 완전 삭제 시)
- **NOT in _evict_to_free_space()** (chunks가 다른 tier로 이동할 수 있음)

**Rule 3: Tier-wise Eviction**
- GPU에서 pop → CPU로 move
- CPU에서 pop → DROPPED로 move
- 각 move 후에도 session_chunks에 남아있음 ✓

---

## Eviction Flow with Correct session_chunks Handling

### GPU Eviction (Detailed)

```
_evict_to_free_space(required_bytes, location=GPU)
│
├─ Get eviction candidates from GPU
│
├─ For each candidate (by retention value):
│  │
│  ├─ Skip if pinned
│  │
│  ├─ chunk = gpu_cache.pop(chunk_key)
│  │  └─ ✓ Remove from GPU
│  │     (session_chunks NOT touched!)
│  │
│  ├─ gpu_used_bytes -= chunk.size_bytes
│  │
│  ├─ Try: Move to CPU
│  │  ├─ If CPU has space:
│  │  │  └─ cpu_cache[chunk_key] = chunk
│  │  │     └─ ✓ Chunk still in session_chunks
│  │  │
│  │  └─ If CPU full:
│  │     ├─ Evict from CPU to DROPPED
│  │     │  └─ For each cpu_chunk to drop:
│  │     │     └─ dropped_chunks[drop_key] = drop_chunk
│  │     │        └─ ✓ keep in session_chunks
│  │     │
│  │     └─ Move current chunk to CPU
│  │        └─ ✓ Chunk still in session_chunks
│  │
│  └─ freed += chunk.size_bytes
│
└─ Return freed bytes
```

### CPU Eviction

```
_evict_to_free_space(required_bytes, location=CPU)
│
├─ chunk = cpu_cache.pop(chunk_key)
│  └─ Remove from CPU
│
├─ cpu_used_bytes -= chunk.size_bytes
│
├─ dropped_chunks[chunk_key] = chunk
│  └─ Move to DROPPED
│
├─ ✓ session_chunks not modified
│   (chunk still tracked in session_chunks)
│
└─ freed += chunk.size_bytes
```

### Session Cleanup (Final)

```
evict_session(session_id)
│
├─ Get: chunk_keys = copy of session_chunks[session_id]
│
├─ For each chunk_key:
│  ├─ Skip if pinned
│  ├─ Pop from gpu_cache/cpu_cache/dropped_chunks
│  └─ Update memory counters
│
├─ ✅ Delete: session_chunks[session_id]
│   └─ NOW we remove from tracking!
│
└─ Return freed_bytes (GPU+CPU only)
```

---

## Code Changes

### File: src/pensieve/core/cache.py

**Change 1: _evict_to_free_space() - Lines 506-508**

**Before**:
```python
# ✅ CRITICAL: Remove from session_chunks tracking
if chunk.session_id in self.session_chunks:
    if chunk_key in self.session_chunks[chunk.session_id]:
        self.session_chunks[chunk.session_id].remove(chunk_key)
```

**After**:
```python
# ⚠️ DO NOT remove from session_chunks yet!
# Chunk will move to another tier (CPU or DROPPED)
# session_chunks tracks all tiers, so keep it there
```

**Rationale**: Chunks can move between tiers, so session_chunks should always have the chunk_key regardless of which tier it's in.

**Change 2: evict_session() - Lines 376-379**

**Code**:
```python
elif chunk_key in self.dropped_chunks:
    # ✅ CRITICAL: Also remove from DROPPED
    self.dropped_chunks.pop(chunk_key)
    # session_chunks is deleted AFTER all chunks removed
```

---

## Memory Tracking Invariants

After this design:

### Invariant 1: session_chunks is Master Index

```
∀ session_id, chunk_key in session_chunks[session_id]:
  ∃ chunk in (gpu_cache ∪ cpu_cache ∪ dropped_chunks)
  where chunk.key == chunk_key
```

**Meaning**: If it's in session_chunks, it's in exactly ONE tier.

### Invariant 2: No Orphaned Chunks

```
∀ chunk in (gpu_cache ∪ cpu_cache ∪ dropped_chunks):
  chunk.key ∈ session_chunks[chunk.session_id]
```

**Meaning**: Every chunk in cache is tracked in session_chunks.

### Invariant 3: Clean Deletion

```
When session_id deleted:
  ∀ chunk_key ∈ session_chunks[session_id] (before deletion):
    chunk_key NOT in (gpu_cache ∪ cpu_cache ∪ dropped_chunks)
  AND
    session_chunks[session_id] deleted
```

---

## Tier Movement Sequence

Example: Chunk's journey through cache tiers

```
Timeline: CPU fills up, GPU needs to move chunk to CPU
═══════════════════════════════════════════════════════

t=0: GPU eviction triggered
    chunk = gpu_cache.pop("s1:chunk:0:layer:0")
    gpu_used_bytes -= size

    session_chunks["s1"] still contains "s1:chunk:0:layer:0" ✓

t=1: CPU has no space, needs to make room
    For each cpu_chunk by retention value:
        cpu_cache.pop(cpu_chunk_key)
        cpu_used_bytes -= size
        dropped_chunks[cpu_chunk_key] = cpu_chunk

        session_chunks["s1"] still contains cpu_chunk_key ✓

t=2: Now CPU has space
    cpu_cache["s1:chunk:0:layer:0"] = chunk
    cpu_used_bytes += size

    session_chunks["s1"] still contains "s1:chunk:0:layer:0" ✓
    (no change needed!)

t=3: Later, session_1 ends
    evict_session("s1"):
        For each "s1:*" key:
            Pop from gpu_cache or cpu_cache or dropped_chunks
            Update memory counters

        Finally: del session_chunks["s1"]
        └─ All chunks are now gone from all tiers
```

---

## Why This Design Works

### Problem with Previous Approach
```
Remove from session_chunks at gpu_cache.pop():
├─ chunk moves to cpu_cache
│  └─ But it's NOT in session_chunks! ❌
├─ chunk moves to dropped_chunks
│  └─ But it's NOT in session_chunks! ❌
└─ Result: Cannot find chunk later
```

### Solution with New Approach
```
Keep in session_chunks until evict_session():
├─ chunk moves: GPU → CPU
│  └─ session_chunks tracks it ✓
├─ chunk moves: CPU → DROPPED
│  └─ session_chunks tracks it ✓
├─ chunk accessed later: get_session_chunks()
│  └─ Finds it in whichever tier ✓
└─ session ends: evict_session()
   └─ Comprehensive cleanup of all tiers ✓
```

---

## Testing session_chunks Consistency

```python
def test_session_chunks_consistency():
    """Verify session_chunks tracks all chunks regardless of tier."""
    cache = TwoTierCache(gpu_capacity_gb=0.1, cpu_capacity_gb=0.2)

    # Add chunks to session
    for i in range(5):
        chunk = create_test_chunk("s1", i)
        cache.store_chunk(chunk, location=GPU)

    # Force GPU full → move to CPU
    for i in range(5, 10):
        chunk = create_test_chunk("s2", i-5)
        cache.store_chunk(chunk, location=GPU)

    # Some s1 chunks should be in CPU now
    s1_in_session_chunks = len(cache.session_chunks.get("s1", []))

    # Count s1 chunks in all tiers
    s1_in_gpu = sum(1 for k in cache.gpu_cache if "s1" in k)
    s1_in_cpu = sum(1 for k in cache.cpu_cache if "s1" in k)
    s1_in_dropped = sum(1 for k in cache.dropped_chunks if "s1" in k)

    s1_actual_total = s1_in_gpu + s1_in_cpu + s1_in_dropped

    # MUST be equal
    assert s1_in_session_chunks == s1_actual_total, \
        f"session_chunks has {s1_in_session_chunks} but actual is {s1_actual_total}"
```

---

## Summary

**Design Decision**:
- ✅ session_chunks = Master index of all chunks (all tiers)
- ✅ _evict_to_free_space() = Manage tier transitions (don't touch session_chunks)
- ✅ evict_session() = Final cleanup (delete from session_chunks)
- ✅ end_session() = Trigger cleanup from server level

**Result**:
- No orphaned chunks
- Consistent tracking across tier movements
- Clean session deletion
- Correct get_session_chunks() behavior
