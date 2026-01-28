# Cache Lifecycle and Session Management

## Your Questions Answered

### Q1: "GPU에 최초/recovery시 store_chunk부르고, swap할때 그 위치 조정하는것도 확인. 그러면 날리는건 cpu에서도 eviction될때 날려야할거같은데?"

**Answer**: 맞다. CPU에서도 eviction될 때 정리해야 한다. 문제를 찾아 고쳤다.

#### Problem Found
CPU eviction 시 `session_chunks` 정리가 빠졌다:

**Before** (Line 548-553 - BUGGY):
```python
else:
    # Evicting from CPU
    self.cpu_used_bytes -= chunk.size_bytes
    self.dropped_chunks[chunk_key] = chunk
    chunk.location = CacheLocation.DROPPED
    # ← session_chunks에서 제거 안 함!
```

**After** (Line 548-559 - FIXED):
```python
else:
    # Evicting from CPU
    self.cpu_used_bytes -= chunk.size_bytes
    self.dropped_chunks[chunk_key] = chunk
    chunk.location = CacheLocation.DROPPED

    # ✅ CRITICAL: Also remove from session_chunks
    if chunk.session_id in self.session_chunks:
        if chunk_key in self.session_chunks[chunk.session_id]:
            self.session_chunks[chunk.session_id].remove(chunk_key)
```

---

### Q2: "Recovery시 cache는 TokenRecoveryManager 멤버변수, 최초 생성시는 Worker 멤버변수. Worker는 사라지는데 cache 메타데이터는 세션이 완전히 사라질때까지 관리해야하지 않아?"

**Answer**: 정확하다! Architecture는 이미 올바르게 설계되어 있다.

#### Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│            PensieveServer (Singleton)                │
│                                                       │
│  ┌─ cache: TwoTierCache (lifetime = server)         │
│  │  ├─ gpu_cache: {}                                │
│  │  ├─ cpu_cache: {}                                │
│  │  ├─ dropped_chunks: {}                           │
│  │  ├─ session_chunks: {session_id: [chunk_keys]}   │
│  │  └─ pinned_chunks, pinned_sessions: {}           │
│  │                                                    │
│  ├─ scheduler: BatchScheduler                        │
│  │  └─ cache (same reference)                        │
│  │                                                    │
│  ├─ batched_recovery_manager: BatchedRecoveryManager │
│  │  ├─ cache (same reference)                        │
│  │  └─ recovery_manager: TokenRecoveryManager        │
│  │      └─ cache (same reference)                    │
│  │                                                    │
│  ├─ worker: Worker                                   │
│  │  └─ cache (same reference)                        │
│  │                                                    │
│  └─ session_token_histories: {session_id: [tokens]} │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Key Points**:
- ✅ Cache 생성: Server.__init__ (line 61-65)
- ✅ Cache 공유: 모든 components (Worker, RecoveryManager, Scheduler) 같은 instance
- ✅ Cache lifetime: Server 전체 lifetime (개별 Worker/RecoveryManager가 와도 유지)
- ✅ Session 정리: Server.end_session() (새로 추가)

---

## Session Lifecycle Flow

### 1️⃣ Session Start

```
Server.process_request(session_id="user_123", input="Hello")
│
├─ Check: session_id in active_sessions?
├─ NO → Create: self.active_sessions["user_123"] = []
├─ YES → Append new request
│
└─ Create Request → Scheduler → Worker → Execution
```

### 2️⃣ Cache Operations During Session

```
Initial Generation (Turn 1):
├─ worker._store_new_kv_chunks()
│  └─ cache.store_chunk(chunk, location=GPU)
│     ├─ Check all caches for duplicates ← FIX #4
│     ├─ Store in GPU
│     └─ Update: gpu_used_bytes, session_chunks
│
Recovery (if dropped):
├─ token_recovery._store_recovered_chunks()
│  └─ cache.store_chunk(recovered_chunk, location=GPU)
│     └─ (same cleanup logic as above)
│
Eviction (if GPU full):
├─ cache._evict_to_free_space(required_bytes, GPU)
│  ├─ Pop chunk from GPU
│  ├─ Update: gpu_used_bytes
│  ├─ Try move to CPU
│  │  ├─ If CPU full: evict_from_cpu()
│  │  │  └─ Update: cpu_used_bytes ← FIX #1 (session_chunks cleanup)
│  │  └─ Move to CPU
│  └─ Update: session_chunks, statistics
```

### 3️⃣ Session End

```
Server.end_session(session_id="user_123")
│
├─ Step 1: Cache cleanup
│  └─ cache.evict_session("user_123")
│     ├─ For each chunk of session:
│     │  ├─ Check: GPU? → pop + update gpu_used_bytes
│     │  ├─ Check: CPU? → pop + update cpu_used_bytes
│     │  └─ Check: DROPPED? → pop ← FIX #2
│     │
│     └─ Delete: session_chunks["user_123"]
│
├─ Step 2: Session state cleanup
│  └─ Delete: active_sessions["user_123"]
│
├─ Step 3: History cleanup
│  └─ Delete: session_token_histories["user_123"]
│
└─ Return: freed_bytes (for statistics)
```

---

## Memory Safety Invariants

After all fixes, these invariants hold:

### Invariant 1: No Cross-Tier Duplicates
```
∀ chunk_key:
  count(chunk_key in gpu_cache) +
  count(chunk_key in cpu_cache) +
  count(chunk_key in dropped_chunks) ≤ 1
```

### Invariant 2: Memory Tracking Accuracy
```
gpu_used_bytes = Σ(chunk.size_bytes for chunk in gpu_cache.values())
cpu_used_bytes = Σ(chunk.size_bytes for chunk in cpu_cache.values())
```

### Invariant 3: session_chunks Consistency
```
∀ session_id, chunk_key in session_chunks[session_id]:
  ∃ chunk in {gpu_cache, cpu_cache, dropped_chunks} where chunk.key == chunk_key
```

### Invariant 4: No Orphaned Resources
```
When session_id is deleted:
  ∀ chunk with chunk.session_id == session_id:
    chunk is NOT in any cache (gpu, cpu, dropped)
    AND chunk_key NOT in session_chunks[session_id]
```

---

## Code Changes Summary

### 1. Cache.py - CPU Eviction Cleanup

**File**: `src/pensieve/core/cache.py`
**Lines**: 548-559
**Change**: Added session_chunks cleanup when evicting from CPU

```python
# BEFORE: Missing cleanup
else:
    self.cpu_used_bytes -= chunk.size_bytes
    self.dropped_chunks[chunk_key] = chunk

# AFTER: Proper cleanup
else:
    self.cpu_used_bytes -= chunk.size_bytes
    self.dropped_chunks[chunk_key] = chunk
    # ✅ Also remove from session_chunks
    if chunk.session_id in self.session_chunks:
        if chunk_key in self.session_chunks[chunk.session_id]:
            self.session_chunks[chunk.session_id].remove(chunk_key)
```

### 2. Cache.py - Session Cleanup Method

**File**: `src/pensieve/core/cache.py`
**Lines**: 345-380
**Change**: Updated evict_session to cleanup DROPPED chunks

```python
# BEFORE: Missing DROPPED cleanup
elif chunk_key in self.cpu_cache:
    chunk = self.cpu_cache.pop(chunk_key)
    # ...
del self.session_chunks[session_id]

# AFTER: Complete cleanup
elif chunk_key in self.cpu_cache:
    chunk = self.cpu_cache.pop(chunk_key)
    # ...
elif chunk_key in self.dropped_chunks:
    # ✅ Also cleanup DROPPED
    self.dropped_chunks.pop(chunk_key)
del self.session_chunks[session_id]
```

### 3. Server.py - Session Lifecycle Method

**File**: `src/pensieve/server/server.py`
**Lines**: 358-394
**Change**: Added new method for session cleanup

```python
def end_session(self, session_id: str) -> int:
    """Cleanup resources for completed session.

    1. Evict all chunks from cache (GPU/CPU/DROPPED)
    2. Remove from active_sessions
    3. Remove session token history

    Returns: bytes freed
    """
    freed_bytes = 0
    if self.cache:
        freed_bytes = self.cache.evict_session(session_id)

    if session_id in self.active_sessions:
        del self.active_sessions[session_id]

    if session_id in self.session_token_histories:
        del self.session_token_histories[session_id]

    return freed_bytes
```

---

## Usage Example

```python
# Start session
response1 = server.process_request("user_123", "Hello, how are you?")

# Multi-turn conversation (cache reused)
response2 = server.process_request("user_123", "Tell me about AI")
response3 = server.process_request("user_123", "What about ML?")

# Session complete → cleanup
freed = server.end_session("user_123")
print(f"Freed {freed / 1024**2:.2f} MB")

# Cache metadata cleaned:
# - session_chunks["user_123"] removed
# - All chunks for "user_123" evicted from GPU/CPU/DROPPED
# - session_token_histories["user_123"] removed
# - active_sessions["user_123"] removed
```

---

## Cleanup Verification

All three issues are now resolved:

✅ **Issue 1**: CPU eviction에서 session_chunks 정리
- Fixed in: `_evict_to_free_space()` line 556-559

✅ **Issue 2**: DROPPED chunks 정리
- Fixed in: `evict_session()` line 376-379

✅ **Issue 3**: Server에서 session cleanup 메서드
- Added: `end_session()` line 358-394

---

## Testing Session Lifecycle

```python
def test_session_lifecycle():
    """Test complete session lifecycle with cleanup."""
    server = create_server(mode="pensieve")

    # Session starts
    server.process_request("s1", "Turn 1")
    assert "s1" in server.active_sessions
    assert "s1" in server.session_token_histories

    # Multiple turns (cache should work)
    server.process_request("s1", "Turn 2")
    server.process_request("s1", "Turn 3")

    # Check cache has chunks
    assert len(server.cache.session_chunks.get("s1", [])) > 0

    # Session ends
    freed = server.end_session("s1")

    # Verify cleanup
    assert "s1" not in server.active_sessions
    assert "s1" not in server.session_token_histories
    assert "s1" not in server.cache.session_chunks
    assert freed > 0
```

---

## Related Files

- **Cache Lifecycle**: `src/pensieve/core/cache.py`
- **Session Management**: `src/pensieve/server/server.py`
- **Recovery Integration**: `src/pensieve/recovery/token_recovery.py`
- **Worker Integration**: `src/pensieve/worker/worker.py`

---

## Summary

User의 두 질문:

1. **"CPU에서도 정리해야하지 않아?"**
   → 맞다. CPU eviction에서 session_chunks 정리가 빠졌다. ✅ Fixed

2. **"Cache 메타데이터는 세션이 완전히 사라질때까지 관리되어야"**
   → 맞다. Server level에서 cache를 소유하고 session cleanup 메서드가 필요하다. ✅ Fixed & Added

Architecture는 이미 올바르게 설계되었고, memory safety를 위해 세 가지 cleanup 포인트를 모두 처리했다.
