# Multi-Turn KV Cache Shape Mismatch - Complete Debug Summary

## Problem Statement

When running multi-turn conversations (Turn 2+), the system crashes with:
```
Expected size 38 but got size 81 for tensor number 1 in the list
```

This occurs when storing KV chunks in `_store_new_kv_chunks()` method.

## Root Cause

The batch generation process (`_custom_generate()`) returns KV tensors with **batch_size=2** (containing outputs for multiple sessions in the batch). However, when storing these in the session-specific cache, we were storing the **full batch** KV instead of extracting the **individual session's** KV (batch_size=1).

**Example:**
- Batch contains: [session_1_request, session_2_request]
- Generation outputs: `final_past_kv = [batch=2, seq_len=81, heads=12, dim=64]`
- Need to store for session_1: only `[batch=1, seq_len=38, heads=12, dim=64]`

## Fixes Implemented

### Location 1: `_custom_generate()` - Line 319-323
**Purpose**: Pack request index with KV for later extraction

```python
# Before:
final_past_kv_per_session[session_id] = session_past_kv

# After:
final_past_kv_per_session[session_id] = (req_idx, session_past_kv)
```

**What it does**: Stores a tuple `(req_idx, kv)` so we know which batch position corresponds to this session.

### Location 2: `_process_outputs()` - Line 530-568
**Purpose**: Extract batch_size=1 KV using req_idx before storing

```python
# Before:
self._store_new_kv_chunks(batch, kv_data, session_id)

# After:
if isinstance(kv_data, tuple) and len(kv_data) == 2:
    req_idx, session_kv = kv_data
    session_kv_single = tuple(
        (
            k[req_idx:req_idx+1, ...] if k is not None else None,
            v[req_idx:req_idx+1, ...] if v is not None else None,
        )
        for k, v in session_kv
    )
    self._store_new_kv_chunks(batch, session_kv_single, session_id)
```

**What it does**:
1. Unpacks the tuple to get `req_idx` and full batch KV
2. Extracts single request's KV using `[req_idx:req_idx+1, ...]` slicing
3. Stores only the session-specific KV

### Location 3: `_store_new_kv_chunks()` - Line 662-669
**Purpose**: Debug output to verify shapes

```python
if layer_idx == 0:
    print(f"[DEBUG] Layer {layer_idx}: k.shape={k.shape}, v.shape={v.shape}")
    print(f"[DEBUG] num_generated={num_generated}, fill_last={fill_last}")
    if fill_last > 0 and last_chunk_id >= 0:
        last_chunk = self.cache.get_chunk(last_chunk_key)
        if last_chunk:
            print(f"[DEBUG] last_chunk.key_tensor.shape={last_chunk.key_tensor.shape}")
```

### Location 4: `_custom_generate()` - Line 320-324
**Purpose**: Verify tuple packing at source

```python
if session_past_kv and len(session_past_kv) > 0:
    first_k, first_v = session_past_kv[0]
    if first_k is not None:
        print(f"[DEBUG _custom_generate] Packed session_id={session_id}, req_idx={req_idx}, kv[0].shape={first_k.shape}")
```

### Location 5: `_process_outputs()` - Line 542-548
**Purpose**: Verify extraction at destination

```python
print(f"[DEBUG _process_outputs] session_id={session_id}, req_idx={req_idx}, kv_data is tuple")
# ... extraction happens ...
first_k, first_v = session_kv_single[0]
if first_k is not None:
    print(f"[DEBUG _process_outputs] After extraction: first_k.shape={first_k.shape}, first_v.shape={first_v.shape}")
```

## How to Test

### Option 1: Quick Test (Recommended)
```bash
cd /Users/sj/pensieve
python test_multiturn_debug.py
```

This will:
- Create a Pensieve server with GPT-2
- Run Turn 1: "Hello, how are you?"
- Run Turn 2: "Tell me about machine learning"
- Print debug output showing shapes at each stage

### Option 2: Simple Demo
```bash
python main.py --mode pensieve --model gpt2 --max-new-tokens 20
```

### Option 3: With Output Capture
```bash
python test_multiturn_debug.py 2>&1 | tee multiturn_debug.log
```

Then examine the log:
```bash
grep "\[DEBUG" multiturn_debug.log
```

## Expected Debug Output

### ✅ Working Case (After Fix)

```
[DEBUG _custom_generate] Packed session_id=test_session_1, req_idx=0, kv[0].shape=torch.Size([2, 25, 12, 64])

[DEBUG _process_outputs] session_id=test_session_1, req_idx=0, kv_data is tuple
[DEBUG _process_outputs] After extraction: first_k.shape=torch.Size([1, 25, 12, 64]), first_v.shape=torch.Size([1, 25, 12, 64])

[DEBUG] Layer 0: k.shape=torch.Size([1, 25, 12, 64]), v.shape=torch.Size([1, 25, 12, 64])
[DEBUG] num_generated=20, fill_last=12
[DEBUG] last_chunk.key_tensor.shape=torch.Size([1, 8, 12, 64])

[DEBUG _custom_generate] Packed session_id=test_session_1, req_idx=0, kv[0].shape=torch.Size([1, 45, 12, 64])

[DEBUG _process_outputs] session_id=test_session_1, req_idx=0, kv_data is tuple
[DEBUG _process_outputs] After extraction: first_k.shape=torch.Size([1, 45, 12, 64]), first_v.shape=torch.Size([1, 45, 12, 64])

[DEBUG] Layer 0: k.shape=torch.Size([1, 45, 12, 64]), v.shape=torch.Size([1, 45, 12, 64])
[DEBUG] num_generated=20, fill_last=12
[DEBUG] last_chunk.key_tensor.shape=torch.Size([1, 8, 12, 64])

✓ Multi-turn test PASSED
```

**Key observations**:
- `[DEBUG _custom_generate]` shows batch=2 (correct - full batch at generation)
- `[DEBUG _process_outputs] After extraction` shows batch=1 (correct - extracted to single request)
- `[DEBUG] Layer 0` shows batch=1 (correct - storage receives single batch)

### ⚠️ Error Case (Extraction Not Working)

```
[DEBUG _custom_generate] Packed session_id=test_session_1, req_idx=0, kv[0].shape=torch.Size([2, 25, 12, 64])

[DEBUG _process_outputs] session_id=test_session_1, req_idx=0, kv_data is tuple
[DEBUG _process_outputs] After extraction: first_k.shape=torch.Size([2, 25, 12, 64])  ← STILL batch=2!

[DEBUG] Layer 0: k.shape=torch.Size([2, 25, 12, 64]), v.shape=torch.Size([2, 25, 12, 64])
[DEBUG] num_generated=20, fill_last=12
[DEBUG] last_chunk.key_tensor.shape=torch.Size([1, 8, 12, 64])

ERROR: Expected size 38 but got size 81 for tensor number 1 in the list
```

**Problem**: Batch extraction didn't work, still storing batch=2 KV for session

**Next debugging step**: Check if slicing `k[req_idx:req_idx+1, ...]` is working correctly

### 🔍 Fallback Case

```
[DEBUG _process_outputs] session_id=test_session_1, kv_data is NOT tuple (type=<class 'dict'>), using fallback

ERROR: Expected size 38 but got size 81...
```

**Problem**: Tuple packing not happening, fallback path triggered

**Next debugging step**: Verify `_custom_generate()` is actually returning the tuple

## Troubleshooting Decision Tree

**Q1: Are the debug prints showing up?**
- No → Print statements might not reach output, check buffering or stderr redirection
- Yes → Continue to Q2

**Q2: What does `[DEBUG _process_outputs]` show?**
- `kv_data is NOT tuple` → Tuple packing failed in `_custom_generate()`
  - Check line 319: Is the assignment actually setting `(req_idx, kv)`?
  - Check: Is `_custom_generate()` being called?
- `kv_data is tuple` → Continue to Q3

**Q3: What shape is reported in "After extraction"?**
- Still `[2, ...]` (batch=2) → Slicing not working
  - Try: `k.narrow(0, req_idx, 1)` instead of `k[req_idx:req_idx+1, ...]`
  - Check: Is `req_idx` correct? Print its value
- `[1, ...]` (batch=1) → Continue to Q4

**Q4: What shape is in `[DEBUG] Layer 0`?**
- `[1, ...]` → Extraction worked! Mismatch might be elsewhere
  - Check: `fill_last` logic, last_chunk merge, context_length calculation
  - Try: Different approach to merging incomplete chunks
- `[2, ...]` → Something went wrong after extraction
  - Check: Is there another code path modifying the tensors?

## Files Modified

1. `/Users/sj/pensieve/src/pensieve/worker/worker.py`
   - Line 319-323: Tuple packing in `_custom_generate()`
   - Line 530-568: Tuple unpacking and extraction in `_process_outputs()`
   - Line 542-548: Debug output in `_process_outputs()`
   - Line 662-669: Debug output in `_store_new_kv_chunks()`

2. `/Users/sj/pensieve/test_multiturn_debug.py` (NEW)
   - Quick test script for debugging

3. `/Users/sj/pensieve/DEBUG_GUIDE.md` (NEW)
   - Detailed debugging guide

## Next Steps After Debugging

1. **Run the test** and capture output
2. **Analyze the debug output** using the decision tree above
3. **Identify the exact failure point** (packing, extraction, or storage)
4. **Apply targeted fix** based on failure point
5. **Verify the fix** by running test again
6. **Remove debug prints** once working
7. **Run full benchmark** to confirm multi-turn performance

## Alternative Approaches (If Fix Doesn't Work)

### Approach A: Store Each Layer Separately
Instead of tuple of all layers, store per-layer:
```python
final_past_kv_per_session[session_id] = {
    "req_idx": req_idx,
    "kv_by_layer": {layer_idx: (k_single, v_single) for layer_idx, (k, v) in enumerate(session_past_kv)}
}
```

### Approach B: Pre-extract During Generation
Instead of extracting in `_process_outputs()`, extract immediately after generation:
```python
# Inside _custom_generate(), right after generating past_key_values
session_past_kv = tuple(
    (full_kv[0][req_idx:req_idx+1, ...], full_kv[1][req_idx:req_idx+1, ...])
    for full_kv in final_past_kv
)
final_past_kv_per_session[session_id] = session_past_kv  # Already extracted
```

### Approach C: Use Different Data Structure
Instead of tuples, use a class:
```python
@dataclass
class SessionKVPair:
    req_idx: int
    kv: tuple

final_past_kv_per_session[session_id] = SessionKVPair(req_idx, session_past_kv)
```

## Performance Expectation

Once fixed, multi-turn conversations should:
- ✅ Complete without shape mismatch errors
- ✅ Show increasing speedup in later turns (cache reuse)
- ✅ Pass all internal consistency checks
- ✅ Show proper cache statistics (GPU cache hits)

Expected speedup over stateless baseline:
- Turn 1: ~1.0x (no cache reuse yet)
- Turn 2: ~1.3-1.5x (some cache reuse)
- Turn 5+: ~2.0-3.0x (full cache benefit)
