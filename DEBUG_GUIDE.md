# Multi-Turn KV Cache Shape Mismatch - Debugging Guide

## Current Issue

When running multi-turn conversations, we get:
```
Expected size 38 but got size 81 for tensor number 1 in the list
```

This occurs during KV chunk storage in `_store_new_kv_chunks()`.

## Root Cause Analysis

The problem is in how we handle KV tensors from batched generation:

1. **Generation Phase**: `_custom_generate()` processes a batch with multiple sessions
   - Input: `[batch_size=2, ...]` containing requests from different sessions
   - Output: `final_past_kv_per_session` dict with per-session KV

2. **Storage Phase**: We need to extract the specific request's KV from full batch
   - **Wrong**: Storing full batch KV `[batch=2, seq_len, ...]` directly
   - **Right**: Extract single batch KV `[batch=1, seq_len, ...]` using req_idx

## Fixes Implemented

### Fix 1: Store request index with KV (Line 319)
```python
# Before:
final_past_kv_per_session[session_id] = session_past_kv

# After:
final_past_kv_per_session[session_id] = (req_idx, session_past_kv)
```

### Fix 2: Extract batch_size=1 KV in `_process_outputs` (Lines 547-553)
```python
if isinstance(kv_data, tuple) and len(kv_data) == 2:
    req_idx, session_kv = kv_data
    if session_kv:
        session_kv_single = tuple(
            (
                k[req_idx:req_idx+1, ...] if k is not None else None,
                v[req_idx:req_idx+1, ...] if v is not None else None,
            )
            for k, v in session_kv
        )
        self._store_new_kv_chunks(batch, session_kv_single, session_id)
```

### Fix 3: Debug output (Lines 662-669)
Prints actual tensor shapes to identify mismatch source.

## Running the Debug

### Quick Test (Simple Demo)
```bash
# Navigate to project root
cd /Users/sj/pensieve

# Run simple demo with 2 sessions
python main.py --mode pensieve --model gpt2 --max-new-tokens 20

# Expected output will include:
# [DEBUG] Layer 0: k.shape=..., v.shape=...
# [DEBUG] num_generated=..., fill_last=...
# [DEBUG] last_chunk.key_tensor.shape=...
```

### What to Look For in Debug Output

**Normal Case (Working)**:
```
[DEBUG] Layer 0: k.shape=torch.Size([1, 45, 12, 64]), v.shape=torch.Size([1, 45, 12, 64])
[DEBUG] num_generated=20, fill_last=12
[DEBUG] last_chunk.key_tensor.shape=torch.Size([1, 20, 12, 64])
```

**Error Case (Shape Mismatch)**:
```
[DEBUG] Layer 0: k.shape=torch.Size([2, 45, 12, 64])  ← BATCH=2, should be 1!
[DEBUG] num_generated=20, fill_last=12
[DEBUG] last_chunk.key_tensor.shape=torch.Size([1, 20, 12, 64])
```

### What the Numbers Mean

- **k.shape[0]** = batch size (should be **1** after extraction, not 2)
- **k.shape[1]** = sequence length (includes all previous context + new tokens)
- **k.shape[2]** = num_heads (model config)
- **k.shape[3]** = head_dim (model config)

### Understanding the Error Numbers (38 vs 81)

These are likely **sequence length values**:
- `38` = last_chunk.key_tensor.shape[1] (tokens in last incomplete chunk)
- `81` = k.shape[1] (total sequence length from full batch)

This suggests the extraction might not be working correctly.

## Possible Outcomes

### ✅ Success: Extraction Works
- Debug shows `k.shape[0] = 1` (batch correctly extracted)
- Shape mismatch resolves
- Multi-turn conversations complete successfully

### ⚠️ Extraction Not Working
- Debug shows `k.shape[0] = 2` (still full batch)
- Indicates the tuple unpacking didn't work as expected
- Possible causes:
  - `_custom_generate()` not returning tuple correctly
  - `_process_outputs()` not receiving the tuple
  - Fallback path being triggered instead

### 🔍 Different Mismatch
- Debug shows `k.shape[0] = 1` but still errors
- Mismatch might be in:
  - Last chunk tensor shape calculation
  - Context length computation
  - Incomplete chunk merging logic

## Next Steps

1. **Run the simple demo** and capture the debug output
2. **Compare k.shape** with expected values
3. **If batch extraction works** but error persists:
   - Check `fill_last` logic (should split new tokens correctly)
   - Verify last_chunk tensor dimensions
   - Inspect the torch.cat merge operation

4. **If batch extraction doesn't work**:
   - Add debug print in `_process_outputs` to confirm tuple unpacking
   - Check if `_custom_generate()` is actually returning the tuple
   - Verify the fallback path isn't being triggered

## Code Locations

- **Tuple packing**: `/Users/sj/pensieve/src/pensieve/worker/worker.py:319`
- **Tuple unpacking**: `/Users/sj/pensieve/src/pensieve/worker/worker.py:547-553`
- **Debug output**: `/Users/sj/pensieve/src/pensieve/worker/worker.py:662-669`
- **Storage method**: `/Users/sj/pensieve/src/pensieve/worker/worker.py:566` (start of `_store_new_kv_chunks`)

## Command to Run with Full Output

```bash
# Redirect stderr to see all prints
python main.py --mode pensieve --model gpt2 --max-new-tokens 20 2>&1 | tee debug_output.log
```

Then search the log for `[DEBUG]` to see all debug output.
