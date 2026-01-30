"""GPU execution worker for Pensieve inference batches."""

import torch
import time
import logging
import os
from typing import Tuple, List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from pensieve.core.types import (
    Batch,
    Request,
    BatchResult,
    CachePlan,
    Phase,
    KVChunk,
    CacheLocation,
)
from pensieve.core.cache import TwoTierCache
from pensieve.worker.custom_cache import PensieveCacheFactory

# Setup logging - control with PENSIEVE_DEBUG environment variable
logger = logging.getLogger(__name__)
_debug_enabled = os.getenv("PENSIEVE_DEBUG", "0") == "1"
_cache_debug_enabled = os.getenv("PENSIEVE_CACHE_DEBUG", "0") == "1"  # Cache-specific detailed logging
if _debug_enabled or _cache_debug_enabled:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    if _cache_debug_enabled:
        handler.setFormatter(logging.Formatter('[CACHE] %(message)s'))
    else:
        handler.setFormatter(logging.Formatter('[DEBUG] %(message)s'))
    if not logger.handlers:  # Avoid duplicate handlers
        logger.addHandler(handler)
else:
    logger.setLevel(logging.WARNING)


class Worker:
    """GPU execution worker for batched inference.

    Responsibilities:
    - Execute cache swaps from cache plan
    - Run model forward pass with custom KV cache
    - Store new KV chunks after generation
    - Handle dropped token recovery (delegated to RecoveryManager)

    Design:
    - Separates cache management (scheduler) from GPU execution (worker)
    - Pure GPU work: forward pass, KV extraction, tensor operations
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cache: TwoTierCache,
        device: str = "cuda:0",
        batched_recovery_manager=None,  # BatchedRecoveryManager (optional)
    ):
        """Initialize worker.

        Args:
            model: HuggingFace language model (already loaded)
            tokenizer: Tokenizer for encoding/decoding
            cache: TwoTierCache instance
            device: GPU device string
            batched_recovery_manager: BatchedRecoveryManager for batch-level recovery
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.device = device
        self.batched_recovery_manager = batched_recovery_manager

        # Store model name for generation config decisions
        self.model_name = getattr(model.config, 'name_or_path', 'unknown').lower()

        # Model config
        self.num_layers = (
            model.config.num_hidden_layers
            if hasattr(model.config, "num_hidden_layers")
            else model.config.n_layer
        )
        self.hidden_size = model.config.hidden_size

    def _get_seq_len_from_kv(self, kv_tensor: torch.Tensor) -> int:
        """Get sequence length from KV cache tensor, handling different model formats.

        Different transformer models use different tensor shapes for KV cache:
        - Gemma-2 format: [batch, num_heads, seq_len, head_dim] → seq at dim=2
        - Standard HF: [batch, seq_len, num_heads, head_dim] → seq at dim=1

        Uses heuristic: if shape[1] < 256 (likely num_heads), use shape[2] for seq_len
        Otherwise, use shape[1] for seq_len.
        """
        if kv_tensor is None or len(kv_tensor.shape) < 2:
            return 0

        shape = kv_tensor.shape
        if len(shape) == 4:
            # 4D tensor: determine which dimension is sequence
            # Heuristic: if dim[1] is small (num_heads 8-128), then dim=2 is seq
            if shape[1] < 256:  # shape[1] is likely num_heads
                return shape[2]  # Gemma format: [batch, heads, seq, head_dim]
            else:
                return shape[1]  # Standard format: [batch, seq, heads, head_dim]
        else:
            # For other shapes, default to dim[1]
            return shape[1]

    def _get_seq_dim_from_kv(self, kv_tensor: torch.Tensor) -> int:
        """Get the dimension index where sequence length is located.

        Returns:
            Dimension index: 2 for Gemma format, 1 for standard format
        """
        if kv_tensor is None or len(kv_tensor.shape) < 4:
            return 1  # Default

        shape = kv_tensor.shape
        # Heuristic: if dim[1] is small (num_heads typically 8-128), then dim=2 is seq
        if shape[1] < 256:  # shape[1] is likely num_heads
            return 2  # Gemma format: [batch, heads, seq, head_dim]
        else:
            return 1  # Standard format: [batch, seq, heads, head_dim]

    def _inspect_cache_thoroughly(self, session_id: str, pensieve_cache) -> str:
        """Thoroughly inspect and dump all cached KV chunks for a session.

        Returns:
            Detailed inspection report as string
        """
        report = []
        report.append("\n" + "="*80)
        report.append(f"[CACHE INSPECTION] Session: {session_id}")
        report.append("="*80)

        if pensieve_cache is None:
            report.append("❌ pensieve_cache is None")
            return "\n".join(report)

        if pensieve_cache.is_empty():
            report.append("⚠️  pensieve_cache is empty (no chunks stored)")
            return "\n".join(report)

        # 1. Get all chunks for this session from cache manager
        try:
            gpu_cache = pensieve_cache.cache_manager.gpu_cache
            cpu_cache = pensieve_cache.cache_manager.cpu_cache

            report.append(f"\n📊 Cache Storage Status:")
            report.append(f"   GPU cache size: {len(gpu_cache)} chunks")
            report.append(f"   CPU cache size: {len(cpu_cache)} chunks")

            # 2. Find all chunks for this session
            session_chunks_gpu = []
            session_chunks_cpu = []

            for key, chunk in gpu_cache.items():
                if chunk.session_id == session_id:
                    session_chunks_gpu.append(chunk)

            for key, chunk in cpu_cache.items():
                if chunk.session_id == session_id:
                    session_chunks_cpu.append(chunk)

            # Sort by chunk_id for readable output
            session_chunks_gpu.sort(key=lambda c: c.chunk_id)
            session_chunks_cpu.sort(key=lambda c: c.chunk_id)

            report.append(f"\n🔍 Chunks for session '{session_id}':")
            report.append(f"   GPU: {len(session_chunks_gpu)} chunks")
            report.append(f"   CPU: {len(session_chunks_cpu)} chunks")
            report.append(f"   Total: {len(session_chunks_gpu) + len(session_chunks_cpu)} chunks")

            # 3. Dump GPU chunks details
            if session_chunks_gpu:
                report.append(f"\n📍 GPU Chunks (from newest to oldest):")
                total_gpu_tokens = 0
                for chunk in sorted(session_chunks_gpu, key=lambda c: -c.chunk_id):
                    num_tokens = chunk.num_tokens if hasattr(chunk, 'num_tokens') else self._get_seq_len_from_kv(chunk.key_tensor)
                    total_gpu_tokens += num_tokens
                    report.append(
                        f"   chunk_id={chunk.chunk_id:3d} | "
                        f"layer={chunk.layer_idx:2d} | "
                        f"tokens={num_tokens:3d} | "
                        f"shape={tuple(chunk.key_tensor.shape)} | "
                        f"dtype={chunk.key_tensor.dtype}"
                    )
                report.append(f"   └─ Total GPU tokens: {total_gpu_tokens}")

            # 4. Dump CPU chunks details
            if session_chunks_cpu:
                report.append(f"\n📍 CPU Chunks (from newest to oldest):")
                total_cpu_tokens = 0
                for chunk in sorted(session_chunks_cpu, key=lambda c: -c.chunk_id):
                    num_tokens = chunk.num_tokens if hasattr(chunk, 'num_tokens') else self._get_seq_len_from_kv(chunk.key_tensor)
                    total_cpu_tokens += num_tokens
                    report.append(
                        f"   chunk_id={chunk.chunk_id:3d} | "
                        f"layer={chunk.layer_idx:2d} | "
                        f"tokens={num_tokens:3d} | "
                        f"shape={tuple(chunk.key_tensor.shape)} | "
                        f"dtype={chunk.key_tensor.dtype}"
                    )
                report.append(f"   └─ Total CPU tokens: {total_cpu_tokens}")

            # 5. Calculate total tokens and verify consistency
            all_chunks = session_chunks_gpu + session_chunks_cpu
            if all_chunks:
                max_chunk_id = max(c.chunk_id for c in all_chunks)
                expected_total_tokens = (max_chunk_id + 1) * 32  # Rough estimate
                report.append(f"\n📈 Cache Statistics:")
                report.append(f"   Max chunk_id: {max_chunk_id}")
                report.append(f"   Expected ~total tokens: {expected_total_tokens}")

                # Check for layer coverage
                layers_in_cache = set()
                for chunk in all_chunks:
                    layers_in_cache.add(chunk.layer_idx)
                report.append(f"   Layers covered: {sorted(layers_in_cache)}")
                report.append(f"   Number of layers: {len(layers_in_cache)}")

            # 6. Check session metadata
            metadata = pensieve_cache.cache_manager.get_session_metadata(session_id)
            if metadata:
                report.append(f"\n📋 Session Metadata:")
                report.append(f"   Total tokens stored: {metadata.total_tokens}")
                report.append(f"   Last chunk size: {metadata.last_chunk_size}")
                report.append(f"   Last access: {metadata.last_access_time}")
            else:
                report.append(f"\n⚠️  No metadata for session '{session_id}'")

        except Exception as e:
            report.append(f"❌ Error during inspection: {e}")
            import traceback
            report.append(traceback.format_exc())

        report.append("="*80 + "\n")
        return "\n".join(report)

    def execute_batch(
        self,
        batch: Batch,
        cache_plan: CachePlan,
    ) -> BatchResult:
        """Execute a batch of requests.

        Steps:
        1. PIN all sessions in batch (prevent concurrent eviction)
        2. Execute cache swaps from plan (GPU ←→ CPU, CPU → DROPPED)
        3. Handle dropped token recovery (if any)
        4. Create PensieveCache for this batch
        5. Prepare batch inputs
        6. Run custom generation loop with proper KV cache integration
        7. Extract and store new KV chunks
        8. UNPIN all sessions (allow eviction again)
        9. Return results

        CRITICAL: Pinning prevents concurrent requests from evicting this batch's chunks
        while the batch is being executed. This ensures cache consistency.

        Args:
            batch: Batch to execute
            cache_plan: Cache swap operations

        Returns:
            BatchResult with generated tokens and statistics
        """
        start_time = time.time()
        logger.debug(f"[execute_batch] START: batch_id={batch.batch_id}, num_requests={len(batch.requests)}, max_new_tokens will be extracted from requests")

        # 1. PIN all sessions in this batch to protect from concurrent eviction
        session_ids = [req.session_id for req in batch.requests]
        for session_id in session_ids:
            self.cache.pin_session(session_id)

        try:
            # 2. Execute cache swaps (including recovery)
            prefill_start = time.time()
            self._execute_cache_plan(cache_plan, batch)
            prefill_end = time.time()
            prefill_time_elapsed = prefill_end - prefill_start
            # 3. Prepare batch inputs
            input_ids, attention_mask, original_input_lengths = self._prepare_batch_inputs(batch)

            # 4. Create custom cache for this batch (always, even if empty)
            # Design: Pensieve always has cache object, but passes None to model if empty
            # - Turn 1: cache object exists, but is empty → pass None to model
            # - Turn 2+: cache object exists with chunks → pass to model
            pensieve_cache = PensieveCacheFactory.create(
                cache_manager=self.cache,
                batch_requests=batch.requests,
                num_layers=self.num_layers,
            )

            # 5. Run custom generation loop with KV cache integration
            with torch.no_grad():
                try:
                    # Extract max_new_tokens from first request (all requests in batch should have same value)
                    max_new_tokens = batch.requests[0].max_new_tokens if batch.requests else 32
                    outputs = self._custom_generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pensieve_cache=pensieve_cache,
                        batch=batch,
                        max_new_tokens=max_new_tokens,
                        original_input_lengths=original_input_lengths,
                    )
                except Exception as e:
                    print(f"Error during custom generation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return empty result on error
                    return BatchResult(batch_id=batch.batch_id)

            # ⏱️ Separate prefill and generation timing
            # Prefill was measured above: cache plan execution only
            # Generation: remaining time in _custom_generate (first forward + token loop)
            total_end = time.time()
            total_elapsed = total_end - start_time

            # prefill_time_elapsed was already calculated at line 123 (cache plan only)
            # Generation time is the rest
            generation_time_elapsed = total_elapsed - prefill_time_elapsed

            # 6. Extract generated tokens and store new KV chunks
            results = self._process_outputs(batch, outputs)

            results.execution_time = total_elapsed
            results.prefill_time = prefill_time_elapsed
            results.generation_time = generation_time_elapsed

            # Store TTFT (Time To First Token) per request if available
            if hasattr(outputs, 'ttft') and outputs.ttft:
                results.ttft_per_request = outputs.ttft

            return results

        finally:
            # 8. UNPIN all sessions (allow concurrent requests to evict them if needed)
            for session_id in session_ids:
                self.cache.unpin_session(session_id)

    def _custom_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pensieve_cache,
        batch: Batch,
        max_new_tokens: int = 32,
        original_input_lengths: List[int] = None,
    ) -> Dict:
        """Custom generation loop - processes each session independently.

        CRITICAL: To avoid cross-session semantic interference, we process each
        session's generation loop separately. While different token embeddings
        are mathematically different, attention mechanism computes non-zero scores
        for all positions, causing 5-15% average cross-session influence.

        Processing approach:
        - For each request (session), run independent generation loop
        - Each session's query/key/value interact only with own context
        - No semantic leakage between concurrent sessions
        - Maintains correctness for multi-session batches

        Args:
            input_ids: [batch_size, seq_len] input tokens (may be padded)
            attention_mask: [batch_size, seq_len] attention mask
            pensieve_cache: PensieveCache instance (maps session_id to cache)
            batch: Original batch
            max_new_tokens: Max tokens to generate
            original_input_lengths: List of original input lengths BEFORE padding

        Returns:
            Dictionary with:
            - sequences: [batch_size, seq_len+max_new_tokens] all tokens
            - past_key_values: Final KV cache (not used in per-session mode)
            - ttft: Dict of TTFT per request_id (seconds)
            - original_input_lengths: Original input lengths before padding (for token extraction)
        """
        batch_size = len(batch.requests)
        device = input_ids.device

        # Get EOS token ID safely
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            # Fallback: use model config
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
        if eos_token_id is None:
            # Last resort: use common EOS tokens
            # 50256 = GPT-2/GPT-3, 2 = common default, 0 = last resort
            eos_token_id = 50256

        logger.debug(f"[_custom_generate] Using EOS token ID: {eos_token_id}, max_new_tokens: {max_new_tokens}")

        # Track results per request
        generated_ids = [[] for _ in range(batch_size)]
        ttft_per_request = {}
        final_past_kv_per_session = {}  # ✅ Track final KV for each session
        generation_start_time = time.time()

        # Process each session independently
        for req_idx, req in enumerate(batch.requests):
            session_id = req.session_id

            # Get this request's inputs
            req_input_ids = input_ids[req_idx:req_idx+1]  # [1, seq_len]
            req_attention_mask = attention_mask[req_idx:req_idx+1]  # [1, seq_len]

            # ✅ DEBUG: Check batch extraction (Scenario 2)
            # logger.debug(f"_custom_generate] Session {session_id}: req_input_ids.shape={req_input_ids.shape}, req_attention_mask.shape={req_attention_mask.shape}")

            # Get cached KV for this session
            # ✅ PensieveCache handles session-specific chunk gathering internally
            # KEY: Pass PensieveCache to model for all sessions in batch
            # PensieveCache.__getitem__(layer_idx) automatically:
            # - Filters chunks by session_id (only returns this session's chunks)
            # - Concatenates chunks from all positions for the layer
            # - Returns [batch_size, seq_len, num_heads, head_dim] KV tensors
            session_cache = None
            try:
                # Check if pensieve_cache has any cached chunks
                if hasattr(pensieve_cache, 'is_empty') and pensieve_cache.is_empty():
                    # Cache is empty, pass None to model (will compute new KV)
                    session_cache = None
                    if _cache_debug_enabled:
                        logger.debug(f"[DEBUG] {session_id}: pensieve_cache.is_empty()=True → session_cache=None")
                else:
                    # Cache has chunks, pass full PensieveCache
                    # (it will filter to only this session's chunks)
                    session_cache = pensieve_cache
                    # session_cache=None
                    if _cache_debug_enabled:
                        logger.debug(f"[DEBUG] {session_id}: pensieve_cache has chunks → session_cache=PensieveCache")
            except Exception as e:
                logger.warning(f"Failed to get session cache for {session_id}: {e}")
                session_cache = None

            # ✅ CACHE INSPECTION: Thoroughly examine cached chunks before generation
            if session_cache is not None and not session_cache.is_empty():
                inspection_report = self._inspect_cache_thoroughly(session_id, session_cache)
                print(inspection_report)  # Print to console for immediate visibility
                logger.debug(inspection_report)

            # Generation loop for this session only
            session_past_kv = None
            ttft_recorded = False

            for step in range(max_new_tokens):
                # Prepare input for this step
                if step == 0:
                    step_input_ids = req_input_ids
                    # ✅ CRITICAL FIX: When reusing cache from previous turn, attention_mask must cover ALL tokens
                    # Turn 1: session_cache=None, mask covers input only (correct)
                    # Turn 2+: session_cache has cached KV, mask must be None (model auto-extends)
                    if session_cache is not None and not session_cache.is_empty():
                        # Turn 2+: Cache exists, let model handle attention for cached + new tokens
                        step_attention_mask = None
                    else:
                        # Turn 1: No cache, use provided mask
                        step_attention_mask = req_attention_mask

                    # ✅ DEBUG: Log exact input to model at step 0
                    logger.debug(f"[STEP 0 INPUT] {session_id}: step_input_ids.shape={step_input_ids.shape}, attention_mask_shape={req_attention_mask.shape if req_attention_mask is not None else 'None'}")
                    if req_attention_mask is not None:
                        logger.debug(f"[STEP 0 INPUT] {session_id}: attention_mask sum (non-padded tokens)={req_attention_mask.sum().item()}")
                else:
                    step_input_ids = next_token_ids.unsqueeze(1)  # [1, 1]
                    # CRITICAL: When using past_key_values, don't constrain attention_mask
                    # Let HuggingFace handle it internally (attend to all past + current)
                    step_attention_mask = None

                # DEBUG: Print input shapes
                # if step <= 2:
                #     mask_info = f"shape={step_attention_mask.shape}, sum={step_attention_mask.sum().item()}" if step_attention_mask is not None else "None (auto)"
                #     print(f"  [Step {step}] input_ids.shape={step_input_ids.shape}, mask={mask_info}")

                # ✅ Ensure tensors are on correct device for model
                # Handle both single GPU and device_map='auto' (distributed) scenarios
                try:
                    # For device_map='auto', model.device might not be reliable
                    # Try to infer device from model parameters
                    model_device = next(self.model.parameters()).device
                except StopIteration:
                    model_device = device

                # Debug: Print device info on first step
                # if step == 0:
                #     print(f"    Device check: model_device={model_device}, input_device={device}")

                # Move input tensors to model device
                step_input_ids = step_input_ids.to(model_device)
                if step_attention_mask is not None:
                    step_attention_mask = step_attention_mask.to(model_device)

                # ✅ DEBUG: KV cache accumulation tracking
                # Check input KV cache (what we're passing to the model)
                input_cache = session_cache if step == 0 else session_past_kv
                print("!!!!!!! SJ input cache !!!",input_cache)
                # input_cache=session_past_kv
                if input_cache is not None and len(input_cache) > 0:
                    # Handle both PensieveCache and standard HuggingFace cache tuples
                    if hasattr(input_cache, 'get_seq_length'):
                        # PensieveCache object
                        input_cache_len = input_cache.get_seq_length()
                    else:
                        # Standard HuggingFace cache (tuple of tuples)
                        try:
                            input_cache_len = self._get_seq_len_from_kv(input_cache[0][0])
                        except (TypeError, IndexError):
                            input_cache_len = 0
                else:
                    input_cache_len = 0
                
                input_seq_len = step_input_ids.shape[1]  # Current input token count
                # input_len = len(req.input_ids)
                # ✅ VERIFY PENSIEVE WORKING: Only log when Step 0 has cache from previous turns
                # This proves multi-turn cache reuse is working (not same-turn cache)
                # Step 0 + input_cache_len > 0 = cross-turn cache reuse (NEW TURN using previous cache)
                if _cache_debug_enabled and step == 0:
                    # DEBUG: Check Step 0 state for ALL turns
                    logger.debug(f"[DEBUG Step 0] {session_id}: cache={input_cache_len} tokens, input={input_seq_len} tokens, session_cache={session_cache is not None}")
                    # ✅ DEBUG: Show batch_info positions to verify chunking
                    if hasattr(input_cache, 'batch_info'):
                        for info in input_cache.batch_info.values():
                            if info.get('session_id') == session_id:
                                positions = info.get('positions', [])
                                logger.debug(f"[DEBUG] session_id={session_id}: batch_info positions={positions} (chunk count={len(positions)})")
                                # ✅ CRITICAL: Show actual chunk sizes in cache
                                if positions:
                                    max_chunk_id = max(positions)
                                    for cache_dict in [input_cache.cache_manager.gpu_cache, input_cache.cache_manager.cpu_cache]:
                                        for chunk in cache_dict.values():
                                            if chunk.session_id == session_id and chunk.chunk_id == max_chunk_id:
                                                logger.debug(f"[DEBUG] Last chunk ({max_chunk_id}): num_tokens={chunk.num_tokens} (expected ~32 or partial)")
                    if input_cache_len > 0:
                        # NEW TURN with cached KV from previous turns - CORE PENSIEVE FEATURE
                        logger.debug(f"[Pensieve {session_id}] ⭐ NEW TURN REUSES CACHE: Forward input=[1, {input_seq_len}] (new query) + cached=[1, {input_cache_len}] (from previous turns)")
                # print(input_cache.len())/
                # print(input_cache.shape)
                # print(input_cache)
                # print(input_cache)
                
                # 🔴 DEBUG: Check what we're passing as past_key_values
                if step == 0:  # Only on first step
                    cache_type = type(input_cache).__name__ if input_cache is not None else "None"
                    cache_len = len(input_cache) if input_cache is not None and hasattr(input_cache, '__len__') else "?"
                    print(f"\n🔴 [FORWARD PASS] Passing past_key_values:", flush=True)
                    print(f"  type: {cache_type}", flush=True)
                    print(f"  len: {cache_len}", flush=True)
                    print(f"  is PensieveCache: {hasattr(input_cache, 'cache_manager')}", flush=True)
                    print(f"  has __getitem__: {hasattr(input_cache, '__getitem__')}", flush=True)

                    # ✅ VALIDATION: If using PensieveCache, verify chunks exist and have correct shapes
                    if input_cache is not None and hasattr(input_cache, 'cache_manager'):
                        print(f"\n🔍 [CACHE VALIDATION] Verifying PensieveCache contents before forward pass:", flush=True)
                        try:
                            # For each layer, verify we can fetch valid chunks
                            for layer_idx in range(min(self.num_layers, 2)):  # Check first 2 layers
                                print(f"\n  Layer {layer_idx}:", flush=True)
                                try:
                                    # Try to get KV for this layer (calls __getitem__)
                                    k, v = input_cache[layer_idx]
                                    print(f"    ✅ Retrieved: k.shape={k.shape}, v.shape={v.shape}, device={k.device}", flush=True)

                                    # Validate shapes
                                    if k.numel() == 0:
                                        logger.warning(f"⚠️ Layer {layer_idx} KV is EMPTY!")
                                    if len(k.shape) != 4:
                                        logger.error(f"❌ Layer {layer_idx} k has wrong dimensions: {k.shape}")
                                    if k.shape[-1] == 0:
                                        logger.error(f"❌ Layer {layer_idx} k has head_dim=0: {k.shape}")
                                        raise ValueError(f"Layer {layer_idx} has malformed tensor with head_dim=0")

                                except Exception as e:
                                    print(f"    ❌ FAILED to retrieve: {e}", flush=True)
                                    logger.error(f"❌ Cache retrieval failed for layer {layer_idx}: {e}")
                                    raise
                        except Exception as e:
                            logger.error(f"❌ CACHE VALIDATION FAILED: {e}")
                            raise

                # Forward pass - with session-specific cache
                # print("@@@@@@@@@@@@@@@@@@@@ sj SJSJ input_cache",input_cache)
                outputs = self.model(
                    step_input_ids,
                    attention_mask=step_attention_mask,
                    past_key_values=input_cache,
                    use_cache=True,
                    return_dict=True,
                )

                # Check output KV cache (what the model produced)
                if outputs.past_key_values is not None and len(outputs.past_key_values) > 0:
                    output_cache_len = self._get_seq_len_from_kv(outputs.past_key_values[0][0])
                else:
                    output_cache_len = 0

                # Expected: input_cache_len + input_seq_len = output_cache_len
                expected_len = input_cache_len + input_seq_len

                # ✅ DEBUG OUTPUT: Track KV cache growth (only when cache exists - multi-turn scenario)
                # Skip: Step 0 (prefill) and first turn of any session (input_cache_len = 0)
                # Log: Step 1+ where cache is being reused (input_cache_len > 0)
                # if _cache_debug_enabled and step > 0 and input_cache_len > 0:
                #     cache_reuse_pct = 100 * input_cache_len / expected_len
                #     logger.debug(f"[Step {step}] Session {session_id}: {input_cache_len}↓ cached + {input_seq_len} new → {output_cache_len} total ({cache_reuse_pct:.1f}% reuse)")

                # ✅ CORRECTNESS CHECK: Ensure no duplication
                if output_cache_len != expected_len:
                    logger.error(f"❌ KV CACHE MISMATCH! Expected {expected_len}, got {output_cache_len}")
                    logger.error(f"   This indicates cache duplication or incorrect accumulation!")

                # ✅ WARNING: If cache exists but input_seq_len > 1, possible duplication
                if input_cache_len > 0 and input_seq_len > 1:
                    logger.warning(f"⚠️ Cache exists ({input_cache_len}), but input has {input_seq_len} tokens. Possible duplication?")

                # Extract outputs
                logits = outputs.logits
                session_past_kv = outputs.past_key_values

                # ✅ DEBUG: Track KV cache shape at each step to find where token is lost
                if session_past_kv and len(session_past_kv) > 0:
                    first_k, first_v = session_past_kv[0]
                    if first_k is not None:
                        kv_seq_len = self._get_seq_len_from_kv(first_k)
                        if step == 0:
                            logger.debug(f"[KV TRACKING] {session_id} Step {step}: input_len={step_input_ids.shape[1]}, model_kv_seq_len={kv_seq_len}, first_k.shape={first_k.shape}")
                        elif step % 5 == 0 or step == max_new_tokens - 1:  # Log every 5 steps and last step
                            logger.debug(f"[KV TRACKING] {session_id} Step {step}: generated_so_far={step}, model_kv_seq_len={kv_seq_len}")

                # ✅ DEBUG: Check model KV output (Scenario 1, 2, 3)
                # if step == 0 and session_past_kv:
                #     first_k, first_v = session_past_kv[0]
                #     if first_k is not None:
                #         logger.debug(f"_custom_generate] After forward step {step}: first_k.shape={first_k.shape} (batch={first_k.shape[0]})")
                #         logger.debug(f"_custom_generate] dtype: first_k={first_k.dtype}, first_v={first_v.dtype}")
                #         if first_k.shape[0] == 0:
                #             logger.error(f"❌ ERROR: KV batch size is 0! Input was [1, {step_input_ids.shape[1]}]")
                #             logger.error(f"   step_input_ids.shape={step_input_ids.shape}, logits.shape={logits.shape}")

                # Get next token logits
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]

                # DEBUG: Check top-5 predictions
                # if step <= 1:
                #     top_k = 5
                #     _, top_indices = torch.topk(next_token_logits, top_k, dim=-1)
                #     top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices[0]]
                #     print(f"  [Step {step}] Top-5 predictions: {list(zip(top_indices[0].tolist(), top_tokens))}")

                # ✅ Token selection: GREEDY DECODING (deterministic)
                # Paper requirement: "Output matches a stateless baseline (vLLM)"
                # Using greedy (argmax) ensures deterministic output - same logits = same tokens
                # This is critical for correctness verification vs vLLM baseline
                # (Sampling can be enabled later by setting PENSIEVE_USE_SAMPLING=1)
                next_token_ids = torch.argmax(next_token_logits, dim=-1)  # [1]

                # Record TTFT
                if step == 0 and not ttft_recorded:
                    ttft_recorded = True
                    ttft_per_request[req.request_id] = time.time() - generation_start_time

                # Store generated token
                token_id = next_token_ids.item()
                generated_ids[req_idx].append(token_id)

                # Debug: Print generated tokens
                # if step < 3:  # Only print first 3 tokens for debugging
                #     token_str = self.tokenizer.decode([token_id])
                #     print(f"  [Step {step}] Selected Token ID: {token_id}, Token: '{token_str}'")

                # Check for EOS
                if token_id == eos_token_id:
                    logger.debug(f"[_custom_generate] Session {session_id}: EOS reached at step {step}, generated {len(generated_ids[req_idx])} tokens")
                    break

            # Log generation summary for this session
            tokens_generated = len(generated_ids[req_idx])
            logger.debug(f"[_custom_generate] Session {session_id}: Generated {tokens_generated} tokens (max allowed: {max_new_tokens})")

            # ✅ DEBUG: Log final KV shape before storage
            if session_past_kv and len(session_past_kv) > 0:
                final_k, final_v = session_past_kv[0]
                if final_k is not None:
                    final_seq_len = self._get_seq_len_from_kv(final_k)
                    input_seq_len = req_input_ids.shape[1] if req_input_ids.dim() > 1 else 1
                    expected_final_seq = input_seq_len + tokens_generated
                    logger.debug(f"[FINAL KV] {session_id}: final_k.shape={final_k.shape}, seq_len={final_seq_len}, expected={expected_final_seq}")
                    if final_seq_len != expected_final_seq:
                        logger.error(f"❌ KV SEQ LEN MISMATCH! Expected {expected_final_seq} but got {final_seq_len}")

                        # ✅ TOKEN RECOVERY: One missing token detected, recover it
                        token_loss = expected_final_seq - final_seq_len
                        if tokens_generated > 0 and token_loss > 0:
                            logger.warning(f"[TOKEN RECOVERY] Detecting {token_loss} token loss. Attempting recovery with last generated token...")

                            try:
                                # Get the last generated token
                                last_token_id = generated_ids[req_idx][-1]
                                last_token_tensor = torch.tensor([[last_token_id]], device=model_device, dtype=torch.long)

                                # Forward pass with just the last token to get its KV representation
                                recovery_outputs = self.model(
                                    last_token_tensor,
                                    past_key_values=session_past_kv,
                                    use_cache=True,
                                    return_dict=True,
                                )

                                # Update KV cache with the recovered token
                                session_past_kv = recovery_outputs.past_key_values

                                # Verify recovery
                                if session_past_kv and len(session_past_kv) > 0:
                                    recovered_k, recovered_v = session_past_kv[0]
                                    if recovered_k is not None:
                                        recovered_seq_len = self._get_seq_len_from_kv(recovered_k)
                                        logger.debug(f"[TOKEN RECOVERY] After recovery: seq_len={recovered_seq_len}, expected={expected_final_seq}")
                                        if recovered_seq_len == expected_final_seq:
                                            logger.info(f"✅ TOKEN RECOVERY SUCCESS! Recovered {token_loss} missing token(s). Now has {recovered_seq_len} tokens")
                                        else:
                                            logger.warning(f"⚠️ TOKEN RECOVERY INCOMPLETE: Still {expected_final_seq - recovered_seq_len} tokens short")
                            except Exception as e:
                                logger.error(f"❌ TOKEN RECOVERY FAILED: {e}")

            # Cache status after generation
            if _cache_debug_enabled and pensieve_cache is not None:
                try:
                    stats = pensieve_cache.get_statistics()
                    logger.debug(f"  Cache after session: GPU={stats.num_gpu_chunks} chunks ({stats.gpu_used_bytes/(1024**2):.1f}MB) | "
                                f"CPU={stats.num_cpu_chunks} chunks ({stats.cpu_used_bytes/(1024**2):.1f}MB) | "
                                f"Drops={stats.num_dropped_chunks} | "
                                f"GPU hits={stats.gpu_hit_count}, CPU hits={stats.cpu_hit_count}, Misses={stats.miss_count}")
                except Exception as e:
                    logger.debug(f"  (Could not get cache stats: {e})")

            # ✅ Store final KV for this session (already per-session, no req_idx needed)
            # CRITICAL: session_past_kv is already [1, num_heads, seq, head_dim] from per-session processing
            # Do NOT store req_idx - it will cause batch extraction issues when req_idx > 0
            final_past_kv_per_session[session_id] = session_past_kv
            # ✅ DEBUG: Confirm tuple packing
            # if session_past_kv and len(session_past_kv) > 0:
            #     first_k, first_v = session_past_kv[0]
            #     if first_k is not None:
            #         logger.debug(f"_custom_generate] Packed session_id={session_id}, kv[0].shape={first_k.shape}")

        # ✅ BATCH SUMMARY: Multi-token generation overview
        if _cache_debug_enabled:
            total_generated = sum(len(ids) for ids in generated_ids)
            avg_tokens = total_generated / batch_size if batch_size > 0 else 0
            logger.debug(f"\n[BATCH GENERATION COMPLETE]")
            logger.debug(f"  Batch size: {batch_size} sessions")
            logger.debug(f"  Total tokens generated: {total_generated}")
            logger.debug(f"  Average tokens per session: {avg_tokens:.1f}")
            if pensieve_cache is not None:
                try:
                    stats = pensieve_cache.get_statistics()
                    logger.debug(f"  Final Cache State:")
                    logger.debug(f"    GPU: {stats.num_gpu_chunks} chunks ({stats.gpu_used_bytes/(1024**2):.1f}MB)")
                    logger.debug(f"    CPU: {stats.num_cpu_chunks} chunks ({stats.cpu_used_bytes/(1024**2):.1f}MB)")
                    logger.debug(f"    Dropped: {stats.num_dropped_chunks} chunks")
                    logger.debug(f"    Hit Rate: GPU={stats.gpu_hit_rate:.1%} CPU={stats.cpu_hit_rate:.1%} Misses={stats.miss_rate:.1%}")
                except Exception as e:
                    logger.debug(f"    (Could not get stats: {e})")
            logger.debug("")

        # Reconstruct sequences
        all_sequences = []
        for i in range(batch_size):
            full_seq = torch.cat(
                [
                    input_ids[i],
                    torch.tensor(
                        generated_ids[i], device=device, dtype=input_ids.dtype
                    ),
                ]
            )
            req = batch.requests[i]
            if _cache_debug_enabled:
                logger.debug(f"[SEQ {i}] req_id={req.request_id}, input={len(input_ids[i])}, generated={len(generated_ids[i])}, total={len(full_seq)}")
            else:
                logger.debug(f"[_custom_generate SUMMARY] req_id={req.request_id}, input_len={len(input_ids[i])}, generated_len={len(generated_ids[i])}, total_len={len(full_seq)}, max_allowed={max_new_tokens}")
            all_sequences.append(full_seq)

        # Pad to same length
        max_len = max(len(s) for s in all_sequences)
        padded_sequences = []
        for seq in all_sequences:
            if len(seq) < max_len:
                padding = torch.full(
                    (max_len - len(seq),),
                    self.tokenizer.pad_token_id or 0,
                    device=device,
                    dtype=seq.dtype,
                )
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)

        sequences = torch.stack(padded_sequences)

        # ✅ CRITICAL: Return generated_ids_per_request directly (already computed, ground truth)
        # This avoids trying to extract from padded batch sequences which causes token misalignment
        generated_ids_per_request = [torch.tensor(ids, device=device, dtype=input_ids.dtype) for ids in generated_ids]

        return type("obj", (object,), {
            "sequences": sequences,
            "past_key_values": final_past_kv_per_session,  # ✅ Return per-session KV!
            "ttft": ttft_per_request,
            "original_input_lengths": original_input_lengths,  # ✅ Keep for reference
            "generated_ids_per_request": generated_ids_per_request  # ✅ NEW: Direct token IDs (ground truth)
        })()

    def _execute_cache_plan(self, cache_plan: CachePlan, batch: Batch = None) -> None:
        """Execute swap operations from cache plan.

        Steps:
        1. Swap out chunks (GPU → CPU) to make space
        2. Swap in chunks (CPU → GPU) for this batch
        3. Batch-level recovery of dropped chunks (respects all dependencies)

        Args:
            cache_plan: Cache operations to execute
            batch: Current batch (needed for recovery)
        """
        # 1. Swap out chunks first (GPU → CPU)
        for chunk_key in cache_plan.chunks_to_swap_out:
            try:
                self.cache.evict_to_cpu(chunk_key)
            except Exception as e:
                print(f"Warning: Failed to evict {chunk_key}: {e}")

        # 2. Swap in chunks (CPU → GPU)
        for chunk_key in cache_plan.chunks_to_swap_in:
            try:
                self.cache.swap_chunk_to_gpu(chunk_key)
            except Exception as e:
                print(f"Warning: Failed to swap in {chunk_key}: {e}")

        # 3. ✅ Batch-level recovery with full context dependency
        # BatchedRecoveryManager handles multiple sessions efficiently,
        # respecting both layer-wise and token-wise dependencies
        if cache_plan.chunks_to_recompute and self.batched_recovery_manager and batch:
            print(
                f"🔧 Batch Recovery: {len(cache_plan.chunks_to_recompute)} "
                f"sessions need dropped chunk recovery"
            )

            # Batch-level recovery: Process all sessions' dropped chunks together
            # Each session's recovery respects:
            # - Layer dependency: previous layers' cached KV passed as context
            # - Token dependency: previous chunks loaded before current chunk recovery
            recovery_results = self.batched_recovery_manager.recover_batch(
                batch.requests
            )

            if recovery_results:
                recovered_count = sum(
                    1 for plan in recovery_results.values() if plan is not None
                )
                print(f"✓ Recovered {recovered_count} requests with dropped chunks")

    def _prepare_batch_inputs(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Prepare batch inputs for model.

        CRITICAL: Sessions must NOT attend to each other!

        Strategy:
        - Process each session independently
        - Create separate attention masks to block cross-session attention
        - Each session only attends to its own tokens

        Why separate processing?
        - Different sessions have different contexts and questions
        - Allowing cross-session attention would mix semantics
        - Session 1 should not be influenced by Session 2's tokens
        - Even with different embeddings, attention weights are non-zero
        - Average cross-session interference: 5-15% without masking

        Args:
            batch: Batch with requests (each from different session)

        Returns:
            (input_ids, attention_mask, original_input_lengths) tensors
        """
        # Track original input lengths BEFORE padding
        original_input_lengths = []

        # Find max sequence length
        max_len = 0
        for req in batch.requests:
            seq_len = len(req.input_ids) if req.input_ids.dim() > 0 else 0
            original_input_lengths.append(seq_len)
            max_len = max(max_len, seq_len)

        # ✅ DEBUG: Check for empty input scenario
        # logger.debug(f"_prepare_batch_inputs] batch_size={len(batch.requests)}, max_seq_len={max_len}")
        # if max_len == 0:
        #     logger.warning(f"⚠️ WARNING: All requests have empty input_ids! batch_size={len(batch.requests)}")
        #     for i, req in enumerate(batch.requests):
        #         logger.warning(f"   Request {i}: input_ids.shape={req.input_ids.shape}, dim={req.input_ids.dim()}")

        input_ids_list = []

        # Pad all sequences to max length
        for req in batch.requests:
            if req.input_ids.dim() == 0:
                # Handle scalar tensor
                input_ids = req.input_ids.unsqueeze(0)
            else:
                input_ids = req.input_ids

            # Pad to max_len
            if len(input_ids) < max_len:
                padding_len = max_len - len(input_ids)
                padding = torch.full(
                    (padding_len,),
                    self.tokenizer.pad_token_id or 0,
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, padding])

            input_ids_list.append(input_ids)

        # Stack into batch tensor
        batch_input_ids = torch.stack(input_ids_list)
        batch_input_ids = batch_input_ids.to(self.device)

        # Create 2D attention mask for padding positions
        # (Each session only attends to its own real tokens + padding is ignored)
        batch_size = len(batch.requests)
        batch_attention_mask = torch.ones(batch_size, max_len, dtype=torch.long)

        for i, req in enumerate(batch.requests):
            seq_len = (
                len(req.input_ids) if req.input_ids.dim() > 0 else 1
            )
            # Mask out padding (0 = ignore, 1 = attend)
            batch_attention_mask[i, seq_len:] = 0

        batch_attention_mask = batch_attention_mask.to(self.device)

        return batch_input_ids, batch_attention_mask, original_input_lengths

    def _process_outputs(
        self,
        batch: Batch,
        outputs,
    ) -> BatchResult:
        """Process model outputs and store new KV chunks.

        Steps:
        1. Extract generated token sequences
        2. Update request states (generated_tokens, finished)
        3. Extract KV cache from model's internal state
        4. Split into chunks (32 tokens each)
        5. Store chunks in cache (GPU tier)

        Args:
            batch: Original batch
            outputs: Model generation outputs

        Returns:
            BatchResult with generated tokens
        """
        result = BatchResult(batch_id=batch.batch_id)

        # 1. Extract generated tokens per request
        # ✅ CRITICAL: Use generated_ids_per_request directly (computed during generation, ground truth)
        # This avoids trying to extract from padded batch sequences which causes token misalignment
        if hasattr(outputs, "generated_ids_per_request") and outputs.generated_ids_per_request:
            generated_ids_per_request = outputs.generated_ids_per_request
        else:
            # Fallback to old extraction method (shouldn't happen with new code)
            generated_sequences = outputs.sequences
            generated_ids_per_request = []
            for i, req in enumerate(batch.requests):
                input_len = len(req.input_ids) if req.input_ids.dim() > 0 else 1
                generated_ids = generated_sequences[i][input_len:]
                generated_ids_per_request.append(generated_ids)

        for i, req in enumerate(batch.requests):
            # Use the ground-truth generated token IDs
            generated_ids = generated_ids_per_request[i]

            # Decode to string
            response_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # DEBUG: Token counting verification
            input_len = len(req.input_ids) if req.input_ids.dim() > 0 else 1
            logger.debug(f"[Pensieve DEBUG FIXED] req_id={req.request_id}, req_idx={i}, input_len={input_len}, generated_tokens={len(generated_ids)}, response_len={len(response_text)}")
            # if len(generated_ids) > 20:
            #     logger.debug(f"  First 20 tokens: {generated_ids[:20].tolist()}")

            # Update request
            req.generated_tokens = generated_ids.tolist()
            req.finished = True

            # Store in results
            result.request_results[req.request_id] = {
                "response": response_text,
                "tokens_generated": len(generated_ids),
                "finished": True,
            }

        # 2. Store KV cache for each session (from custom generation loop)
        # ✅ FIXED: custom_generate() now returns per-session KV with batch index
        if hasattr(outputs, "past_key_values") and outputs.past_key_values:
            try:
                # past_key_values is now a dict: {session_id: (req_idx, final_past_kv)}
                past_kv_dict = outputs.past_key_values
                if isinstance(past_kv_dict, dict):
                    # Per-session KV storage (no batch extraction needed - already per-session)
                    for session_id, session_kv in past_kv_dict.items():
                        if session_kv:
                            # ✅ FIXED: session_kv is already [1, num_heads, seq, head_dim] from per-session processing
                            # Do NOT slice by req_idx - it's already per-session!
                            # DEBUG: Print first layer shape
                            first_k, _ = session_kv[0]
                            if first_k is not None:
                                logger.debug(f"_process_outputs] session_id={session_id}, kv[0].shape={first_k.shape} (already per-session, batch_size={first_k.shape[0]})")

                            # Store directly without slicing
                            self._store_new_kv_chunks(batch, session_kv, session_id)
                else:
                    # Fallback for old code path (shouldn't happen now)
                    self._store_new_kv_chunks(batch, past_kv_dict)
            except Exception as e:
                print(f"Warning: Failed to store new KV chunks: {e}")

        return result

    def _store_new_kv_chunks(
        self,
        batch: Batch,
        past_key_values,
        target_session_id: str = None,  # ✅ NEW: Store only this session's KV
    ) -> None:
        """Store newly generated KV chunks in cache with last chunk merge.
        
        ✅ CRITICAL: Handles incomplete last chunks correctly!

        Process:
        1. For each layer, extract key and value tensors
        2. Merge with incomplete last chunk (if exists)
        3. Split into 32-token chunks
        4. Create KVChunk objects with correct context_length
        5. Store in GPU cache

        Example (Turn 2 after Turn 1 with 145 tokens):
        - Turn 1: chunks 0-3 (128), chunk 4 (17 tokens)
        - Turn 2: generate 30 new tokens
        - Last chunk needs: 32 - 17 = 15 more tokens
        - So: take 15 from new 30 → fill chunk 4
        - Remaining: 15 new tokens → create chunk 5

        Args:
            batch: Batch that was executed
            past_key_values: Model's KV cache (tuple per layer)
            target_session_id: If specified, only store KV for this session
        """
        if not past_key_values:
            return

        chunk_size = self.cache.CHUNK_SIZE

        # ✅ If target_session_id specified, find that request only
        target_reqs = []
        if target_session_id:
            for req in batch.requests:
                if req.session_id == target_session_id:
                    target_reqs.append(req)
        else:
            target_reqs = batch.requests

        for req in target_reqs:
            session_id = req.session_id

            # ✅ KEY: Extract ONLY newly generated tokens from past_key_values
            # past_key_values contains: [cached_from_history + input + newly_generated]
            # We need to find where newly_generated starts

            input_len = len(req.input_ids) if req.input_ids.dim() > 0 else 1
            num_generated = len(req.generated_tokens)

            # ✅ DEBUG: Log actual input/generated counts
            logger.debug(f"[DEBUG _store] session_id={session_id}: req.input_ids.shape={req.input_ids.shape}, len={input_len}")
            logger.debug(f"[DEBUG _store] session_id={session_id}: req.generated_tokens len={num_generated}, expected_total={input_len + num_generated}")

            # ✅ DEBUG: Will compare expected vs actual after we see model output
            expected_total_tokens = input_len + num_generated

            # ✅ DEBUG: Check input/generated token counts (Scenario 3)
            # logger.debug(f"_store_new_kv_chunks] session_id={session_id}: input_len={input_len}, num_generated={num_generated}")

            if num_generated == 0:
                # No new tokens generated
                # logger.debug(f"_store_new_kv_chunks] session_id={session_id}: No tokens generated, returning early")
                return

            # Determine new chunk_ids based on existing chunks
            existing_positions = self.cache.get_session_positions(session_id)

            # ✅ NEW: Get actual last chunk size from SessionMetadata
            metadata = self.cache.get_session_metadata(session_id)
            last_chunk_id = max(existing_positions) if existing_positions else -1
            last_chunk_size = metadata.last_chunk_size if metadata else 32

            # Calculate how many tokens needed to complete last chunk
            remaining_to_fill_last = 0 if last_chunk_id == -1 else (chunk_size - last_chunk_size)

            # Split new tokens: some fill last chunk, rest create new chunks
            # Note: tokens_to_store will be calculated below after we extract KV tensors
            if remaining_to_fill_last > 0 and num_generated > 0:
                # How many new tokens can fill the last chunk?
                fill_last = min(remaining_to_fill_last, num_generated)
                remaining_new = num_generated - fill_last  # After filling last chunk (will be recalculated below)
                next_chunk_id = last_chunk_id + 1  # Next new chunk after last_chunk_id
            else:
                # Last chunk doesn't exist or is already full
                fill_last = 0
                remaining_new = num_generated  # Will be recalculated below for Turn 1
                next_chunk_id = last_chunk_id + 1 if last_chunk_id >= 0 else 0

            # ✅ DEBUG: Track chunk allocation
            logger.debug(f"[DEBUG _store_new_kv_chunks] session_id={session_id}, num_generated={num_generated}, fill_last={fill_last}, remaining_new={remaining_new}, last_chunk_id={last_chunk_id}, existing_positions={existing_positions}")

            # ✅ Calculate actual context_length considering metadata
            if metadata:
                actual_context_before = metadata.total_tokens - last_chunk_size
            else:
                actual_context_before = len(existing_positions) * chunk_size

            # Total tokens after this generation
            total_tokens = metadata.total_tokens + num_generated if metadata else (len(existing_positions) * chunk_size + input_len + num_generated)
            total_chunks = (total_tokens + chunk_size - 1) // chunk_size

            # ✅ DEBUG: Log after all values calculated
            logger.debug(f"[DEBUG _store_new_kv_chunks] metadata exists: {metadata is not None}, actual_context_before={actual_context_before}, total_tokens={total_tokens}, total_chunks={total_chunks}")

            # Process each layer and split into 32-token chunks
            for layer_idx, (k, v) in enumerate(past_key_values):
                if k is None or v is None:
                    continue
                print("sj @@@@ layer_idx",layer_idx)
                # ✅ DEBUG: Print actual shapes to diagnose mismatch
                if layer_idx == 0:
                    # logger.debug(f"Layer {layer_idx}: k.shape={k.shape}, v.shape={v.shape}")
                    # logger.debug(f"num_generated={num_generated}, fill_last={fill_last}")
                    if fill_last > 0 and last_chunk_id >= 0:
                        last_chunk_key = f"{session_id}:chunk:{last_chunk_id}:layer:{layer_idx}"
                        last_chunk = self.cache.get_chunk(last_chunk_key)
                        # if last_chunk:
                        #     logger.debug(f"last_chunk.key_tensor.shape={last_chunk.key_tensor.shape}")

                # k, v shapes: [batch, num_heads, seq_len, head_dim]
                # (HuggingFace format for some models/versions)
                # seq_len (dim=2) includes everything: prev_context + input + new_generated

                # Calculate where new tokens start
                total_seq_len = self._get_seq_len_from_kv(k)  # Total sequence length

                # ✅ CRITICAL FIX: On Turn 1 (no existing chunks), store ALL tokens, not just generated
                # Turn 1: existing_positions=[] → store everything (input + generated)
                # Turn 2+: existing_positions=[...] → store only newly generated
                if not existing_positions:
                    # Turn 1: Store ALL tokens from the beginning
                    new_tokens_start = 0
                else:
                    # Turn 2+: Store only newly generated tokens
                    new_tokens_start = total_seq_len - num_generated

                # Extract tokens to store - use correct dimension based on tensor format
                seq_dim = self._get_seq_dim_from_kv(k)

                if seq_dim == 2:
                    # Gemma format: [batch, heads, seq, head_dim]
                    new_key = k[:, :, new_tokens_start:, :]  # [batch, heads, tokens_to_store, head_dim]
                    new_value = v[:, :, new_tokens_start:, :]  # [batch, heads, tokens_to_store, head_dim]
                else:
                    # Standard format: [batch, seq, heads, head_dim]
                    new_key = k[:, new_tokens_start:, :, :]  # [batch, tokens_to_store, heads, head_dim]
                    new_value = v[:, new_tokens_start:, :, :]  # [batch, tokens_to_store, heads, head_dim]

                # ✅ DEBUG for Turn 2+: Verify extracted token count
                if existing_positions and layer_idx == 0:
                    tokens_extracted = self._get_seq_len_from_kv(new_key)
                    logger.debug(f"[DEBUG TURN 2+] session_id={session_id}: num_generated={num_generated}, total_seq_len={total_seq_len}, new_tokens_start={new_tokens_start}, tokens_extracted={tokens_extracted}")
                    if tokens_extracted != num_generated:
                        logger.error(f"❌ TURN 2+ TOKEN MISMATCH! Expected to extract {num_generated} new tokens but got {tokens_extracted}")

                # ✅ RECALCULATE remaining_new for Turn 1 case
                if not existing_positions:
                    # Turn 1: We're storing ALL tokens (total_seq_len), not just generated
                    tokens_stored = self._get_seq_len_from_kv(new_key)  # Actual tokens in extracted tensor
                    if layer_idx == 0:  # Only recalculate once per request
                        logger.debug(f"[DEBUG TURN 1] session_id={session_id}, layer_idx={layer_idx}: input_len={input_len}, num_generated={num_generated}")
                        logger.debug(f"[DEBUG TURN 1] k.shape={k.shape}, total_seq_len={total_seq_len}, new_tokens_start={new_tokens_start}")
                        logger.debug(f"[DEBUG TURN 1] new_key.shape={new_key.shape}, tokens_stored={tokens_stored}")

                        # ✅ CRITICAL: Compare expected vs actual token counts
                        if tokens_stored != expected_total_tokens:
                            logger.error(f"❌ TOKEN LOSS DETECTED! Expected {expected_total_tokens} tokens ({input_len} input + {num_generated} generated) but model output has {tokens_stored} tokens. LOSS: {expected_total_tokens - tokens_stored} token(s)")
                        else:
                            logger.debug(f"✅ Token count matches: {tokens_stored} tokens stored")

                        remaining_new = tokens_stored - fill_last
                        # Update total_tokens to reflect actual stored tokens
                        total_tokens = tokens_stored
                        total_chunks = (total_tokens + chunk_size - 1) // chunk_size
                        logger.debug(f"[DEBUG TURN 1] Recalculated: remaining_new={remaining_new}, total_tokens={total_tokens}, total_chunks={total_chunks}")

                # ✅ DEBUG: Check batch size of extracted new tokens (Scenario 1, 2, 3)
                # if layer_idx == 0:
                #     logger.debug(f"_store_new_kv_chunks] After extraction: new_key.shape={new_key.shape}, new_value.shape={new_value.shape}")
                #     logger.debug(f"_store_new_kv_chunks] dtype: new_key={new_key.dtype}, new_value={new_value.dtype}")
                #     if new_key.shape[0] == 0:
                #         logger.error(f"❌ ERROR FOUND: new_key batch size is 0!")
                #         logger.error(f"   k.shape={k.shape} (batch={k.shape[0]})")
                #         logger.error(f"   total_seq_len={total_seq_len}, new_tokens_start={new_tokens_start}, num_generated={num_generated}")
                #         logger.error(f"   This suggests k.shape[0] was already 0 from model output (Scenario 1 or 2)")

                # ✅ CRITICAL: Handle last chunk merge
                if fill_last > 0 and last_chunk_id >= 0:
                    # Get last chunk to update it
                    last_chunk_key = f"{session_id}:chunk:{last_chunk_id}:layer:{layer_idx}"
                    last_chunk = self.cache.get_chunk(last_chunk_key)

                    if layer_idx == 0:
                        logger.debug(f"[DEBUG MERGE] session_id={session_id}: fill_last={fill_last}, last_chunk exists={last_chunk is not None}")

                    if last_chunk:
                        # Extract tokens to fill last chunk (use correct dimension based on tensor format)
                        if seq_dim == 2:
                            # Gemma format: [batch, heads, seq, head_dim]
                            fill_key = new_key[:, :, :fill_last, :]  # [batch, heads, fill_last, head_dim]
                            fill_value = new_value[:, :, :fill_last, :]  # [batch, heads, fill_last, head_dim]
                        else:
                            # Standard format: [batch, seq, heads, head_dim]
                            fill_key = new_key[:, :fill_last, :, :]  # [batch, fill_last, heads, head_dim]
                            fill_value = new_value[:, :fill_last, :, :]  # [batch, fill_last, heads, head_dim]

                        # ✅ Chunks are stored on GPU for performance
                        # Both fill_key and last_chunk.key_tensor are on GPU
                        # No need to move to CPU - keep everything on GPU for faster merging

                        # ✅ DEBUG: Capture the merge scenario (just before error)
                        # if layer_idx == 0:
                        #     logger.debug(f"_store_new_kv_chunks] About to merge:")
                        #     logger.debug(f"   last_chunk.key_tensor.shape={last_chunk.key_tensor.shape} (batch={last_chunk.key_tensor.shape[0]})")
                        #     logger.debug(f"   fill_key.shape={fill_key.shape} (batch={fill_key.shape[0]})")
                        #     if last_chunk.key_tensor.shape[0] != fill_key.shape[0]:
                        #         logger.error(f"❌ BATCH MISMATCH: {last_chunk.key_tensor.shape[0]} != {fill_key.shape[0]}")
                        #         logger.error(f"   This is SCENARIO 3: Cached chunk structure mismatch with new KV")

                        # Concatenate with existing last chunk KV (concatenate along correct seq_len dimension)
                        # Both on GPU - merge on GPU for performance
                        merged_key = torch.cat(
                            [last_chunk.key_tensor, fill_key], dim=seq_dim
                        )  # Concatenate along sequence dimension
                        merged_value = torch.cat(
                            [last_chunk.value_tensor, fill_value], dim=seq_dim
                        )

                        # ✅ Update last chunk with merged KV (keep on GPU for performance)
                        merged_key_gpu = merged_key.detach()
                        merged_value_gpu = merged_value.detach()

                        # ✅ VALIDATION: Check merged tensor dimensions BEFORE storing
                        if layer_idx == 0:
                            logger.debug(f"[MERGE VALIDATION] session_id={session_id}, chunk_id={last_chunk_id}:")
                            logger.debug(f"  last_chunk.key_tensor.shape={last_chunk.key_tensor.shape}")
                            logger.debug(f"  fill_key.shape={fill_key.shape}")
                            logger.debug(f"  merged_key_gpu.shape={merged_key_gpu.shape}")

                        # Check for malformed tensors
                        if len(merged_key_gpu.shape) != 4:
                            logger.error(f"❌ Merged key has wrong dimensions! Expected 4D, got {merged_key_gpu.shape}")
                        if merged_key_gpu.shape[-1] == 0:
                            logger.error(f"❌ Merged key has head_dim=0! Shape: {merged_key_gpu.shape}")
                            raise ValueError(f"Merge operation created tensor with head_dim=0: {merged_key_gpu.shape}")

                        # ✅ DEBUG: Log dtype when merging chunks
                        # if layer_idx == 0:
                        #     logger.debug(f"_store_new_kv_chunks] Merging chunk: merged_key_gpu={merged_key_gpu.dtype}, merged_value_gpu={merged_value_gpu.dtype}")
                        #     logger.debug(f"  → last_chunk original: key={last_chunk.key_tensor.dtype}, value={last_chunk.value_tensor.dtype}")

                        updated_chunk = KVChunk(
                            session_id=session_id,
                            chunk_id=last_chunk_id,
                            layer_idx=layer_idx,
                            key_tensor=merged_key_gpu,
                            value_tensor=merged_value_gpu,
                            context_length=last_chunk.context_length,
                            session_total_chunks=total_chunks,
                            num_layers=self.num_layers,
                        )

                        try:
                            self.cache.store_chunk(updated_chunk, location=CacheLocation.GPU)
                        except Exception as e:
                            print(f"Warning: Failed to update last chunk {last_chunk_key}: {e}")

                # ✅ Process remaining new tokens as full chunks
                # Use correct dimension for tensor format
                if seq_dim == 2:
                    # Gemma format: [batch, heads, seq, head_dim]
                    remaining_key = new_key[:, :, fill_last:, :]  # [batch, heads, remaining_new, head_dim]
                    remaining_value = new_value[:, :, fill_last:, :]
                else:
                    # Standard format: [batch, seq, heads, head_dim]
                    remaining_key = new_key[:, fill_last:, :, :]  # [batch, remaining_new, heads, head_dim]
                    remaining_value = new_value[:, fill_last:, :, :]

                if layer_idx == 0 and not existing_positions:
                    logger.debug(f"[DEBUG CHUNKS] session_id={session_id}: remaining_key.shape={remaining_key.shape}, fill_last={fill_last}, remaining_new={remaining_new}")
                    logger.debug(f"[DEBUG CHUNKS] Will create {(remaining_new + chunk_size - 1) // chunk_size} chunks")

                for chunk_idx in range((remaining_new + chunk_size - 1) // chunk_size):
                    # Calculate token range for this chunk
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, remaining_new)

                    if layer_idx == 0 and not existing_positions:
                        logger.debug(f"[DEBUG CHUNKS] chunk_idx={chunk_idx}: chunk_start={chunk_start}, chunk_end={chunk_end}, extracted_size={chunk_end - chunk_start}")

                    # Extract chunk tokens (from correct seq_len dimension)
                    if seq_dim == 2:
                        # Gemma format: seq at dim=2
                        chunk_key = remaining_key[:, :, chunk_start:chunk_end, :]
                        chunk_value = remaining_value[:, :, chunk_start:chunk_end, :]
                    else:
                        # Standard format: seq at dim=1
                        chunk_key = remaining_key[:, chunk_start:chunk_end, :, :]
                        chunk_value = remaining_value[:, chunk_start:chunk_end, :, :]

                    # Determine chunk_id
                    chunk_id = next_chunk_id + chunk_idx

                    # ✅ CRITICAL: Correct context_length calculation
                    # = tokens before this chunk considering the merge
                    # = actual_context_before (완전한 이전 청크들)
                    #   + last_chunk_size (채우기 전 chunk 0)
                    #   + fill_last (chunk 0에 추가된 토큰)
                    #   + (chunk_idx * chunk_size) (현재 루프의 청크들)
                    context_length = actual_context_before + last_chunk_size + fill_last + (chunk_idx * chunk_size)

                    # Create chunk for this layer - KEEP DATA ON GPU
                    chunk_key_gpu = chunk_key.detach()
                    chunk_value_gpu = chunk_value.detach()

                    # ✅ DEBUG: Log tensor shape when storing chunks
                    if layer_idx == 0:
                        logger.debug(f"[DEBUG CHUNK STORE] session_id={session_id}, chunk_id={chunk_id}: shape={chunk_key_gpu.shape}, num_tokens_expected={chunk_end - chunk_start}")

                    # ✅ VALIDATION: Check dimension integrity BEFORE creating chunk
                    if len(chunk_key_gpu.shape) != 4:
                        logger.error(f"❌ CRITICAL: chunk_key has wrong number of dimensions! Expected 4D, got {len(chunk_key_gpu.shape)}D with shape {chunk_key_gpu.shape}")

                    # Check for head_dim=0 which causes attention failure
                    if chunk_key_gpu.shape[-1] == 0:
                        logger.error(f"❌ CRITICAL: head_dim is 0! Shape: {chunk_key_gpu.shape}")
                        logger.error(f"   seq_dim={seq_dim}, chunk_start={chunk_start}, chunk_end={chunk_end}")
                        logger.error(f"   remaining_key.shape={remaining_key.shape}")
                        raise ValueError(f"Cannot store chunk with head_dim=0: {chunk_key_gpu.shape}")

                    # Verify device
                    if chunk_key_gpu.device.type != 'cuda':
                        logger.warning(f"⚠️ Chunk tensor is not on CUDA! Device: {chunk_key_gpu.device}")

                    chunk = KVChunk(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        layer_idx=layer_idx,
                        key_tensor=chunk_key_gpu,
                        value_tensor=chunk_value_gpu,
                        context_length=context_length,
                        session_total_chunks=total_chunks,
                        num_layers=self.num_layers,
                    )

                    # ✅ DEBUG: Verify num_tokens calculation
                    if layer_idx == 0:
                        logger.debug(f"[DEBUG CHUNK STORE] chunk.num_tokens={chunk.num_tokens} (should be {chunk_end - chunk_start})")

                    # ✅ VALIDATION: After chunk creation, verify it was stored correctly
                    if chunk.num_tokens != (chunk_end - chunk_start):
                        logger.error(f"❌ CHUNK NUM_TOKENS MISMATCH! Expected {chunk_end - chunk_start}, got {chunk.num_tokens}")
                        logger.error(f"   key_tensor.shape={chunk.key_tensor.shape}, value_tensor.shape={chunk.value_tensor.shape}")

                    # Store in cache
                    try:
                        self.cache.store_chunk(chunk, location=CacheLocation.GPU)
                    except Exception as e:
                        print(f"Warning: Failed to store chunk {chunk.key}: {e}")

            # ✅ CRITICAL: Update SessionMetadata with new token count
            # This is essential for:
            # - eviction policy to use correct position weights (SessionMetadata.total_chunks)
            # - recovery to know exact last chunk size (SessionMetadata.last_chunk_size)
            # - context_length calculation to be accurate in next turn
            #
            # Note: Each session appears only once per batch due to pinned_sessions logic,
            # so each session_id update is called exactly once per batch execution.
            self.cache.update_session_tokens(
                session_id=session_id,
                input_tokens=input_len,
                generated_tokens=num_generated,
            )

    def reset(self) -> None:
        """Reset worker state (if any)."""
        # Worker is stateless, nothing to reset
        pass
