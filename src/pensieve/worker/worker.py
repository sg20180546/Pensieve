"""GPU execution worker for Pensieve inference batches."""

import torch
import time
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

        # Model config
        self.num_layers = (
            model.config.num_hidden_layers
            if hasattr(model.config, "num_hidden_layers")
            else model.config.n_layer
        )
        self.hidden_size = model.config.hidden_size

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

        # 1. PIN all sessions in this batch to protect from concurrent eviction
        session_ids = [req.session_id for req in batch.requests]
        for session_id in session_ids:
            self.cache.pin_session(session_id)

        try:
            # 2. Execute cache swaps (including recovery)
            self._execute_cache_plan(cache_plan, batch)

            # 3. Prepare batch inputs
            input_ids, attention_mask = self._prepare_batch_inputs(batch)

            # 4. Create custom cache for this batch
            pensieve_cache = PensieveCacheFactory.create(
                cache_manager=self.cache,
                batch_requests=batch.requests,
                num_layers=self.num_layers,
            )

            # 5. Run custom generation loop with KV cache integration
            with torch.no_grad():
                try:
                    outputs = self._custom_generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pensieve_cache=pensieve_cache,
                        batch=batch,
                        max_new_tokens=32,
                    )
                except Exception as e:
                    print(f"Error during custom generation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return empty result on error
                    return BatchResult(batch_id=batch.batch_id)

            # 6. Extract generated tokens and store new KV chunks
            results = self._process_outputs(batch, outputs)

            elapsed = time.time() - start_time
            results.execution_time = elapsed

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
            input_ids: [batch_size, seq_len] input tokens
            attention_mask: [batch_size, seq_len] attention mask
            pensieve_cache: PensieveCache instance (maps session_id to cache)
            batch: Original batch
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with:
            - sequences: [batch_size, seq_len+max_new_tokens] all tokens
            - past_key_values: Final KV cache (not used in per-session mode)
            - ttft: Dict of TTFT per request_id (seconds)
        """
        batch_size = len(batch.requests)
        device = input_ids.device
        eos_token_id = self.tokenizer.eos_token_id or 2

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

            # Get cached KV for this session (if available in pensieve_cache)
            # PensieveCache.__getitem__ returns per-session cache
            session_cache = None
            try:
                # Attempt to get session-specific cache
                # (if PensieveCache is session-aware)
                if hasattr(pensieve_cache, 'get_session_cache'):
                    session_cache = pensieve_cache.get_session_cache(session_id)
                else:
                    # Fallback: use full pensieve_cache for first step
                    session_cache = pensieve_cache
            except Exception:
                session_cache = None

            # Generation loop for this session only
            session_past_kv = None
            ttft_recorded = False

            for step in range(max_new_tokens):
                # Prepare input for this step
                if step == 0:
                    step_input_ids = req_input_ids
                    step_attention_mask = req_attention_mask
                else:
                    step_input_ids = next_token_ids.unsqueeze(1)  # [1, 1]
                    step_attention_mask = torch.ones(1, 1, device=device, dtype=torch.long)

                # Forward pass - with session-specific cache
                outputs = self.model(
                    step_input_ids,
                    attention_mask=step_attention_mask,
                    past_key_values=session_cache if step == 0 else session_past_kv,
                    use_cache=True,
                    return_dict=True,
                )

                # Extract outputs
                logits = outputs.logits
                session_past_kv = outputs.past_key_values

                # Get next token
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                next_token_ids = torch.argmax(next_token_logits, dim=-1)  # [1]

                # Record TTFT
                if step == 0 and not ttft_recorded:
                    ttft_recorded = True
                    ttft_per_request[req.request_id] = time.time() - generation_start_time

                # Store generated token
                generated_ids[req_idx].append(next_token_ids.item())

                # Check for EOS
                if next_token_ids.item() == eos_token_id:
                    break

            # ✅ Store final KV for this session
            final_past_kv_per_session[session_id] = session_past_kv

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

        return type("obj", (object,), {
            "sequences": sequences,
            "past_key_values": final_past_kv_per_session,  # ✅ Return per-session KV!
            "ttft": ttft_per_request
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            (input_ids, attention_mask) tensors
        """
        # Find max sequence length
        max_len = 0
        for req in batch.requests:
            seq_len = len(req.input_ids) if req.input_ids.dim() > 0 else 0
            max_len = max(max_len, seq_len)

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

        return batch_input_ids, batch_attention_mask

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
        generated_sequences = outputs.sequences
        input_len_per_req = [
            len(req.input_ids) if req.input_ids.dim() > 0 else 1
            for req in batch.requests
        ]

        for i, req in enumerate(batch.requests):
            # Extract tokens generated for this request
            # (Everything after the input)
            input_len = input_len_per_req[i]
            generated_ids = generated_sequences[i][input_len:]

            # Decode to string
            response_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

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
        # ✅ FIXED: custom_generate() now returns per-session KV
        if hasattr(outputs, "past_key_values") and outputs.past_key_values:
            try:
                # past_key_values is now a dict: {session_id: final_past_kv}
                past_kv_dict = outputs.past_key_values
                if isinstance(past_kv_dict, dict):
                    # Per-session KV storage
                    for req in batch.requests:
                        session_id = req.session_id
                        if session_id in past_kv_dict:
                            session_kv = past_kv_dict[session_id]
                            if session_kv:
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
        """Store newly generated KV chunks in cache.

        Process:
        1. For each layer, extract key and value tensors
        2. Split into 32-token chunks
        3. Create KVChunk objects
        4. Store in GPU cache

        Args:
            batch: Batch that was executed
            past_key_values: Model's KV cache (tuple per layer)
            target_session_id: If specified, only store KV for this session
        """
        if not past_key_values:
            return

        chunk_size = 32

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

            if num_generated == 0:
                # No new tokens generated
                return

            # Determine new chunk_ids based on existing chunks
            existing_positions = self.cache.get_session_positions(session_id)
            next_chunk_id = max(existing_positions) + 1 if existing_positions else 0

            # Calculate total tokens and chunks in session (after this generation)
            prev_context_length = sum(len(chunk) for chunk in self.cache.get_session_positions(session_id)) * 32
            total_tokens = prev_context_length + input_len + num_generated
            total_chunks = (total_tokens + 31) // 32

            # Process each layer and split into 32-token chunks
            for layer_idx, (k, v) in enumerate(past_key_values):
                if k is None or v is None:
                    continue

                # k, v shapes: [batch, seq_len, num_heads, head_dim]
                # seq_len includes everything: prev_context + input + new_generated

                # Calculate where new tokens start
                # Note: past_key_values seq_len = len(cached_context) + input_len + num_generated
                total_seq_len = k.shape[1]  # Total sequence length
                new_tokens_start = total_seq_len - num_generated

                # Extract ONLY new tokens
                new_key = k[:, new_tokens_start:, :, :]  # [batch, num_generated, heads, dim]
                new_value = v[:, new_tokens_start:, :, :]  # [batch, num_generated, heads, dim]

                # ✅ Split into 32-token chunks
                chunk_size = 32

                for chunk_idx in range((num_generated + chunk_size - 1) // chunk_size):
                    # Calculate token range for this chunk
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, num_generated)

                    # Extract chunk tokens
                    chunk_key = new_key[:, chunk_start:chunk_end, :, :]
                    chunk_value = new_value[:, chunk_start:chunk_end, :, :]

                    # Determine chunk_id
                    # new_chunk_id = chunk from which position?
                    # = existing tokens / 32 + chunk_idx
                    chunk_id = next_chunk_id + chunk_idx

                    # context_length = tokens before this chunk
                    context_length = prev_context_length + (chunk_idx * chunk_size)

                    # Create chunk for this layer
                    chunk = KVChunk(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        layer_idx=layer_idx,
                        key_tensor=chunk_key.detach().cpu(),
                        value_tensor=chunk_value.detach().cpu(),
                        context_length=context_length,
                        session_total_chunks=total_chunks,
                        num_layers=self.num_layers,
                    )

                    # Store in cache
                    try:
                        self.cache.store_chunk(chunk, location=CacheLocation.GPU)
                    except Exception as e:
                        print(f"Warning: Failed to store chunk {chunk.key}: {e}")

    def reset(self) -> None:
        """Reset worker state (if any)."""
        # Worker is stateless, nothing to reset
        pass
