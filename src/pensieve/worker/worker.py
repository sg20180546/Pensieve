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
    ):
        """Initialize worker.

        Args:
            model: HuggingFace language model (already loaded)
            tokenizer: Tokenizer for encoding/decoding
            cache: TwoTierCache instance
            device: GPU device string
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.device = device

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
        1. Execute cache swaps from plan (GPU ←→ CPU, CPU → DROPPED)
        2. Handle dropped token recovery (if any)
        3. Create PensieveCache for this batch
        4. Prepare batch inputs
        5. Run custom generation loop with proper KV cache integration
        6. Extract and store new KV chunks
        7. Return results

        Args:
            batch: Batch to execute
            cache_plan: Cache swap operations

        Returns:
            BatchResult with generated tokens and statistics
        """
        start_time = time.time()

        # 1. Execute cache swaps
        self._execute_cache_plan(cache_plan)

        # 2. Prepare batch inputs
        input_ids, attention_mask = self._prepare_batch_inputs(batch)

        # 3. Create custom cache for this batch
        pensieve_cache = PensieveCacheFactory.create(
            cache_manager=self.cache,
            batch_requests=batch.requests,
            num_layers=self.num_layers,
        )

        # 4. Run custom generation loop with KV cache integration
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

        # 5. Extract generated tokens and store new KV chunks
        results = self._process_outputs(batch, outputs)

        elapsed = time.time() - start_time
        results.execution_time = elapsed

        # Store TTFT (Time To First Token) per request if available
        if hasattr(outputs, 'ttft') and outputs.ttft:
            results.ttft_per_request = outputs.ttft

        return results

    def _custom_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pensieve_cache,
        batch: Batch,  # For future use in multi-request coordination
        max_new_tokens: int = 32,
    ) -> Dict:
        """Custom generation loop with proper HuggingFace KV cache integration.

        This replaces model.generate() to have full control over KV cache handling.

        Tracks TTFT (Time To First Token) for each request in the batch.

        Steps per token:
        1. Forward pass with current input and cached KV
        2. Extract logits for last token
        3. Select next token (greedy)
        4. Extract new KV from model state
        5. Update PensieveCache with new KV
        6. Accumulate results

        Args:
            input_ids: [batch_size, seq_len] input tokens
            attention_mask: [batch_size, seq_len] attention mask
            pensieve_cache: PensieveCache instance
            batch: Original batch
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with:
            - sequences: [batch_size, seq_len+max_new_tokens] all tokens
            - past_key_values: Final KV cache from model
            - ttft: Dict of TTFT per request_id (seconds)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        eos_token_id = self.tokenizer.eos_token_id or 2  # Default to 2 (GPT-2 EOS)

        # Track generated tokens per request
        generated_ids = [[] for _ in range(batch_size)]

        # Track TTFT (Time To First Token) - measured when first token is generated
        ttft_per_request = {}
        generation_start_time = time.time()

        # Initialize past_key_values from cache
        past_key_values = None

        # Generation loop
        for step in range(max_new_tokens):
            # Prepare input for this step
            if step == 0:
                # First step: use full input (prefill phase)
                step_input_ids = input_ids
            else:
                # Subsequent steps: only last generated token (decoding phase)
                step_input_ids = next_token_ids.unsqueeze(1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                batch_size, 1, device=device, dtype=attention_mask.dtype
                            ),
                        ],
                        dim=1,
                    )

            # Forward pass
            outputs = self.model(
                step_input_ids,
                attention_mask=attention_mask,
                past_key_values=pensieve_cache if step == 0 else past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # Extract logits and KV
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Get next token (greedy: argmax of last position)
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)  # [batch_size]

            # Record TTFT for each request on their first generated token (step == 0)
            if step == 0:
                first_token_time = time.time()
                for i, req in enumerate(batch.requests):
                    ttft_per_request[req.request_id] = first_token_time - generation_start_time

            # Update generated tokens
            for i in range(batch_size):
                generated_ids[i].append(next_token_ids[i].item())

            # Check for EOS
            if (next_token_ids == eos_token_id).all():
                break

        # Concatenate all tokens
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

        # Return in format compatible with model.generate()
        return type("obj", (object,), {
            "sequences": sequences,
            "past_key_values": past_key_values,
            "ttft": ttft_per_request  # TTFT per request_id
        })()

    def _execute_cache_plan(self, cache_plan: CachePlan) -> None:
        """Execute swap operations from cache plan.

        Steps:
        1. Swap out chunks (GPU → CPU) to make space
        2. Swap in chunks (CPU → GPU) for this batch
        3. Handle eviction chain if needed

        Args:
            cache_plan: Cache operations to execute
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

        # 3. Dropped chunks will be handled during recovery (Phase 4.5)
        # For now, just log what's dropped
        if cache_plan.chunks_to_recompute:
            print(
                f"Note: {len(cache_plan.chunks_to_recompute)} sessions have dropped chunks"
            )

    def _prepare_batch_inputs(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch inputs for model.

        Strategy:
        - Concatenate input_ids from all requests
        - Create attention_mask accounting for batch composition
        - Handle variable sequence lengths (ragged batch)

        For now: Pad to max length (simple)
        Later: Use nested tensors or custom masking

        Args:
            batch: Batch with requests

        Returns:
            (input_ids, attention_mask) tensors
        """
        input_ids_list = []
        max_len = 0

        # Find max sequence length
        for req in batch.requests:
            seq_len = len(req.input_ids) if req.input_ids.dim() > 0 else 0
            max_len = max(max_len, seq_len)

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

        # Create attention mask (1 for real tokens, 0 for padding)
        batch_attention_mask = torch.ones_like(batch_input_ids)
        for i, req in enumerate(batch.requests):
            seq_len = (
                len(req.input_ids) if req.input_ids.dim() > 0 else 1
            )
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

        # 2. Try to extract and store KV cache (if available in outputs)
        # Note: HuggingFace model.generate() doesn't return past_key_values by default
        # This is a limitation that Phase 4.3 custom generation loop will fix
        if hasattr(outputs, "past_key_values") and outputs.past_key_values:
            try:
                self._store_new_kv_chunks(batch, outputs.past_key_values)
            except Exception as e:
                print(f"Warning: Failed to store new KV chunks: {e}")

        return result

    def _store_new_kv_chunks(
        self,
        batch: Batch,
        past_key_values,
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
        """
        if not past_key_values:
            return

        chunk_size = 32

        for req in batch.requests:
            session_id = req.session_id

            # Determine new chunk_id based on existing chunks
            existing_positions = self.cache.get_session_positions(session_id)
            next_chunk_id = max(existing_positions) + 1 if existing_positions else 0

            # Determine total chunks in session
            # (In real system, would track this separately)
            total_chunks = next_chunk_id + 1

            # Extract context_length (tokens before this new chunk)
            context_length = req.seq_len - len(req.generated_tokens)

            # Process each layer
            for layer_idx, (k, v) in enumerate(past_key_values):
                if k is None or v is None:
                    continue

                # k, v shapes: [batch, seq_len, num_heads, head_dim]
                # Extract for this request in batch (for simplicity, store full tensors)

                # Create chunk for this layer
                chunk = KVChunk(
                    session_id=session_id,
                    chunk_id=next_chunk_id,
                    layer_idx=layer_idx,
                    key_tensor=k.detach().cpu(),  # Store on CPU, will swap as needed
                    value_tensor=v.detach().cpu(),
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
