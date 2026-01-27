"""Dropped token recovery manager for Pensieve.

When KV cache chunks are evicted (dropped), we need to recompute them when
the session returns to avoid correctness issues.

Key challenge (Paper §4.3.4):
- Dropped chunks create NON-CONSECUTIVE regions in cache
- Must handle multiple compute regions in single forward pass
- Efficient recovery of leading tokens (which are cheap to recompute)

Design:
- Check if request has dropped chunks
- Recompute dropped tokens (without KV cache)
- Merge with new prompt for attention computation
- Store recovered chunks in GPU cache
"""

import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from pensieve.core.types import (
    Request,
    KVChunk,
    CacheLocation,
)
from pensieve.core.cache import TwoTierCache


@dataclass
class RecoveryPlan:
    """Plan for recovering dropped tokens.

    Stores information about which chunks need to be recomputed
    and what tokens to use for recomputation.
    """

    dropped_positions: List[int]  # Chunk positions that are dropped
    raw_tokens: torch.Tensor  # Token IDs to recompute
    session_id: str  # Session these tokens belong to


class TokenRecoveryManager:
    """Manages recomputation of dropped KV chunks.

    Key insight (Paper Figure 5):
    - When session returns after eviction, leading chunks are cheapest to recover
    - Recompute dropped tokens in batch before main forward pass
    - Merge recovered KV with cached KV for full context

    Design:
    - Per-request: Check for dropped chunks
    - Create recovery plan with raw tokens
    - Recompute via small forward pass
    - Store recovered chunks back in cache
    """

    def __init__(
        self,
        model,
        tokenizer,
        cache: TwoTierCache,
        device: str = "cuda:0",
    ):
        """Initialize recovery manager.

        Args:
            model: HuggingFace language model
            tokenizer: Tokenizer for encoding
            cache: TwoTierCache instance
            device: GPU device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.device = device
        self.num_layers = (
            model.config.num_hidden_layers
            if hasattr(model.config, "num_hidden_layers")
            else model.config.n_layer
        )

    def create_recovery_plan(
        self,
        request: Request,
    ) -> Optional[RecoveryPlan]:
        """Check if request needs dropped token recovery.

        Process:
        1. Get all positions in session
        2. Check if any chunks are DROPPED
        3. If yes, create recovery plan with raw tokens

        Args:
            request: Request to check

        Returns:
            RecoveryPlan if chunks are dropped, None otherwise
        """
        session_id = request.session_id
        dropped_positions = []

        # Get all positions in this session
        session_positions = self.cache.get_session_positions(session_id)

        # Check each position for dropped chunks
        for pos in session_positions:
            # Check layer 0 as representative (all layers at position drop together)
            chunk_key = f"{session_id}:chunk:{pos}:layer:0"
            chunk = self.cache.get_chunk(chunk_key)

            if chunk is None:
                # Check dropped chunks
                chunk = self.cache.dropped_chunks.get(chunk_key)
                if chunk and chunk.location == CacheLocation.DROPPED:
                    dropped_positions.append(pos)

        if not dropped_positions:
            return None

        # Get raw tokens for dropped positions
        raw_tokens = self._fetch_raw_tokens(session_id, dropped_positions)

        if raw_tokens is None or len(raw_tokens) == 0:
            return None

        return RecoveryPlan(
            dropped_positions=dropped_positions,
            raw_tokens=raw_tokens,
            session_id=session_id,
        )

    def recompute_dropped_chunks(
        self,
        recovery_plan: RecoveryPlan,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Recompute dropped chunks and store in cache.

        Process:
        1. Run forward pass on dropped tokens (no past_key_values)
        2. Extract KV cache from model's internal state
        3. Store chunks in GPU cache with proper metadata

        Args:
            recovery_plan: Recovery plan with dropped tokens

        Returns:
            Dict mapping position to (key, value) tensors
        """
        if len(recovery_plan.raw_tokens) == 0:
            return {}

        # Prepare input
        input_ids = recovery_plan.raw_tokens
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension

        input_ids = input_ids.to(self.device, dtype=torch.long)

        # Run forward pass (recompute KV, no past_key_values)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=True,
                return_dict=True,
            )

        # Extract KV cache
        past_key_values = outputs.past_key_values
        if not past_key_values:
            print(f"Warning: Model didn't return past_key_values")
            return {}

        chunk_size = 32
        stored_chunks = {}

        # Split into 32-token chunks and store
        for layer_idx, (k, v) in enumerate(past_key_values):
            if k is None or v is None:
                continue

            # k, v shape: [batch, num_heads, seq_len, head_dim]
            seq_len = k.shape[2]
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            for chunk_id_idx, dropped_pos in enumerate(
                recovery_plan.dropped_positions[:num_chunks]
            ):
                start = chunk_id_idx * chunk_size
                end = min(start + chunk_size, seq_len)

                chunk_k = k[:, :, start:end, :].cpu()
                chunk_v = v[:, :, start:end, :].cpu()

                # Create and store KVChunk
                chunk = KVChunk(
                    session_id=recovery_plan.session_id,
                    chunk_id=dropped_pos,
                    layer_idx=layer_idx,
                    key_tensor=chunk_k,
                    value_tensor=chunk_v,
                    context_length=start,
                    session_total_chunks=len(recovery_plan.dropped_positions),
                    num_layers=self.num_layers,
                )

                try:
                    # Store in GPU cache (recovery assumes space available)
                    self.cache.store_chunk(chunk, location=CacheLocation.GPU)
                    stored_chunks[dropped_pos] = (chunk_k, chunk_v)
                except Exception as e:
                    print(f"Warning: Failed to store recovered chunk: {e}")

        return stored_chunks

    def merge_for_prefill(
        self,
        request: Request,
        recovery_plan: Optional[RecoveryPlan],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Merge dropped tokens with new prompt for prefill.

        After recovery, need to process both recovered + new prompt.
        Mark boundaries for proper masking during attention.

        Layout after merge:
        [DROPPED_TOKENS | NEW_PROMPT]

        Args:
            request: Request with new input
            recovery_plan: Recovery plan (if any)

        Returns:
            (merged_input_ids, boundaries)
            - merged_input_ids: Concatenated tokens
            - boundaries: List of indices marking region boundaries
        """
        if recovery_plan is None or len(recovery_plan.raw_tokens) == 0:
            # No recovery needed
            boundaries = [len(request.input_ids) if request.input_ids.dim() > 0 else 1]
            return request.input_ids, boundaries

        # Concatenate: dropped_tokens + new_prompt
        dropped_len = len(recovery_plan.raw_tokens)
        new_prompt_len = len(request.input_ids) if request.input_ids.dim() > 0 else 1

        merged = torch.cat([recovery_plan.raw_tokens, request.input_ids])

        # Boundaries mark transitions between regions
        # [0 ... dropped_len) = recovery region
        # [dropped_len ... dropped_len+new_prompt_len) = new prompt region
        boundaries = [dropped_len, len(merged)]

        return merged, boundaries

    def _fetch_raw_tokens(
        self,
        session_id: str,
        positions: List[int],
    ) -> Optional[torch.Tensor]:
        """Fetch raw token IDs for dropped positions.

        In a real system, these would be stored with session history.
        For this prototype, return dummy tokens.

        Args:
            session_id: Session to fetch tokens for
            positions: Chunk positions to recover

        Returns:
            Tensor of token IDs, or None if unavailable
        """
        # In production: Retrieve from session store
        # For now: Return dummy tokens (32 per chunk position)
        tokens = []
        for pos in positions:
            # Chunk size = 32 tokens
            tokens.extend([1] * 32)  # Dummy token IDs (1 = common token)

        if not tokens:
            return None

        return torch.tensor(tokens, dtype=torch.long)

    def estimate_recovery_cost(
        self,
        recovery_plan: RecoveryPlan,
    ) -> float:
        """Estimate cost of recovering dropped chunks.

        Returns time estimate in seconds based on token count.

        Args:
            recovery_plan: Recovery plan

        Returns:
            Estimated recovery time in seconds
        """
        num_tokens = len(recovery_plan.raw_tokens)

        # Rough estimate: ~0.01ms per token on modern GPU
        # (Very approximate, would need actual profiling)
        ms_per_token = 0.01
        estimated_ms = num_tokens * ms_per_token

        return estimated_ms / 1000  # Convert to seconds

    def should_recover(
        self,
        recovery_plan: RecoveryPlan,
        max_recovery_cost: float = 0.1,
    ) -> bool:
        """Decide whether to recover dropped chunks.

        Recovery is beneficial if cost < cost_of_attention_without_chunks.
        Simple heuristic: Always recover if cost < threshold.

        Args:
            recovery_plan: Recovery plan
            max_recovery_cost: Maximum acceptable recovery time (seconds)

        Returns:
            True if should recover, False otherwise
        """
        cost = self.estimate_recovery_cost(recovery_plan)
        return cost < max_recovery_cost


class BatchedRecoveryManager:
    """Batch-level recovery for multiple requests.

    Efficiently recomputes dropped tokens for multiple requests in parallel.
    """

    def __init__(
        self,
        model,
        tokenizer,
        cache: TwoTierCache,
        device: str = "cuda:0",
    ):
        """Initialize batched recovery manager.

        Args:
            model: Language model
            tokenizer: Tokenizer
            cache: TwoTierCache instance
            device: GPU device
        """
        self.recovery_manager = TokenRecoveryManager(
            model, tokenizer, cache, device
        )

    def recover_batch(
        self,
        requests: List[Request],
    ) -> Dict[str, Optional[RecoveryPlan]]:
        """Recover dropped tokens for multiple requests.

        Args:
            requests: List of requests

        Returns:
            Dict mapping request_id to recovery plan (or None)
        """
        recovery_plans = {}

        for req in requests:
            plan = self.recovery_manager.create_recovery_plan(req)
            recovery_plans[req.request_id] = plan

            if plan:
                # Recompute this request's dropped chunks
                self.recovery_manager.recompute_dropped_chunks(plan)

        return recovery_plans
