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
from transformers import DynamicCache

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
    """Manages recomputation of dropped KV chunks with dependency awareness.

    CRITICAL: Respects layer-wise and token-wise dependencies!

    Key insights:
    1. Layer Dependency: Layer N's KV depends on Layer N-1's outputs
       → Must use previous layers' cached KV as past_key_values

    2. Token Dependency: Tokens[32:64] depend on Tokens[0:32]
       → Must include all previous tokens' cached KV

    3. Recovery Process:
       - Load all PREVIOUS chunks' cached KV (chunks 0 to N-1)
       - Forward pass ONLY current chunk's tokens
       - Extract new KV for all layers (full context considered)
       - Store recovered chunks

    Design:
    - Per-request: Check for dropped chunks
    - Create recovery plan with raw tokens from server's session history
    - Recompute via forward pass with full previous context
    - Store recovered chunks back in cache
    """

    def __init__(
        self,
        model,
        tokenizer,
        cache: TwoTierCache,
        server=None,  # PensieveServer reference
        device: str = "cuda:0",
    ):
        """Initialize recovery manager.

        Args:
            model: HuggingFace language model
            tokenizer: Tokenizer for encoding
            cache: TwoTierCache instance
            server: PensieveServer reference (for session_token_histories)
            device: GPU device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.server = server  # ← For accessing session token history
        self.device = device
        self.num_layers = (
            model.config.num_hidden_layers
            if hasattr(model.config, "num_hidden_layers")
            else model.config.n_layer
        )
        self.chunk_size = 32  # tokens per chunk
        self._seq_dim = None  # Detected lazily from first KV tensor

    def _get_seq_dim(self, kv_tensor: torch.Tensor) -> int:
        """Get dimension index for seq_len in KV tensor.

        Llama/Gemma: [batch, num_heads, seq_len, head_dim] → dim=2
        Standard HF: [batch, seq_len, num_heads, head_dim] → dim=1
        """
        if kv_tensor is None or len(kv_tensor.shape) < 4:
            return 1
        if kv_tensor.shape[1] < 256:  # shape[1] is likely num_heads
            return 2
        return 1

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

        # Check each position for dropped chunks (per-position chunks)
        for pos in session_positions:
            chunk_key = f"{session_id}:chunk:{pos}"
            chunk = self.cache.get_chunk(chunk_key)

            if chunk is None:
                continue
            elif chunk.location == CacheLocation.DROPPED:
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
        session_id: str,
        recovery_plan: RecoveryPlan,
    ) -> None:
        """Recompute dropped chunks considering LAYER and TOKEN dependencies.

        Algorithm (respects dependencies!):
        1. For each dropped chunk position (in order):
           a. Load all PREVIOUS chunks' cached KV (Layer 0..N)
           b. Extract current chunk's tokens from session history
           c. Forward pass ONLY current chunk's tokens
           d. Extract new KV for all layers from outputs
           e. Store in GPU cache

        Why this works:
        - Past KV from chunks 0..N-1 provides full context for attention
        - Forward pass on current tokens computes correct KV (considering full history)
        - Layer-wise: Each layer uses previous layer's outputs (via HF model)
        - Token-wise: Attention uses all previous tokens (via past_key_values)

        Args:
            session_id: Session ID
            recovery_plan: Recovery plan with dropped positions
        """
        if not recovery_plan or not recovery_plan.dropped_positions:
            return

        # Get session's accumulated token history from server
        if not self.server or session_id not in self.server.session_token_histories:
            print(f"⚠️  No token history for session {session_id}")
            return

        all_tokens = self.server.session_token_histories[session_id]
        print(
            f"📋 Recovering {len(recovery_plan.dropped_positions)} chunks for {session_id} "
            f"(total {len(all_tokens)} tokens)"
        )

        # Process each dropped chunk IN ORDER (respects token dependency)
        for dropped_chunk_id in sorted(recovery_plan.dropped_positions):
            print(
                f"  ↻ Chunk {dropped_chunk_id} (tokens {dropped_chunk_id*32}:{(dropped_chunk_id+1)*32})"
            )

            # Calculate token range for this chunk
            start_token_idx = dropped_chunk_id * self.chunk_size
            end_token_idx = (dropped_chunk_id + 1) * self.chunk_size

            # Boundary check
            if end_token_idx > len(all_tokens):
                print(
                    f"    ⚠️  Insufficient tokens ({end_token_idx} > {len(all_tokens)})"
                )
                continue

            # 🔑 KEY STEP 1: Load PREVIOUS chunks' cached KV (respects layer dependency)
            prev_cached_kv = None
            if dropped_chunk_id > 0:
                try:
                    prev_cached_kv = self._load_prev_chunks_kv(
                        session_id,
                        start_chunk_id=0,
                        end_chunk_id=dropped_chunk_id,  # Exclusive
                    )
                    if prev_cached_kv:
                        print(f"    ✓ Loaded cached KV from chunks 0-{dropped_chunk_id-1}")
                except Exception as e:
                    print(f"    ⚠️  Failed to load prev cached KV: {e}")
                    prev_cached_kv = None

            # 🔑 KEY STEP 2: Extract this chunk's tokens
            chunk_token_ids = all_tokens[start_token_idx:end_token_idx]
            chunk_tokens = torch.tensor(
                chunk_token_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)  # [1, chunk_size]

            # 🔑 KEY STEP 3: Forward pass (layer dependency handled by model)
            try:
                # Convert tuple to DynamicCache for HuggingFace models
                past_kv_cache = None
                if prev_cached_kv is not None:
                    past_kv_cache = DynamicCache.from_legacy_cache(prev_cached_kv)

                with torch.no_grad():
                    outputs = self.model(
                        chunk_tokens,
                        past_key_values=past_kv_cache,  # ← Full context as DynamicCache!
                        use_cache=True,
                        return_dict=True,
                    )

                # 🔑 KEY STEP 4: Store recovered KV for all layers
                past_kv = outputs.past_key_values
                if past_kv:
                    # Convert DynamicCache back to tuple for storage
                    if isinstance(past_kv, DynamicCache):
                        past_kv_tuple = past_kv.to_legacy_cache()
                    else:
                        past_kv_tuple = past_kv

                    # ✅ DEBUG: Log dtype from model output
                    if past_kv_tuple and len(past_kv_tuple) > 0:
                        k, v = past_kv_tuple[0]
                        if k is not None:
                            print(f"      [dtype-check] Model output KV: key={k.dtype}, value={v.dtype}")

                    self._store_recovered_chunks(
                        session_id=session_id,
                        chunk_id=dropped_chunk_id,
                        past_key_values=past_kv_tuple,
                        num_prev_tokens=start_token_idx,
                        num_layers=self.num_layers,
                    )
                    print(
                        f"    ✓ Stored recovered KV across {self.num_layers} layers"
                    )

            except Exception as e:
                print(f"    ❌ Recomputation failed: {e}")
                import traceback
                traceback.print_exc()

    def _load_prev_chunks_kv(
        self,
        session_id: str,
        start_chunk_id: int = 0,
        end_chunk_id: int = None,
    ) -> Optional[Tuple]:
        """Load cached KV from previous chunks (respects layer dependency).

        This is CRITICAL: To compute correct KV for chunk N, we need
        the full context from chunks 0 to N-1 as past_key_values.

        Process:
        1. Load per-position chunks in order
        2. Extract each layer's KV from each position chunk
        3. Concatenate along seq_len dimension per layer
        4. Return as tuple of (key, value) per layer

        Args:
            session_id: Session ID
            start_chunk_id: Start chunk ID (inclusive)
            end_chunk_id: End chunk ID (exclusive)

        Returns:
            past_key_values tuple for all layers, or None
        """
        if end_chunk_id is None or end_chunk_id == 0:
            return None

        # Collect KV for each layer
        collected_keys = [[] for _ in range(self.num_layers)]
        collected_values = [[] for _ in range(self.num_layers)]
        
        # Load position chunks in order (respects token dependency)
        for chunk_id in range(start_chunk_id, end_chunk_id):
            chunk_key = f"{session_id}:chunk:{chunk_id}"

            try:
                chunk = self.cache.get_chunk(chunk_key)
                if chunk is None or chunk.layer_kv is None:
                    print(f"      ⚠️  Chunk {chunk_key} not found or has no tensors")
                    continue

                # Extract each layer's KV from this position chunk
                for layer_idx in range(self.num_layers):
                    if layer_idx not in chunk.layer_kv:
                        continue

                    k, v = chunk.layer_kv[layer_idx]

                    # Move to device (preserve original dtype)
                    key_tensor = k.to(self.device)
                    value_tensor = v.to(self.device)

                    if chunk_id == 0 and layer_idx == 0:
                        print(f"      [dtype-check] Loaded KV: key={key_tensor.dtype}, value={value_tensor.dtype}")

                    # Ensure shape: [batch=1, seq_len, num_heads, head_dim]
                    if key_tensor.dim() == 3:
                        key_tensor = key_tensor.unsqueeze(0)
                    if value_tensor.dim() == 3:
                        value_tensor = value_tensor.unsqueeze(0)

                    collected_keys[layer_idx].append(key_tensor)
                    collected_values[layer_idx].append(value_tensor)

            except Exception as e:
                print(f"      ❌ Error loading {chunk_key}: {e}")
                return None

        # Detect seq_dim from first collected tensor
        seq_dim = None
        for keys_list in collected_keys:
            if keys_list:
                seq_dim = self._get_seq_dim(keys_list[0])
                break
        if seq_dim is None:
            seq_dim = 2  # Default for Llama

        # Concatenate chunks for each layer (respects token dependency)
        past_key_values = []
        for layer_idx in range(self.num_layers):
            if not collected_keys[layer_idx]:
                past_key_values.append((None, None))
            else:
                try:
                    concatenated_key = torch.cat(
                        collected_keys[layer_idx],
                        dim=seq_dim,
                    )
                    concatenated_value = torch.cat(
                        collected_values[layer_idx],
                        dim=seq_dim,
                    )

                    if layer_idx == 0:
                        print(f"      [dtype-check] Concatenated KV for recovery: key={concatenated_key.dtype}, value={concatenated_value.dtype}")
                    # print(concatenated_key.shape, concatenated_value.shape)
                    past_key_values.append((concatenated_key, concatenated_value))
                    
                except Exception as e:
                    print(f"      ❌ Error concatenating layer {layer_idx}: {e}")
                    # print(past_key_values)
                    return None

        return tuple(past_key_values)

    def _store_recovered_chunks(
        self,
        session_id: str,
        chunk_id: int,
        past_key_values: Tuple,
        num_prev_tokens: int,
        num_layers: int,
    ) -> None:
        """Store recovered KV as a single per-position chunk (all layers).

        Extracts the NEW chunk's KV from full past_key_values
        (which includes prev chunks + new chunk).

        Args:
            session_id: Session ID
            chunk_id: Chunk ID to store
            past_key_values: Full KV from forward pass
            num_prev_tokens: Number of tokens in previous chunks
            num_layers: Number of model layers
        """
        try:
            # Bundle all layers' KV for this position
            layer_kv = {}
            for layer_idx, (key, value) in enumerate(past_key_values):
                if key is None or value is None:
                    continue

                # Extract ONLY the NEW chunk's tokens
                # past_key_values has all tokens: [prev ... new]
                # We want: just the new (last chunk_size tokens)
                if self._seq_dim is None:
                    self._seq_dim = self._get_seq_dim(key)
                seq_dim = self._seq_dim

                if seq_dim == 2:
                    new_key = key[:, :, -self.chunk_size:, :]
                    new_value = value[:, :, -self.chunk_size:, :]
                else:
                    new_key = key[:, -self.chunk_size:, :, :]
                    new_value = value[:, -self.chunk_size:, :, :]
                layer_kv[layer_idx] = (new_key.detach(), new_value.detach())

            chunk = KVChunk(
                session_id=session_id,
                chunk_id=chunk_id,
                layer_kv=layer_kv,
                context_length=num_prev_tokens,
                session_total_chunks=(num_prev_tokens + 2 * self.chunk_size - 1) // self.chunk_size,
                num_layers=num_layers,
            )

            self.cache.store_chunk(chunk, location=CacheLocation.GPU)

        except Exception as e:
            print(f"      ❌ Failed to store chunk {chunk_id}: {e}")

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
        chunks_to_recompute: Dict[str, List[str]],
    ) -> Dict[str, Optional[RecoveryPlan]]:
        """Recover dropped tokens for multiple requests.

        Only recovers chunks that were pre-determined by create_cache_plan.
        No independent scanning - recovery decisions are made once in the
        scheduler and executed here exactly as planned.

        Args:
            requests: List of requests (from different sessions)
            chunks_to_recompute: Pre-computed {session_id: [chunk_keys]} from cache_plan.

        Returns:
            Dict mapping request_id to recovery plan (or None)
        """
        recovery_plans = {}
        print(f"recover_batch: {sum(len(v) for v in chunks_to_recompute.values())} chunks to recompute")

        for req in requests:
            session_id = req.session_id

            if session_id not in chunks_to_recompute:
                recovery_plans[req.request_id] = None
                continue

            dropped_chunk_keys = chunks_to_recompute[session_id]
            # Extract position IDs from chunk keys ("session:chunk:pos" → pos)
            dropped_positions = []
            for ck in dropped_chunk_keys:
                parts = ck.split(":")
                if len(parts) >= 3:
                    dropped_positions.append(int(parts[2]))
            dropped_positions.sort()

            if not dropped_positions:
                recovery_plans[req.request_id] = None
                continue

            raw_tokens = self.recovery_manager._fetch_raw_tokens(
                session_id, dropped_positions
            )
            plan = RecoveryPlan(
                dropped_positions=dropped_positions,
                raw_tokens=raw_tokens if raw_tokens is not None else torch.tensor([], dtype=torch.long),
                session_id=session_id,
            )

            recovery_plans[req.request_id] = plan
            print("plan", plan)
            if plan.raw_tokens is not None and len(plan.raw_tokens) > 0:
                self.recovery_manager.recompute_dropped_chunks(
                    session_id=session_id,
                    recovery_plan=plan,
                )

        return recovery_plans
