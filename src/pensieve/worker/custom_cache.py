"""Custom KV cache class for HuggingFace Transformers integration."""

import torch
from typing import Dict, List, Optional, Tuple

from transformers import DynamicCache

class PensieveCache(DynamicCache):
    """Custom KV cache implementing HuggingFace Cache interface (layer-wise chunking).

    This class enables HuggingFace models to use our two-tier KV cache
    by implementing the standard Cache interface (__getitem__, __len__).

    Design: Each KVChunk represents a single layer (layer-wise chunking).
    When gathering KV for a layer, we search for all chunks at all positions
    with matching (session_id, layer_idx) and concatenate them.

    When a HuggingFace model does:
        layer_past = past_key_values[layer_idx]

    This calls our __getitem__(layer_idx) which:
    1. Finds all chunks for this layer across all positions in batch
    2. Gathers K and V tensors from each chunk
    3. Concatenates them to form layer's full KV cache
    4. Returns to model for attention computation
    """

    def __init__(self, cache_manager, batch_info: Dict, num_layers: int):
        """Initialize Pensieve cache.

        Args:
            cache_manager: TwoTierCache instance managing actual cache
            batch_info: Information about current batch and its requests
                {request_id: {session_id, positions, context_lengths, ...}}
            num_layers: Number of transformer layers in model
        """
        self.cache_manager = cache_manager
        self.batch_info = batch_info
        self.num_layers = num_layers

        # Track KV tensors for current forward pass
        self._layer_kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._seq_length = 0

        # HuggingFace Cache interface requires 'layers' attribute
        self.layers = [None] * num_layers

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache for a specific layer.

        Since chunks are now layer-indexed, we gather all chunks at all positions
        for this specific layer, ensuring they're ordered correctly.

        Called by HuggingFace models when they need KV cache during forward pass.
        Example: k, v = past_key_values[layer_idx]

        Args:
            layer_idx: Layer index (0 to num_layers-1)

        Returns:
            Tuple of (key_tensor, value_tensor)
            - Shape: [batch, heads, seq, head_dim] for Gemma-2
                  or [batch, seq, heads, head_dim] for Llama/Standard
        """
        if layer_idx in self._layer_kv_cache:
            # Already cached in this forward pass
            return self._layer_kv_cache[layer_idx]

        # Gather KV from all requests in batch for this specific layer
        # Chunks are ordered by (session_id, chunk_id, layer_idx)
        all_keys = []
        all_values = []

        # ✅ DEBUG: Log batch_info at start
        if layer_idx == 0:
            print(f"\n[PensieveCache.__getitem__] layer_idx={layer_idx}, batch_info keys={list(self.batch_info.keys())}", flush=True)
            for req_id, info in self.batch_info.items():
                print(f"  {req_id}: session={info.get('session_id')}, positions={info.get('positions', [])}", flush=True)

        for request_info in self.batch_info.values():
            session_id = request_info.get('session_id')
            positions = request_info.get('positions', [])  # chunk_ids

            # Gather chunks for this layer at all positions in order
            for position in sorted(positions):
                # Search for chunk at (session_id, position, layer_idx)
                chunk_found = False
                for cache_dict in [self.cache_manager.gpu_cache, self.cache_manager.cpu_cache]:
                    for chunk in cache_dict.values():
                        if (chunk.session_id == session_id and
                            chunk.chunk_id == position and
                            chunk.layer_idx == layer_idx):
                            chunk_found = True
                            chunk.update_access_time()
                            # ✅ Only add non-empty chunks
                            if chunk.key_tensor is not None and chunk.key_tensor.numel() > 0:
                                all_keys.append(chunk.key_tensor)
                                all_values.append(chunk.value_tensor)
                            break
                    if chunk_found:
                        break

        # Fallback: If no chunks found via batch_info.positions, scan cache directly
        # This handles cases where chunk_keys wasn't populated in Request object
        if not all_keys:
            # Get all session_ids from batch_info
            session_ids = {info.get('session_id') for info in self.batch_info.values()}

            if layer_idx == 0:
                print(f"[PensieveCache.__getitem__] PRIMARY path found 0 chunks! Triggering fallback...", flush=True)
                print(f"  Looking for session_ids={session_ids}, layer_idx={layer_idx}", flush=True)
                print(f"  GPU cache has {len(self.cache_manager.gpu_cache)} chunks", flush=True)
                print(f"  CPU cache has {len(self.cache_manager.cpu_cache)} chunks", flush=True)

            # Collect all chunks for these sessions and this layer, sorted by chunk_id
            found_chunks = {}  # {(session_id, chunk_id): chunk}
            for cache_dict in [self.cache_manager.gpu_cache, self.cache_manager.cpu_cache]:
                for chunk in cache_dict.values():
                    if chunk.session_id in session_ids and chunk.layer_idx == layer_idx:
                        key = (chunk.session_id, chunk.chunk_id)
                        found_chunks[key] = chunk

            # Add chunks in order of chunk_id (grouped by session_id)
            for session_id in sorted(session_ids):
                session_chunks = [chunk for (sid, _), chunk in found_chunks.items() if sid == session_id]
                session_chunks.sort(key=lambda c: c.chunk_id)
                for chunk in session_chunks:
                    chunk.update_access_time()
                    # ✅ Only add non-empty chunks
                    if chunk.key_tensor is not None and chunk.key_tensor.numel() > 0:
                        all_keys.append(chunk.key_tensor)
                        all_values.append(chunk.value_tensor)

        # Concatenate all KV tensors for this layer
        # They may be non-contiguous in GPU memory (that's the whole point!)
        if layer_idx == 0:
            print(f"[PensieveCache.__getitem__] Found {len(all_keys)} chunks for layer_idx={layer_idx}", flush=True)

        if all_keys:
            # ✅ CRITICAL: Detect actual tensor shape to concatenate at correct dimension
            # Gemma-2 uses [batch, heads, seq, head_dim]
            # Standard HF uses [batch, seq, heads, head_dim]
            sample_key = all_keys[0]
            if sample_key.dim() == 4:
                # 4D tensor - determine which dimension is sequence
                # For Gemma: [batch, heads, seq, head_dim] → concat at dim=2
                # For Standard: [batch, seq, heads, head_dim] → concat at dim=1
                # Heuristic: if dim[1] is small (num_heads typically 8-32), then dim=2 is seq
                if sample_key.shape[1] < 256:  # Likely heads dimension (usually 8-32)
                    seq_dim = 2  # Gemma format: [batch, heads, seq, head_dim]
                else:
                    seq_dim = 1  # Standard format: [batch, seq, heads, head_dim]
            else:
                # Default to dim=1 for 2D or other cases
                seq_dim = 1

            keys = torch.cat(all_keys, dim=seq_dim)  # Concatenate along detected sequence dimension
            values = torch.cat(all_values, dim=seq_dim)

            # ✅ GQA EXPAND: Llama uses 8 KV heads but 32 Query heads
            # Repeat each KV head 4 times to match query head count
            if keys.shape[1] == 8:  # Llama: 8 KV heads
                keys = keys.repeat_interleave(4, dim=1)  # [1, 8, seq, 128] → [1, 32, seq, 128]
                values = values.repeat_interleave(4, dim=1)

            # ✅ VALIDATION: After concatenation, verify result is valid
            if keys.shape[-1] == 0:
                raise ValueError(f"Concatenation resulted in head_dim=0: {keys.shape}")
            if keys.numel() == 0:
                raise ValueError(f"Concatenation resulted in empty tensor: {keys.shape}")
        else:
            # ✅ CRITICAL FIX: Return None for both k and v to signal "no cache"
            # HuggingFace models handle (None, None) correctly as "compute new KV"
            # Don't return empty tensors - they cause shape mismatches in attention!
            keys = None
            values = None

        # Cache for this forward pass
        self._layer_kv_cache[layer_idx] = (keys, values)

        # ✅ FIX: Detect tensor format correctly for both Gemma and Llama
        # Gemma-2: [batch, heads, seq, head_dim] → seq at dim=2
        # Llama: [batch, heads, seq, head_dim] → seq at dim=2 (heads at dim=1, which is small)
        # Standard: [batch, seq, heads, head_dim] → seq at dim=1 (seq at dim=1, which is large)
        if keys is not None and len(keys.shape) == 4 and keys.shape[1] < 256:
            # shape[1] is small → likely num_heads → seq is at dim=2 (Gemma/Llama format)
            self._seq_length = keys.shape[2]
        elif keys is not None:
            # shape[1] is large → likely seq → seq is at dim=1 (Standard format)
            self._seq_length = keys.shape[1] if len(keys.shape) > 1 else 0
        else:
            # No cache for this layer
            self._seq_length = 0

        return keys, values

    def is_empty(self) -> bool:
        """Check if this cache has any cached KV chunks FOR THIS BATCH'S SESSIONS.

        ✅ CRITICAL FIX: Only check chunks for sessions in current batch.
        This prevents other sessions' cached chunks from being seen as "cache exists"
        for a session on its first turn.

        Example:
        - Session A completes Turn 1 → chunks stored in cache_manager
        - Session B's first request arrives
        - OLD CODE: is_empty() checks cache_manager globally → finds Session A's chunks → returns False
        - NEW CODE: is_empty() checks only for Session B → finds nothing → returns True
        - Now Session B correctly gets input_cache=None for its first turn!

        Returns:
            True if no cached chunks for this batch's sessions, False if has cached chunks
        """
        # Get session_ids for sessions in this batch
        session_ids = {info.get('session_id') for info in self.batch_info.values()}

        # Check if any chunks exist for ONLY these sessions
        for cache_dict in [self.cache_manager.gpu_cache, self.cache_manager.cpu_cache]:
            for chunk in cache_dict.values():
                if chunk.session_id in session_ids:
                    return False  # Found at least one chunk for a batch session

        return True  # No chunks for any batch session

    def __len__(self) -> int:
        """Return number of layers (required by Cache interface)."""
        return self.num_layers

    def get_mask_sizes(
        self,
        cache_position: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Get cache sizes for mask calculation (required by HuggingFace).

        Args:
            cache_position: Cache position tensor
            layer_idx: Layer index

        Returns:
            Tuple of (kv_length, kv_offset)
        """
        # ✅ CRITICAL: Calculate actual cached seq_length if _seq_length hasn't been set yet
        # __getitem__ might not have been called yet, so we need to calculate from actual chunks
        if self._seq_length == 0:
            actual_seq = self.calculate_cached_seq_length()
            return actual_seq, 0

        return self._seq_length, 0

    def to(self, device, **kwargs):
        """Move cache to device (required by HuggingFace).

        Since we manage our own tensors, just return self.
        """
        return self

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with newly computed KV tensors.

        Called by HuggingFace models after computing new KV tokens during forward pass.

        Args:
            key_states: New key tensor from model
            value_states: New value tensor from model
            layer_idx: Which layer these tensors are from
            cache_position: Optional cache position tensor (newer HuggingFace versions)
            **kwargs: Additional arguments for compatibility with different HuggingFace versions

        Returns:
            Tuple of (key_states, value_states) for HuggingFace models to use
        """
        # Store in our layer cache
        self._layer_kv_cache[layer_idx] = (key_states, value_states)
        # Return the updated states (required by HuggingFace)
        return key_states, value_states

    def get_seq_length(self) -> int:
        """Get current sequence length in cache.

        Returns:
            Total sequence length
        """
        
        # If _seq_length hasn't been updated yet (before first __getitem__),
        # calculate from actual cache chunks
        if self._seq_length == 0:
            return self.calculate_cached_seq_length()
        return self._seq_length

    def calculate_cached_seq_length(self) -> int:
        """Calculate cached sequence length from actual cache chunks.

        This is used before __getitem__ is called in the forward pass,
        so _seq_length hasn't been updated yet.

        Returns:
            Total sequence length from cache chunks
        """
        from pensieve.core.types import CHUNK_SIZE

        max_tokens = 0

        # Try method 1: Use batch_info positions if available
        for request_id, request_info in self.batch_info.items():
            positions = request_info.get('positions', [])
            if positions:
                max_chunk_id = max(positions)
                session_id = request_info.get('session_id')
                # Find actual size of last chunk
                for cache_dict in [self.cache_manager.gpu_cache,
                                  self.cache_manager.cpu_cache]:
                    for chunk in cache_dict.values():
                        if (chunk.session_id == session_id and
                            chunk.chunk_id == max_chunk_id):
                            last_chunk_tokens = chunk.num_tokens
                            total_tokens = (max_chunk_id * CHUNK_SIZE) + last_chunk_tokens
                            max_tokens = max(max_tokens, total_tokens)
                            break

        # Method 2 fallback: If batch_info is incomplete, scan cache_manager directly
        # This handles the case where chunk_keys wasn't properly populated in Request
        if max_tokens == 0:
            # Get session_ids from batch_info
            session_ids = {info.get('session_id') for info in self.batch_info.values()}
            # Scan all chunks for these sessions
            for cache_dict in [self.cache_manager.gpu_cache, self.cache_manager.cpu_cache]:
                for chunk in cache_dict.values():
                    if chunk.session_id in session_ids:
                        # Found a chunk for this session
                        last_chunk_tokens = chunk.num_tokens
                        total_tokens = (chunk.chunk_id * CHUNK_SIZE) + last_chunk_tokens
                        max_tokens = max(max_tokens, total_tokens)
        return max_tokens

    def reset(self) -> None:
        """Reset cache for new forward pass."""
        self._layer_kv_cache.clear()
        self._seq_length = 0
        # Reset layers attribute for HuggingFace compatibility
        self.layers = [None] * self.num_layers


class PensieveCacheFactory:
    """Factory for creating PensieveCache instances."""

    @staticmethod
    def create(
        cache_manager,
        batch_requests: List,
        num_layers: int,
    ) -> PensieveCache:
        """Create a PensieveCache for a batch.

        With layer-wise chunking, each KVChunk covers one layer.
        This factory extracts session and position info from requests.

        Args:
            cache_manager: TwoTierCache instance
            batch_requests: List of Request objects in batch
            num_layers: Number of transformer layers

        Returns:
            PensieveCache instance
        """
        # Build batch info dict mapping request_id to request info
        batch_info = {}
        for req in batch_requests:
            # Extract unique positions (chunk_ids) from chunk_keys
            # Format: "session:chunk:id:layer:idx"
            positions = set()

            for chunk_key in req.chunk_keys:
                # Parse chunk_key to extract chunk_id
                parts = chunk_key.split(':')
                if len(parts) >= 3:
                    chunk_id = int(parts[2])
                    positions.add(chunk_id)

            batch_info[req.request_id] = {
                'session_id': req.session_id,
                'positions': sorted(list(positions)),  # Ordered chunk positions
                'context_length': len(req.input_ids) if hasattr(req, 'input_ids') else 0,
                'seq_len': req.seq_len if hasattr(req, 'seq_len') else 0,
            }

        return PensieveCache(cache_manager, batch_info, num_layers)


class SimpleCacheWrapper:
    """Wrapper around standard HuggingFace Cache for baseline."""

    def __init__(self, num_layers: int):
        """Initialize simple cache wrapper.

        Args:
            num_layers: Number of transformer layers
        """
        self.num_layers = num_layers
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __getitem__(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV for layer."""
        return self._cache.get(layer_idx, None)

    def __setitem__(self, layer_idx: int, kv: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set cached KV for layer."""
        self._cache[layer_idx] = kv

    def __len__(self) -> int:
        """Get number of layers."""
        return self.num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new KV.

        Args:
            key_states: New key tensor from model
            value_states: New value tensor from model
            layer_idx: Which layer these tensors are from
            cache_position: Optional cache position tensor (newer HuggingFace versions)
            **kwargs: Additional arguments for compatibility with different HuggingFace versions

        Returns:
            Tuple of (key_states, value_states) for HuggingFace models to use
        """
        self._cache[layer_idx] = (key_states, value_states)
        # Return the updated states (required by HuggingFace)
        return key_states, value_states

    def reset(self) -> None:
        """Reset cache."""
        self._cache.clear()
