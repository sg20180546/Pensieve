"""Custom KV cache class for HuggingFace Transformers integration."""

import torch
from typing import Dict, List, Optional, Tuple
import sys
import os
import traceback

# Try to import from transformers
try:
    from transformers import Cache
except ImportError:
    # Fallback if not available
    Cache = object


class PensieveCache(Cache):
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

    def _write_debug_log(self, message: str) -> None:
        """Write to debug log with immediate disk sync."""
        debug_file = "/home/elicer/pensieve_cache_debug.log"
        try:
            with open(debug_file, "a", buffering=1) as f:
                f.write(message)
                f.flush()
                os.fsync(f.fileno())  # Force immediate disk write
        except Exception:
            pass  # Silently ignore file write errors

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
            - Shape: [batch_size, seq_len, num_heads, head_dim]
        """
        import sys
        import traceback

        # CRITICAL: Use FILE-BASED logging to bypass buffering issues
        debug_file = "/home/elicer/pensieve_cache_debug.log"
        try:
            with open(debug_file, "a", buffering=1) as f:  # Line buffering
                f.write(f"\n========== [CACHE_DEBUG] __getitem__ START layer_idx={layer_idx} ==========\n")
                f.write(f"batch_info keys: {list(self.batch_info.keys())}\n")
                f.write(f"batch_info values: {self.batch_info}\n")
                f.flush()
                os.fsync(f.fileno())  # ✅ Force sync to disk immediately
        except Exception as e:
            try:
                with open(debug_file, "a", buffering=1) as f:
                    f.write(f"ERROR in debug write: {e}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except:
                pass

        # Also try stderr for redundancy
        try:
            print(f"\n========== [CACHE_DEBUG] __getitem__ START layer_idx={layer_idx} ==========", file=sys.stderr, flush=True)
            sys.stderr.flush()
        except Exception as e:
            pass

        if layer_idx in self._layer_kv_cache:
            # Already cached in this forward pass
            with open("/home/elicer/pensieve_cache_debug.log", "a", buffering=1) as f:
                f.write(f"[CACHE_DEBUG] layer_idx={layer_idx} FOUND IN _layer_kv_cache, returning cached\n")
                f.flush()
                os.fsync(f.fileno())
            return self._layer_kv_cache[layer_idx]

        # Debug: Print cache state on first call (for visibility)
        if layer_idx == 0:
            print(f"\n[CACHE_DEBUG] __getitem__(layer_idx=0) START", flush=True)
            print(f"[CACHE_DEBUG] batch_info: {self.batch_info}", flush=True)
            batch_positions = [info.get('positions', []) for info in self.batch_info.values()]
            print(f"[CACHE_DEBUG] batch_info.positions: {batch_positions}", flush=True)

            # List all chunks in cache_manager
            all_chunks = {}
            for cache_dict in [self.cache_manager.gpu_cache, self.cache_manager.cpu_cache]:
                for key, chunk in cache_dict.items():
                    if key not in all_chunks:
                        all_chunks[key] = chunk
            print(f"[CACHE_DEBUG] Total chunks in system: {len(all_chunks)}", flush=True)
            print(f"[CACHE_DEBUG] Chunks by (session_id, chunk_id, layer_idx):", flush=True)
            for c in all_chunks.values():
                print(f"[CACHE_DEBUG]   ({c.session_id}, {c.chunk_id}, {c.layer_idx}) - kv_shape=({c.key_tensor.shape if c.key_tensor is not None else 'None'}, {c.value_tensor.shape if c.value_tensor is not None else 'None'})", flush=True)

        # Gather KV from all requests in batch for this specific layer
        # Chunks are ordered by (session_id, chunk_id, layer_idx)
        all_keys = []
        all_values = []

        with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
            f.write(f"[CACHE_DEBUG] Starting primary path for layer_idx={layer_idx}\n")

        for request_id, request_info in self.batch_info.items():
            session_id = request_info.get('session_id')
            positions = request_info.get('positions', [])  # chunk_ids

            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Primary path: request_id={request_id}, session_id={session_id}, positions={positions}\n")

            # Gather chunks for this layer at all positions in order
            for position in sorted(positions):
                # Search for chunk at (session_id, position, layer_idx)
                for cache_dict in [self.cache_manager.gpu_cache,
                                  self.cache_manager.cpu_cache]:
                    for chunk in cache_dict.values():
                        if (chunk.session_id == session_id and
                            chunk.chunk_id == position and
                            chunk.layer_idx == layer_idx):
                            chunk.update_access_time()
                            # ✅ Only add non-empty chunks
                            if chunk.key_tensor is not None and chunk.key_tensor.numel() > 0:
                                all_keys.append(chunk.key_tensor)
                                all_values.append(chunk.value_tensor)
                                with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                                    f.write(f"[CACHE_DEBUG] ✓ Found in primary path: chunk({session_id}, {position}, {layer_idx}), shape={chunk.key_tensor.shape}\n")
                            else:
                                with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                                    f.write(f"[CACHE_DEBUG] ⚠️ Skipping empty chunk: ({session_id}, {position}, {layer_idx})\n")
                            break

        # Fallback: If no chunks found via batch_info.positions, scan cache directly
        # This handles cases where chunk_keys wasn't populated in Request object
        if not all_keys:
            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Primary path found 0 chunks, triggering fallback scan for layer_idx={layer_idx}...\n")

            # Get all session_ids from batch_info
            session_ids = {info.get('session_id') for info in self.batch_info.values()}

            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Fallback: Looking for session_ids={session_ids}, layer_idx={layer_idx}\n")
                f.write(f"[CACHE_DEBUG] Available in cache: GPU={len(self.cache_manager.gpu_cache)}, CPU={len(self.cache_manager.cpu_cache)} chunks\n")

            # Collect all chunks for these sessions and this layer, sorted by chunk_id
            found_chunks = {}  # {(session_id, chunk_id): chunk}
            chunk_count = 0
            for cache_dict in [self.cache_manager.gpu_cache, self.cache_manager.cpu_cache]:
                for chunk in cache_dict.values():
                    chunk_count += 1
                    if layer_idx == 0 and chunk_count <= 5:
                        with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                            f.write(f"[CACHE_DEBUG] Scanning chunk {chunk_count}: session_id={chunk.session_id}, chunk_id={chunk.chunk_id}, layer_idx={chunk.layer_idx}, matches_session={chunk.session_id in session_ids}, matches_layer={chunk.layer_idx == layer_idx}\n")
                    if chunk.session_id in session_ids and chunk.layer_idx == layer_idx:
                        key = (chunk.session_id, chunk.chunk_id)
                        found_chunks[key] = chunk
                        if layer_idx == 0:
                            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                                f.write(f"[CACHE_DEBUG] ✓ Found chunk in fallback: {key}\n")
            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Fallback scan complete: Total chunks scanned={chunk_count}, matched={len(found_chunks)}\n")

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
                    else:
                        with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                            f.write(f"[CACHE_DEBUG] ⚠️ Fallback: Skipping empty chunk: ({session_id}, {chunk.chunk_id}, {layer_idx})\n")

            if layer_idx == 0:
                print(f"[CACHE_DEBUG] Fallback found {len(all_keys)} KV pairs for layer {layer_idx}", flush=True)

        # Concatenate all KV tensors for this layer
        # They may be non-contiguous in GPU memory (that's the whole point!)
        if all_keys:
            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Concatenating {len(all_keys)} KV pairs for layer_idx={layer_idx}\n")
            keys = torch.cat(all_keys, dim=1)  # Concatenate along sequence dimension
            values = torch.cat(all_values, dim=1)
            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Result: keys.shape={keys.shape}, values.shape={values.shape}\n")
        else:
            # No cached KV for this layer, return empty tensors
            # Model will treat this as no past_key_values for this layer
            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] !!!WARNING!!! No KV pairs found for layer_idx={layer_idx}, returning EMPTY tensors\n")
                f.write(f"all_keys length: {len(all_keys)}\n")
                f.write(f"all_values length: {len(all_values)}\n")
                f.write(f"session_ids from batch_info: {set(info.get('session_id') for info in self.batch_info.values())}\n")
            keys = torch.empty(0, dtype=torch.float16)
            values = torch.empty(0, dtype=torch.float16)
            with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
                f.write(f"[CACHE_DEBUG] Empty tensors: keys.shape={keys.shape}, values.shape={values.shape}\n")

        # Cache for this forward pass
        self._layer_kv_cache[layer_idx] = (keys, values)
        self._seq_length = keys.shape[1] if len(keys.shape) > 1 else 0

        with open("/home/elicer/pensieve_cache_debug.log", "a") as f:
            f.write(f"[CACHE_DEBUG] __getitem__ RETURNING layer_idx={layer_idx}: keys.shape={keys.shape}, values.shape={values.shape}\n")
        return keys, values

    def is_empty(self) -> bool:
        """Check if this cache has any cached KV chunks.

        Returns:
            True if no cached chunks, False if has cached chunks
        """
        # Check actual cache manager, not just current forward pass state
        # This ensures we detect cache from previous turns correctly
        has_gpu_chunks = len(self.cache_manager.gpu_cache) > 0
        has_cpu_chunks = len(self.cache_manager.cpu_cache) > 0
        return not (has_gpu_chunks or has_cpu_chunks)

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
        # Return current sequence length and offset
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
    ) -> None:
        """Update cache with newly computed KV tensors.

        Called by HuggingFace models after computing new KV tokens during forward pass.

        Args:
            key_states: New key tensor from model
            value_states: New value tensor from model
            layer_idx: Which layer these tensors are from
        """
        # Store in our layer cache
        self._layer_kv_cache[layer_idx] = (key_states, value_states)

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
    ) -> None:
        """Update cache with new KV."""
        self._cache[layer_idx] = (key_states, value_states)

    def reset(self) -> None:
        """Reset cache."""
        self._cache.clear()
