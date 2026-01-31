"""Two-tier KV cache management for Pensieve."""

import torch
import time
from typing import Dict, List, Optional, Tuple
from .types import KVChunk, CacheLocation, CacheStatistics, SessionMetadata
from .eviction import RetentionValuePolicy
from collections import OrderedDict


class TwoTierCache:
    """Two-tier GPU-CPU KV cache manager."""

    CHUNK_SIZE = 32  # Tokens per chunk

    def __init__(
        self,
        gpu_capacity_gb: float = 40,
        cpu_capacity_gb: float = 100,
        num_layers: int = 40,
        device: str = 'cuda:0',
    ):
        """Initialize two-tier cache.

        Args:
            gpu_capacity_gb: GPU memory for KV cache (GB)
            cpu_capacity_gb: CPU memory for KV cache (GB)
            num_layers: Number of transformer layers
            device: GPU device string
        """
        self.gpu_capacity_bytes = int(gpu_capacity_gb * 1024**3)
        self.cpu_capacity_bytes = int(cpu_capacity_gb * 1024**3)
        self.num_layers = num_layers
        self.device = device

        # Cache storage
        self.gpu_cache: Dict[str, KVChunk] = OrderedDict()
        self.cpu_cache: Dict[str, KVChunk] = OrderedDict()
        self.dropped_chunks: Dict[str, KVChunk] = {}

        # Memory tracking
        self.gpu_used_bytes = 0
        self.cpu_used_bytes = 0

        # Statistics
        self.stats = CacheStatistics(
            gpu_free_bytes=self.gpu_capacity_bytes,
            cpu_free_bytes=self.cpu_capacity_bytes,
        )

        # Session tracking
        self.session_chunks: Dict[str, List[str]] = {}  # {session_id: [chunk_keys]}
        self.session_metadata: Dict[str, SessionMetadata] = {}  # {session_id: SessionMetadata}

        # Pinning mechanism: protect chunks from eviction during execution
        # This prevents chunks from being evicted while a batch is being processed
        self.pinned_chunks: set = set()  # {chunk_key: True} for chunks that cannot be evicted
        self.pinned_sessions: set = set()  # {session_id: True} for sessions whose chunks cannot be evicted

        # Eviction policy (retention value based)
        self.eviction_policy = RetentionValuePolicy()

    def store_chunks_for_position(
        self,
        session_id: str,
        chunk_id: int,
        layer_kv_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        context_length: int,
        session_total_chunks: int,
        num_layers: int,
        location: CacheLocation = CacheLocation.GPU,
    ) -> bool:
        """Store KV cache for all layers at a specific position.

        This is the primary method for storing KV cache from a forward pass.
        Creates separate KVChunk objects for each layer (layer-wise chunking).

        Args:
            session_id: Session/conversation ID
            chunk_id: Position in conversation (0, 1, 2, ...)
            layer_kv_dict: {layer_idx: (key_tensor, value_tensor)} for all layers
            context_length: Tokens before this chunk
            session_total_chunks: Total chunks in this session (for position weighting)
            num_layers: Total number of layers in model
            location: Where to store (GPU or CPU)

        Returns:
            True if all layers stored successfully, False if space unavailable
        """
        success = True
        for layer_idx, (k, v) in layer_kv_dict.items():
            chunk = KVChunk(
                session_id=session_id,
                chunk_id=chunk_id,
                layer_idx=layer_idx,
                key_tensor=k,
                value_tensor=v,
                context_length=context_length,
                session_total_chunks=session_total_chunks,
                num_layers=num_layers,
            )
            if not self.store_chunk(chunk, location):
                success = False

        return success

    def store_chunk(
        self,
        chunk: KVChunk,
        location: CacheLocation = CacheLocation.GPU,
    ) -> bool:
        """Store a single KV chunk in cache (internal, use store_chunks_for_position for storage).

        CRITICAL: If chunk already exists in ANY cache tier, replace it with proper cleanup.
        This ensures correctness for BOTH:
        - Recovery path: _store_recovered_chunks() recreates dropped chunks
        - Initial generation path: _store_new_kv_chunks() creates new chunks

        Memory Safety:
        - Checks ALL caches (GPU/CPU/DROPPED) for duplicates before storing
        - Properly updates memory counters when removing old chunks
        - Prevents cross-tier duplication (e.g., chunk in CPU and GPU simultaneously)

        Args:
            chunk: KVChunk to store (typically created by store_chunks_for_position)
            location: Where to store (GPU or CPU)

        Returns:
            True if stored successfully, False if no space
        """
        chunk_size = chunk.size_bytes
        chunk_key = chunk.key

        # Determine target cache
        if location == CacheLocation.GPU:
            target_cache = self.gpu_cache
            current_used = self.gpu_used_bytes
            capacity = self.gpu_capacity_bytes
        else:
            target_cache = self.cpu_cache
            current_used = self.cpu_used_bytes
            capacity = self.cpu_capacity_bytes

        # ✅ CRITICAL: Check ALL caches (not just target) for duplicates
        # This handles cases where:
        # 1. Same tier replacement (recovery recreates a chunk)
        # 2. Cross-tier duplication (chunk evicted to CPU, then recovery stores in GPU)
        # Both paths call store_chunk(), so cleanup must handle both!
        for cache_dict, cache_location in [
            (self.gpu_cache, CacheLocation.GPU),
            (self.cpu_cache, CacheLocation.CPU),
            (self.dropped_chunks, CacheLocation.DROPPED),
        ]:
            if chunk_key in cache_dict:
                old_chunk = cache_dict[chunk_key]
                freed_bytes = old_chunk.size_bytes

                # Update memory tracking (remove old chunk from its current location)
                if cache_location == CacheLocation.GPU:
                    self.gpu_used_bytes -= freed_bytes
                elif cache_location == CacheLocation.CPU:
                    self.cpu_used_bytes -= freed_bytes
                # DROPPED chunks don't have memory tracking

                # Remove from wherever it currently is
                del cache_dict[chunk_key]
                # ✅ Note: Don't remove from session_chunks yet
                # (will be re-added below if needed, with check to avoid duplicates)

        # Make space if needed
        current_used = self.gpu_used_bytes if location == CacheLocation.GPU else self.cpu_used_bytes
        if current_used + chunk_size > capacity:
            freed = self._evict_to_free_space(chunk_size, location)
            if freed < chunk_size:
                print(f"Warning: Could not free enough space for chunk {chunk_key}")

        # Store new chunk
        target_cache[chunk_key] = chunk
        chunk.location = location

        # Update session tracking (only add if new)
        if chunk.session_id not in self.session_chunks:
            self.session_chunks[chunk.session_id] = []
        if chunk_key not in self.session_chunks[chunk.session_id]:
            self.session_chunks[chunk.session_id].append(chunk_key)

        # Update memory tracking
        if location == CacheLocation.GPU:
            self.gpu_used_bytes += chunk_size
        else:
            self.cpu_used_bytes += chunk_size

        self._update_statistics()
        return True

    def get_chunks_for_position(
        self,
        session_id: str,
        chunk_id: int,
    ) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Get KV tensors for all layers at a specific position.

        This gathers all layer chunks at a position and returns them as
        a layer_idx -> (key, value) dictionary.

        Args:
            session_id: Session ID
            chunk_id: Position in session

        Returns:
            Dict[layer_idx: (key_tensor, value_tensor)] if all layers found,
            None if any layer is missing or evicted
        """
        layer_kv = {}

        # Search across all possible layers
        # We assume chunks know their layer_idx, so we can search by pattern
        for cache_dict in [self.gpu_cache, self.cpu_cache]:
            for chunk_key, chunk in cache_dict.items():
                if (chunk.session_id == session_id and
                    chunk.chunk_id == chunk_id):
                    chunk.update_access_time()
                    if cache_dict is self.gpu_cache:
                        self.stats.gpu_hit_count += 1
                    else:
                        self.stats.cpu_hit_count += 1
                    layer_kv[chunk.layer_idx] = (chunk.key_tensor, chunk.value_tensor)

        # Check if we have all layers
        if not layer_kv:
            self.stats.miss_count += 1
            return None

        # If any layer is missing, treat as miss
        # (In dropped chunks, we'd need to recover)
        return layer_kv if len(layer_kv) > 0 else None

    def get_chunk(self, chunk_key: str) -> Optional[KVChunk]:
        """Get a single chunk by key (internal method, use get_chunks_for_position for retrieval).

        Args:
            chunk_key: Key of chunk to retrieve (format: "session:chunk:id:layer:idx")

        Returns:
            KVChunk if found, None otherwise
        """
        # Check GPU cache
        if chunk_key in self.gpu_cache:
            chunk = self.gpu_cache[chunk_key]
            chunk.update_access_time()
            self.stats.gpu_hit_count += 1
            return chunk

        # Check CPU cache
        if chunk_key in self.cpu_cache:
            chunk = self.cpu_cache[chunk_key]
            chunk.update_access_time()
            self.stats.cpu_hit_count += 1
            # Could implement swap-in here
            return chunk

        # Check dropped chunks
        if chunk_key in self.dropped_chunks:
            chunk = self.dropped_chunks[chunk_key]
            # ✅ Return dropped chunks for recovery (token_recovery.py needs them)
            # Don't update access_time (recovery is special case)
            self.stats.miss_count += 1  # Considered a miss since recovery needed
            return chunk

        self.stats.miss_count += 1
        return None

    def get_session_positions(self, session_id: str) -> List[int]:
        """Get all chunk positions (chunk_ids) for a session.

        Since chunks are now layer-indexed, multiple chunks exist per position.
        This returns the unique positions.

        Args:
            session_id: Session ID

        Returns:
            List of chunk_ids (sorted)
        """
        positions = set()
        for cache_dict in [self.gpu_cache, self.cpu_cache, self.dropped_chunks]:
            for chunk in cache_dict.values():
                if chunk.session_id == session_id:
                    positions.add(chunk.chunk_id)
        return sorted(list(positions))

    def pin_session(self, session_id: str) -> None:
        print("@@@@@ pin_session ",session_id)
        """Pin all chunks of a session to prevent eviction.

        Called when a batch execution starts to protect the session's chunks
        from being evicted by other sessions' cache requests.

        Args:
            session_id: Session ID to pin
        """
        self.pinned_sessions.add(session_id)
        if session_id in self.session_chunks:
            for chunk_key in self.session_chunks[session_id]:
                self.pinned_chunks.add(chunk_key)

    def unpin_session(self, session_id: str) -> None:
        """Unpin all chunks of a session to allow eviction.

        Called when batch execution completes to allow the session's chunks
        to be evicted if needed by other sessions.

        Args:
            session_id: Session ID to unpin
        """
        print("unpin session",session_id)
        self.pinned_sessions.discard(session_id)
        if session_id in self.session_chunks:
            for chunk_key in self.session_chunks[session_id]:
                self.pinned_chunks.discard(chunk_key)

    def pin_chunks(self, chunk_keys: List[str]) -> None:
        """Pin specific chunks to prevent eviction.

        Args:
            chunk_keys: List of chunk keys to pin
        """
        for key in chunk_keys:
            self.pinned_chunks.add(key)

    def unpin_chunks(self, chunk_keys: List[str]) -> None:
        """Unpin specific chunks to allow eviction.

        Args:
            chunk_keys: List of chunk keys to unpin
        """
        for key in chunk_keys:
            self.pinned_chunks.discard(key)

    def is_pinned(self, chunk_key: str) -> bool:
        """Check if a chunk is pinned.

        Args:
            chunk_key: Key of chunk to check

        Returns:
            True if pinned, False otherwise
        """
        return chunk_key in self.pinned_chunks

    def evict_session(self, session_id: str) -> int:
        """Evict all chunks of a session from ALL tiers (GPU/CPU/DROPPED).

        Cleanup at session end:
        - Removes chunks from GPU cache
        - Removes chunks from CPU cache
        - Removes chunks from DROPPED list
        - Clears session_chunks tracking

        Since chunks are layer-indexed, this evicts all layer variants
        for all positions in the session.

        Args:
            session_id: Session ID to evict

        Returns:
            Number of bytes freed (GPU + CPU, not DROPPED)
        """
        freed = 0
        if session_id not in self.session_chunks:
            return freed

        chunk_keys = self.session_chunks[session_id][:]  # Copy to avoid modification during iteration
        for chunk_key in chunk_keys:
            # Skip pinned chunks (cannot evict while session is executing)
            if self.is_pinned(chunk_key):
                continue

            if chunk_key in self.gpu_cache:
                chunk = self.gpu_cache.pop(chunk_key)
                freed += chunk.size_bytes
                self.gpu_used_bytes -= chunk.size_bytes
            elif chunk_key in self.cpu_cache:
                chunk = self.cpu_cache.pop(chunk_key)
                freed += chunk.size_bytes
                self.cpu_used_bytes -= chunk.size_bytes
            elif chunk_key in self.dropped_chunks:
                # ✅ CRITICAL: Also remove from DROPPED (no memory to free, just cleanup)
                self.dropped_chunks.pop(chunk_key)

        del self.session_chunks[session_id]
        self._update_statistics()
        return freed

    def swap_chunk_to_cpu(self, chunk_key: str) -> bool:
        """Move chunk from GPU to CPU.

        Args:
            chunk_key: Key of chunk to swap

        Returns:
            True if successful
        """
        print("swap_chunk_to_cpu" ,chunk_key)
        if chunk_key not in self.gpu_cache:
            return False

        chunk = self.gpu_cache.pop(chunk_key)
        chunk_size = chunk.size_bytes

        # Make space in CPU if needed
        if self.cpu_used_bytes + chunk_size > self.cpu_capacity_bytes:
            freed = self._evict_to_free_space(chunk_size, CacheLocation.CPU)
            if freed < chunk_size:
                # Can't fit in CPU, drop instead
                self.dropped_chunks[chunk_key] = chunk
                chunk.location = CacheLocation.DROPPED
                self.gpu_used_bytes -= chunk_size
                self._update_statistics()
                return False

        # Move to CPU
        chunk.move_to_cpu()
        self.cpu_cache[chunk_key] = chunk
        self.gpu_used_bytes -= chunk_size
        self.cpu_used_bytes += chunk_size

        self._update_statistics()
        return True

    def swap_chunk_to_gpu(self, chunk_key: str) -> bool:
        """Move chunk from CPU to GPU.

        Args:
            chunk_key: Key of chunk to swap

        Returns:
            True if successful
        """
        if chunk_key not in self.cpu_cache:
            return False
        print("swap_chunk_to_gpu", chunk_key)
        chunk = self.cpu_cache.pop(chunk_key)
        chunk_size = chunk.size_bytes

        # Make space in GPU if needed
        if self.gpu_used_bytes + chunk_size > self.gpu_capacity_bytes:
            freed = self._evict_to_free_space(chunk_size, CacheLocation.GPU)
            if freed < chunk_size:
                print(f"Warning: Could not free enough GPU space for chunk {chunk_key}")

        # Move to GPU
        chunk.move_to_gpu(self.device)
        self.gpu_cache[chunk_key] = chunk
        self.cpu_used_bytes -= chunk_size
        self.gpu_used_bytes += chunk_size

        self._update_statistics()
        return True

    def _evict_to_free_space(self, required_bytes: int, location: CacheLocation) -> int:
        """Evict chunks to free requested space using retention value policy.

        Uses the retention value eviction policy from the paper:
        V = Cost(context_length) / time_inactive

        Chunks with LOW retention value are evicted first (cheap to recompute).
        Key insight: Leading tokens have low context_length → evicted first!

        CRITICAL: Pinned chunks are NEVER evicted, ensuring cache consistency
        when multiple sessions are executing concurrently.

        Args:
            required_bytes: Amount of space needed
            location: Which tier to evict from (GPU or CPU)

        Returns:
            Amount of space freed
        """
        freed = 0

        if location == CacheLocation.GPU:
            cache = self.gpu_cache
        else:
            cache = self.cpu_cache

        # Get all chunks in this tier
        chunks_to_rank = list(cache.values())
        if not chunks_to_rank:
            return freed

        # Get eviction candidates ranked by retention value
        # select_chunks_to_evict returns chunks sorted by retention value
        # ✅ Pass cache=self to ensure SessionMetadata is used for position weights
        eviction_candidates = self.eviction_policy.select_chunks_to_evict(
            chunks_to_rank, required_bytes, cache=self
        )
        print("eviction_candidates")
        print(eviction_candidates)
        # Evict chunks in order (skip pinned chunks)
        for chunk_key in eviction_candidates:
            if freed >= required_bytes:
                break

            if chunk_key not in cache:
                # print("")
                continue

            # IMPORTANT: Skip pinned chunks (cannot evict while being executed)
            if self.is_pinned(chunk_key):
                # print("reason pin")
                continue

            chunk = cache.pop(chunk_key)
            freed += chunk.size_bytes

            # ⚠️ DO NOT remove from session_chunks yet!
            # chunk will move to another tier (CPU or DROPPED)
            # We'll add it back after move completes

            if location == CacheLocation.GPU:
                self.gpu_used_bytes -= chunk.size_bytes
                # Try to move to CPU (hierarchical eviction)
                if self.cpu_used_bytes + chunk.size_bytes <= self.cpu_capacity_bytes:
                    # CPU has space, move chunk there
                    chunk.move_to_cpu()
                    self.cpu_cache[chunk_key] = chunk
                    self.cpu_used_bytes += chunk.size_bytes
                    # ✅ Chunk still in session_chunks (now in CPU tier)
                else:
                    # CPU is full - need to evict from CPU to make space
                    # This implements the hierarchical structure: GPU → CPU → DROPPED
                    cpu_chunks = list(self.cpu_cache.values())
                    if cpu_chunks:
                        # Use retention value policy to select what to drop from CPU
                        # ✅ Pass cache=self to ensure SessionMetadata is used
                        cpu_evict_candidates = self.eviction_policy.select_chunks_to_evict(
                            cpu_chunks, chunk.size_bytes, cache=self
                        )

                        # Drop chunks from CPU using retention value ranking
                        cpu_freed = 0
                        for drop_key in cpu_evict_candidates:
                            if drop_key not in self.cpu_cache:
                                continue

                            drop_chunk = self.cpu_cache.pop(drop_key)
                            # ⚠️ IMPORTANT: DROPPED chunks stay in CPU memory!
                            # They're in dropped_chunks dict but still take CPU space
                            # Don't subtract from cpu_used_bytes - the space is still used
                            cpu_freed += drop_chunk.size_bytes

                            # Move to DROPPED tier (but tensors remain in CPU memory)
                            self.dropped_chunks[drop_key] = drop_chunk
                            drop_chunk.location = CacheLocation.DROPPED
                            # ✅ Keep in session_chunks (tracking all tiers)
                            # ✅ Keep in cpu_used_bytes (DROPPED chunks use CPU memory)

                            if cpu_freed >= chunk.size_bytes:
                                break

                    # Now try to move chunk to CPU (space should be available)
                    if self.cpu_used_bytes + chunk.size_bytes <= self.cpu_capacity_bytes:
                        chunk.move_to_cpu()
                        self.cpu_cache[chunk_key] = chunk
                        self.cpu_used_bytes += chunk.size_bytes
                        # ✅ Chunk still in session_chunks (now in CPU tier)
                    else:
                        # Still no space, drop the chunk directly
                        self.dropped_chunks[chunk_key] = chunk
                        chunk.location = CacheLocation.DROPPED
                        # ✅ Keep in session_chunks (tracking DROPPED tier too)

            else:
                # Evicting from CPU → DROPPED tier
                # ⚠️ IMPORTANT: Don't subtract from cpu_used_bytes!
                # DROPPED chunks stay in CPU memory (recovery cache)
                # Chunk moves: cpu_cache → dropped_chunks (both CPU memory)
                # Therefore: cpu_used_bytes unchanged

                self.dropped_chunks[chunk_key] = chunk
                chunk.location = CacheLocation.DROPPED

                # ✅ Keep in session_chunks (chunk still exists, now DROPPED)
                # (Will be cleaned up when session ends via evict_session())

        self._update_statistics()
        return freed

    def get_session_chunks(self, session_id: str) -> List[KVChunk]:
        """Get all chunks for a session (all layers, all positions).

        With layer-wise chunking, returns chunks for all positions and all layers
        in the session.

        Args:
            session_id: Session ID

        Returns:
            List of KVChunk objects for session (order: position, then layer)
        """
        chunks = []
        if session_id in self.session_chunks:
            for chunk_key in self.session_chunks[session_id]:
                chunk = None
                if chunk_key in self.gpu_cache:
                    chunk = self.gpu_cache[chunk_key]
                elif chunk_key in self.cpu_cache:
                    chunk = self.cpu_cache[chunk_key]
                elif chunk_key in self.dropped_chunks:
                    chunk = self.dropped_chunks[chunk_key]

                if chunk:
                    chunks.append(chunk)

        return chunks

    def get_checkerboard_summary(self) -> str:
        """Get a text summary of cache state in checkerboard view.

        Visualization:
                    Layer0  Layer1  Layer2 ... Layer39
        Session1:    ■       ■       ■             ■
        Session2:    ▓       ▓       ▓             ▓   (CPU)
        Session3:    ░       ░       ░             ░   (DROPPED)

        ■ = GPU (hot)
        ▓ = CPU (warm)
        ░ = DROPPED (cold)
        """
        sessions = set()
        for cache_dict in [self.gpu_cache, self.cpu_cache, self.dropped_chunks]:
            for chunk in cache_dict.values():
                sessions.add(chunk.session_id)

        sessions = sorted(list(sessions))

        lines = []
        lines.append("Cache Checkerboard View (Layer-wise):")
        lines.append("")

        for session_id in sessions:
            session_chunks = self.get_session_chunks(session_id)

            # Group by position
            positions = {}
            for chunk in session_chunks:
                if chunk.chunk_id not in positions:
                    positions[chunk.chunk_id] = {}
                positions[chunk.chunk_id][chunk.layer_idx] = chunk

            # Format: session_id | chunk0 | chunk1 | chunk2 | ...
            line = f"{session_id:15} |"
            for pos in sorted(positions.keys()):
                # Just show summary per position
                num_layers_cached = len(positions[pos])
                if num_layers_cached == chunk.num_layers:
                    status = "✓"  # All layers cached
                else:
                    status = f"{num_layers_cached}/{chunk.num_layers}"
                line += f" P{pos}:{status} |"

            lines.append(line)

        lines.append("")
        lines.append(f"Summary:")
        lines.append(f"  GPU chunks: {len(self.gpu_cache)}")
        lines.append(f"  CPU chunks: {len(self.cpu_cache)}")
        lines.append(f"  DROPPED chunks: {len(self.dropped_chunks)}")

        return "\n".join(lines)

    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics.

        Returns:
            CacheStatistics object
        """
        return self.stats

    def _update_statistics(self) -> None:
        """Update statistics."""
        self.stats.gpu_used_bytes = self.gpu_used_bytes
        self.stats.gpu_free_bytes = self.gpu_capacity_bytes - self.gpu_used_bytes
        self.stats.cpu_used_bytes = self.cpu_used_bytes
        self.stats.cpu_free_bytes = self.cpu_capacity_bytes - self.cpu_used_bytes
        self.stats.num_gpu_chunks = len(self.gpu_cache)
        self.stats.num_cpu_chunks = len(self.cpu_cache)
        self.stats.num_dropped_chunks = len(self.dropped_chunks)

    def reset(self) -> None:
        """Clear all caches."""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.dropped_chunks.clear()
        self.session_chunks.clear()
        self.gpu_used_bytes = 0
        self.cpu_used_bytes = 0
        self._update_statistics()

    def get_memory_stats_str(self) -> str:
        """Get formatted memory statistics string.

        Returns:
            Formatted statistics string
        """
        gpu_gb = self.gpu_used_bytes / (1024**3)
        gpu_cap = self.gpu_capacity_bytes / (1024**3)
        cpu_gb = self.cpu_used_bytes / (1024**3)
        cpu_cap = self.cpu_capacity_bytes / (1024**3)

        return (
            f"GPU: {gpu_gb:.2f}/{gpu_cap:.2f} GB | "
            f"CPU: {cpu_gb:.2f}/{cpu_cap:.2f} GB | "
            f"GPU chunks: {len(self.gpu_cache)} | "
            f"CPU chunks: {len(self.cpu_cache)} | "
            f"Dropped chunks: {len(self.dropped_chunks)}"
        )

    def update_session_tokens(
        self,
        session_id: str,
        input_tokens: int,
        generated_tokens: int,
    ) -> None:
        """Update cumulative token count for a session.

        Called after each generation to track total tokens in session.
        This enables accurate token recovery and chunk management.

        Args:
            session_id: Session ID
            input_tokens: New input tokens in this request
            generated_tokens: New generated tokens in this request
        """
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = SessionMetadata(session_id)

        metadata = self.session_metadata[session_id]
        metadata.total_input_tokens += input_tokens
        metadata.total_generated_tokens += generated_tokens
        metadata.update_last_accessed()

    def get_session_total_tokens(self, session_id: str) -> int:
        """Get cumulative total tokens for a session.

        Args:
            session_id: Session ID

        Returns:
            Total number of tokens (input + generated) for this session
        """
        if session_id not in self.session_metadata:
            return 0
        return self.session_metadata[session_id].total_tokens

    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Get metadata for a session.

        Args:
            session_id: Session ID

        Returns:
            SessionMetadata if exists, None otherwise
        """
        return self.session_metadata.get(session_id, None)
