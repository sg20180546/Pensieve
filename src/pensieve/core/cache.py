"""Two-tier KV cache management for Pensieve."""

import torch
import time
import threading
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

        # Thread synchronization lock for all cache operations
        # Protects: gpu_cache, cpu_cache, dropped_chunks, gpu_used_bytes, cpu_used_bytes
        self.cache_lock = threading.RLock()  # RLock allows same thread to acquire multiple times

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
        capacity = self.gpu_capacity_bytes if location == CacheLocation.GPU else self.cpu_capacity_bytes

        # PHASE 1: Check space needed (quick, under lock)
        with self.cache_lock:
            current_used = self.gpu_used_bytes if location == CacheLocation.GPU else self.cpu_used_bytes
            need_eviction = (current_used + chunk_size > capacity)

        # PHASE 2: Evict if needed (heavy work, NO lock)
        if need_eviction:
            freed = self._evict_to_free_space(chunk_size, location)
            if freed < chunk_size:
                # Eviction failed, try cascade
                if location == CacheLocation.GPU:
                    success = self._demote_to_cpu_with_eviction(chunk)
                    if success:
                        return True
                    else:
                        print(f"Warning: Could not store chunk {chunk_key} - all tiers full")
                        return False
                else:
                    print(f"Warning: Could not free enough space for chunk {chunk_key} - CPU full")
                    return False

        # PHASE 3: Store chunk (quick, under lock)
        with self.cache_lock:
            # ✅ CRITICAL: Check ALL caches for duplicates
            for cache_dict, cache_location in [
                (self.gpu_cache, CacheLocation.GPU),
                (self.cpu_cache, CacheLocation.CPU),
                (self.dropped_chunks, CacheLocation.DROPPED),
            ]:
                if chunk_key in cache_dict:
                    old_chunk = cache_dict[chunk_key]
                    freed_bytes = old_chunk.size_bytes
                    if cache_location == CacheLocation.GPU:
                        self.gpu_used_bytes -= freed_bytes
                    elif cache_location == CacheLocation.CPU:
                        self.cpu_used_bytes -= freed_bytes
                    del cache_dict[chunk_key]

            # Store new chunk
            if location == CacheLocation.GPU:
                self.gpu_cache[chunk_key] = chunk
                self.gpu_used_bytes += chunk_size
            else:
                self.cpu_cache[chunk_key] = chunk
                self.cpu_used_bytes += chunk_size
            chunk.location = location

            # Update session tracking
            if chunk.session_id not in self.session_chunks:
                self.session_chunks[chunk.session_id] = []
            if chunk_key not in self.session_chunks[chunk.session_id]:
                self.session_chunks[chunk.session_id].append(chunk_key)

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
        # PHASE 1: Snapshot caches (quick, under lock)
        with self.cache_lock:
            gpu_snapshot = dict(self.gpu_cache)
            cpu_snapshot = dict(self.cpu_cache)

        # PHASE 2: Search in snapshots (no lock, can iterate safely)
        layer_kv = {}
        for cache_dict in [gpu_snapshot, cpu_snapshot]:
            for chunk_key, chunk in cache_dict.items():
                if (chunk.session_id == session_id and
                    chunk.chunk_id == chunk_id):
                    layer_kv[chunk.layer_idx] = (chunk.key_tensor, chunk.value_tensor)

        # PHASE 3: Update access time and stats (quick, under lock)
        with self.cache_lock:
            if layer_kv:
                # Update access times for found chunks
                for cache_dict in [self.gpu_cache, self.cpu_cache]:
                    for chunk in cache_dict.values():
                        if (chunk.session_id == session_id and
                            chunk.chunk_id == chunk_id):
                            chunk.update_access_time()
                            if cache_dict is self.gpu_cache:
                                self.stats.gpu_hit_count += 1
                            else:
                                self.stats.cpu_hit_count += 1
            else:
                self.stats.miss_count += 1

        return layer_kv if layer_kv else None

    def get_chunk(self, chunk_key: str) -> Optional[KVChunk]:
        """Get a single chunk by key (internal method, use get_chunks_for_position for retrieval).

        Args:
            chunk_key: Key of chunk to retrieve (format: "session:chunk:id:layer:idx")

        Returns:
            KVChunk if found, None otherwise
        """
        with self.cache_lock:
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
        with self.cache_lock:
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

    def _demote_to_cpu_with_eviction(self, chunk: KVChunk) -> bool:
        """Demote chunk to CPU tier, evicting from CPU if necessary.

        Cascade strategy when GPU is full:
        1. Try to move chunk to CPU
        2. If CPU is full, evict cheapest chunk from CPU to DROPPED
        3. Then move the original chunk to CPU

        Args:
            chunk: KVChunk to demote from GPU to CPU

        Returns:
            True if successfully demoted to CPU, False if even CPU is full
        """
        # THREAD-SAFE: Acquire lock to protect cache modifications
        # RLock allows reentrant acquisition (safe if already held by caller)
        with self.cache_lock:
            chunk_size = chunk.size_bytes
            chunk_key = chunk.key
            print("_demote_to_cpu_with_eviction",chunk_key)
            # Check if CPU has space
            if self.cpu_used_bytes + chunk_size <= self.cpu_capacity_bytes:
                # CPU has space, move directly
                chunk.move_to_cpu()
                self.cpu_cache[chunk_key] = chunk
                self.cpu_used_bytes += chunk_size
                chunk.location = CacheLocation.CPU
                self._update_statistics()
                return True

            # CPU is full - evict cheapest chunk from CPU to DROPPED
            cpu_chunks = list(self.cpu_cache.values())
            if not cpu_chunks:
                # No chunks in CPU to evict, can't demote
                return False

            # Get eviction candidates from CPU (cost-based ranking)
            cpu_evict_candidates = self.eviction_policy.select_chunks_to_evict(
                cpu_chunks, chunk_size, cache=self
            )

            # Evict from CPU until we have space
            cpu_freed = 0
            for evict_key in cpu_evict_candidates:
                if cpu_freed >= chunk_size:
                    break

                if evict_key not in self.cpu_cache:
                    continue

                # Skip pinned chunks
                if self.is_pinned(evict_key):
                    continue

                # Move from CPU to DROPPED
                evict_chunk = self.cpu_cache.pop(evict_key)
                evict_chunk.location = CacheLocation.DROPPED
                self.dropped_chunks[evict_key] = evict_chunk
                cpu_freed += evict_chunk.size_bytes
                # ✅ Keep cpu_used_bytes unchanged (DROPPED chunks use CPU memory)

            # Now try to move original chunk to CPU
            if self.cpu_used_bytes + chunk_size <= self.cpu_capacity_bytes:
                chunk.move_to_cpu()
                self.cpu_cache[chunk_key] = chunk
                self.cpu_used_bytes += chunk_size
                chunk.location = CacheLocation.CPU
                self._update_statistics()
                return True

            return False

    def swap_chunk_to_cpu(self, chunk_key: str) -> bool:
        """Move chunk from GPU to CPU.

        Args:
            chunk_key: Key of chunk to swap

        Returns:
            True if successful
        """
        print("swap_chunk_to_cpu" ,chunk_key)

        # PHASE 1: Check if chunk exists and get size (quick, under lock)
        with self.cache_lock:
            if chunk_key not in self.gpu_cache:
                return False
            chunk = self.gpu_cache[chunk_key]
            chunk_size = chunk.size_bytes
            need_eviction = (self.cpu_used_bytes + chunk_size > self.cpu_capacity_bytes)

        # PHASE 2: Evict if needed (heavy work, NO lock)
        if need_eviction:
            freed = self._evict_to_free_space(chunk_size, CacheLocation.CPU)
            if freed < chunk_size:
                # Can't fit, drop to DROPPED tier
                with self.cache_lock:
                    if chunk_key in self.gpu_cache:
                        chunk = self.gpu_cache.pop(chunk_key)
                        self.dropped_chunks[chunk_key] = chunk
                        chunk.location = CacheLocation.DROPPED
                        self.gpu_used_bytes -= chunk_size
                        self._update_statistics()
                print("WHY FAILE?? swap_chunk_to_cpu")
                return False

        # PHASE 3: Move to CPU (quick, under lock)
        with self.cache_lock:
            # Re-check chunk still exists
            if chunk_key not in self.gpu_cache:
                return False

            chunk = self.gpu_cache.pop(chunk_key)
            chunk.move_to_cpu()
            self.cpu_cache[chunk_key] = chunk
            self.gpu_used_bytes -= chunk_size
            self.cpu_used_bytes += chunk_size
            self._update_statistics()

        return True

    def swap_chunk_to_gpu(self, chunk_key: str) -> bool:
        """Move chunk from CPU to GPU with cascade fallback.

        If GPU is full:
        1. Try to evict chunks from GPU
        2. If still not enough space, keep chunk in CPU (don't move)
        3. Return False to indicate swap failed

        Args:
            chunk_key: Key of chunk to swap

        Returns:
            True if successful, False if GPU still full (chunk stays in CPU)
        """
        print("swap_chunk_to_gpu",chunk_key)

        # PHASE 1: Check if chunk exists and get size (quick, under lock)
        with self.cache_lock:
            if chunk_key not in self.cpu_cache:
                print("WHY@@@@@ swap_chunk_to_gpu")
                return False
            chunk = self.cpu_cache[chunk_key]
            chunk_size = chunk.size_bytes
            need_eviction = (self.gpu_used_bytes + chunk_size > self.gpu_capacity_bytes)

        # PHASE 2: Evict if needed (heavy work, NO lock)
        if need_eviction:
            freed = self._evict_to_free_space(chunk_size, CacheLocation.GPU)
            if freed < chunk_size:
                return False

        # PHASE 3: Move chunk (quick, under lock)
        with self.cache_lock:
            # Re-check chunk still exists (another thread might have moved it)
            if chunk_key not in self.cpu_cache:
                return False

            chunk = self.cpu_cache.pop(chunk_key)
            chunk.move_to_gpu(self.device)
            self.gpu_cache[chunk_key] = chunk
            self.cpu_used_bytes -= chunk_size
            self.gpu_used_bytes += chunk_size
            self._update_statistics()
            print("swap_chunk_to_gpu return",chunk_key)

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
        # THREAD-SAFE: Acquire lock to protect cache modifications
        # RLock allows reentrant acquisition (safe if already held by caller)
        with self.cache_lock:
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
            # print("eviction_candidates")
            # print(eviction_candidates)
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
                print("ok i am evicted",chunk_key)
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
        with self.cache_lock:
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
        with self.cache_lock:
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

    def get_session_total_chunk_size(self, session_id: str) -> int:
        """Get total size in bytes for all chunks of a session (all tiers).

        Calculates the total memory requirement for a session across
        GPU, CPU, and DROPPED tiers.

        Thread-safe: Creates snapshots to avoid iteration issues during
        concurrent eviction/removal by other threads.

        Args:
            session_id: Session ID

        Returns:
            Total size in bytes
        """
        total_size = 0

        # ✅ SNAPSHOT: Create local copy of chunk_keys to avoid iteration issues
        # during concurrent eviction by other threads
        chunk_keys = []
        if session_id in self.session_chunks:
            chunk_keys = self.session_chunks[session_id][:]  # Shallow copy of list

        # ✅ SNAPSHOT: Create local copies of all cache dicts
        gpu_cache_snapshot = dict(self.gpu_cache)
        cpu_cache_snapshot = dict(self.cpu_cache)
        dropped_chunks_snapshot = dict(self.dropped_chunks)

        # Now search in snapshots (safe from concurrent modifications)
        for chunk_key in chunk_keys:
            chunk = None
            # Search in all tiers (via snapshots)
            for cache_dict in [gpu_cache_snapshot, cpu_cache_snapshot, dropped_chunks_snapshot]:
                if chunk_key in cache_dict:
                    chunk = cache_dict[chunk_key]
                    break
            if chunk:
                total_size += chunk.size_bytes

        return total_size

    def print_session_chunks_status(self, session_id: str) -> str:
        """Print detailed status of all chunks for a session.

        Shows each chunk's tier location (GPU/CPU/DROPPED), size, and other metadata.
        Thread-safe: Uses snapshots to handle concurrent modifications.

        Args:
            session_id: Session ID to display

        Returns:
            Formatted status string
        """
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"Session Chunks Status: {session_id}")
        lines.append(f"{'='*80}")

        # ✅ SNAPSHOT: Get chunk keys and cache snapshots
        chunk_keys = []
        if session_id in self.session_chunks:
            chunk_keys = self.session_chunks[session_id][:]

        gpu_cache_snapshot = dict(self.gpu_cache)
        cpu_cache_snapshot = dict(self.cpu_cache)
        dropped_chunks_snapshot = dict(self.dropped_chunks)

        if not chunk_keys:
            lines.append(f"No chunks found for session {session_id}")
            result = "\n".join(lines)
            print(result)
            return result

        # Count chunks by tier
        gpu_count = 0
        cpu_count = 0
        dropped_count = 0
        total_size = 0

        # Detailed chunk info
        lines.append(f"\nDetailed Chunk List:")
        lines.append(f"{'-'*80}")
        lines.append(f"{'Chunk Key':<50} {'Tier':<10} {'Size (MB)':<12}")
        lines.append(f"{'-'*80}")

        for chunk_key in chunk_keys:
            chunk = None
            tier = "MISSING"

            # Search in all tiers
            if chunk_key in gpu_cache_snapshot:
                chunk = gpu_cache_snapshot[chunk_key]
                tier = "GPU"
                gpu_count += 1
            elif chunk_key in cpu_cache_snapshot:
                chunk = cpu_cache_snapshot[chunk_key]
                tier = "CPU"
                cpu_count += 1
            elif chunk_key in dropped_chunks_snapshot:
                chunk = dropped_chunks_snapshot[chunk_key]
                tier = "DROPPED"
                dropped_count += 1

            if chunk:
                size_mb = chunk.size_bytes / (1024**2)
                total_size += chunk.size_bytes
                lines.append(f"{chunk_key:<50} {tier:<10} {size_mb:<12.2f}")
            else:
                lines.append(f"{chunk_key:<50} {tier:<10} {'N/A':<12}")

        lines.append(f"{'-'*80}")
        lines.append(f"\nSummary:")
        lines.append(f"  Total chunks: {len(chunk_keys)}")
        lines.append(f"  GPU chunks: {gpu_count}")
        lines.append(f"  CPU chunks: {cpu_count}")
        lines.append(f"  DROPPED chunks: {dropped_count}")
        lines.append(f"  Total size: {total_size / (1024**3):.2f}GB")
        lines.append(f"  GPU capacity: {self.gpu_capacity_bytes / (1024**3):.2f}GB")
        lines.append(f"  CPU capacity: {self.cpu_capacity_bytes / (1024**3):.2f}GB")
        lines.append(f"{'='*80}\n")

        result = "\n".join(lines)
        print(result)
        return result

    def print_all_sessions_status(self) -> str:
        """Print status of all sessions and their chunks in summary format.

        Shows all sessions with their chunk distribution across tiers.
        Thread-safe: Uses snapshots to handle concurrent modifications.

        Returns:
            Formatted status string
        """
        lines = []
        lines.append(f"\n{'='*100}")
        lines.append(f"All Sessions Cache Status")
        lines.append(f"{'='*100}")

        # ✅ SNAPSHOT: Get all session_ids and cache snapshots
        all_session_ids = set(self.session_chunks.keys())

        gpu_cache_snapshot = dict(self.gpu_cache)
        cpu_cache_snapshot = dict(self.cpu_cache)
        dropped_chunks_snapshot = dict(self.dropped_chunks)

        if not all_session_ids:
            lines.append("No sessions found")
            result = "\n".join(lines)
            print(result)
            return result

        # Header
        lines.append(f"\n{'Session ID':<30} {'GPU Chunks':<15} {'CPU Chunks':<15} {'DROPPED':<15} {'Total Size (GB)':<20}")
        lines.append(f"{'-'*100}")

        total_gpu_chunks = 0
        total_cpu_chunks = 0
        total_dropped_chunks = 0
        total_overall_size = 0

        # Process each session
        for session_id in sorted(all_session_ids):
            chunk_keys = self.session_chunks[session_id]

            gpu_count = 0
            cpu_count = 0
            dropped_count = 0
            session_size = 0

            # Count chunks by tier for this session
            for chunk_key in chunk_keys:
                chunk = None

                if chunk_key in gpu_cache_snapshot:
                    chunk = gpu_cache_snapshot[chunk_key]
                    gpu_count += 1
                elif chunk_key in cpu_cache_snapshot:
                    chunk = cpu_cache_snapshot[chunk_key]
                    cpu_count += 1
                elif chunk_key in dropped_chunks_snapshot:
                    chunk = dropped_chunks_snapshot[chunk_key]
                    dropped_count += 1

                if chunk:
                    session_size += chunk.size_bytes

            size_gb = session_size / (1024**3)
            lines.append(
                f"{session_id:<30} {gpu_count:<15} {cpu_count:<15} {dropped_count:<15} {size_gb:<20.2f}"
            )

            total_gpu_chunks += gpu_count
            total_cpu_chunks += cpu_count
            total_dropped_chunks += dropped_count
            total_overall_size += session_size

        # Summary footer
        lines.append(f"{'-'*100}")
        lines.append(
            f"{'TOTAL':<30} {total_gpu_chunks:<15} {total_cpu_chunks:<15} {total_dropped_chunks:<15} "
            f"{total_overall_size / (1024**3):<20.2f}"
        )
        lines.append(f"\nCapacity Status:")
        lines.append(
            f"  GPU: {self.gpu_used_bytes / (1024**3):.2f}GB / {self.gpu_capacity_bytes / (1024**3):.2f}GB "
            f"(used: {100 * self.gpu_used_bytes / self.gpu_capacity_bytes:.1f}%)"
        )
        lines.append(
            f"  CPU: {self.cpu_used_bytes / (1024**3):.2f}GB / {self.cpu_capacity_bytes / (1024**3):.2f}GB "
            f"(used: {100 * self.cpu_used_bytes / self.cpu_capacity_bytes:.1f}%)"
        )
        lines.append(f"{'='*100}\n")

        result = "\n".join(lines)
        print(result)
        return result
