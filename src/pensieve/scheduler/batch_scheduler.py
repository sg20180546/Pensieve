"""Iteration-level batch scheduler for Pensieve inference."""

import torch
from typing import List, Tuple, Dict, Optional
from collections import deque
import time

from pensieve.core.types import (
    Request,
    Batch,
    CachePlan,
    CacheLocation,
)
from pensieve.core.cache import TwoTierCache


class BatchScheduler:
    """Manages request queue and forms batches for each iteration.

    Key features (Paper §4.2):
    - Iteration-level batching: Add new requests to queue
    - Unified scheduling: Scheduler automatically mixes requests
    - Cache planning: Determine swap operations BEFORE execution
    - Memory-aware: Respects GPU/CPU capacity constraints

    Design:
    - request_queue: All incoming requests (unified handling)
    - completed_requests: Track finished requests
    - Worker's _custom_generate() automatically handles PREFILL/GENERATION phases
      based on step number (step 0 = prefill, step > 0 = generation)
    """

    def __init__(
        self,
        cache: TwoTierCache,
        max_batch_size: int = 8,
    ):
        """Initialize batch scheduler.

        Args:
            cache: TwoTierCache instance for memory checking
            max_batch_size: Maximum requests per batch
        """
        self.cache = cache
        self.max_batch_size = max_batch_size

        # Request management
        self.request_queue: deque = deque()  # All incoming requests
        self.completed_requests: Dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """Add new request to queue.

        Args:
            request: Request to add (no phase management needed)

        Note:
            All requests are treated equally. Worker will automatically
            handle PREFILL (step 0) vs GENERATION (step > 0) based on
            generation loop progress.
        """
        self.request_queue.append(request)

    def add_requests(self, requests: List[Request]) -> None:
        """Add multiple requests to queue at once.

        ✅ Used for async batching: collect multiple requests and add together.

        Args:
            requests: List of Request objects to add
        """
        self.request_queue.extend(requests)

    def form_next_batch(self) -> Tuple[Batch, CachePlan]:
        """Form batch for next iteration.

        Strategy (Unified with Pinning Awareness):
        1. PREFER requests from unpinned sessions (avoid eviction conflicts)
        2. Pull up to max_batch_size requests from queue
        3. Create cache plan for batch
        4. All requests handled uniformly - Worker automatically handles
           PREFILL (step 0) vs GENERATION (step > 0) in generation loop

        CRITICAL: If another batch is still executing (pinned), we avoid
        adding requests from currently-executing sessions to prevent
        eviction conflicts.

        Returns:
            batch: Batch with requests (no phase distinction)
            cache_plan: Swap operations needed before execution

        Note:
            True unified batching - scheduler doesn't distinguish between
            prefill/generation. Worker's generation loop naturally handles
            phase transitions via step counter.

            By avoiding pinned sessions, we ensure eviction only affects
            unpinned chunks, preventing stalls when all chunks are protected.
        """
        batch = Batch(batch_id=f"batch_{int(time.time() * 1000)}")

        # Strategy: Prefer unpinned sessions to avoid eviction conflicts
        # This prevents situations where all chunks are pinned and new requests cannot be served
        #
        # Algorithm: Round-robin through queue, taking unpinned requests first
        # If a request is from a pinned session, defer it to back of queue
        skipped_reqs = []

        while len(batch.requests) < self.max_batch_size and len(self.request_queue) > 0:
            req = self.request_queue.popleft()

            # Check if this request's session is currently pinned (being executed)
            if req.session_id in self.cache.pinned_sessions:
                # Defer pinned requests to back of queue
                skipped_reqs.append(req)
            else:
                # Add unpinned requests to batch
                batch.add_request(req)
                # print("batch len",len(batch))
                # print(batch)
                break

        # Return skipped requests to back of queue for next batch
        for req in skipped_reqs:
            self.request_queue.append(req)

        # Create cache plan for this batch
        cache_plan = self.create_cache_plan(batch)

        return batch, cache_plan

    def create_cache_plan(self, batch: Batch) -> CachePlan:
        """Create plan for cache operations before batch execution.

        Strategy for each request:
        1. Check which chunks are currently in cache (GPU/CPU/DROPPED)
        2. Identify chunks needed for this batch
        3. Plan swaps: GPU → CPU if needed, CPU → GPU if needed
        4. Handle dropped chunks (will need recovery during execution)

        Args:
            batch: Batch to plan cache operations for

        Returns:
            CachePlan with swap operations organized by priority
        """
        cache_plan = CachePlan(batch_id=batch.batch_id)

        # Track chunks we need and their current locations
        chunks_needed: Dict[str, str] = {}  # {chunk_key: current_location}
        chunks_to_swap_in: List[str] = []
        chunks_to_swap_out: List[str] = []

        # 1. Identify all chunks needed for this batch
        for req in batch.requests:
            session_id = req.session_id
            # Get all available positions (chunks) for this session
            positions = self.cache.get_session_positions(session_id)

            # Check each position's chunk status (per-position chunks contain all layers)
            for pos in positions:
                chunk_key = f"{session_id}:chunk:{pos}"
                chunk = self.cache.get_chunk(chunk_key)

                if chunk is None:
                    continue

                if chunk.location == CacheLocation.GPU:
                    chunks_needed[chunk_key] = "GPU"
                elif chunk.location == CacheLocation.CPU:
                    chunks_needed[chunk_key] = "CPU"
                elif chunk.location == CacheLocation.DROPPED:
                    chunks_needed[chunk_key] = "DROPPED"

        # PHASE 1: Snapshot cache state (quick, under lock)
        with self.cache.cache_lock:
            cpu_cache_snapshot = dict(self.cache.cpu_cache)
            gpu_cache_snapshot = dict(self.cache.gpu_cache)
            dropped_chunks_snapshot = dict(self.cache.dropped_chunks)
            gpu_used_bytes = self.cache.gpu_used_bytes
            gpu_capacity_bytes = self.cache.gpu_capacity_bytes

        # PHASE 2: Identify dropped chunks FIRST (need their size for eviction planning)
        dropped_chunks = {
            key: chunk
            for key, chunk in dropped_chunks_snapshot.items()
            if key in chunks_needed
        }
        for chunk_key, chunk in dropped_chunks.items():
            parts = chunk_key.split(":")
            if len(parts) >= 2:
                session_id = parts[0]
                if session_id not in cache_plan.chunks_to_recompute:
                    cache_plan.chunks_to_recompute[session_id] = []
                cache_plan.chunks_to_recompute[session_id].append(chunk_key)

        # Estimate recovery size: dropped chunks have size_bytes=0,
        # so use average size of existing GPU/CPU chunks from same session
        recovery_estimated_bytes = 0
        if cache_plan.chunks_to_recompute:
            # Build session → avg chunk size map from live chunks
            session_avg_size: Dict[str, int] = {}
            all_live_chunks = list(gpu_cache_snapshot.values()) + list(cpu_cache_snapshot.values())
            session_sizes: Dict[str, List[int]] = {}
            for c in all_live_chunks:
                if c.size_bytes > 0:
                    if c.session_id not in session_sizes:
                        session_sizes[c.session_id] = []
                    session_sizes[c.session_id].append(c.size_bytes)
            for sid, sizes in session_sizes.items():
                session_avg_size[sid] = sum(sizes) // len(sizes) if sizes else 0

            # Global fallback if session has no live chunks
            all_sizes = [s for sizes in session_sizes.values() for s in sizes]
            global_avg = sum(all_sizes) // len(all_sizes) if all_sizes else 0

            for session_id, chunk_keys in cache_plan.chunks_to_recompute.items():
                avg = session_avg_size.get(session_id, global_avg)
                recovery_estimated_bytes += avg * len(chunk_keys)

        # PHASE 3: Plan swap-in (CPU→GPU) + account for recovery space
        chunks_in_cpu = [
            key for key, loc in chunks_needed.items() if loc == "CPU"
        ]

        total_evict_amount = 0
        chunks_needing_space = []

        accumulated_gpu_used = gpu_used_bytes
        for chunk_key in chunks_in_cpu:
            chunk = cpu_cache_snapshot.get(chunk_key)
            if chunk:
                if (
                    accumulated_gpu_used + chunk.size_bytes
                    <= gpu_capacity_bytes
                ):
                    chunks_to_swap_in.append(chunk_key)
                    accumulated_gpu_used += chunk.size_bytes
                else:
                    evict_amount = (
                        accumulated_gpu_used + chunk.size_bytes
                        - gpu_capacity_bytes
                    )
                    chunks_needing_space.append((chunk_key, chunk, evict_amount))
                    total_evict_amount += evict_amount
                    accumulated_gpu_used += chunk.size_bytes

        # Include recovery size in eviction budget
        remaining_after_swaps = gpu_capacity_bytes - accumulated_gpu_used
        if recovery_estimated_bytes > remaining_after_swaps:
            total_evict_amount += recovery_estimated_bytes - max(remaining_after_swaps, 0)

        # Evict once based on total need (swap-in + recovery)
        if total_evict_amount > 0:
            evicted = self.cache.eviction_policy.select_chunks_to_evict(
                list(gpu_cache_snapshot.values()), total_evict_amount * 1.01, cache=self.cache
            )
            chunks_to_swap_out.extend(evicted)

        # Add all chunks that needed space to swap_in
        for chunk_key, chunk, _ in chunks_needing_space:
            chunks_to_swap_in.append(chunk_key)

        # PHASE 4: Build cache plan
        cache_plan.chunks_to_swap_in = chunks_to_swap_in
        cache_plan.chunks_to_swap_out = chunks_to_swap_out

        print("gpu_used_bytes ", gpu_used_bytes)
        with self.cache.cache_lock:
            cpu_used_bytes = self.cache.cpu_used_bytes
        print("cpu_used_bytes ", cpu_used_bytes)
        print("chunks_already_in", len(chunks_needed) - len(cache_plan.chunks_to_swap_in) - sum(len(v) for v in cache_plan.chunks_to_recompute.values()))
        print("cache_plan.chunks_to_swap_in ", len(cache_plan.chunks_to_swap_in))
        print("cache_plan.chunks_to_swap_out ", len(cache_plan.chunks_to_swap_out))
        print("cache_plan.chunks_to_recompute ", sum(len(v) for v in cache_plan.chunks_to_recompute.values()))
        if recovery_estimated_bytes > 0:
            print(f"recovery_estimated_bytes {recovery_estimated_bytes / 1024**2:.1f} MB")
        print(cache_plan.chunks_to_recompute)
        return cache_plan

    def update_running_requests(
        self, batch_results: Dict[str, any]
    ) -> None:
        """Update running requests based on batch results.

        Args:
            batch_results: Results from executing batch
                Format: {request_id: {"finished": bool, "tokens_generated": int}}
        """
        # Mark requests that finished
        finished_request_ids = set()
        for req_id, result in batch_results.items():
            if result.get("finished", False):
                finished_request_ids.add(req_id)

                # Find request and mark as completed
                for i, req in enumerate(self.running_requests):
                    if req.request_id == req_id:
                        req.finished = True
                        self.completed_requests[req_id] = req
                        break

        # Remove finished requests from running list
        self.running_requests = [
            req
            for req in self.running_requests
            if req.request_id not in finished_request_ids
        ]

    def get_batch_info_dict(self, batch: Batch) -> Dict:
        """Extract batch info for PensieveCacheFactory.

        Returns:
            Dict mapping request_id to {session_id, positions, ...}
            Used by custom cache to gather chunks
        """
        batch_info = {}

        for req in batch.requests:
            # Get positions (chunk_ids) available for this session
            positions = self.cache.get_session_positions(req.session_id)

            batch_info[req.request_id] = {
                "session_id": req.session_id,
                "positions": positions,
                "context_length": req.seq_len if hasattr(req, "seq_len") else 0,
                "phase": req.phase,
            }

        return batch_info

    def should_continue_serving(self) -> bool:
        """Check if there are more requests to process.

        Returns:
            True if queue has requests
        """
        return len(self.request_queue) > 0

    def reset(self) -> None:
        """Reset scheduler state."""
        self.request_queue.clear()
        self.completed_requests.clear()
