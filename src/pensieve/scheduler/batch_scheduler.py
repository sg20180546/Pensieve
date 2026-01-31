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

            # Check each position and EACH LAYER's chunk status
            for pos in positions:
                # ✅ Check all layers (each layer can have different availability)
                for layer_idx in range(self.cache.num_layers):
                    chunk_key = f"{session_id}:chunk:{pos}:layer:{layer_idx}"
                    chunk = self.cache.get_chunk(chunk_key)

                    if chunk is None:
                        # Not found anywhere
                        continue

                    # ✅ Explicitly determine chunk location from chunk.location
                    # (get_chunk now returns DROPPED chunks too)
                    if chunk.location == CacheLocation.GPU:
                        chunks_needed[chunk_key] = "GPU"
                    elif chunk.location == CacheLocation.CPU:
                        chunks_needed[chunk_key] = "CPU"
                    elif chunk.location == CacheLocation.DROPPED:
                        chunks_needed[chunk_key] = "DROPPED"

        # 2. Plan swaps based on memory pressure
        stats = self.cache.get_statistics()
        gpu_free_ratio = stats.gpu_free_ratio

        # If GPU is filling up, plan to move CPU chunks to GPU first
        # (Otherwise, chunks from CPU might stay there during execution)
        chunks_in_cpu = [
            key for key, loc in chunks_needed.items() if loc == "CPU"
        ]

        # Priority: Swap in chunks that will be used in this batch
        # For now, swap in all CPU chunks that are needed
        for chunk_key in chunks_in_cpu:
            # Check if chunk size fits in GPU
            chunk = self.cache.cpu_cache.get(chunk_key)
            if chunk:
                # Only swap in if GPU has space
                if (
                    self.cache.gpu_used_bytes + chunk.size_bytes
                    <= self.cache.gpu_capacity_bytes
                ):
                    chunks_to_swap_in.append(chunk_key)
                else:
                    # print("needs to be evicted")
                    # GPU is full, need to evict something first
                    # Plan to evict the least valuable chunks
                    evict_amount = (
                        self.cache.gpu_used_bytes + chunk.size_bytes
                        - self.cache.gpu_capacity_bytes
                    )
                    # ✅ Pass cache=self.cache to ensure SessionMetadata is used
                    evicted = self.cache.eviction_policy.select_chunks_to_evict(
                        list(self.cache.gpu_cache.values()), evict_amount, cache=self.cache
                    )
                    # print(evicted)
                    chunks_to_swap_out.extend(evicted)
                    chunks_to_swap_in.append(chunk_key)

        # 3. Build cache plan
        cache_plan.chunks_to_swap_in = chunks_to_swap_in
        cache_plan.chunks_to_swap_out = chunks_to_swap_out

        # 4. Identify dropped chunks needing recovery
        dropped_chunks = {
            key: chunk
            for key, chunk in self.cache.dropped_chunks.items()
            if key in chunks_needed
        }
        # Store dropped chunk info for worker to handle recovery
        for chunk_key, chunk in dropped_chunks.items():
            # Extract session_id from chunk_key
            # Format: "session:chunk:id:layer:idx"
            parts = chunk_key.split(":")
            if len(parts) >= 2:
                session_id = parts[0]
                # Add to recovery dict for this session
                if session_id not in cache_plan.chunks_to_recompute:
                    cache_plan.chunks_to_recompute[session_id] = []
                cache_plan.chunks_to_recompute[session_id].append(chunk_key)
        # print(chunks_needed)
        print("gpu_used_bytes ",self.cache.gpu_used_bytes)
        print("cpu_used_bytes ",self.cache.cpu_used_bytes)
        print("chunks_already_in",len(chunks_needed)-len(cache_plan.chunks_to_swap_in)-len(cache_plan.chunks_to_recompute))
        print("chunks_to_swap_in ",len(cache_plan.chunks_to_swap_in))
        print("chunks_to_swap_out ",len(cache_plan.chunks_to_swap_out))
        print("chunks_to_recompute ",len(cache_plan.chunks_to_recompute))
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
