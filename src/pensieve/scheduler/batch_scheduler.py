"""Iteration-level batch scheduler for Pensieve inference."""

import torch
from typing import List, Tuple, Dict, Optional
from collections import deque
import time

from pensieve.core.types import (
    Request,
    Batch,
    CachePlan,
    Phase,
    CacheLocation,
)
from pensieve.core.cache import TwoTierCache


class BatchScheduler:
    """Manages request queue and forms batches for each iteration.

    Key features (Paper §4.2):
    - Iteration-level batching: Add new requests when generation step completes
    - Unified scheduling: Mix prefill + generation requests in same batch
    - Cache planning: Determine swap operations BEFORE execution
    - Memory-aware: Respects GPU/CPU capacity constraints

    Design:
    - request_queue: Incoming prefill requests
    - running_requests: Requests currently in generation phase
    - Each iteration forms a batch from both queues
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
        self.request_queue: deque = deque()  # Incoming prefill requests
        self.running_requests: List[Request] = []  # In-flight generation requests
        self.completed_requests: Dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """Add new request to queue.

        Args:
            request: Request to add (should be PREFILL phase)
        """
        request.phase = Phase.PREFILL
        self.request_queue.append(request)

    def form_next_batch(self) -> Tuple[Batch, CachePlan]:
        """Form batch for next iteration with mixed prefill/generation.

        Strategy:
        1. Try to keep running generation requests in batch (GENERATION phase)
        2. Add new prefill requests until batch is full
        3. Create cache plan for combined batch

        Returns:
            batch: Batch with mixed prefill/generation requests
            cache_plan: Swap operations needed before execution

        Raises:
            None (returns empty batch if queue empty and no running requests)
        """
        batch = Batch(batch_id=f"batch_{int(time.time() * 1000)}")

        # Add running requests first (generation phase)
        for req in self.running_requests[:self.max_batch_size]:
            batch.add_request(req)

        # Add new prefill requests to fill batch
        remaining_capacity = self.max_batch_size - len(batch.requests)
        while remaining_capacity > 0 and len(self.request_queue) > 0:
            req = self.request_queue.popleft()
            batch.add_request(req)
            remaining_capacity -= 1

        # Create cache plan for this batch
        cache_plan = self.create_cache_plan(batch)

        # Update running requests for next iteration
        # (In real system, would separate by whether they finished generation)
        self.running_requests = [
            req for req in batch.requests if req.phase == Phase.GENERATION
        ]

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

            # Check each position's chunk status
            for pos in positions:
                # For now, check layer 0 as representative
                # (all layers at same position have similar availability)
                chunk_key = f"{session_id}:chunk:{pos}:layer:0"
                chunk = self.cache.get_chunk(chunk_key)

                if chunk is None:
                    # Check dropped chunks
                    chunk = self.cache.dropped_chunks.get(chunk_key)
                    if chunk:
                        chunks_needed[chunk_key] = "DROPPED"
                    continue

                # Chunk exists in GPU or CPU
                if chunk_key in self.cache.gpu_cache:
                    chunks_needed[chunk_key] = "GPU"
                elif chunk_key in self.cache.cpu_cache:
                    chunks_needed[chunk_key] = "CPU"
                else:
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
                    # GPU is full, need to evict something first
                    # Plan to evict the least valuable chunks
                    evict_amount = (
                        self.cache.gpu_used_bytes + chunk.size_bytes
                        - self.cache.gpu_capacity_bytes
                    )
                    evicted = self.cache.eviction_policy.select_chunks_to_evict(
                        list(self.cache.gpu_cache.values()), evict_amount
                    )
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
            True if queue has requests or running requests exist
        """
        return len(self.request_queue) > 0 or len(self.running_requests) > 0

    def reset(self) -> None:
        """Reset scheduler state."""
        self.request_queue.clear()
        self.running_requests.clear()
        self.completed_requests.clear()
