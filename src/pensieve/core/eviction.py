"""Retention value eviction policy for KV cache."""

import time
from typing import List, Tuple, Dict, Optional
from .types import KVChunk, CacheLocation
from dataclasses import dataclass


@dataclass
class CostProfile:
    """Cost profile for a model (obtained from offline profiling)."""
    # Attention cost: cost_attention(l) = alpha * l + beta
    # where l is the context length (tokens before this chunk)
    alpha: float  # Cost per token of context
    beta: float  # Baseline attention cost

    # Non-attention cost (constant)
    const_non_attention: float


class RetentionValuePolicy:
    """Eviction policy based on retention value (§4.3.1 of paper).

    The retention value of a chunk is:
        V = Cost(s, l) / T

    where:
        - Cost(s, l) = Cost_attention(l) + Cost_other(s)
        - Cost_attention(l) = alpha * l + beta (linear in context length)
        - Cost_other(s) = constant (independent of context)
        - l = context_length (tokens BEFORE this chunk)
        - T = time_inactive (how long since chunk was accessed)

    Chunks with LOWER retention value are evicted first.
    Key insight: Leading tokens (small l) have low cost, so evict first!
    """

    CHUNK_SIZE = 32  # Tokens per chunk

    def __init__(self):
        """Initialize eviction policy."""
        # Default cost profile (from paper profiling on OPT-13B)
        # These values are estimated; in practice would profile per model
        self.cost_profile = CostProfile(
            alpha=0.001,  # Cost per token of context (seconds)
            beta=0.01,    # Baseline attention cost (seconds)
            const_non_attention=0.005,  # Non-attention operations cost
        )

    def set_cost_profile(self, cost_profile: CostProfile) -> None:
        """Set cost profile from profiling results.

        Args:
            cost_profile: CostProfile object from offline profiling
        """
        self.cost_profile = cost_profile

    def calculate_cost(self, chunk: KVChunk) -> float:
        """Calculate recomputation cost for a chunk (layer-wise model).

        Cost = layer_weight × position_weight × base_cost

        where:
        - layer_weight: Scales by layer depth (later layers have pipelined lower cost)
            formula: (num_layers - layer_idx) / num_layers
            → Layer 0: cost = 1.0 × ...  (highest cost, critical path)
            → Layer 39: cost = 0.03 × ...  (lowest cost, pipelined)

        - position_weight: Session-relative position (leading tokens cheaper to recompute)
            formula: (chunk_id + 1) / session_total_chunks
            → Chunk 0: 0.01 (cheapest, evict first)
            → Chunk 99: 1.0 (most expensive, evict last)

        - base_cost = attention_cost + non_attention_cost
            - attention_cost = alpha * context_length + beta
            - non_attention_cost = const_non_attention per chunk

        Args:
            chunk: KVChunk to estimate cost for

        Returns:
            Estimated recomputation cost (dimensionless, for relative ranking)
        """
        # Base cost: attention cost scales with context length
        l = chunk.context_length  # Tokens before this chunk
        cost_attention = self.cost_profile.alpha * l + self.cost_profile.beta
        cost_non_attention = self.cost_profile.const_non_attention

        # Layer weight: Earlier layers are more critical (pipelining assumption)
        # Layer 0 = 1.0 (on critical path)
        # Layer 39 = 0.025 (fully pipelined, minimal impact)
        if chunk.num_layers > 0:
            layer_weight = (chunk.num_layers - chunk.layer_idx) / chunk.num_layers
        else:
            layer_weight = 1.0

        # Position weight: Session-relative position
        # Chunk 0 = 0.01 (leading, cheapest to recompute)
        # Chunk 99 = 1.0 (trailing, expensive to recompute)
        if chunk.session_total_chunks > 0:
            position_weight = (chunk.chunk_id + 1) / chunk.session_total_chunks
        else:
            position_weight = 1.0

        # Combined cost with weights
        base_cost = cost_attention + cost_non_attention
        weighted_cost = layer_weight * position_weight * base_cost

        return weighted_cost

    def calculate_retention_value(self, chunk: KVChunk) -> float:
        """Calculate retention value for a chunk.

        Retention value V = Cost(layer, position, context) / T

        where:
        - Cost incorporates layer depth and session-relative position
        - T is time since last access
        - Higher retention value = more important to keep
        - Lower retention value = cheaper to recompute, evict first

        The LRU component (T) ensures recently accessed sessions are preserved,
        while the cost component (Cost) ensures optimal eviction order within
        similar recency levels.

        Args:
            chunk: KVChunk to calculate retention value for

        Returns:
            Retention value (lower = evict first)
        """
        cost = self.calculate_cost(chunk)

        # Time since last access (LRU component)
        time_inactive = time.time() - chunk.last_accessed
        if time_inactive <= 0:
            time_inactive = 0.001  # Avoid division by zero

        retention_value = cost / time_inactive

        return retention_value

    def rank_chunks_for_eviction(self, chunks: List[KVChunk]) -> List[Tuple[str, float]]:
        """Rank chunks by retention value for eviction.

        Chunks are sorted in ascending order of retention value.
        Lowest retention value = evict first.

        Args:
            chunks: List of KVChunk objects to rank

        Returns:
            List of (chunk_key, retention_value) tuples sorted by value
        """
        scored_chunks = []
        for chunk in chunks:
            if chunk.location==CacheLocation.DROPPED:
                continue
            value = self.calculate_retention_value(chunk)
            scored_chunks.append((chunk.key, value, chunk.context_length, chunk.session_id))

        # Sort by retention value (ascending) then by context_length (ascending)
        # This prefers evicting chunks with low cost and old sessions
        scored_chunks.sort(key=lambda x: (x[1], x[2]))

        # Return just (key, value) pairs
        return [(key, value) for key, value, _, _ in scored_chunks]

    def get_eviction_candidates(
        self,
        chunks: List[KVChunk],
        num_candidates: int = 10,
    ) -> List[str]:
        """Get top candidates for eviction.

        Args:
            chunks: List of chunks available to evict
            num_candidates: Number of candidates to return

        Returns:
            List of chunk keys to evict (in order)
        """
        ranked = self.rank_chunks_for_eviction(chunks)
        return [key for key, _ in ranked[:num_candidates]]

    def select_chunks_to_evict(
        self,
        chunks: List[KVChunk],
        target_bytes: int,
    ) -> List[str]:
        """Select chunks to evict to free a target amount of memory.

        Args:
            chunks: List of chunks available to evict
            target_bytes: Target amount of memory to free

        Returns:
            List of chunk keys to evict
        """
        ranked = self.rank_chunks_for_eviction(chunks)

        to_evict = []
        freed = 0

        for chunk_key, retention_value in ranked:
            if freed >= target_bytes:
                break

            # Find the chunk object to get its size
            chunk = next((c for c in chunks if c.key == chunk_key), None)
            if chunk:
                to_evict.append(chunk_key)
                freed += chunk.size_bytes

        return to_evict

    def profile_attention_cost(
        self,
        model,
        tokenizer,
        device: str = 'cuda:0',
        context_sizes: Optional[List[int]] = None,
        chunk_size: int = 32,
    ) -> CostProfile:
        """Profile attention cost for a model (offline).

        This function should be run once per model to calibrate the cost function.

        Args:
            model: HuggingFace language model
            tokenizer: Tokenizer for the model
            device: GPU device
            context_sizes: List of context sizes to profile (default: powers of 2)
            chunk_size: Chunk size to profile

        Returns:
            CostProfile with fitted alpha and beta
        """
        import torch
        import numpy as np

        if context_sizes is None:
            # Profile at these context sizes
            context_sizes = [256, 512, 1024, 2048, 4096, 8192]

        costs = []
        context_lengths = []

        model.eval()

        try:
            with torch.no_grad():
                for context_size in context_sizes:
                    # Create dummy input
                    # Input: [context_size + chunk_size]
                    input_ids = torch.randint(0, 1000, (1, context_size + chunk_size))
                    input_ids = input_ids.to(device)

                    # Time the forward pass
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    import time
                    start = time.time()

                    outputs = model(input_ids, return_dict=True)

                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    elapsed = time.time() - start

                    costs.append(elapsed)
                    context_lengths.append(context_size)

                    print(f"Context size {context_size}: {elapsed:.4f}s")

            # Fit linear model: cost = alpha * context_length + beta
            # Use numpy polyfit (degree 1 = linear)
            coeffs = np.polyfit(context_lengths, costs, deg=1)
            alpha = coeffs[0]
            beta = coeffs[1]

            # Estimate non-attention cost (roughly constant part)
            const_non_attention = beta / chunk_size

            profile = CostProfile(
                alpha=alpha,
                beta=beta,
                const_non_attention=const_non_attention,
            )

            return profile

        except Exception as e:
            print(f"Warning: Profiling failed, using default values: {e}")
            return self.cost_profile


def profile_and_save(model_name: str, output_path: str = "cost_profiles.json"):
    """Helper function to profile a model and save cost profile.

    This is a standalone utility to be run offline on a server.

    Args:
        model_name: HuggingFace model name
        output_path: Path to save profiles JSON
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Profiling {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        device = 'cuda:0'
        model = model.to(device)
    else:
        device = 'cpu'
        print("Warning: CUDA not available, profiling on CPU (will be slow)")

    policy = RetentionValuePolicy()
    profile = policy.profile_attention_cost(model, tokenizer, device=device)

    # Save to JSON
    data = {
        model_name: {
            "alpha": profile.alpha,
            "beta": profile.beta,
            "const_non_attention": profile.const_non_attention,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved profile to {output_path}")
    print(f"Profile: alpha={profile.alpha:.6f}, beta={profile.beta:.6f}")
