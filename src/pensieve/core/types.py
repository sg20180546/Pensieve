"""Core data types for Pensieve system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import time


# Global constant for KV cache chunking
CHUNK_SIZE = 32  # Tokens per chunk


class CacheLocation(Enum):
    """Location of KV cache."""
    GPU = "gpu"
    CPU = "cpu"
    DROPPED = "dropped"


class Phase(Enum):
    """Phase of request processing."""
    PREFILL = "prefill"          # Processing input prompt
    GENERATION = "generation"    # Generating output tokens


@dataclass
class RequestConfig:
    """Configuration for a request."""
    session_id: str
    user_input: str
    max_new_tokens: int = 128


@dataclass
class SessionMetadata:
    """Track cumulative state of a conversation session."""
    session_id: str
    total_input_tokens: int = 0        # 입력 토큰 누적
    total_generated_tokens: int = 0    # 생성 토큰 누적
    chunk_keys: List[str] = field(default_factory=list)  # 캐시된 chunks
    last_accessed: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """전체 누적 토큰 개수"""
        return self.total_input_tokens + self.total_generated_tokens

    @property
    def total_chunks(self) -> int:
        """전체 청크 개수 (32개 토큰 = 1청크)"""
        return (self.total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE

    @property
    def last_chunk_size(self) -> int:
        """마지막 청크의 실제 토큰 개수.

        예시:
        - 145개 토큰 → 5개 청크, 마지막은 17개
        - 128개 토큰 → 4개 청크, 마지막은 32개 (정확히 나누어떨어짐)

        Returns:
            1-CHUNK_SIZE 사이의 정수
        """
        remainder = self.total_tokens % CHUNK_SIZE
        return remainder if remainder > 0 else CHUNK_SIZE

    def update_last_accessed(self) -> None:
        """마지막 접근 시간 업데이트"""
        self.last_accessed = time.time()


@dataclass
class Request:
    """Represents a single user request in a conversation."""
    session_id: str
    request_id: str
    input_ids: torch.Tensor  # Shape: [seq_len]
    max_new_tokens: int = 128
    generated_tokens: List[int] = field(default_factory=list)
    finished: bool = False

    # KV cache management
    chunk_keys: List[str] = field(default_factory=list)  # Keys of cached chunks
    dropped_token_ids: Optional[torch.Tensor] = None  # Dropped tokens to recompute

    # Conversation history tracking
    original_text: str = ""  # User input text (for multi-turn conversation tracking)
    response_text: str = ""  # Generated response text (for building conversation history)

    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    ended_at: float = 0.0

    def __post_init__(self):
        """Validate request."""
        if self.input_ids is not None and isinstance(self.input_ids, list):
            self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)

    @property
    def seq_len(self) -> int:
        """Current sequence length (input + generated)."""
        return len(self.input_ids) + len(self.generated_tokens)

    @property
    def elapsed_time(self) -> float:
        """Time since request started."""
        if self.started_at == 0:
            return 0.0
        end_time = self.ended_at if self.ended_at > 0 else time.time()
        return end_time - self.started_at


@dataclass
class Batch:
    """Represents a batch of requests for execution."""
    requests: List[Request] = field(default_factory=list)
    batch_id: str = ""

    def add_request(self, req: Request) -> None:
        """Add request to batch."""
        self.requests.append(req)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in batch."""
        return sum(req.seq_len for req in self.requests)

    @property
    def num_requests(self) -> int:
        """Number of requests in batch."""
        return len(self.requests)


@dataclass
class KVChunk:
    """
    Represents a single layer's KV cache for a 32-token chunk.

    Design: Layer-wise chunking enables fine-grained eviction control.
    - Each chunk covers one layer (layer_idx)
    - Chunks with same (session_id, chunk_id) but different layer_idx are separate objects
    - This enables "checkerboard" eviction: evict by layer first, then by position within layer

    Key fields:
    - session_id: Which conversation
    - chunk_id: Position within conversation (0, 1, 2, ...)
    - layer_idx: Which transformer layer (0 to num_layers-1)
    - key_tensor, value_tensor: KV cache for this layer (shape: [1, seq_len, num_heads, head_dim])
    - context_length: Tokens BEFORE this chunk (same for all layers in same chunk_id)
    - session_total_chunks: Total chunks in this session (for relative position weighting)
    - num_layers: Total layers in model (for layer cost weighting)
    """
    session_id: str
    chunk_id: int               # Position in session (0, 1, 2, ...)
    layer_idx: int              # Which layer (0 to num_layers-1)
    key_tensor: torch.Tensor    # Shape: [1, seq_len, num_heads, head_dim]
    value_tensor: torch.Tensor  # Shape: [1, seq_len, num_heads, head_dim]
    context_length: int         # Tokens BEFORE this chunk
    session_total_chunks: int   # Total chunks in this session
    num_layers: int             # Total layers in model
    last_accessed: float = field(default_factory=time.time)
    location: CacheLocation = CacheLocation.GPU

    def __post_init__(self):
        """Calculate size after initialization."""
        self._recalculate_size()

    def _recalculate_size(self) -> None:
        """Recalculate memory footprint for this single layer's KV tensors."""
        self._size_bytes = 0
        if self.key_tensor is not None:
            self._size_bytes += self.key_tensor.element_size() * self.key_tensor.numel()
        if self.value_tensor is not None:
            self._size_bytes += self.value_tensor.element_size() * self.value_tensor.numel()

    @property
    def size_bytes(self) -> int:
        """Memory footprint in bytes (single layer only)."""
        return self._size_bytes

    @property
    def num_tokens(self) -> int:
        """Number of tokens in this chunk (always 32, except possibly last)."""
        if self.key_tensor is None:
            return 0

        # ✅ FIX: Detect tensor format correctly
        # Different models use different shapes:
        # - Gemma-2: [batch, heads, seq, head_dim] → seq at dim=2
        # - Standard HF: [batch, seq, heads, head_dim] → seq at dim=1

        shape = self.key_tensor.shape
        if len(shape) < 2:
            return 0

        if len(shape) == 4:
            # 4D tensor: determine which dimension is sequence
            # Heuristic: if dim[1] is small (num_heads 8-128), then dim=2 is seq
            if shape[1] < 256:  # shape[1] is likely num_heads
                seq_len = shape[2]  # Gemma format: [batch, heads, seq, head_dim]
            else:
                seq_len = shape[1]  # Standard format: [batch, seq, heads, head_dim]
            return seq_len
        else:
            # For other shapes, default to dim[1] (e.g., 2D: [batch, seq] or [seq, heads])
            return shape[1]

    @property
    def key(self) -> str:
        """Unique identifier for this chunk (includes layer)."""
        return f"{self.session_id}:chunk:{self.chunk_id}:layer:{self.layer_idx}"

    def update_access_time(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()

    def move_to_cpu(self) -> None:
        """Move tensors to CPU."""
        if self.location == CacheLocation.GPU:
            self.key_tensor = self.key_tensor.cpu()
            self.value_tensor = self.value_tensor.cpu()
            self.location = CacheLocation.CPU

    def move_to_gpu(self, device: str = 'cuda:0') -> None:
        """Move tensors to GPU."""
        if self.location == CacheLocation.CPU:
            self.key_tensor = self.key_tensor.to(device)
            self.value_tensor = self.value_tensor.to(device)
            self.location = CacheLocation.GPU


@dataclass
class CachePlan:
    """Plan for cache operations in a batch."""
    batch_id: str
    chunks_to_swap_in: List[str] = field(default_factory=list)  # CPU -> GPU
    chunks_to_swap_out: List[str] = field(default_factory=list)  # GPU -> CPU
    chunks_to_recompute: Dict[str, torch.Tensor] = field(default_factory=dict)  # Dropped

    @property
    def total_operations(self) -> int:
        """Total number of cache operations."""
        return len(self.chunks_to_swap_in) + len(self.chunks_to_swap_out) + len(self.chunks_to_recompute)


@dataclass
class BatchResult:
    """Results from executing a batch."""
    batch_id: str
    request_results: Dict[str, Dict] = field(default_factory=dict)  # {request_id: result}
    generated_tokens: List[int] = field(default_factory=list)  # All generated tokens
    execution_time: float = 0.0  # Total execution time (tail latency)
    ttft_per_request: Dict[str, float] = field(default_factory=dict)  # {request_id: ttft in seconds}
    prefill_time: float = 0.0  # Time for prefill (cache plan + first forward pass)
    generation_time: float = 0.0  # Time for token generation (decode loop)


@dataclass
class CacheStatistics:
    """Cache statistics for monitoring."""
    gpu_used_bytes: int = 0
    gpu_free_bytes: int = 0
    cpu_used_bytes: int = 0
    cpu_free_bytes: int = 0
    num_gpu_chunks: int = 0
    num_cpu_chunks: int = 0
    num_dropped_chunks: int = 0
    gpu_hit_count: int = 0
    cpu_hit_count: int = 0
    miss_count: int = 0

    @property
    def gpu_hit_rate(self) -> float:
        """GPU cache hit rate."""
        total = self.gpu_hit_count + self.cpu_hit_count + self.miss_count
        return self.gpu_hit_count / total if total > 0 else 0.0

    @property
    def cpu_hit_rate(self) -> float:
        """CPU cache hit rate."""
        total = self.gpu_hit_count + self.cpu_hit_count + self.miss_count
        return self.cpu_hit_count / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Cache miss rate."""
        total = self.gpu_hit_count + self.cpu_hit_count + self.miss_count
        return self.miss_count / total if total > 0 else 0.0

    @property
    def gpu_free_ratio(self) -> float:
        """Ratio of free GPU memory."""
        total = self.gpu_used_bytes + self.gpu_free_bytes
        return self.gpu_free_bytes / total if total > 0 else 1.0

    @property
    def cpu_free_ratio(self) -> float:
        """Ratio of free CPU memory."""
        total = self.cpu_used_bytes + self.cpu_free_bytes
        return self.cpu_free_bytes / total if total > 0 else 1.0
