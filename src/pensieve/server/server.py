"""Main server implementation for Pensieve with vLLM baseline support."""

import torch
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pensieve.core import (
    Request,
    Batch,
    TwoTierCache,
    KVChunk,
    CacheLocation,
    Phase,
    RequestConfig,
)
from pensieve.scheduler import BatchScheduler
from pensieve.worker import Worker


class InferenceMode(Enum):
    """Inference mode selection."""
    PENSIEVE = "pensieve"  # Stateful with KV cache
    VLLM_BASELINE = "vllm"  # Stateless baseline (simulated)


class PensieveServer:
    """Pensieve serving system with optional vLLM baseline."""

    def __init__(
        self,
        model_name: str = "gpt2",
        inference_mode: InferenceMode = InferenceMode.PENSIEVE,
        gpu_capacity_gb: float = 40,
        cpu_capacity_gb: float = 100,
        device: str = "cuda:0",
    ):
        """Initialize Pensieve server.

        Args:
            model_name: HuggingFace model name
            inference_mode: PENSIEVE (stateful) or VLLM_BASELINE (stateless)
            gpu_capacity_gb: GPU cache capacity in GB
            cpu_capacity_gb: CPU cache capacity in GB
            device: GPU device string
        """
        self.model_name = model_name
        self.inference_mode = inference_mode
        self.device = device

        # Lazy load model (actual loading happens on server startup)
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

        # For Pensieve mode
        if inference_mode == InferenceMode.PENSIEVE:
            self.cache = TwoTierCache(
                gpu_capacity_gb=gpu_capacity_gb,
                cpu_capacity_gb=cpu_capacity_gb,
                device=device,
            )
            # Phase 4: Initialize scheduler and worker
            self.scheduler = BatchScheduler(self.cache, max_batch_size=8)
            self.worker = None  # Lazy load with model
        else:
            self.cache = None
            self.scheduler = None
            self.worker = None

        # Request management
        self.request_queue: List[Request] = []
        self.active_sessions: Dict[str, List[Request]] = {}  # {session_id: [requests]}
        self.completed_requests: Dict[str, Request] = {}

        # Statistics
        self.total_prefill_time = 0.0
        self.total_generation_time = 0.0
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.last_ttft_per_request = {}  # TTFT for last batch {request_id: ttft in seconds}

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _get_worker(self):
        """Lazy load worker for Pensieve mode."""
        if self.inference_mode == InferenceMode.PENSIEVE and self.worker is None:
            self.worker = Worker(
                model=self.model,
                tokenizer=self.tokenizer,
                cache=self.cache,
                device=self.device,
            )
        return self.worker

    def process_request(self, session_id: str, user_input: str, max_new_tokens: int = 32) -> str:
        """Process a single request.

        Args:
            session_id: Unique session identifier
            user_input: User's input text
            max_new_tokens: Max tokens to generate

        Returns:
            Generated response text
        """
        if self.inference_mode == InferenceMode.PENSIEVE:
            return self._process_pensieve(session_id, user_input, max_new_tokens)
        else:
            return self._process_vllm_baseline(session_id, user_input, max_new_tokens)

    def _process_pensieve(self, session_id: str, user_input: str, max_new_tokens: int) -> str:
        """Process request using Pensieve (stateful with Phase 4 scheduler + worker).

        Args:
            session_id: Session ID
            user_input: User input
            max_new_tokens: Max new tokens

        Returns:
            Generated text
        """
        request_id = f"{session_id}_{self.total_requests}"

        # Get or create session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []

        # Get conversation history
        history = self._get_session_history(session_id)

        # Encode input
        full_input = history + user_input
        input_ids = self.tokenizer.encode(full_input, return_tensors='pt')
        input_ids = input_ids.squeeze(0)

        # Create request for Phase 4 pipeline
        request = Request(
            session_id=session_id,
            request_id=request_id,
            input_ids=input_ids,
            phase=Phase.PREFILL,
            max_new_tokens=max_new_tokens,
            original_text=user_input,  # Store user input for history tracking
        )

        # Time prefill + generation
        start_time = time.time()

        try:
            # Phase 4 Pipeline: Scheduler → Worker
            self.scheduler.add_request(request)
            batch, cache_plan = self.scheduler.form_next_batch()

            # Get worker and execute batch
            worker = self._get_worker()
            batch_result = worker.execute_batch(batch, cache_plan)

            elapsed = time.time() - start_time
            self.total_prefill_time += elapsed

            # Extract response from batch result
            if request_id in batch_result.request_results:
                result_dict = batch_result.request_results[request_id]
                response = result_dict.get("response", "")
                tokens_generated = result_dict.get("tokens_generated", 0)
            else:
                response = ""
                tokens_generated = 0

            # Store response in request for conversation history tracking
            request.response_text = response

            # Update statistics
            self.total_requests += 1
            self.total_tokens_generated += tokens_generated
            request.finished = True
            self.completed_requests[request_id] = request

            # Store TTFT (Time To First Token) from batch result if available
            if hasattr(batch_result, 'ttft_per_request') and batch_result.ttft_per_request:
                self.last_ttft_per_request = batch_result.ttft_per_request

            # Store request in session
            self.active_sessions[session_id].append(request)

            return response

        except Exception as e:
            print(f"Error in Pensieve processing: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _process_vllm_baseline(self, session_id: str, user_input: str, max_new_tokens: int) -> str:
        """Process request using vLLM baseline (stateless).

        In stateless mode:
        - Recompute entire conversation history with each request
        - No KV cache reuse across requests
        - Simulates vLLM behavior (which reprocesses history)

        Args:
            session_id: Session ID
            user_input: User input
            max_new_tokens: Max new tokens

        Returns:
            Generated text
        """
        request_id = f"{session_id}_{self.total_requests}"

        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []

        # Get conversation history (full recomputation in vLLM mode)
        history = self._get_session_history(session_id)

        # Full input includes all history + new user input
        full_input = history + user_input
        input_ids = self.tokenizer.encode(full_input, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # Create request
        request = Request(
            session_id=session_id,
            request_id=request_id,
            input_ids=input_ids.squeeze(0),
            phase=Phase.PREFILL,
            max_new_tokens=max_new_tokens,
            original_text=user_input,  # Store user input for history tracking
        )

        # Time the entire process (simulating vLLM stateless processing)
        start_time = time.time()

        with torch.no_grad():
            # Standard generation (no cache reuse)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=False,
            )

        elapsed = time.time() - start_time
        self.total_prefill_time += elapsed  # vLLM counts everything as prefill cost

        # Decode output
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Store response in request for conversation history tracking
        request.response_text = response

        # Update statistics
        self.total_requests += 1
        self.total_tokens_generated += len(generated_ids)
        request.finished = True
        self.completed_requests[request_id] = request

        # Store request in session
        self.active_sessions[session_id].append(request)

        return response

    def _get_session_history(self, session_id: str) -> str:
        """Get conversation history for a session.

        Builds history from all completed requests in session:
        - User input (original_text)
        - Assistant response (response_text)

        Args:
            session_id: Session ID

        Returns:
            Conversation history as string (previous inputs + responses)
        """
        if session_id not in self.active_sessions:
            return ""

        history_parts = []
        for req in self.active_sessions[session_id]:
            # Add user input (original text)
            if req.original_text:
                history_parts.append(f"User: {req.original_text}")

            # Add assistant response
            if req.response_text:
                history_parts.append(f"Assistant: {req.response_text}")

        # Add newline separator between history and new input
        if history_parts:
            return "\n".join(history_parts) + "\n"
        return ""

    def get_statistics_str(self) -> str:
        """Get formatted statistics string.

        Returns:
            Statistics string
        """
        mode_str = "Pensieve (Stateful)" if self.inference_mode == InferenceMode.PENSIEVE else "vLLM (Stateless)"

        stats = f"""
=== Pensieve Server Statistics ===
Inference Mode: {mode_str}
Total Requests: {self.total_requests}
Total Tokens Generated: {self.total_tokens_generated}
Avg Prefill Time: {self.total_prefill_time / max(self.total_requests, 1):.3f}s
Total Prefill Time: {self.total_prefill_time:.3f}s
Active Sessions: {len(self.active_sessions)}
"""

        if self.inference_mode == InferenceMode.PENSIEVE and self.cache:
            stats += f"\nCache Statistics:\n{self.cache.get_memory_stats_str()}\n"
            cache_stats = self.cache.get_statistics()
            stats += f"GPU Hit Rate: {cache_stats.gpu_hit_rate:.1%}\n"
            stats += f"CPU Hit Rate: {cache_stats.cpu_hit_rate:.1%}\n"
            stats += f"Miss Rate: {cache_stats.miss_rate:.1%}\n"

        return stats

    def reset(self) -> None:
        """Reset server state."""
        self.request_queue.clear()
        self.active_sessions.clear()
        self.completed_requests.clear()
        if self.cache:
            self.cache.reset()
        self.total_prefill_time = 0.0
        self.total_generation_time = 0.0
        self.total_requests = 0
        self.total_tokens_generated = 0


def create_server(
    model_name: str = "gpt2",
    mode: str = "pensieve",
    gpu_capacity_gb: float = 40,
    cpu_capacity_gb: float = 100,
    device: str = "cuda:0",
) -> PensieveServer:
    """Factory function to create server with specified mode.

    Args:
        model_name: HuggingFace model name
        mode: "pensieve" or "vllm" (baseline)
        gpu_capacity_gb: GPU cache capacity (only used in pensieve mode)
        cpu_capacity_gb: CPU cache capacity (only used in pensieve mode)
        device: GPU device

    Returns:
        PensieveServer instance

    Example:
        # Create Pensieve server
        server = create_server(mode="pensieve")

        # Create vLLM baseline server
        baseline = create_server(mode="vllm")

        # Process request
        response = server.process_request("session_1", "Hello, how are you?")
    """
    try:
        inference_mode = InferenceMode(mode.lower())
    except ValueError:
        raise ValueError(f"Invalid mode: {mode}. Must be 'pensieve' or 'vllm'")

    return PensieveServer(
        model_name=model_name,
        inference_mode=inference_mode,
        gpu_capacity_gb=gpu_capacity_gb,
        cpu_capacity_gb=cpu_capacity_gb,
        device=device,
    )
