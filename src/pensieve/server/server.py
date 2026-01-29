"""Main server implementation for Pensieve with vLLM baseline support."""

import torch
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys
import os
from queue import Queue
from threading import Lock
import threading

# Suppress HuggingFace warnings for cleaner output
import warnings
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()  # Only show errors, not warnings/info
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pensieve.core import (
    Request,
    Batch,
    TwoTierCache,
    KVChunk,
    CacheLocation,
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

        # ✅ Async queue-based batching
        self.async_request_queue = Queue()  # Incoming requests
        self.pending_requests: Dict[str, Dict] = {}  # {request_id: request_data}
        self.request_results: Dict[str, str] = {}  # {request_id: response}
        self.request_lock = Lock()  # Thread-safe access
        self.batch_timeout = 0.05  # 50ms wait for batch collection
        self.max_batch_size = 8  # Or collect earlier if batch is full
        self.batch_collection_thread = None
        self.batch_collection_running = False

        # ✅ Session token history for recovery (Phase 4.5)
        self.session_token_histories: Dict[str, List[int]] = {}  # {session_id: [all_tokens]}

        # Recovery manager (initialized in _get_worker)
        self.recovery_manager = None

        # Statistics
        self.total_prefill_time = 0.0
        self.total_generation_time = 0.0
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.all_ttfts = []  # ✅ All TTFT measurements [ttft1, ttft2, ...] for aggregation
        self.last_ttft_per_request = {}  # TTFT for all requests {request_id: ttft in seconds}

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            from transformers import AutoModelForCausalLM

            # Determine appropriate dtype based on model size
            # Use bfloat16 for larger models (llama, etc.) to save memory
            torch_dtype = torch.bfloat16 if 'llama' in self.model_name.lower() else torch.float32

            print(f"Loading {self.model_name} with dtype={torch_dtype} on device={self.device}")

            # Load model with proper dtype and device handling
            # For memory efficiency and compatibility with our cache management,
            # load to single specified device (not device_map='auto')
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch_dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
                device_map=None,  # Don't auto-distribute, we handle device placement
            )

            # Move model to specified device
            self._model = self._model.to(self.device)

            self._model.eval()
            print(f"Model loaded successfully on device(s)")
        return self._model

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Configure pad and eos tokens for models that don't have them
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # GPT-2 specific: ensure eos_token is set
            if self._tokenizer.eos_token is None:
                # Use standard EOS tokens based on model
                if 'gpt2' in self.model_name.lower():
                    self._tokenizer.eos_token = '<|endoftext|>'
                else:
                    self._tokenizer.eos_token = '</s>'

            # Ensure pad_token_id is set (fallback to eos_token_id)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        return self._tokenizer

    def _get_worker(self):
        """Lazy load worker for Pensieve mode with batch-level recovery manager."""
        if self.inference_mode == InferenceMode.PENSIEVE and self.worker is None:
            # ✅ Initialize batch-level recovery manager (Phase 4.5 optimization)
            # BatchedRecoveryManager handles multiple sessions' dropped chunks efficiently,
            # respecting both layer-wise and token-wise dependencies
            from pensieve.recovery.token_recovery import BatchedRecoveryManager

            self.batched_recovery_manager = BatchedRecoveryManager(
                model=self.model,
                tokenizer=self.tokenizer,
                cache=self.cache,
                device=self.device,
            )

            # Pass server reference to TokenRecoveryManager inside BatchedRecoveryManager
            # for accessing session_token_histories
            self.batched_recovery_manager.recovery_manager.server = self

            # Initialize worker with batched recovery manager
            self.worker = Worker(
                model=self.model,
                tokenizer=self.tokenizer,
                cache=self.cache,
                device=self.device,
                batched_recovery_manager=self.batched_recovery_manager,  # ← Inject batch recovery manager
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
        # print("@@@@ _process_pensieve do i called?")
        request_id = f"{session_id}_{self.total_requests}"

        # Get or create session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []

        # Get conversation history
        history = self._get_session_history(session_id)
        # print("Penseive history",history)
        # Encode input
        full_input = history + user_input
        input_ids = self.tokenizer.encode(full_input, return_tensors='pt')
        input_ids = input_ids.squeeze(0)

        # Create request for unified batching pipeline
        request = Request(
            session_id=session_id,
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            original_text=user_input,  # Store user input for history tracking
        )

        # Time prefill + generation


        try:
            # Phase 4 Pipeline: Scheduler → Worker
            self.scheduler.add_request(request)
            batch, cache_plan = self.scheduler.form_next_batch()

            # Get worker and execute batch
            worker = self._get_worker()
            batch_result = worker.execute_batch(batch, cache_plan)

            # Accumulate timing metrics (thread-safe with lock)
            with self.request_lock:
                self.total_prefill_time += batch_result.prefill_time
                self.total_generation_time += batch_result.generation_time

                # Store TTFT from batch result
                if batch_result.ttft_per_request:
                    self.last_ttft_per_request.update(batch_result.ttft_per_request)
                    # ✅ Accumulate all TTFTs for aggregation
                    self.all_ttfts.extend(batch_result.ttft_per_request.values())

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

            # ✅ Update session token history for recovery (Phase 4.5)
            if session_id not in self.session_token_histories:
                self.session_token_histories[session_id] = []

            # Add input tokens to history
            input_token_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
            self.session_token_histories[session_id].extend(input_token_ids)

            # Add generated tokens to history (if available)
            if request_id in batch_result.request_results:
                gen_tokens = batch_result.request_results[request_id].get(
                    'tokens_generated', []
                )
                if isinstance(gen_tokens, list):
                    self.session_token_histories[session_id].extend(gen_tokens)

            # Update statistics
            self.total_requests += 1
            self.total_tokens_generated += tokens_generated
            request.finished = True
            self.completed_requests[request_id] = request

            # Store TTFT (Time To First Token) from batch result if available
            if hasattr(batch_result, 'ttft_per_request') and batch_result.ttft_per_request:
                # print("pensieve internal ttft",batch_result.ttft_per_request)
                self.last_ttft_per_request = batch_result.ttft_per_request

            # Store request in session
            self.active_sessions[session_id].append(request)

            return response

        except Exception as e:
            print(f"Error in Pensieve processing: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _vllm_generate_with_timing(self, input_ids, gen_kwargs):
        """Simple wrapper around model.generate() - timing removed for stability.

        Args:
            input_ids: Input token IDs
            gen_kwargs: Arguments for model.generate()

        Returns:
            Tuple of (outputs, prefill_time, first_token_time, remaining_decode_time)
            Note: Timing values are dummy (0.0) to maintain interface compatibility
        """
        # Simply call model.generate without timing hooks
        outputs = self.model.generate(input_ids, **gen_kwargs)

        # Return dummy timing values (to maintain interface compatibility)
        return outputs, 0.0, 0.0, 0.0

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
        print("vllm history",history)

        # Full input includes all history + new user input
        full_input = history + user_input
        input_ids = self.tokenizer.encode(full_input, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # Create attention mask (all ones = attend to all tokens)
        attention_mask = torch.ones_like(input_ids)

        # Create request
        request = Request(
            session_id=session_id,
            request_id=request_id,
            input_ids=input_ids.squeeze(0),
            max_new_tokens=max_new_tokens,
            original_text=user_input,  # Store user input for history tracking
        )

        # Time the entire process (simulating vLLM stateless processing)

        with torch.no_grad():
            # Standard generation (no cache reuse)
            # For Instruct models: use model's default config (already optimized)
            # For base models (especially small ones): use sampling to avoid repetition
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "attention_mask": attention_mask,
                "return_dict_in_generate": True,
                "output_attentions": False,
            }

            # ✅ GREEDY DECODING (deterministic)
            # Paper requirement: "Output matches a stateless baseline (Pensieve)"
            # Using greedy (do_sample=False) ensures deterministic output
            # Same logits = same tokens for correctness verification
            # gen_kwargs stays with default greedy behavior (do_sample=False implicit)

            # Disable sampling for correctness verification
            gen_kwargs.update({
                "do_sample": False,
            })
            # Note: Sampling can be re-enabled by setting do_sample=True + temperature/top_p
            # but for now we use greedy to match Pensieve's deterministic output
                
                
                
            # Use helper to measure prefill and decode time separately
            outputs, prefill_time, first_token_time, remaining_decode_time = self._vllm_generate_with_timing(input_ids, gen_kwargs)

        # Calculate TTFT (Time to First Token) = prefill + first token generation
        ttft = prefill_time + first_token_time

        # Accumulate measured times (thread-safe with lock)
        with self.request_lock:
            self.total_prefill_time += prefill_time
            self.total_generation_time += remaining_decode_time + first_token_time  # Total generation (first + remaining)
            # Store TTFT for this request
            self.last_ttft_per_request[request_id] = ttft
            self.all_ttfts.append(ttft)  # ✅ Accumulate all TTFTs for aggregation
            self.last_ttft = ttft  # Track latest TTFT for client workers

        # Decode output
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Store response in request for conversation history tracking
        request.response_text = response

        # Note: vLLM blocking generate() cannot measure TTFT accurately (first token timing lost)
        # We measure tail latency instead (full generation time)
        # TTFT comparison between Pensieve and vLLM is not applicable here due to architectural differences
        # - Pensieve: measures TTFT from batch prefill (can interrupt after first token)
        # - vLLM: blocking generate() completes all tokens before returning
        # self.last_ttft_per_request = {}  # Don't report TTFT for vLLM

        # DEBUG: Token counting verification
        # print(f"[vLLM DEBUG] input_ids shape: {input_ids.shape}, input_len: {input_ids.shape[1]}, generated_tokens: {len(generated_ids)}, response_len: {len(response)}")

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

    def end_session(self, session_id: str) -> int:
        """Cleanup resources for a completed session.

        Removes all session state and cache data:
        - Session from active_sessions
        - All chunks from cache (GPU/CPU/DROPPED)
        - Session token history
        - Any pinned chunks for this session

        CRITICAL: Called when session is finished to prevent memory leaks.

        Args:
            session_id: Session ID to cleanup

        Returns:
            Number of bytes freed from cache
        """
        freed_bytes = 0

        # 1. Cleanup cache (GPU/CPU/DROPPED chunks)
        if self.inference_mode == InferenceMode.PENSIEVE and self.cache:
            try:
                freed_bytes = self.cache.evict_session(session_id)
                print(f"✓ Freed {freed_bytes / 1024**2:.2f} MB from cache for session {session_id}")
            except Exception as e:
                print(f"Warning: Failed to evict session {session_id}: {e}")

        # 2. Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        # 3. Remove session token history (used for recovery)
        if session_id in self.session_token_histories:
            del self.session_token_histories[session_id]

        print(f"✓ Session {session_id} cleaned up (freed {freed_bytes / 1024**2:.2f} MB)")
        return freed_bytes

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
Avg Generation Time: {self.total_generation_time / max(self.total_requests, 1):.3f}s
Total Generation Time: {self.total_generation_time:.3f}s
Active Sessions: {len(self.active_sessions)}
"""

        if self.inference_mode == InferenceMode.PENSIEVE and self.cache:
            stats += f"\nCache Statistics:\n{self.cache.get_memory_stats_str()}\n"
            cache_stats = self.cache.get_statistics()
            stats += f"GPU Hit Rate: {cache_stats.gpu_hit_rate:.1%}\n"
            stats += f"CPU Hit Rate: {cache_stats.cpu_hit_rate:.1%}\n"
            stats += f"Miss Rate: {cache_stats.miss_rate:.1%}\n"

        return stats

    # ============================================================================
    # ✅ ASYNC REQUEST INTERFACE (For unified batching)
    # ============================================================================

    def submit_request_async(
        self, session_id: str, user_input: str, max_new_tokens: int = 32
    ) -> str:
        """Non-blocking request submission for unified batching.

        This method queues a request without waiting for response.
        Results can be retrieved with get_request_result() or poll_results().

        Args:
            session_id: Session ID
            user_input: User input text
            max_new_tokens: Max tokens to generate

        Returns:
            request_id for later result retrieval
        """
        request_id = f"{session_id}_{self.total_requests}_{time.time()}"

        # Queue request data (non-blocking)
        self.async_request_queue.put({
            'request_id': request_id,
            'session_id': session_id,
            'user_input': user_input,
            'max_new_tokens': max_new_tokens,
        })

        # Mark as pending
        with self.request_lock:
            self.pending_requests[request_id] = {
                'session_id': session_id,
                'status': 'queued',
            }

        return request_id

    def start_batch_collection_thread(self):
        """Start background thread for batch collection and execution.

        This thread:
        1. Collects requests from queue (with timeout)
        2. Forms unified batches (mixed prefill + generation)
        3. Executes batches via scheduler + worker
        4. Stores results for retrieval

        Call this once at server startup.
        """
        if self.batch_collection_running:
            return

        self.batch_collection_running = True
        self.batch_collection_thread = threading.Thread(
            target=self._batch_collection_loop,
            daemon=True,
        )
        self.batch_collection_thread.start()
        print("✓ Batch collection thread started")

    def start_immediate_request_processing_thread(self):
        """Start background thread for immediate request processing (vLLM mode).

        This thread processes requests immediately without batching.
        Each request is processed individually (stateless).

        Call this once at server startup for vLLM baseline.
        """
        if self.batch_collection_running:
            return

        self.batch_collection_running = True
        self.batch_collection_thread = threading.Thread(
            target=self._immediate_request_processing_loop,
            daemon=True,
        )
        self.batch_collection_thread.start()
        print("✓ Immediate request processing thread started (vLLM stateless mode)")

    def _immediate_request_processing_loop(self):
        """Background loop for immediate request processing (no batching).

        For vLLM baseline: process each request individually as it arrives.
        """
        while self.batch_collection_running:
            try:
                # Get single request (non-blocking check)
                try:
                    req_data = self.async_request_queue.get(timeout=0.01)
                except:  # Queue.Empty
                    continue

                # Process immediately (stateless, no batching)
                request_id = req_data['request_id']
                session_id = req_data['session_id']
                user_input = req_data['user_input']
                max_new_tokens = req_data['max_new_tokens']

                # Process synchronously (immediate execution)
                response = self.process_request(
                    session_id,
                    user_input,
                    max_new_tokens=max_new_tokens,
                )

                # Store result
                with self.request_lock:
                    self.request_results[request_id] = response

            except Exception as e:
                print(f"Error in immediate request processing loop: {e}")
                import traceback
                traceback.print_exc()

    def _batch_collection_loop(self):
        """Background loop for batch collection and execution."""
        while self.batch_collection_running:
            try:
                # Collect batch
                batch_requests = self._collect_batch()

                if not batch_requests:
                    continue

                # Execute batch
                self._execute_batch_async(batch_requests)

            except Exception as e:
                print(f"Error in batch collection loop: {e}")
                import traceback
                traceback.print_exc()

    def _collect_batch(self) -> List[Dict]:
        """Collect multiple requests into a batch.

        Strategy:
        - Wait up to batch_timeout for requests
        - Return early if max_batch_size reached
        - Return partial batch if timeout expires

        Returns:
            List of request dicts, or empty list if queue empty
        """
        batch = []
        start_time = time.time()

        while len(batch) < self.max_batch_size:
            elapsed = time.time() - start_time
            timeout = max(0.001, self.batch_timeout - elapsed)

            try:
                req_data = self.async_request_queue.get(timeout=timeout)
                batch.append(req_data)
            except:  # Queue.Empty
                break

        return batch

    def _execute_batch_async(self, batch_requests: List[Dict]):
        """Execute a batch of requests asynchronously.

        Args:
            batch_requests: List of request data dicts
        """
        try:
            # Convert request dicts to Request objects
            requests = []
            for req_data in batch_requests:
                session_id = req_data['session_id']
                user_input = req_data['user_input']
                max_new_tokens = req_data['max_new_tokens']
                request_id = req_data['request_id']

                # Get or create session
                if session_id not in self.active_sessions:
                    self.active_sessions[session_id] = []

                # Get conversation history
                history = self._get_session_history(session_id)
                print("Penesieve history",history)
                # Encode input
                full_input = history + user_input
                input_ids = self.tokenizer.encode(full_input, return_tensors='pt')
                input_ids = input_ids.squeeze(0)

                # Create request
                request = Request(
                    session_id=session_id,
                    request_id=request_id,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    original_text=user_input,
                )
                requests.append(request)

            # ✅ Execute via unified scheduler + worker
            if self.inference_mode == InferenceMode.PENSIEVE:
                self.scheduler.add_requests(requests)  # Add all to scheduler
                batch, cache_plan = self.scheduler.form_next_batch()

                worker = self._get_worker()
                batch_result = worker.execute_batch(batch, cache_plan)
                # print(batch_result.prefill_time)
                # print(batch_result)
                # Store results with timing
                with self.request_lock:
                    # print("@@@@@. sj")
                    # ✅ Accumulate prefill and generation time separately
                    self.total_prefill_time += batch_result.prefill_time
                    # print(self.total_prefill_time)
                    self.total_generation_time += batch_result.generation_time

                    # Store TTFT from batch result
                    if batch_result.ttft_per_request:
                        self.last_ttft_per_request.update(batch_result.ttft_per_request)
                        # ✅ Accumulate all TTFTs for aggregation
                        self.all_ttfts.extend(batch_result.ttft_per_request.values())

                    for req in requests:
                        request_id = req.request_id
                        session_id = req.session_id

                        if request_id in batch_result.request_results:
                            result_dict = batch_result.request_results[request_id]
                            response = result_dict.get("response", "")
                            tokens_generated = result_dict.get("tokens_generated", 0)
                        else:
                            response = ""
                            tokens_generated = 0

                        # Store response
                        self.request_results[request_id] = response

                        # Update session history
                        if session_id not in self.session_token_histories:
                            self.session_token_histories[session_id] = []

                        input_token_ids = req.input_ids.tolist() if isinstance(req.input_ids, torch.Tensor) else req.input_ids
                        self.session_token_histories[session_id].extend(input_token_ids)

                        # Mark as completed
                        self.pending_requests[request_id]['status'] = 'completed'
                        self.total_requests += 1
                        self.total_tokens_generated += tokens_generated
                        req.response_text = response
                        self.active_sessions[session_id].append(req)

        except Exception as e:
            print(f"Error executing batch: {e}")
            import traceback
            traceback.print_exc()

    def get_request_result(self, request_id: str, timeout: float = 30.0) -> Optional[str]:
        """Retrieve result for a specific request (blocking).

        Args:
            request_id: Request ID from submit_request_async()
            timeout: Max time to wait (seconds)

        Returns:
            Response text, or None if not ready or timed out
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.request_lock:
                if request_id in self.request_results:
                    return self.request_results[request_id]

            time.sleep(0.01)  # Poll interval

        return None

    def poll_results(self) -> Dict[str, str]:
        """Get all currently available results (non-blocking).

        Returns:
            Dict mapping request_id to response (only completed requests)
        """
        with self.request_lock:
            return dict(self.request_results)

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
        self.all_ttfts.clear()
        self.last_ttft_per_request.clear()


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
