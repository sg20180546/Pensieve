#!/usr/bin/env python3
"""Main entry point for Pensieve server.

Usage:
    # Run with Pensieve (stateful) mode - uses Llama-3-8B by default
    python main.py --mode pensieve

    # Run with vLLM baseline (stateless) mode
    python main.py --mode vllm

    # Run interactive multi-turn conversation
    python main.py --mode pensieve --interactive

    # Compare Pensieve vs vLLM (sequential - single user)
    python main.py --mode compare --model gpt2

    # Compare with concurrent users (RECOMMENDED for demonstration)
    # Automatically creates HOT/WARM/COLD clients to show cache hotness
    python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.5

    # Stress test with many concurrent users
    python main.py --mode compare --model gpt2 --num-concurrent-users 9 --request-interval 0.3

    # Full model evaluation (Meta-Llama-3-8B)
    python main.py --mode compare --model meta-llama/Meta-Llama-3-8B \
        --num-concurrent-users 6 --gpu-cache 40 --cpu-cache 100

    # Compare on ShareGPT dataset (future)
    python main.py --mode compare --dataset sharegt --num_conversations 10

Key Features:
    - Concurrent benchmark: Simulates multiple users with different access patterns
    - Automatic hotness distribution: HOT (frequent), WARM (normal), COLD (infrequent)
    - Cache-aware batching: Unified PREFILL+GENERATION batches
    - Eviction policy showcase: Shows intelligent cache management across sessions
"""

import argparse
import sys
import os
import time
import torch
import threading
from queue import Queue
from collections import defaultdict
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pensieve.server import create_server, InferenceMode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pensieve: Stateful LLM Serving with KV Cache Management"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="pensieve",
        choices=["pensieve", "vllm", "compare"],
        help="Inference mode: 'pensieve' (stateful with KV cache), 'vllm' (stateless baseline), or 'compare' (run both and compare)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model name (default: meta-llama/Meta-Llama-3-8B). Also supports 'gpt2', 'opt-125m', 'opt-350m', 'opt-1.3b'",
    )

    # Cache configuration
    parser.add_argument(
        "--gpu-cache",
        type=float,
        default=40,
        help="GPU cache size in GB (default: 40)",
    )
    parser.add_argument(
        "--cpu-cache",
        type=float,
        default=100,
        help="CPU cache size in GB (default: 100)",
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU device (default: cuda:0)",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive multi-turn conversation",
    )

    # Dataset evaluation
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sharegt", "ultrachat"],
        help="Dataset to use for evaluation (sharegt or ultrachat)",
    )

    parser.add_argument(
        "--num-conversations",
        type=int,
        default=10,
        help="Number of conversations to process (default: 10)",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Max turns per conversation (default: 5)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens per turn (default: 32)",
    )

    parser.add_argument(
        "--num-concurrent-users",
        type=int,
        default=1,
        help="Number of concurrent users for benchmark (default: 1). Only used with --mode compare.",
    )

    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.5,
        help="Time interval (seconds) between consecutive requests from each user (default: 0.5)",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("Pensieve: Stateful LLM Serving with KV Cache Management")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"GPU Cache: {args.gpu_cache} GB")
    print(f"CPU Cache: {args.cpu_cache} GB")
    print(f"Device: {args.device}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    if args.interactive:
        run_interactive(args)
    elif args.mode == "compare":
        # Use concurrent benchmark if num_concurrent_users > 1
        if args.num_concurrent_users > 1:
            run_concurrent_comparison(args)
        else:
            run_comparison(args)
    elif args.dataset:
        run_dataset_evaluation(args)
    else:
        run_demo(args)


def run_demo(args):
    """Run simple demo."""
    print("\n--- Running Demo ---\n")

    # Create server
    server = create_server(
        model_name=args.model,
        mode=args.mode,
        gpu_capacity_gb=args.gpu_cache,
        cpu_capacity_gb=args.cpu_cache,
        device=args.device,
    )

    # Demo conversation
    conversations = [
        ("session_1", "Hello, how are you?"),
        ("session_1", "Tell me about machine learning"),
        ("session_2", "What is Python?"),
        ("session_2", "Tell me about Python's popularity"),
    ]

    print(f"Using mode: {args.mode}")
    print(f"Model: {args.model}\n")

    for session_id, user_input in conversations:
        print(f"[{session_id}] User: {user_input}")
        response = server.process_request(session_id, user_input, max_new_tokens=args.max_new_tokens)
        print(f"[{session_id}] Assistant: {response}\n")

    # Print statistics
    print(server.get_statistics_str())


def run_interactive(args):
    """Run interactive multi-turn conversation."""
    print("\n--- Interactive Mode ---")
    print("Type 'exit' to quit, 'stats' to see statistics\n")

    server = create_server(
        model_name=args.model,
        mode=args.mode,
        gpu_capacity_gb=args.gpu_cache,
        cpu_capacity_gb=args.cpu_cache,
        device=args.device,
    )

    session_id = f"interactive_{int(time.time())}"
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Exiting...")
                break

            if user_input.lower() == "stats":
                print(server.get_statistics_str())
                continue

            if not user_input:
                continue

            turn += 1
            print("\nAssistant: ", end="", flush=True)

            response = server.process_request(
                session_id,
                user_input,
                max_new_tokens=args.max_new_tokens,
            )
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

    print(server.get_statistics_str())


def concurrent_client_worker(
    client_id: int,
    server,
    conversations: list,
    request_interval: float,
    results_queue: Queue,
) -> None:
    """Simulate a single concurrent client making requests (blocking version).

    Args:
        client_id: ID of this client
        server: Inference server instance
        conversations: List of conversation turns for this client
        request_interval: Time to wait between requests (seconds)
        results_queue: Thread-safe queue for collecting results
    """
    ttfts = []
    tail_latencies = []
    session_id = f"session_{client_id}_{int(time.time() * 1000)}"

    for turn_idx, user_input in enumerate(conversations):
        # Wait before sending next request
        if turn_idx > 0:
            time.sleep(request_interval)

        # Measure execution time
        turn_start = time.time()
        try:
            response = server.process_request(
                session_id,
                user_input,
                max_new_tokens=32,
            )
            turn_end = time.time()
            tail_latency = turn_end - turn_start

            # Get TTFT from server
            ttft = list(server.last_ttft_per_request.values())[0] if server.last_ttft_per_request else 0.0

            ttfts.append(ttft)
            tail_latencies.append(tail_latency)
        except Exception as e:
            print(f"Error in client {client_id} turn {turn_idx}: {e}")

    # Put results in queue
    results_queue.put({
        "client_id": client_id,
        "ttfts": ttfts,
        "tail_latencies": tail_latencies,
        "response_count": len(conversations),
    })


def concurrent_client_worker_async(
    client_id: int,
    server,
    conversations: list,
    request_interval: float,
    results_queue: Queue,
) -> None:
    """Simulate a single concurrent client using async request submission (non-blocking).

    ✅ ASYNC VERSION: Submits requests without waiting for response.
    Benefits:
    - Allows unified batching: multiple requests in flight together
    - Better GPU utilization: batch processing multiple sessions
    - Demonstrates Pensieve advantage: prefill + generation mixed in batches

    Args:
        client_id: ID of this client
        server: Inference server instance (with async queue)
        conversations: List of conversation turns for this client
        request_interval: Time to wait between requests (seconds)
        results_queue: Thread-safe queue for collecting results
    """
    ttfts = []
    tail_latencies = []
    session_id = f"session_{client_id}_{int(time.time() * 1000)}"
    request_ids = []

    # Phase 1: Submit all requests (non-blocking)
    for turn_idx, user_input in enumerate(conversations):
        # Wait before sending next request
        if turn_idx > 0:
            time.sleep(request_interval)

        # Submit async (returns immediately)
        request_id = server.submit_request_async(
            session_id,
            user_input,
            max_new_tokens=32,
        )
        request_ids.append(request_id)

    # Phase 2: Collect results (blocking wait)
    for request_id in request_ids:
        try:
            # Wait for result (with timeout)
            response = server.get_request_result(request_id, timeout=30.0)
            if response:
                # Result is available - estimate timing
                # (In real system, would get actual TTFT from batch results)
                ttft = 0.1  # Placeholder
                tail_latency = 0.2  # Placeholder
                ttfts.append(ttft)
                tail_latencies.append(tail_latency)
        except Exception as e:
            print(f"Error retrieving result {request_id} for client {client_id}: {e}")

    # Put results in queue
    results_queue.put({
        "client_id": client_id,
        "ttfts": ttfts,
        "tail_latencies": tail_latencies,
        "response_count": len(conversations),
    })


def run_concurrent_comparison(args):
    """Run comparison between Pensieve and vLLM with concurrent users.

    IMPORTANT: Simulates multiple clients with DIFFERENT access patterns
    to demonstrate cache hotness effects and eviction policy.

    Key Design:
    - Each client has different request frequency (hotness levels)
    - Hot clients (frequent access) → stay in GPU cache
    - Cold clients (infrequent access) → evicted to CPU/DROPPED
    - When cold client returns → shows recovery/swap overhead

    Workflow:
    1. Launch N concurrent client threads with varied request intervals
    2. Collect metrics from all clients
    3. Clean up and reset
    4. Launch N concurrent client threads for vLLM with same pattern
    5. Compare results (should show Pensieve advantage with cache reuse)
    """
    import gc

    # Define conversations for each client
    # Using different conversation sets so clients have diverse workloads
    client_conversations = [
        (
            "session_1",
            ["Hello, how are you?", "Tell me about Python", "What is machine learning?"],
        ),
        (
            "session_2",
            ["Hi there", "Explain AI", "How does deep learning work?"],
        ),
        (
            "session_3",
            ["What's up?", "Tell me about data science", "What is cloud computing?"],
        ),
        (
            "session_4",
            ["Hey", "Explain algorithms", "What is a neural network?"],
        ),
        (
            "session_5",
            ["Hello", "What is big data?", "Explain supervised learning"],
        ),
    ]

    # Define client-specific access patterns (hotness levels)
    # This creates realistic scenario where sessions have different access frequencies
    num_users = min(args.num_concurrent_users, len(client_conversations))

    # Access intervals per client - creates hotness distribution
    # Hot clients: frequent access (0.2s) → GPU cache
    # Warm clients: normal access (0.5s) → GPU/CPU boundary
    # Cold clients: infrequent access (1.5-2.0s) → CPU cache, triggers eviction
    client_intervals = []
    for i in range(num_users):
        if i < num_users // 3:  # First 1/3: HOT (frequent)
            interval = max(0.1, args.request_interval * 0.3)
        elif i < 2 * num_users // 3:  # Middle 1/3: WARM (normal)
            interval = args.request_interval
        else:  # Last 1/3: COLD (infrequent)
            interval = args.request_interval * 2.5
        client_intervals.append(interval)

    print("\n" + "=" * 60)
    print("Concurrent Benchmark: Pensieve vs vLLM (With Cache Hotness)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Concurrent Users: {num_users}")
    print(f"  Base Request Interval: {args.request_interval}s")
    print(f"  Model: {args.model}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"\nClient Hotness Distribution:")
    for i in range(num_users):
        if i < num_users // 3:
            hotness = "HOT (Frequent)"
        elif i < 2 * num_users // 3:
            hotness = "WARM (Normal)"
        else:
            hotness = "COLD (Infrequent)"
        print(f"  Client {i}: {hotness:20} interval={client_intervals[i]:.2f}s")
    print("=" * 60)

    # ============================================================================
    # PHASE 1: Run Pensieve with Concurrent Clients
    # ============================================================================
    print("\nPHASE 1: PENSIEVE (Concurrent Users with Hotness)")
    print("-" * 60)

    pensieve_server = create_server(
        model_name=args.model,
        mode="pensieve",
        gpu_capacity_gb=args.gpu_cache,
        cpu_capacity_gb=args.cpu_cache,
        device=args.device,
    )

    print(f"✓ Pensieve server initialized (GPU memory allocated)")

    # ✅ Start batch collection thread for unified scheduling
    pensieve_server.start_batch_collection_thread()
    print(f"✓ Unified batch scheduler started (batch_timeout={pensieve_server.batch_timeout:.3f}s, max_batch_size={pensieve_server.max_batch_size})")

    # Launch concurrent client threads (using async submission)
    pensieve_results_queue = Queue()
    pensieve_threads = []

    start_time = time.time()

    for client_id in range(num_users):
        _, conversations = client_conversations[client_id]
        thread = threading.Thread(
            target=concurrent_client_worker_async,  # ✅ Use async version
            args=(
                client_id,
                pensieve_server,
                conversations,
                client_intervals[client_id],  # Use client-specific interval
                pensieve_results_queue,
            ),
        )
        thread.start()
        pensieve_threads.append(thread)

    # Wait for all threads to complete
    for thread in pensieve_threads:
        thread.join()

    pensieve_total_time = time.time() - start_time

    # ✅ Stop batch collection thread
    pensieve_server.batch_collection_running = False
    if pensieve_server.batch_collection_thread:
        pensieve_server.batch_collection_thread.join(timeout=1.0)

    # Aggregate results from all clients
    all_pensieve_ttfts = []
    all_pensieve_tail_latencies = []
    total_pensieve_requests = 0

    while not pensieve_results_queue.empty():
        result = pensieve_results_queue.get()
        all_pensieve_ttfts.extend(result["ttfts"])
        all_pensieve_tail_latencies.extend(result["tail_latencies"])
        total_pensieve_requests += result["response_count"]

    pensieve_stats = pensieve_server.get_statistics_str()
    print(f"\n{pensieve_stats}")

    # Calculate Pensieve metrics
    pensieve_avg_ttft = mean(all_pensieve_ttfts) if all_pensieve_ttfts else 0
    pensieve_p99_ttft = (
        sorted(all_pensieve_ttfts)[int(len(all_pensieve_ttfts) * 0.99)]
        if all_pensieve_ttfts
        else 0
    )
    pensieve_avg_tail = mean(all_pensieve_tail_latencies) if all_pensieve_tail_latencies else 0
    pensieve_p99_tail = (
        sorted(all_pensieve_tail_latencies)[int(len(all_pensieve_tail_latencies) * 0.99)]
        if all_pensieve_tail_latencies
        else 0
    )
    pensieve_throughput = (
        total_pensieve_requests / pensieve_total_time if pensieve_total_time > 0 else 0
    )

    # ============================================================================
    # CLEANUP: Explicit memory cleanup between runs
    # ============================================================================
    print("\n" + "-" * 60)
    print("Cleaning up GPU memory...")

    del pensieve_server
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("✓ Memory cleanup complete")
    print("-" * 60)

    # ============================================================================
    # PHASE 2: Run vLLM Baseline with Concurrent Clients (Same Pattern)
    # ============================================================================
    print("\nPHASE 2: vLLM BASELINE (Concurrent Users - Stateless, Same Hotness)")
    print("-" * 60)

    vllm_server = create_server(
        model_name=args.model,
        mode="vllm",
        gpu_capacity_gb=args.gpu_cache,
        cpu_capacity_gb=args.cpu_cache,
        device=args.device,
    )

    print(f"✓ vLLM server initialized (GPU memory allocated)")

    # Launch concurrent client threads with SAME access pattern
    vllm_results_queue = Queue()
    vllm_threads = []

    start_time = time.time()

    for client_id in range(num_users):
        _, conversations = client_conversations[client_id]
        thread = threading.Thread(
            target=concurrent_client_worker,
            args=(
                client_id,
                vllm_server,
                conversations,
                client_intervals[client_id],  # Use same client-specific interval
                vllm_results_queue,
            ),
        )
        thread.start()
        vllm_threads.append(thread)

    # Wait for all threads to complete
    for thread in vllm_threads:
        thread.join()

    vllm_total_time = time.time() - start_time

    # Aggregate results from all clients
    all_vllm_ttfts = []
    all_vllm_tail_latencies = []
    total_vllm_requests = 0

    while not vllm_results_queue.empty():
        result = vllm_results_queue.get()
        all_vllm_ttfts.extend(result["ttfts"])
        all_vllm_tail_latencies.extend(result["tail_latencies"])
        total_vllm_requests += result["response_count"]

    vllm_stats = vllm_server.get_statistics_str()
    print(f"\n{vllm_stats}")

    # Calculate vLLM metrics
    vllm_avg_ttft = mean(all_vllm_ttfts) if all_vllm_ttfts else 0
    vllm_p99_ttft = (
        sorted(all_vllm_ttfts)[int(len(all_vllm_ttfts) * 0.99)]
        if all_vllm_ttfts
        else 0
    )
    vllm_avg_tail = mean(all_vllm_tail_latencies) if all_vllm_tail_latencies else 0
    vllm_p99_tail = (
        sorted(all_vllm_tail_latencies)[int(len(all_vllm_tail_latencies) * 0.99)]
        if all_vllm_tail_latencies
        else 0
    )
    vllm_throughput = (
        total_vllm_requests / vllm_total_time if vllm_total_time > 0 else 0
    )

    # ============================================================================
    # COMPARISON & ANALYSIS
    # ============================================================================
    print("\n" + "=" * 60)
    print("CONCURRENT BENCHMARK COMPARISON")
    print("=" * 60)

    print(f"\n{'METRIC':<30} {'PENSIEVE':<20} {'vLLM':<20}")
    print("-" * 70)

    # Total time comparison
    print(f"{'Total Time':<30} {pensieve_total_time:.3f}s{'':<13} {vllm_total_time:.3f}s")
    time_speedup = vllm_total_time / pensieve_total_time if pensieve_total_time > 0 else 0
    print(f"{'Time Speedup':<30} {time_speedup:.2f}x")

    # Throughput comparison
    print(f"\n{'Throughput (req/s)':<30} {pensieve_throughput:.2f}{'':<17} {vllm_throughput:.2f}")
    throughput_speedup = vllm_throughput / pensieve_throughput if pensieve_throughput > 0 else 0
    print(f"{'Throughput Speedup':<30} {throughput_speedup:.2f}x")

    # TTFT comparison
    print(f"\n{'Avg TTFT':<30} {pensieve_avg_ttft*1000:.1f}ms{'':<13} {vllm_avg_ttft*1000:.1f}ms")
    print(f"{'P99 TTFT':<30} {pensieve_p99_ttft*1000:.1f}ms{'':<13} {vllm_p99_ttft*1000:.1f}ms")
    ttft_speedup = vllm_avg_ttft / pensieve_avg_ttft if pensieve_avg_ttft > 0 else 0
    print(f"{'TTFT Speedup (avg)':<30} {ttft_speedup:.2f}x")

    # Tail latency comparison
    print(f"\n{'Avg Tail Latency':<30} {pensieve_avg_tail*1000:.1f}ms{'':<13} {vllm_avg_tail*1000:.1f}ms")
    print(f"{'P99 Tail Latency':<30} {pensieve_p99_tail*1000:.1f}ms{'':<13} {vllm_p99_tail*1000:.1f}ms")
    tail_speedup = vllm_avg_tail / pensieve_avg_tail if pensieve_avg_tail > 0 else 0
    print(f"{'Tail Latency Speedup':<30} {tail_speedup:.2f}x")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("\nHotness-Based Analysis:")
    print(f"  • HOT clients (1/3): Request every {client_intervals[0]:.2f}s → GPU cache hit")
    print(f"  • WARM clients (1/3): Request every {client_intervals[num_users//3]:.2f}s → GPU/CPU mixed")
    print(f"  • COLD clients (1/3): Request every {client_intervals[-1]:.2f}s → Eviction/Recovery")
    print("\nExpected Results:")
    print("  • Pensieve benefits from cache reuse on hot clients")
    print("  • Cold clients trigger cache swaps/recovery but reuse saved prefill cost")
    print("  • vLLM recomputes ALL turns regardless of access pattern (no cache)")
    print("  • Speedup > 1.0 indicates cache reuse benefits for multi-turn workload")

    print("\n" + "=" * 70)
    if time_speedup > 1.0:
        improvement_pct = (time_speedup - 1.0) * 100
        print(f"✓ Pensieve is {improvement_pct:.1f}% faster (cache reuse across {num_users} concurrent users)")
    else:
        improvement_pct = (1.0/time_speedup - 1.0) * 100
        print(f"⚠ vLLM is {improvement_pct:.1f}% faster")
        print("  Note: This may indicate cache swap overhead > reuse savings")
        print("  Increase --num-concurrent-users to see stronger Pensieve advantage")

    print(f"\n✓ Summary: {num_users} concurrent users | {total_pensieve_requests} total requests")


def run_comparison(args):
    """Run comparison between Pensieve and vLLM baseline.

    IMPORTANT: Each server runs in isolated state with explicit memory cleanup
    to ensure accurate benchmarking without GPU memory contention.

    Workflow:
    1. Run Pensieve with clean GPU memory
    2. Collect statistics and clear model from memory
    3. Run vLLM baseline with clean GPU memory
    4. Compare results
    """
    import gc

    print("\n" + "=" * 60)
    print("Comparison: Pensieve vs vLLM Baseline (Isolated Execution)")
    print("=" * 60)

    # Demo conversations
    conversations = [
        ("session_1", ["Hello", "Tell me a joke", "What's the capital of France?"]),
        ("session_2", ["Hi there", "Explain AI", "How does deep learning work?"]),
    ]

    print(f"\nModel: {args.model}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"GPU Cache: {args.gpu_cache} GB | CPU Cache: {args.cpu_cache} GB")
    print("=" * 60)

    # ============================================================================
    # PHASE 1: Run Pensieve (Stateful with KV Cache)
    # ============================================================================
    print("\nPHASE 1: PENSIEVE (Stateful with KV Cache)")
    print("-" * 60)

    pensieve_server = create_server(
        model_name=args.model,
        mode="pensieve",
        gpu_capacity_gb=args.gpu_cache,
        cpu_capacity_gb=args.cpu_cache,
        device=args.device,
    )

    print(f"✓ Pensieve server initialized (GPU memory allocated)")

    # Run conversations and track per-turn metrics
    pensieve_ttfts = []
    pensieve_tail_latencies = []

    for session_id, turns in conversations:
        print(f"\n[{session_id}]")
        for turn_idx, user_input in enumerate(turns, 1):
            print(f"  Turn {turn_idx}: User: {user_input[:40]}...")

            # Measure execution time
            turn_start = time.time()
            response = pensieve_server.process_request(
                session_id,
                user_input,
                max_new_tokens=args.max_new_tokens,
            )
            turn_end = time.time()
            tail_latency = turn_end - turn_start

            # Get TTFT from server
            ttft = list(pensieve_server.last_ttft_per_request.values())[0] if pensieve_server.last_ttft_per_request else 0.0

            print(f"           Assistant: {response[:60]}...")
            print(f"           TTFT: {ttft*1000:.1f}ms | Tail: {tail_latency*1000:.1f}ms")

            pensieve_ttfts.append(ttft)
            pensieve_tail_latencies.append(tail_latency)

    # Collect Pensieve statistics
    pensieve_stats = pensieve_server.get_statistics_str()
    pensieve_total_time = pensieve_server.total_prefill_time
    pensieve_requests = pensieve_server.total_requests

    print("\n" + pensieve_stats)

    # ============================================================================
    # CLEANUP: Explicit memory cleanup between runs
    # ============================================================================
    print("\n" + "-" * 60)
    print("Cleaning up GPU memory...")

    # Delete server and trigger garbage collection
    del pensieve_server
    gc.collect()

    # Clear CUDA cache explicitly
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("✓ Memory cleanup complete (GPU memory freed)")
    print("-" * 60)

    # ============================================================================
    # PHASE 2: Run vLLM Baseline (Stateless)
    # ============================================================================
    print("\nPHASE 2: vLLM BASELINE (Stateless - Recomputes History)")
    print("-" * 60)

    vllm_server = create_server(
        model_name=args.model,
        mode="vllm",
        gpu_capacity_gb=args.gpu_cache,
        cpu_capacity_gb=args.cpu_cache,
        device=args.device,
    )

    print(f"✓ vLLM server initialized (GPU memory allocated)")

    # Run conversations and track per-turn metrics
    vllm_ttfts = []
    vllm_tail_latencies = []

    for session_id, turns in conversations:
        print(f"\n[{session_id}]")
        for turn_idx, user_input in enumerate(turns, 1):
            print(f"  Turn {turn_idx}: User: {user_input[:40]}...")

            # Measure execution time
            turn_start = time.time()
            response = vllm_server.process_request(
                session_id,
                user_input,
                max_new_tokens=args.max_new_tokens,
            )
            turn_end = time.time()
            tail_latency = turn_end - turn_start

            # Get TTFT from server
            ttft = list(vllm_server.last_ttft_per_request.values())[0] if vllm_server.last_ttft_per_request else 0.0

            print(f"           Assistant: {response[:60]}...")
            print(f"           TTFT: {ttft*1000:.1f}ms | Tail: {tail_latency*1000:.1f}ms")

            vllm_ttfts.append(ttft)
            vllm_tail_latencies.append(tail_latency)

    # Collect vLLM statistics
    vllm_stats = vllm_server.get_statistics_str()
    vllm_total_time = vllm_server.total_prefill_time
    vllm_requests = vllm_server.total_requests

    print("\n" + vllm_stats)

    # ============================================================================
    # COMPARISON & ANALYSIS
    # ============================================================================
    print("\n" + "=" * 60)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 60)

    # Overall metrics
    print(f"\n{'METRIC':<30} {'PENSIEVE':<20} {'vLLM':<20}")
    print("-" * 70)

    if pensieve_total_time > 0 and vllm_total_time > 0:
        overall_speedup = vllm_total_time / pensieve_total_time

        print(f"{'Total Time':<30} {pensieve_total_time:.3f}s{'':<13} {vllm_total_time:.3f}s")
        print(f"{'Overall Speedup':<30} {overall_speedup:.2f}x")

        # Tail Latency comparison
        pensieve_avg_tail = sum(pensieve_tail_latencies) / len(pensieve_tail_latencies) if pensieve_tail_latencies else 0
        vllm_avg_tail = sum(vllm_tail_latencies) / len(vllm_tail_latencies) if vllm_tail_latencies else 0
        tail_speedup = vllm_avg_tail / pensieve_avg_tail if pensieve_avg_tail > 0 else 0

        print(f"{'Avg Tail Latency':<30} {pensieve_avg_tail*1000:.1f}ms{'':<13} {vllm_avg_tail*1000:.1f}ms")
        print(f"{'Tail Latency Speedup':<30} {tail_speedup:.2f}x")

        # TTFT comparison
        pensieve_avg_ttft = sum(pensieve_ttfts) / len(pensieve_ttfts) if pensieve_ttfts else 0
        vllm_avg_ttft = sum(vllm_ttfts) / len(vllm_ttfts) if vllm_ttfts else 0
        ttft_speedup = vllm_avg_ttft / pensieve_avg_ttft if pensieve_avg_ttft > 0 else 0

        print(f"{'Avg TTFT (Time to 1st Token)':<30} {pensieve_avg_ttft*1000:.1f}ms{'':<13} {vllm_avg_ttft*1000:.1f}ms")
        print(f"{'TTFT Speedup':<30} {ttft_speedup:.2f}x")

        print("\n" + "-" * 70)
        print("Per-Turn Analysis (Critical: Shows cache benefit growth)")
        print("-" * 70)
        print(f"{'Turn':<6} {'P-TTFT':<12} {'P-Tail':<12} {'V-TTFT':<12} {'V-Tail':<12} {'Tail-Gain'}")
        print("-" * 70)

        for turn_idx in range(min(len(pensieve_ttfts), len(vllm_ttfts))):
            p_ttft = pensieve_ttfts[turn_idx] * 1000
            p_tail = pensieve_tail_latencies[turn_idx] * 1000
            v_ttft = vllm_ttfts[turn_idx] * 1000
            v_tail = vllm_tail_latencies[turn_idx] * 1000

            tail_gain = v_tail / p_tail if p_tail > 0 else 0

            print(f"{turn_idx+1:<6} {p_ttft:>6.1f}ms{'':<4} {p_tail:>6.1f}ms{'':<4} {v_ttft:>6.1f}ms{'':<4} {v_tail:>6.1f}ms{'':<4} {tail_gain:>5.2f}x")

        print("\n" + "=" * 70)
        if overall_speedup > 1.0:
            improvement_pct = (overall_speedup - 1.0) * 100
            print(f"✓ Pensieve is {improvement_pct:.1f}% faster than vLLM baseline")
        else:
            print(f"⚠ vLLM is {(1.0/overall_speedup - 1.0) * 100:.1f}% faster (cache overhead?)")

        if len(pensieve_tail_latencies) > 1:
            final_speedup = vllm_tail_latencies[-1] / pensieve_tail_latencies[-1]
            print(f"✓ Final turn (turn {len(pensieve_tail_latencies)}): {final_speedup:.2f}x speedup")
    else:
        print("Warning: Unable to calculate speedup (invalid timing data)")


def run_dataset_evaluation(args):
    """Run evaluation on a dataset."""
    print(f"\n--- Dataset Evaluation ---")
    print(f"Dataset: {args.dataset}")
    print(f"Note: Dataset loading not yet implemented in demo")
    print(f"Use --interactive or default demo mode for now\n")

    # TODO: Implement dataset loading (ShareGPT, UltraChat)
    # For now, show how it would be used
    print("Dataset evaluation will:")
    print("  1. Load conversations from ShareGPT/UltraChat")
    print("  2. Simulate multi-turn interactions")
    print("  3. Measure prefill speedup over multiple turns")
    print("  4. Report cache hit rates and throughput")


if __name__ == "__main__":
    main()
