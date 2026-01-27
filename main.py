#!/usr/bin/env python3
"""Main entry point for Pensieve server.

Usage:
    # Run with Pensieve (stateful) mode - uses Llama-3-8B by default
    python main.py --mode pensieve

    # Run with vLLM baseline (stateless) mode
    python main.py --mode vllm

    # Run interactive multi-turn conversation
    python main.py --mode pensieve --interactive

    # Compare Pensieve vs vLLM (main benchmark)
    python main.py --mode compare

    # Use smaller model for faster testing
    python main.py --mode compare --model gpt2

    # Compare on ShareGPT dataset (future)
    python main.py --mode compare --dataset sharegt --num_conversations 10
"""

import argparse
import sys
import os
import time
import torch

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
