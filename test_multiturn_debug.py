#!/usr/bin/env python3
"""Quick test to debug multi-turn KV cache shape mismatch.

Run with: python test_multiturn_debug.py
This will:
1. Load a small model (GPT-2)
2. Run 2 turns with same session
3. Print debug output to identify shape mismatch
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pensieve.server import create_server


def main():
    print("=" * 70)
    print("Multi-Turn KV Cache Debug Test")
    print("=" * 70)
    print()

    # Create Pensieve server with small model
    print("Initializing Pensieve server with GPT-2...")
    server = create_server(
        model_name="gpt2",
        mode="pensieve",
        gpu_capacity_gb=8,
        cpu_capacity_gb=16,
        device="cuda:0",
    )
    print("✓ Server initialized\n")

    # Simulate 2-turn conversation on same session
    session_id = "test_session_1"

    # Turn 1
    print("-" * 70)
    print("TURN 1: Initial request")
    print("-" * 70)
    user_input_1 = "Hello, how are you?"
    print(f"User: {user_input_1}")
    print("\nProcessing...")
    try:
        response_1 = server.process_request(
            session_id,
            user_input_1,
            max_new_tokens=20,
        )
        print(f"Assistant: {response_1}")
    except Exception as e:
        print(f"ERROR in Turn 1: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Turn 2
    print("-" * 70)
    print("TURN 2: Follow-up request (should reuse cache)")
    print("-" * 70)
    user_input_2 = "Tell me about machine learning"
    print(f"User: {user_input_2}")
    print("\nProcessing...")
    try:
        response_2 = server.process_request(
            session_id,
            user_input_2,
            max_new_tokens=20,
        )
        print(f"Assistant: {response_2}")
    except Exception as e:
        print(f"ERROR in Turn 2: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Shape mismatch occurred in Turn 2 (multi-turn KV storage)")
        print("Check the [DEBUG] output above to understand the issue")
        return

    print()
    print("=" * 70)
    print("✓ Multi-turn test PASSED")
    print("=" * 70)

    # Print statistics
    print(server.get_statistics_str())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
