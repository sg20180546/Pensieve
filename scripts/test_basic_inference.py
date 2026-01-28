#!/usr/bin/env python3
"""Test basic model loading and inference."""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pensieve.core import Request, Batch


def test_basic_inference():
    """Test that we can load GPT-2 and run basic inference."""
    print("Testing basic GPT-2 model loading and inference...")

    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Set pad token for GPT-2 (doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Test 1: Simple generation
    print("\n--- Test 1: Simple generation ---")
    prompt = "Hello, how are you?"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    print(f"Input: {prompt}")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0])
    print(f"Output: {generated_text}")

    # Test 2: Multi-turn conversation simulation
    print("\n--- Test 2: Multi-turn simulation ---")
    turn_1_prompt = "Hello"
    turn_1_input = tokenizer.encode(turn_1_prompt, return_tensors='pt').to(device)
    turn_1_attention_mask = torch.ones_like(turn_1_input)

    print(f"Turn 1 input: {turn_1_prompt}")
    with torch.no_grad():
        outputs_1 = model.generate(
            turn_1_input,
            attention_mask=turn_1_attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )

    response_1 = tokenizer.decode(outputs_1[0])
    print(f"Turn 1 response: {response_1}")

    # Turn 2: Append to history
    turn_2_prompt = response_1 + " What is AI?"
    turn_2_input = tokenizer.encode(turn_2_prompt, return_tensors='pt').to(device)
    turn_2_attention_mask = torch.ones_like(turn_2_input)

    print(f"Turn 2 input: {turn_2_prompt[:50]}...")
    with torch.no_grad():
        outputs_2 = model.generate(
            turn_2_input,
            attention_mask=turn_2_attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )

    response_2 = tokenizer.decode(outputs_2[0])
    print(f"Turn 2 response: {response_2[-50:]}")

    # Test 3: Check model architecture
    print("\n--- Test 3: Model architecture ---")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of layers: {model.config.n_layer}")
    print(f"Hidden size: {model.config.n_embd}")
    print(f"Number of heads: {model.config.n_head}")

    # Test 4: Batch creation
    print("\n--- Test 4: Batch creation ---")
    req1 = Request(
        session_id="session_1",
        request_id="req_1",
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
    )
    req2 = Request(
        session_id="session_1",
        request_id="req_2",
        input_ids=torch.tensor([4, 5], dtype=torch.long),
    )

    batch = Batch()
    batch.add_request(req1)
    batch.add_request(req2)

    print(f"Batch size: {batch.num_requests}")
    print(f"Total tokens in batch: {batch.total_tokens}")

    # Test 5: KVChunk creation
    print("\n--- Test 5: KVChunk creation ---")
    from pensieve.core import KVChunk, CacheLocation

    # Create dummy KV tensors for a single layer
    num_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    num_heads = model.config.n_head
    head_dim = hidden_size // num_heads

    # Create chunk for layer 0 (32 tokens)
    k = torch.randn(1, 32, num_heads, head_dim, device='cpu')  # Store on CPU
    v = torch.randn(1, 32, num_heads, head_dim, device='cpu')

    chunk = KVChunk(
        session_id="session_1",
        chunk_id=0,
        layer_idx=0,  # First layer
        key_tensor=k,
        value_tensor=v,
        context_length=0,
        session_total_chunks=5,  # Assume 5 chunks total in session
        num_layers=num_layers,
        location=CacheLocation.GPU,
    )

    print(f"Chunk key: {chunk.key}")
    print(f"Chunk num tokens: {chunk.num_tokens}")
    print(f"Chunk size: {chunk.size_bytes / (1024**2):.2f} MB")

    print("\n✓ All basic tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_basic_inference()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
