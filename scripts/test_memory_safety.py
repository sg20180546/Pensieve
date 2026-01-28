#!/usr/bin/env python3
"""Test memory safety: no cross-tier chunk duplicates.

This test verifies that the fix for cross-tier chunk cleanup works correctly.
Both recovery and initial generation paths should use the same store_chunk() logic.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pensieve.core import TwoTierCache, KVChunk, CacheLocation


def test_cross_tier_cleanup_recovery():
    """Test that recovery recreates chunks without cross-tier duplication."""
    print("\n" + "="*60)
    print("TEST 1: Cross-Tier Cleanup During Recovery")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=0.1, cpu_capacity_gb=0.5)

    # Step 1: Create and store initial chunk in GPU
    chunk_0_v1 = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=3,
        num_layers=40,
    )
    cache.store_chunk(chunk_0_v1, CacheLocation.GPU)
    print(f"✓ Chunk 0 stored in GPU cache")
    assert chunk_0_v1.key in cache.gpu_cache, "Chunk should be in GPU"
    assert chunk_0_v1.key not in cache.cpu_cache, "Chunk should NOT be in CPU"

    # Step 2: Force eviction to CPU
    # Add enough chunks to fill GPU and cause eviction
    for i in range(1, 8):
        chunk = KVChunk(
            session_id='session_1',
            chunk_id=i,
            layer_idx=0,
            key_tensor=torch.randn(1, 32, 8, 64),
            value_tensor=torch.randn(1, 32, 8, 64),
            context_length=i * 32,
            session_total_chunks=3,
            num_layers=40,
        )
        cache.store_chunk(chunk, CacheLocation.GPU)

    print(f"✓ GPU used: {cache.gpu_used_bytes / 1024**2:.2f} MB / {cache.gpu_capacity_bytes / 1024**2:.2f} MB")
    print(f"✓ CPU used: {cache.cpu_used_bytes / 1024**2:.2f} MB / {cache.cpu_capacity_bytes / 1024**2:.2f} MB")

    # Check where chunk_0 ended up
    chunk_0_in_gpu = chunk_0_v1.key in cache.gpu_cache
    chunk_0_in_cpu = chunk_0_v1.key in cache.cpu_cache
    chunk_0_in_dropped = chunk_0_v1.key in cache.dropped_chunks

    print(f"✓ Chunk 0 location: GPU={chunk_0_in_gpu}, CPU={chunk_0_in_cpu}, DROPPED={chunk_0_in_dropped}")

    # Step 3: Simulate recovery - recreate chunk_0
    # This is what happens when recovery recreates a dropped chunk
    chunk_0_v2 = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),  # Different tensor (recomputed)
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=3,
        num_layers=40,
    )

    # Before storing recovered chunk
    initial_gpu_bytes = cache.gpu_used_bytes
    initial_cpu_bytes = cache.cpu_used_bytes

    # Store recovered chunk in GPU (this should trigger cleanup of old version)
    cache.store_chunk(chunk_0_v2, CacheLocation.GPU)
    print(f"\n✓ Recovered chunk 0 stored in GPU")

    # Step 4: Verify no cross-tier duplication
    chunk_0_in_gpu_after = chunk_0_v2.key in cache.gpu_cache
    chunk_0_in_cpu_after = chunk_0_v2.key in cache.cpu_cache
    chunk_0_in_dropped_after = chunk_0_v2.key in cache.dropped_chunks

    print(f"✓ After recovery: GPU={chunk_0_in_gpu_after}, CPU={chunk_0_in_cpu_after}, DROPPED={chunk_0_in_dropped_after}")

    # CRITICAL ASSERTIONS
    count_in_caches = sum([chunk_0_in_gpu_after, chunk_0_in_cpu_after, chunk_0_in_dropped_after])
    assert count_in_caches == 1, f"ERROR: Chunk 0 exists in {count_in_caches} places! Should be exactly 1"
    assert chunk_0_in_gpu_after, "Recovered chunk should be in GPU"
    print(f"✓ PASS: Chunk 0 exists in exactly 1 location (GPU)")

    # Step 5: Verify memory tracking is correct
    if chunk_0_in_cpu_after:
        # If it's still in CPU, memory should have increased
        assert cache.cpu_used_bytes > initial_cpu_bytes, "CPU memory should reflect stored chunk"
    elif chunk_0_in_dropped_after:
        # If dropped, no memory tracking
        pass
    else:
        # If old version was in CPU and now removed, CPU bytes should decrease
        assert cache.cpu_used_bytes <= initial_cpu_bytes, "Old chunk should be freed from CPU"

    print(f"✓ Memory tracking correct")
    print("\n✅ TEST 1 PASSED: No cross-tier duplication during recovery\n")


def test_same_tier_replacement():
    """Test replacement within the same tier (recovery + same location)."""
    print("\n" + "="*60)
    print("TEST 2: Same-Tier Replacement")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=1, cpu_capacity_gb=2)

    # Create initial chunk
    chunk_v1 = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=1,
        num_layers=40,
    )
    cache.store_chunk(chunk_v1, CacheLocation.GPU)
    print(f"✓ Initial chunk stored")

    initial_gpu_bytes = cache.gpu_used_bytes
    initial_chunks_in_gpu = len(cache.gpu_cache)

    # Store same chunk_id but different tensor (simulates recovery)
    chunk_v2 = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),  # Different tensor
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=1,
        num_layers=40,
    )
    cache.store_chunk(chunk_v2, CacheLocation.GPU)
    print(f"✓ Replacement chunk stored in same tier")

    # Verify no accumulation
    final_gpu_bytes = cache.gpu_used_bytes
    final_chunks_in_gpu = len(cache.gpu_cache)

    assert final_chunks_in_gpu == initial_chunks_in_gpu, \
        f"Should have same number of chunks: {initial_chunks_in_gpu} vs {final_chunks_in_gpu}"
    print(f"✓ No duplicate chunks in GPU ({final_chunks_in_gpu} chunks)")

    # Memory should be same (replaced, not accumulated)
    assert final_gpu_bytes == initial_gpu_bytes, \
        f"GPU memory should be same after replacement: {initial_gpu_bytes} vs {final_gpu_bytes}"
    print(f"✓ Memory stable after replacement")

    # Verify it's the new version
    stored_chunk = cache.gpu_cache[chunk_v2.key]
    assert torch.equal(stored_chunk.key_tensor, chunk_v2.key_tensor), \
        "Should store new version, not old"
    print(f"✓ New version correctly stored")

    print("\n✅ TEST 2 PASSED: Same-tier replacement works correctly\n")


def test_initial_generation_with_recovery():
    """Test that initial generation and recovery both use safe store_chunk logic."""
    print("\n" + "="*60)
    print("TEST 3: Initial Generation + Recovery Mixed")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=0.2, cpu_capacity_gb=0.5)

    # Simulate initial generation: chunks 0, 1, 2
    print("Step 1: Initial generation")
    for i in range(3):
        chunk = KVChunk(
            session_id='session_1',
            chunk_id=i,
            layer_idx=0,
            key_tensor=torch.randn(1, 32, 8, 64),
            value_tensor=torch.randn(1, 32, 8, 64),
            context_length=i * 32,
            session_total_chunks=3,
            num_layers=40,
        )
        cache.store_chunk(chunk, CacheLocation.GPU)

    print(f"✓ Generated 3 chunks")
    initial_positions = cache.get_session_positions('session_1')
    print(f"✓ Chunk positions: {initial_positions}")

    # Force some chunks to evict to CPU
    print("\nStep 2: Force eviction to CPU")
    for i in range(3, 7):
        chunk = KVChunk(
            session_id='session_2',
            chunk_id=i-3,
            layer_idx=0,
            key_tensor=torch.randn(1, 32, 8, 64),
            value_tensor=torch.randn(1, 32, 8, 64),
            context_length=0,
            session_total_chunks=4,
            num_layers=40,
        )
        cache.store_chunk(chunk, CacheLocation.GPU)

    gpu_positions_s1 = [p for p in cache.get_session_positions('session_1')
                        if cache.get_chunks_for_position('session_1', p) is not None]
    print(f"✓ Session 1 GPU chunks: {gpu_positions_s1}")

    # Simulate recovery: recreate chunk 0 (which was evicted)
    print("\nStep 3: Recover evicted chunk 0")
    recovered_chunk = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),  # Recomputed
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=3,
        num_layers=40,
    )
    cache.store_chunk(recovered_chunk, CacheLocation.GPU)
    print(f"✓ Recovered chunk 0")

    # Verify no cross-tier duplicates
    print("\nStep 4: Verify consistency")
    chunk_0_key = recovered_chunk.key

    in_gpu = chunk_0_key in cache.gpu_cache
    in_cpu = chunk_0_key in cache.cpu_cache
    in_dropped = chunk_0_key in cache.dropped_chunks

    count = sum([in_gpu, in_cpu, in_dropped])
    assert count == 1, f"ERROR: Chunk 0 in {count} places!"
    print(f"✓ Chunk 0 in exactly 1 location")

    # Verify next generation uses correct next_chunk_id
    print("\nStep 5: Generate next chunk")
    positions = cache.get_session_positions('session_1')
    next_id = max(positions) + 1 if positions else 0
    print(f"✓ Next chunk_id should be: {next_id}")

    new_chunk = KVChunk(
        session_id='session_1',
        chunk_id=next_id,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=next_id * 32,
        session_total_chunks=next_id + 1,
        num_layers=40,
    )
    cache.store_chunk(new_chunk, CacheLocation.GPU)
    print(f"✓ Generated chunk {next_id}")

    # Final check: all positions unique
    final_positions = cache.get_session_positions('session_1')
    assert len(final_positions) == len(set(final_positions)), "Duplicate positions!"
    print(f"✓ Final positions: {final_positions} (all unique)")

    print("\n✅ TEST 3 PASSED: Mixed recovery + generation works safely\n")


def test_session_chunks_tracking():
    """Verify session_chunks tracking stays consistent with actual caches."""
    print("\n" + "="*60)
    print("TEST 4: session_chunks Tracking Consistency")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=0.2, cpu_capacity_gb=0.5)

    # Store chunks
    for i in range(4):
        chunk = KVChunk(
            session_id='session_1',
            chunk_id=i,
            layer_idx=0,
            key_tensor=torch.randn(1, 32, 8, 64),
            value_tensor=torch.randn(1, 32, 8, 64),
            context_length=i * 32,
            session_total_chunks=4,
            num_layers=40,
        )
        cache.store_chunk(chunk, CacheLocation.GPU)

    tracked_chunks = cache.session_chunks.get('session_1', [])
    print(f"✓ session_chunks tracking: {len(tracked_chunks)} entries")

    # Verify each tracked chunk exists in a cache
    for chunk_key in tracked_chunks:
        in_gpu = chunk_key in cache.gpu_cache
        in_cpu = chunk_key in cache.cpu_cache
        in_dropped = chunk_key in cache.dropped_chunks
        in_any = in_gpu or in_cpu or in_dropped

        assert in_any, f"ERROR: Tracked chunk {chunk_key} not in any cache!"

    print(f"✓ All tracked chunks exist in caches")

    # Force eviction and recovery
    for i in range(4, 8):
        chunk = KVChunk(
            session_id='session_2',
            chunk_id=i-4,
            layer_idx=0,
            key_tensor=torch.randn(1, 32, 8, 64),
            value_tensor=torch.randn(1, 32, 8, 64),
            context_length=0,
            session_total_chunks=4,
            num_layers=40,
        )
        cache.store_chunk(chunk, CacheLocation.GPU)

    # Recover first chunk
    recovered = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=4,
        num_layers=40,
    )
    cache.store_chunk(recovered, CacheLocation.GPU)

    # Final verification
    tracked_chunks_final = cache.session_chunks.get('session_1', [])
    assert len(tracked_chunks_final) == len(set(tracked_chunks_final)), \
        "Duplicates in session_chunks!"
    print(f"✓ No duplicate entries in session_chunks")

    print("\n✅ TEST 4 PASSED: session_chunks tracking is consistent\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MEMORY SAFETY TEST SUITE")
    print("Testing cross-tier cleanup and consistency")
    print("="*60)

    try:
        test_cross_tier_cleanup_recovery()
        test_same_tier_replacement()
        test_initial_generation_with_recovery()
        test_session_chunks_tracking()

        print("\n" + "="*60)
        print("✅ ALL MEMORY SAFETY TESTS PASSED!")
        print("="*60)
        print("\nKey guarantees verified:")
        print("  ✓ No cross-tier chunk duplicates")
        print("  ✓ Memory tracking stays accurate")
        print("  ✓ session_chunks has no duplicates")
        print("  ✓ Recovery and generation both work safely")
        print("  ✓ Chunk positions calculated correctly\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
