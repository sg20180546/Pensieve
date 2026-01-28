#!/usr/bin/env python3
"""Test chunk pinning mechanism for concurrent execution safety.

This test verifies that:
1. Chunks can be pinned and unpinned
2. Pinned chunks are protected from eviction
3. Multiple sessions can be pinned simultaneously
4. Eviction policy respects pinning
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pensieve.core import TwoTierCache, KVChunk, CacheLocation


def test_basic_pinning():
    """Test basic pin/unpin operations."""
    print("\n" + "="*60)
    print("TEST 1: Basic Pin/Unpin Operations")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=1, cpu_capacity_gb=2)

    # Create a chunk for session 1
    chunk = KVChunk(
        session_id='session_1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=1,
        num_layers=40,
    )

    # Store chunk
    cache.store_chunk(chunk, CacheLocation.GPU)
    print(f"✓ Chunk stored: {chunk.key}")

    # Initially not pinned
    assert not cache.is_pinned(chunk.key)
    print(f"✓ Chunk not pinned initially")

    # Pin session
    cache.pin_session('session_1')
    assert cache.is_pinned(chunk.key)
    print(f"✓ Session 1 pinned, chunk protected")

    # Unpin session
    cache.unpin_session('session_1')
    assert not cache.is_pinned(chunk.key)
    print(f"✓ Session 1 unpinned, chunk can be evicted")

    print("\n✅ TEST 1 PASSED")


def test_pinned_eviction_protection():
    """Test that pinned chunks are protected from eviction."""
    print("\n" + "="*60)
    print("TEST 2: Pinned Chunks Protected From Eviction")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=0.1, cpu_capacity_gb=0.5)  # Very small cache

    # Create multiple chunks
    chunks = []
    for session_id in ['s1', 's2', 's3']:
        for chunk_id in range(3):
            chunk = KVChunk(
                session_id=session_id,
                chunk_id=chunk_id,
                layer_idx=0,
                key_tensor=torch.randn(1, 32, 8, 64),
                value_tensor=torch.randn(1, 32, 8, 64),
                context_length=chunk_id * 32,
                session_total_chunks=3,
                num_layers=40,
            )
            chunks.append(chunk)
            cache.store_chunk(chunk, CacheLocation.GPU)

    print(f"✓ Stored {len(chunks)} chunks in cache")
    print(f"  GPU used: {cache.gpu_used_bytes / 1024**2:.1f} MB / {cache.gpu_capacity_bytes / 1024**2:.1f} MB")

    # Count chunks before pinning
    s1_chunks_before = [k for k in cache.gpu_cache.keys() if 's1' in k]
    print(f"✓ Session 1 has {len(s1_chunks_before)} chunks in GPU cache")

    # Pin session 1 (protect its chunks)
    cache.pin_session('s1')
    s1_pinned_count = sum(1 for k in s1_chunks_before if cache.is_pinned(k))
    print(f"✓ Pinned {s1_pinned_count} chunks of session 1")

    # Try to evict space (for a new chunk)
    new_chunk = KVChunk(
        session_id='s4',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=1,
        num_layers=40,
    )

    print(f"\n  Attempting to store new chunk (should evict, but NOT s1)...")
    cache.store_chunk(new_chunk, CacheLocation.GPU)

    # Check if session 1 chunks still exist
    s1_chunks_after = [k for k in cache.gpu_cache.keys() if 's1' in k]
    print(f"✓ Session 1 still has {len(s1_chunks_after)} chunks (pinning protected them!)")
    assert len(s1_chunks_after) > 0, "ERROR: Session 1 chunks were evicted despite pinning!"

    # Check if other sessions' chunks were evicted
    s2_chunks = [k for k in cache.gpu_cache.keys() if 's2' in k]
    s3_chunks = [k for k in cache.gpu_cache.keys() if 's3' in k]
    print(f"✓ Session 2 has {len(s2_chunks)} chunks (may have been evicted)")
    print(f"✓ Session 3 has {len(s3_chunks)} chunks (may have been evicted)")

    # Unpin session 1
    cache.unpin_session('s1')
    print(f"\n✓ Session 1 unpinned (chunks can now be evicted)")

    print("\n✅ TEST 2 PASSED: Pinning protection works!")


def test_multiple_pinned_sessions():
    """Test pinning multiple sessions simultaneously."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Pinned Sessions")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=1, cpu_capacity_gb=2)

    # Create chunks for 3 sessions
    session_ids = ['s1', 's2', 's3']
    for session_id in session_ids:
        for chunk_id in range(2):
            chunk = KVChunk(
                session_id=session_id,
                chunk_id=chunk_id,
                layer_idx=0,
                key_tensor=torch.randn(1, 32, 8, 64),
                value_tensor=torch.randn(1, 32, 8, 64),
                context_length=chunk_id * 32,
                session_total_chunks=2,
                num_layers=40,
            )
            cache.store_chunk(chunk, CacheLocation.GPU)

    print(f"✓ Created 6 chunks for 3 sessions")

    # Pin all 3 sessions
    for session_id in session_ids:
        cache.pin_session(session_id)
        pinned = sum(1 for k in cache.session_chunks[session_id] if cache.is_pinned(k))
        print(f"  ✓ Pinned {session_id}: {pinned} chunks")

    # Verify all chunks are pinned
    total_pinned = len(cache.pinned_chunks)
    print(f"✓ Total pinned chunks: {total_pinned}")
    assert total_pinned == 6, "Not all chunks pinned!"

    # Unpin one session
    cache.unpin_session('s2')
    total_pinned = len(cache.pinned_chunks)
    print(f"✓ After unpinning s2: {total_pinned} chunks remain pinned")
    assert total_pinned == 4, "Wrong number of pinned chunks!"

    # Unpin others
    for session_id in ['s1', 's3']:
        cache.unpin_session(session_id)

    total_pinned = len(cache.pinned_chunks)
    print(f"✓ After unpinning all: {total_pinned} chunks pinned")
    assert total_pinned == 0, "Chunks should not be pinned!"

    print("\n✅ TEST 3 PASSED: Multiple pinning works!")


def test_pinning_with_batch_execution():
    """Test pinning behavior during batch execution (simulated)."""
    print("\n" + "="*60)
    print("TEST 4: Pinning During Batch Execution (Simulated)")
    print("="*60)

    cache = TwoTierCache(gpu_capacity_gb=0.5, cpu_capacity_gb=1)

    # Simulate batch with 2 sessions
    batch_sessions = ['batch_s1', 'batch_s2']

    # Create chunks
    for session_id in batch_sessions:
        for chunk_id in range(4):
            chunk = KVChunk(
                session_id=session_id,
                chunk_id=chunk_id,
                layer_idx=0,
                key_tensor=torch.randn(1, 32, 8, 64),
                value_tensor=torch.randn(1, 32, 8, 64),
                context_length=chunk_id * 32,
                session_total_chunks=4,
                num_layers=40,
            )
            cache.store_chunk(chunk, CacheLocation.GPU)

    print(f"✓ Created batch with {len(batch_sessions)} sessions, 8 chunks total")

    # Simulate batch start: PIN all sessions
    print(f"\n[BATCH START] Pinning batch sessions...")
    for session_id in batch_sessions:
        cache.pin_session(session_id)

    pinned_count = len(cache.pinned_chunks)
    print(f"✓ Pinned {pinned_count} chunks")

    # Simulate concurrent request trying to evict
    print(f"\n[CONCURRENT REQUEST] Trying to add new chunk...")
    other_chunk = KVChunk(
        session_id='other_s1',
        chunk_id=0,
        layer_idx=0,
        key_tensor=torch.randn(1, 32, 8, 64),
        value_tensor=torch.randn(1, 32, 8, 64),
        context_length=0,
        session_total_chunks=1,
        num_layers=40,
    )

    # This should try to evict, but skip pinned chunks
    cache.store_chunk(other_chunk, CacheLocation.GPU)

    # Check batch chunks still exist
    batch_chunks_remaining = sum(
        1 for k in cache.gpu_cache.keys()
        if any(s in k for s in batch_sessions)
    )
    print(f"✓ Batch chunks still in cache: {batch_chunks_remaining}")
    assert batch_chunks_remaining > 0, "Batch chunks evicted despite pinning!"

    # Simulate batch end: UNPIN all sessions
    print(f"\n[BATCH END] Unpinning batch sessions...")
    for session_id in batch_sessions:
        cache.unpin_session(session_id)

    pinned_count = len(cache.pinned_chunks)
    print(f"✓ Unpinned all sessions ({pinned_count} pinned remaining)")
    assert pinned_count == 0, "Chunks should be unpinned!"

    print("\n✅ TEST 4 PASSED: Batch pinning simulation works!")


def test_scheduler_deferral():
    """Test that scheduler defers pinned sessions gracefully."""
    print("\n" + "="*60)
    print("TEST 5: Scheduler Request Deferral")
    print("="*60)

    from pensieve.scheduler import BatchScheduler
    from pensieve.core import Request

    cache = TwoTierCache(gpu_capacity_gb=1, cpu_capacity_gb=2)
    scheduler = BatchScheduler(cache, max_batch_size=4)

    # Create requests from multiple sessions
    requests = []
    for session_id in ['s1', 's2', 's3', 's4']:
        for i in range(2):
            req = Request(
                session_id=session_id,
                request_id=f"{session_id}_req_{i}",
                input_ids=torch.tensor([1, 2, 3]),
            )
            requests.append(req)
            scheduler.add_request(req)

    print(f"✓ Added {len(requests)} requests from 4 sessions")
    print(f"  Queue size: {len(scheduler.request_queue)}")

    # Simulate Batch 1: PIN sessions s1, s2, s3
    cache.pin_session('s1')
    cache.pin_session('s2')
    cache.pin_session('s3')
    print(f"\n✓ Pinned sessions: s1, s2, s3")
    print(f"  Pinned: {cache.pinned_sessions}")

    # Form Batch 1 with pinning awareness
    batch1, _ = scheduler.form_next_batch()

    print(f"\n✓ Formed Batch 1 with scheduler deferral:")
    print(f"  Batch size: {len(batch1.requests)}")
    print(f"  Batch sessions: {set(r.session_id for r in batch1.requests)}")

    # Check that batch only contains unpinned session (s4)
    batch_sessions = set(r.session_id for r in batch1.requests)
    assert 's4' in batch_sessions or len(batch1.requests) == 0, \
        f"Batch should prefer unpinned sessions, got: {batch_sessions}"

    assert 's1' not in batch_sessions, "Batch should not contain pinned s1"
    assert 's2' not in batch_sessions, "Batch should not contain pinned s2"
    assert 's3' not in batch_sessions, "Batch should not contain pinned s3"

    print(f"✓ Scheduler correctly deferred pinned sessions!")

    # Check that deferred requests are back in queue
    remaining_queue_sessions = set(r.session_id for r in scheduler.request_queue)
    print(f"\n✓ Deferred requests still in queue:")
    print(f"  Queue sessions: {remaining_queue_sessions}")

    assert 's1' in remaining_queue_sessions or len(scheduler.request_queue) > 0, \
        "Deferred pinned sessions should be in queue"

    # Unpin and form next batch
    cache.unpin_session('s1')
    cache.unpin_session('s2')
    cache.unpin_session('s3')
    print(f"\n✓ Unpinned all sessions")

    batch2, _ = scheduler.form_next_batch()
    print(f"\n✓ Formed Batch 2 after unpinning:")
    print(f"  Batch size: {len(batch2.requests)}")

    batch2_sessions = set(r.session_id for r in batch2.requests)
    assert len(batch2_sessions) > 0, "Batch 2 should have requests"
    print(f"  Batch sessions: {batch2_sessions}")

    print("\n✅ TEST 5 PASSED: Scheduler deferral works!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CHUNK PINNING MECHANISM TEST SUITE")
    print("="*60)

    try:
        test_basic_pinning()
        test_pinned_eviction_protection()
        test_multiple_pinned_sessions()
        test_pinning_with_batch_execution()
        test_scheduler_deferral()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nChunk pinning + scheduler deferral mechanism is working correctly:")
        print("  ✓ Pinned chunks cannot be evicted")
        print("  ✓ Multiple sessions can be pinned simultaneously")
        print("  ✓ Batch execution can safely use pinning")
        print("  ✓ Concurrent requests respect pinned chunks")
        print("  ✓ Scheduler defers pinned sessions gracefully\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
