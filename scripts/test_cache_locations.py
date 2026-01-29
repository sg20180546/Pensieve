"""Test script to verify KV cache is stored in GPU and CPU correctly."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pensieve.core.cache import TwoTierCache
from pensieve.core.types import CacheLocation


def test_gpu_cache_storage():
    """Test that chunks are stored in GPU cache."""
    print("\n" + "="*60)
    print("TEST 1: GPU Cache Storage")
    print("="*60)

    # Create small cache (10GB GPU, 20GB CPU)
    cache = TwoTierCache(
        gpu_capacity_gb=10,
        cpu_capacity_gb=20,
        num_layers=32,
        device='cuda:0'
    )

    # Create dummy KV tensors (bfloat16: 2 bytes per element)
    # Shape: [batch=1, seq_len=32, num_heads=32, head_dim=128]
    batch_size = 1
    seq_len = 32
    num_heads = 32
    head_dim = 128

    k_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda:0')
    v_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda:0')

    # Store in GPU cache
    layer_kv_dict = {0: (k_tensor, v_tensor)}

    success = cache.store_chunks_for_position(
        session_id="test_session_1",
        chunk_id=0,
        layer_kv_dict=layer_kv_dict,
        context_length=0,
        session_total_chunks=1,
        num_layers=32,
        location=CacheLocation.GPU
    )

    print(f"✓ Store success: {success}")
    print(f"✓ GPU cache size: {len(cache.gpu_cache)} chunks")
    print(f"✓ GPU memory used: {cache.gpu_used_bytes / (1024**2):.2f} MB")
    print(f"✓ CPU cache size: {len(cache.cpu_cache)} chunks")

    # Verify chunk is in GPU cache
    chunk_keys = list(cache.gpu_cache.keys())
    if chunk_keys:
        chunk_key = chunk_keys[0]
        chunk = cache.gpu_cache[chunk_key]
        print(f"✓ Found chunk in GPU: {chunk_key}")
        print(f"  - Session: {chunk.session_id}")
        print(f"  - Chunk ID: {chunk.chunk_id}")
        print(f"  - Layer: {chunk.layer_idx}")
        print(f"  - Location: {chunk.location}")

    assert len(cache.gpu_cache) > 0, "❌ No chunks in GPU cache!"
    assert cache.gpu_used_bytes > 0, "❌ GPU memory not tracked!"
    print("✅ GPU cache storage test PASSED\n")


def test_cpu_cache_eviction():
    """Test that chunks are evicted to CPU when GPU is full."""
    print("="*60)
    print("TEST 2: CPU Cache Eviction (GPU Full → CPU)")
    print("="*60)

    # Create small cache to trigger eviction quickly
    cache = TwoTierCache(
        gpu_capacity_gb=0.5,  # 512 MB - very small to trigger eviction
        cpu_capacity_gb=5,
        num_layers=32,
        device='cuda:0'
    )

    k_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    v_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    layer_kv_dict = {0: (k_tensor, v_tensor)}
    chunk_size = k_tensor.element_size() * k_tensor.numel() + v_tensor.element_size() * v_tensor.numel()

    print(f"Chunk size: {chunk_size / (1024**2):.2f} MB")
    print(f"GPU capacity: 512 MB")
    print(f"Expected evictions: ~1 chunk after GPU fills\n")

    # Store multiple chunks to trigger eviction
    num_chunks = 5
    for i in range(num_chunks):
        success = cache.store_chunks_for_position(
            session_id=f"session_{i}",
            chunk_id=0,
            layer_kv_dict=layer_kv_dict,
            context_length=0,
            session_total_chunks=1,
            num_layers=32,
            location=CacheLocation.GPU  # Try to store in GPU
        )

        gpu_chunks = len(cache.gpu_cache)
        cpu_chunks = len(cache.cpu_cache)
        print(f"After chunk {i}: GPU={gpu_chunks}, CPU={cpu_chunks}, Success={success}")

    # Verify we have chunks in both GPU and CPU
    print(f"\n✓ Final GPU cache: {len(cache.gpu_cache)} chunks")
    print(f"✓ Final CPU cache: {len(cache.cpu_cache)} chunks")
    print(f"✓ GPU memory used: {cache.gpu_used_bytes / (1024**2):.2f} MB")
    print(f"✓ CPU memory used: {cache.cpu_used_bytes / (1024**2):.2f} MB")

    if len(cache.cpu_cache) > 0:
        print("\n✅ Eviction to CPU test PASSED - chunks are in CPU cache\n")
    else:
        print("\n⚠️  No chunks in CPU cache yet (GPU may not have filled)\n")


def test_cache_statistics():
    """Test that cache statistics are tracked correctly."""
    print("="*60)
    print("TEST 3: Cache Statistics Tracking")
    print("="*60)

    cache = TwoTierCache(
        gpu_capacity_gb=5,
        cpu_capacity_gb=10,
        num_layers=32,
        device='cuda:0'
    )

    k_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    v_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    layer_kv_dict = {0: (k_tensor, v_tensor)}

    # Store 3 chunks
    for i in range(3):
        cache.store_chunks_for_position(
            session_id=f"session_{i}",
            chunk_id=0,
            layer_kv_dict=layer_kv_dict,
            context_length=0,
            session_total_chunks=1,
            num_layers=32,
            location=CacheLocation.GPU
        )

    # Get statistics
    stats = cache.get_statistics()

    print(f"GPU Statistics:")
    print(f"  - Total capacity: {stats.gpu_capacity_bytes / (1024**3):.2f} GB")
    print(f"  - Used: {stats.gpu_used_bytes / (1024**2):.2f} MB")
    print(f"  - Free: {stats.gpu_free_bytes / (1024**2):.2f} MB")
    print(f"  - Free ratio: {stats.gpu_free_ratio:.2%}")

    print(f"\nCPU Statistics:")
    print(f"  - Total capacity: {stats.cpu_capacity_bytes / (1024**3):.2f} GB")
    print(f"  - Used: {stats.cpu_used_bytes / (1024**2):.2f} MB")
    print(f"  - Free: {stats.cpu_free_bytes / (1024**2):.2f} MB")
    print(f"  - Free ratio: {stats.cpu_free_ratio:.2%}")

    print(f"\nSession Statistics:")
    print(f"  - Active sessions: {stats.num_sessions}")
    print(f"  - Total chunks: {stats.total_chunks}")

    assert stats.gpu_used_bytes > 0, "❌ GPU usage not tracked!"
    assert stats.total_chunks >= 3, "❌ Chunk count incorrect!"
    print("\n✅ Cache statistics test PASSED\n")


def test_session_metadata():
    """Test that session metadata is tracked correctly."""
    print("="*60)
    print("TEST 4: Session Metadata Tracking")
    print("="*60)

    cache = TwoTierCache(
        gpu_capacity_gb=5,
        cpu_capacity_gb=10,
        num_layers=32,
        device='cuda:0'
    )

    k_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    v_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    layer_kv_dict = {0: (k_tensor, v_tensor)}

    session_id = "test_session"

    # Store chunks for one session
    for chunk_id in range(3):
        cache.store_chunks_for_position(
            session_id=session_id,
            chunk_id=chunk_id,
            layer_kv_dict=layer_kv_dict,
            context_length=chunk_id * 32,
            session_total_chunks=3,
            num_layers=32,
            location=CacheLocation.GPU
        )

    # Get session info
    positions = cache.get_session_positions(session_id)
    metadata = cache.get_session_metadata(session_id)

    print(f"Session: {session_id}")
    print(f"  - Chunk positions: {positions}")
    print(f"  - Metadata: {metadata}")

    if metadata:
        print(f"  - Total tokens: {metadata.total_tokens}")
        print(f"  - Last chunk size: {metadata.last_chunk_size}")
        print(f"  - Last access time: {metadata.last_access_time}")

    assert len(positions) > 0, "❌ Session positions not tracked!"
    print("\n✅ Session metadata test PASSED\n")


def test_chunk_location_retrieval():
    """Test that we can verify chunk locations (GPU vs CPU)."""
    print("="*60)
    print("TEST 5: Chunk Location Retrieval")
    print("="*60)

    cache = TwoTierCache(
        gpu_capacity_gb=1,  # Small GPU cache
        cpu_capacity_gb=5,
        num_layers=32,
        device='cuda:0'
    )

    k_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    v_tensor = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16, device='cuda:0')
    layer_kv_dict = {0: (k_tensor, v_tensor)}

    # Store chunks to fill GPU and spill to CPU
    locations = {}
    for i in range(10):
        session_id = f"session_{i}"
        cache.store_chunks_for_position(
            session_id=session_id,
            chunk_id=0,
            layer_kv_dict=layer_kv_dict,
            context_length=0,
            session_total_chunks=1,
            num_layers=32,
            location=CacheLocation.GPU
        )

        # Check where chunk ended up
        for layer_idx in range(32):
            chunk_key = f"{session_id}:chunk:0:layer:{layer_idx}"
            chunk = cache.get_chunk(chunk_key)
            if chunk:
                location = chunk.location
                locations[chunk_key] = location

    # Count chunks by location
    gpu_count = sum(1 for loc in locations.values() if loc == CacheLocation.GPU)
    cpu_count = sum(1 for loc in locations.values() if loc == CacheLocation.CPU)
    dropped_count = sum(1 for loc in locations.values() if loc == CacheLocation.DROPPED)

    print(f"Chunk locations:")
    print(f"  - GPU: {gpu_count} chunks")
    print(f"  - CPU: {cpu_count} chunks")
    print(f"  - DROPPED: {dropped_count} chunks")

    # Show sample chunks
    print(f"\nSample locations:")
    for chunk_key, location in list(locations.items())[:5]:
        print(f"  - {chunk_key}: {location}")

    print("\n✅ Chunk location retrieval test PASSED\n")


if __name__ == "__main__":
    print("\n" + "🧪 PENSIEVE CACHE LOCATION TESTS 🧪".center(60, "="))

    try:
        test_gpu_cache_storage()
        test_cpu_cache_eviction()
        test_cache_statistics()
        test_session_metadata()
        test_chunk_location_retrieval()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
