# Pensieve: Setup and Execution Guide

## Environment Setup (Run Once)

```bash
# Navigate to project directory
cd /Users/sj/pensieve

# Install dependencies
pip install torch transformers datasets numpy tqdm

# Download the model (one-time, will take a few minutes)
# The model will be cached in ~/.cache/huggingface/hub/
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B'); AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')"
```

## Running Benchmarks

### 1. Basic Sequential Demo (Quick Test)
```bash
python main.py --mode pensieve --model gpt2 --gpu-cache 2 --cpu-cache 4
```

### 2. Sequential Comparison (Pensieve vs vLLM)
```bash
python main.py --mode compare --model gpt2 --gpu-cache 2 --cpu-cache 4
```

### 3. Concurrent Benchmark with 6 Users (Recommended - Tests Cache Hotness)
```bash
# This creates different client access patterns to show cache eviction policy
# 2 HOT clients (frequent, 0.15s), 2 WARM (normal, 0.5s), 2 COLD (infrequent, 1.25s)
python main.py --mode compare --model gpt2 --num-concurrent-users 6 --request-interval 0.5 --gpu-cache 4 --cpu-cache 8
```

### 4. Concurrent Benchmark with Meta-Llama-3-8B (Full Model)
```bash
python main.py --mode compare --model meta-llama/Meta-Llama-3-8B --num-concurrent-users 6 --request-interval 0.5 --gpu-cache 40 --cpu-cache 100
```

### 5. Stress Test with Many Concurrent Users
```bash
# 9 users = 3 HOT, 3 WARM, 3 COLD
# Creates high cache contention to stress eviction policy
python main.py --mode compare --model gpt2 --num-concurrent-users 9 --request-interval 0.3 --gpu-cache 4 --cpu-cache 8
```

### 5. Interactive Mode
```bash
python main.py --mode pensieve --model gpt2 --interactive
```

## Command Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `pensieve` | `pensieve` (stateful), `vllm` (stateless), or `compare` (both) |
| `--model` | `meta-llama/Meta-Llama-3-8B` | HuggingFace model name |
| `--gpu-cache` | 40 | GPU cache size in GB |
| `--cpu-cache` | 100 | CPU cache size in GB |
| `--num-concurrent-users` | 1 | Number of concurrent users (>1 enables concurrent benchmark) |
| `--request-interval` | 0.5 | Seconds between requests from each user |
| `--max-new-tokens` | 32 | Max tokens to generate per turn |
| `--device` | `cuda:0` | GPU device to use |

## Concurrent Benchmark Overview

When running with `--num-concurrent-users > 1`:

1. **Multiple Client Threads**: Launches N concurrent client threads, each simulating an independent user
2. **Automatic Hotness Distribution** (NEW):
   - **HOT clients** (1/3): Frequent requests (0.3× request-interval) → Stay in GPU cache
   - **WARM clients** (1/3): Normal requests (1.0× request-interval) → GPU/CPU boundary
   - **COLD clients** (1/3): Infrequent requests (2.5× request-interval) → Trigger eviction
3. **Cache-Aware Scheduling**: BatchScheduler forms unified batches with proper cache management
4. **Eviction Policy Showcase**: Retention value policy intelligently evicts COLD clients to make room for HOT clients
5. **Aggregate Metrics**: Collects system-wide metrics:
   - **Total Time**: Wall-clock time for entire benchmark
   - **Throughput**: Requests per second (higher with cache reuse)
   - **TTFT**: Time to first token (lower for HOT clients due to cache)
   - **Tail Latency**: Response latency (shows recovery costs for COLD clients)

## Expected Performance

### Sequential Benchmark (Single User)
- Limited batching opportunity
- Primarily shows cache reuse speedup within single session
- **Expected**: 1.2-1.5× speedup by turn 5+

### Concurrent Benchmark (6+ Users with Hotness Distribution)
- Multiple sessions with different access patterns
- BatchScheduler forms unified batches mixing PREFILL + GENERATION
- Cache hotness creates realistic eviction scenarios
- **Expected Improvements**:
  - **Throughput**: 1.5-2.5× faster than vLLM
  - **HOT clients TTFT**: 1.8-2.5× faster (high cache hit rate)
  - **WARM clients TTFT**: 1.3-1.8× faster (mixed cache hits)
  - **COLD clients TTFT**: 1.1-1.5× faster (eviction overhead + recovery cost)
  - **Overall**: Speedup increases as more clients' history is cached

### Why Hotness Matters
- **Without hotness**: All clients treated equally → speedup may be modest
- **With hotness**: Hot clients get GPU space → cold clients evicted → recovery overhead visible
- **Paper's point**: Even with recovery costs, caching leading turns saves more than swap cost
- **Shows Retention Value**: System intelligently evicts infrequent clients to serve frequent ones

## Troubleshooting

### Out of Memory (OOM)
Reduce cache sizes:
```bash
python main.py --mode compare --model gpt2 --num-concurrent-users 4 --gpu-cache 4 --cpu-cache 8
```

### Slow Model Download
The first run downloads the model. Subsequent runs use cached version in `~/.cache/huggingface/hub/`

### CUDA Errors
Ensure:
1. `torch.cuda.is_available()` returns True
2. GPU has ≥ 8GB memory
3. No other processes using GPU (check `nvidia-smi`)

## Key Files

- `main.py` - Entry point with CLI and benchmarking functions
- `src/pensieve/server/server.py` - Main server implementation
- `src/pensieve/scheduler/batch_scheduler.py` - Request batching logic
- `src/pensieve/worker/worker.py` - GPU execution
- `src/pensieve/core/cache.py` - Two-tier KV cache management

## Citation

```bibtex
@inproceedings{yu2025pensieve,
  title={Stateful Large Language Model Serving with Pensieve},
  author={Yu, Lingfan and Lin, Jinkun and Li, Jinyang},
  booktitle={Proceedings of the Twentieth European Conference on Computer Systems},
  pages={144--158},
  year={2025},
  organization={ACM}
}
```
