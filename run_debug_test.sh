#!/bin/bash

# Clear previous debug log
rm -f /tmp/pensieve_cache_debug.log

echo "🔍 Running test with file-based debug logging..."
echo "Debug output will be written to: /tmp/pensieve_cache_debug.log"
echo ""

cd /home/elicer/Pensieve

# Run the test
PENSIEVE_DEBUG=1 python main.py \
  --dataset sharegt \
  --num-concurrent-users 1 \
  --model google/gemma-2-27b-it \
  --gpu-cache 64 \
  --cpu-cache 128 \
  --max_turns 5 \
  --min_turns 5 \
  --max-new-tokens 32 2>&1 | head -200

echo ""
echo "========================================="
echo "📋 DEBUG LOG OUTPUT:"
echo "========================================="
if [ -f /tmp/pensieve_cache_debug.log ]; then
    cat /tmp/pensieve_cache_debug.log
    echo ""
    echo "✅ Full debug log saved to: /tmp/pensieve_cache_debug.log"
else
    echo "❌ Debug log not found! __getitem__() may not have been called."
fi
