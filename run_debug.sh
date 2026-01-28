#!/bin/bash
# Quick script to run multi-turn debug test with debug output enabled

cd /Users/sj/pensieve

echo "======================================================================="
echo "Running Multi-Turn KV Cache Test with DEBUG output"
echo "======================================================================="
echo ""

# Run with PENSIEVE_DEBUG=1 to enable debug logging
PENSIEVE_DEBUG=1 python test_multiturn_debug.py

echo ""
echo "======================================================================="
echo "To run WITHOUT debug output, use:"
echo "  python test_multiturn_debug.py"
echo "======================================================================="
