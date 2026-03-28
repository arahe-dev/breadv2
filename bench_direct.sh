#!/bin/bash

# Test 1: Simple math
echo "Test 1: Simple math"
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model": "qwen3.5:27b", "prompt": "What is 2+2?", "stream": false}' \
  -H "Content-Type: application/json" | grep -o '"eval_count":[0-9]*\|"eval_duration":[0-9]*'

echo ""
echo "Waiting 30s for next prompt..."
sleep 30

# Test 2: Code
echo "Test 2: Code generation"
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model": "qwen3.5:27b", "prompt": "Write a Python factorial function.", "stream": false}' \
  -H "Content-Type: application/json" | grep -o '"eval_count":[0-9]*\|"eval_duration":[0-9]*'

echo ""
echo "Waiting 30s for next prompt..."
sleep 30

# Test 3: Knowledge
echo "Test 3: Knowledge"
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model": "qwen3.5:27b", "prompt": "What is Paris?", "stream": false}' \
  -H "Content-Type: application/json" | grep -o '"eval_count":[0-9]*\|"eval_duration":[0-9]*'

