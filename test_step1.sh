#!/bin/bash
prompt="<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Explain quantum mechanics in 100 words
<|im_end|>
<|im_start|>assistant
"

echo "=== STEP 1: Stream-Ordered Async Allocation ==="
echo "Running single-pass inference test (100 tokens)..."
echo ""

time (echo "$prompt" | ./bread.exe --tokens 100 2>&1 | tail -20)
