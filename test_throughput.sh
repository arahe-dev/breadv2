#!/bin/bash
# Single inference to measure pure generation throughput

prompt="<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Explain quantum mechanics in 100 words
<|im_end|>
<|im_start|>assistant
"

echo "=== Single-pass inference test (100 tokens) ==="
echo "Starting timer..."

time (echo "$prompt" | ./bread.exe --tokens 100)
