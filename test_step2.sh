#!/bin/bash
prompt="<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Explain quantum mechanics in 100 words
<|im_end|>
<|im_start|>assistant
"

echo "=== STEP 2: CPU Arena Allocator + Stream-Ordered GPU ==="
echo ""

time (echo "$prompt" | ./bread.exe --tokens 100 2>&1 | grep -A20 "Benchmark\|DECODE:" | tail -15)
