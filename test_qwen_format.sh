#!/bin/bash

cd /c/bread_v2

echo "=== Test with Qwen3.5 structured format, 200 tokens ==="
(
  cat << 'PROMPT'
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
<think>

</think>

PROMPT
  echo ""
) | timeout 120 ./bread.exe --server --tokens 200 2>&1 | grep -v "^loader:" | grep -v "^gguf:" | grep -v "^weight_cache:" | grep -v "^tokenizer:" | tail -30

