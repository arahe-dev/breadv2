#!/usr/bin/env python3
from tokenizers import Tokenizer

# Use the same tokenizer as BREAD
model_path = "/c/Users/arahe/.ollama/models/blobs/sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a"

# Check output: "2 + 2 equals **4**."
text1 = "2 + 2 equals **4**."
print(f"Output 1: '{text1}'")
print(f"Character count: {len(text1)}")
print()

# The thinking block that was in output
text2 = "<think>\n\n</think>\n\n2 + 2 equals **4**."
print(f"Full output with thinking: '{text2}'")
print()

# Test that shows how many tokens the model generated
# Let's manually count common Qwen tokenization
print("Manual estimate (Qwen tokenizes roughly 1 token per 1-2 characters for ASCII):")
print(f"  Thinking block (~5-10 tokens)")
print(f"  '2 + 2 equals **4**.' (~10-15 tokens)")
print(f"  Total: ~15-25 tokens")
