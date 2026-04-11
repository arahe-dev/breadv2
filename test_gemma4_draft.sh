#!/bin/bash
# Test gemma4:e2b as draft model for BREAD speculative decoding

BREAD_TOKENS=30
OLLAMA_API="http://localhost:11434/api/generate"

echo "======================================"
echo "BREAD vs Gemma4:e2b Acceptance Test"
echo "======================================"
echo ""

# Simple test: generate 30 tokens with both models
# Compare first N tokens to see acceptance rate

BREAD_PROMPT="<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
<think>

</think>

"

echo "Testing with prompt:"
echo "$BREAD_PROMPT"
echo ""

# Test 1: Generate with BREAD
echo "[TEST] Generating $BREAD_TOKENS tokens with BREAD..."
START_TIME=$(date +%s%N)

BREAD_OUTPUT=$(./bread.exe --prompt "$BREAD_PROMPT" --tokens $BREAD_TOKENS 2>&1 | grep -v "^\[" | head -20)

END_TIME=$(date +%s%N)
BREAD_TIME=$(( (END_TIME - START_TIME) / 1000000 ))

echo "$BREAD_OUTPUT" | head -5
echo "[BREAD] Time: ${BREAD_TIME}ms"
echo ""

# Test 2: Generate with Gemma4:e2b
echo "[TEST] Generating with Gemma4:e2b..."
START_TIME=$(date +%s%N)

GEMMA_RESPONSE=$(curl -s -X POST "$OLLAMA_API" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma4:e2b\",\"prompt\":\"$BREAD_PROMPT\",\"stream\":false,\"num_predict\":$BREAD_TOKENS}" \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1)

END_TIME=$(date +%s%N)
GEMMA_TIME=$(( (END_TIME - START_TIME) / 1000000 ))

echo "Gemma response: $GEMMA_RESPONSE" | head -3
echo "[GEMMA] Time: ${GEMMA_TIME}ms"
echo ""

# Summary
echo "======================================"
echo "RESULTS"
echo "======================================"
echo "BREAD generation time:  ${BREAD_TIME}ms"
echo "Gemma4 generation time: ${GEMMA_TIME}ms"
echo ""
echo "Note: Manual token-by-token comparison needed"
echo "Recommendation: Use Python test script for detailed acceptance rate"
