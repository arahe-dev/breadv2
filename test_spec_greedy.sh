#!/bin/bash
# Speculative decoding test with GREEDY sampling

OLLAMA="http://localhost:11434/api/generate"
TARGET="gemma4:26b"
DRAFT="gemma4:e2b"

echo "========================================"
echo "Speculative Decoding (GREEDY sampling)"
echo "========================================"
echo ""

# Test with greedy (temperature=0) for reproducible token selection
PROMPT="The capital of France is"

echo "Prompt: '$PROMPT'"
echo ""

# Get ONE token from target (greedy)
echo "[TARGET] Calling gemma4:26b with temperature=0..."
TARGET_TOKEN=$(curl -s -X POST "$OLLAMA" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$TARGET\",\"prompt\":\"$PROMPT\",\"stream\":false,\"temperature\":0,\"num_predict\":1}" \
  2>/dev/null | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/' | sed 's/[^a-zA-Z0-9 ]//g' | xargs)

echo "Target (26b) token: '$TARGET_TOKEN'"
echo ""

# Get ONE token from draft (greedy)
echo "[DRAFT] Calling gemma4:e2b with temperature=0..."
DRAFT_TOKEN=$(curl -s -X POST "$OLLAMA" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$DRAFT\",\"prompt\":\"$PROMPT\",\"stream\":false,\"temperature\":0,\"num_predict\":1}" \
  2>/dev/null | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/' | sed 's/[^a-zA-Z0-9 ]//g' | xargs)

echo "Draft (e2b) token: '$DRAFT_TOKEN'"
echo ""

# Compare
if [ "$TARGET_TOKEN" = "$DRAFT_TOKEN" ]; then
    echo "✓✓✓ TOKENS MATCH! ✓✓✓"
    echo "Both greedy sampling agrees: '$TARGET_TOKEN'"
    echo ""
    echo "This is EXCELLENT for speculative decoding!"
else
    echo "✗ Tokens differ:"
    echo "  Target: '$TARGET_TOKEN'"
    echo "  Draft:  '$DRAFT_TOKEN'"
    echo ""
    echo "Models diverge even with greedy sampling (temp=0)"
fi
