#!/bin/bash
# Quick speculative decoding test: Gemma4:e2b vs Gemma4:26b

OLLAMA="http://localhost:11434/api/generate"
TARGET="gemma4:26b"
DRAFT="gemma4:e2b"

echo "========================================"
echo "Speculative Decoding: Gemma4 Same-Family"
echo "========================================"
echo "Target: $TARGET (17 GB)"
echo "Draft:  $DRAFT (7.2 GB)"
echo ""

# Test 1: Simple prompt
PROMPT="What is Paris?"
echo "Test 1: $PROMPT"
echo ""

# Get target's response (ground truth)
echo "[1] Getting target model (26b) response..."
TARGET_RESP=$(curl -s -X POST "$OLLAMA" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$TARGET\",\"prompt\":\"$PROMPT\",\"stream\":false,\"num_predict\":10}" \
  2>/dev/null | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/')

echo "Target response: $TARGET_RESP" | head -1
echo ""

# Get draft's response
echo "[2] Getting draft model (e2b) response..."
DRAFT_RESP=$(curl -s -X POST "$OLLAMA" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$DRAFT\",\"prompt\":\"$PROMPT\",\"stream\":false,\"num_predict\":10}" \
  2>/dev/null | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/')

echo "Draft response:  $DRAFT_RESP" | head -1
echo ""

# Compare first few words
echo "[3] Token-level comparison (first 5 words):"
TARGET_WORDS=$(echo "$TARGET_RESP" | head -c 200)
DRAFT_WORDS=$(echo "$DRAFT_RESP" | head -c 200)

echo "Target words: $TARGET_WORDS"
echo "Draft words:  $DRAFT_WORDS"
echo ""

# Check overlap
TARGET_FIRST=$(echo "$TARGET_RESP" | awk '{print $1}')
DRAFT_FIRST=$(echo "$DRAFT_RESP" | awk '{print $1}')

if [ "$TARGET_FIRST" = "$DRAFT_FIRST" ]; then
    echo "✓ First token MATCHES: '$TARGET_FIRST'"
else
    echo "✗ First token DIFFERS: Target='$TARGET_FIRST' vs Draft='$DRAFT_FIRST'"
fi

TARGET_SECOND=$(echo "$TARGET_RESP" | awk '{print $2}')
DRAFT_SECOND=$(echo "$DRAFT_RESP" | awk '{print $2}')

if [ "$TARGET_SECOND" = "$DRAFT_SECOND" ]; then
    echo "✓ Second token MATCHES: '$TARGET_SECOND'"
else
    echo "✗ Second token DIFFERS: Target='$TARGET_SECOND' vs Draft='$DRAFT_SECOND'"
fi

echo ""
echo "INTERPRETATION:"
echo "Both Gemma4 models, so expect high token alignment"
echo "This would enable speculative decoding at high acceptance rate"
