#!/bin/bash
# Simple spec decoding test - factual questions only
# Tests where models SHOULD agree (same correct answer)

OLLAMA="http://localhost:11434/api/generate"
TARGET="gemma4:26b"
DRAFT="gemma4:e2b"

echo "========================================"
echo "SPEC DECODING TEST - FACTUAL QUERIES"
echo "========================================"
echo ""

# Test 1: Capital question
echo "TEST 1: What is the capital of France?"
echo "─────────────────────────────────────"

TARGET_1=$(curl -s -X POST "$OLLAMA" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:26b","prompt":"What is the capital of France?","stream":false,"temperature":0,"num_predict":1}' \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

DRAFT_1=$(curl -s -X POST "$OLLAMA" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e2b","prompt":"What is the capital of France?","stream":false,"temperature":0,"num_predict":1}' \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

echo "Target: $TARGET_1"
echo "Draft:  $DRAFT_1"

if [ "$TARGET_1" = "$DRAFT_1" ]; then
  echo "Result: ✓ MATCH"
  SCORE1=100
else
  echo "Result: ✗ DIFFERENT"
  SCORE1=0
fi
echo ""

# Test 2: Math question
echo "TEST 2: What is 2 plus 2?"
echo "─────────────────────────"

TARGET_2=$(curl -s -X POST "$OLLAMA" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:26b","prompt":"What is 2 plus 2?","stream":false,"temperature":0,"num_predict":1}' \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

DRAFT_2=$(curl -s -X POST "$OLLAMA" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e2b","prompt":"What is 2 plus 2?","stream":false,"temperature":0,"num_predict":1}' \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

echo "Target: $TARGET_2"
echo "Draft:  $DRAFT_2"

if [ "$TARGET_2" = "$DRAFT_2" ]; then
  echo "Result: ✓ MATCH"
  SCORE2=100
else
  echo "Result: ✗ DIFFERENT"
  SCORE2=0
fi
echo ""

# Test 3: Author question
echo "TEST 3: Who wrote Romeo and Juliet?"
echo "────────────────────────────────────"

TARGET_3=$(curl -s -X POST "$OLLAMA" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:26b","prompt":"Who wrote Romeo and Juliet?","stream":false,"temperature":0,"num_predict":1}' \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

DRAFT_3=$(curl -s -X POST "$OLLAMA" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e2b","prompt":"Who wrote Romeo and Juliet?","stream":false,"temperature":0,"num_predict":1}' \
  2>/dev/null | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

echo "Target: $TARGET_3"
echo "Draft:  $DRAFT_3"

if [ "$TARGET_3" = "$DRAFT_3" ]; then
  echo "Result: ✓ MATCH"
  SCORE3=100
else
  echo "Result: ✗ DIFFERENT"
  SCORE3=0
fi
echo ""

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
AVG=$((($SCORE1 + $SCORE2 + $SCORE3) / 3))
echo "Test 1 (Capital):        $SCORE1%"
echo "Test 2 (Math):           $SCORE2%"
echo "Test 3 (Author):         $SCORE3%"
echo ""
echo "Average acceptance rate: $AVG%"
echo ""

if [ $AVG -ge 80 ]; then
  echo "✓✓✓ EXCELLENT for speculative decoding"
elif [ $AVG -ge 50 ]; then
  echo "✓ GOOD alignment for speculation"
elif [ $AVG -gt 0 ]; then
  echo "◐ MODERATE - partial alignment"
else
  echo "✗ POOR - models diverge completely"
fi
