#!/bin/bash
# Speculative decoding acceptance rate test
# Measures how often gemma4:e2b (draft) matches gemma4:26b (target)

OLLAMA="http://localhost:11434/api/generate"
TARGET="gemma4:26b"
DRAFT="gemma4:e2b"

# Configuration
NUM_TOKENS=15
# FACTUAL/DETERMINISTIC prompts only for spec decoding
# (Where both models should give same answer, not creative outputs)
PROMPTS=(
    "What is the capital of France?"
    "What is 2 plus 2?"
    "Who wrote Romeo and Juliet?"
)

echo "========================================"
echo "SPECULATIVE DECODING ACCEPTANCE TEST"
echo "========================================"
echo "Target: $TARGET (17 GB, large)"
echo "Draft:  $DRAFT (7.2 GB, small)"
echo "Sampling: GREEDY (temperature=0)"
echo "Tokens to test: $NUM_TOKENS per prompt"
echo ""

total_accepted=0
total_rejected=0
test_count=0

for prompt in "${PROMPTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Prompt: $prompt"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    context="$prompt"
    accepted=0
    rejected=0

    for i in $(seq 1 $NUM_TOKENS); do
        # Get target token (no timeout)
        target=$(timeout 600 curl -s -X POST "$OLLAMA" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$TARGET\",\"prompt\":\"$context\",\"stream\":false,\"temperature\":0,\"num_predict\":1}" \
            2>/dev/null | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # Get draft token (no timeout)
        draft=$(timeout 600 curl -s -X POST "$OLLAMA" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$DRAFT\",\"prompt\":\"$context\",\"stream\":false,\"temperature\":0,\"num_predict\":1}" \
            2>/dev/null | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # Compare
        if [ "$target" = "$draft" ]; then
            match="✓"
            ((accepted++))
        else
            match="✗"
            ((rejected++))
        fi

        echo "[$i] Target='$target' | Draft='$draft' | $match"

        # Always use target token
        context="$context $target"
    done

    total=$((accepted + rejected))
    if [ $total -gt 0 ]; then
        rate=$(echo "scale=1; $accepted * 100 / $total" | bc)
        echo ""
        echo "  Acceptance rate: $rate% ($accepted/$total)"
    fi

    total_accepted=$((total_accepted + accepted))
    total_rejected=$((total_rejected + rejected))
    test_count=$((test_count + total))

    echo ""
done

# Summary
echo "========================================"
echo "FINAL RESULTS"
echo "========================================"
if [ $test_count -gt 0 ]; then
    final_rate=$(echo "scale=1; $total_accepted * 100 / $test_count" | bc)
    echo "Overall acceptance rate: $final_rate% ($total_accepted/$test_count)"
    echo ""

    if (( $(echo "$final_rate > 80" | bc -l) )); then
        echo "✓✓✓ EXCELLENT!"
        echo "Speculative decoding with Gemma4 models is HIGHLY VIABLE"
        echo "Draft model (e2b) closely matches target (26b)"
        echo "Expected speedup: 2-3x with optimal implementation"
    elif (( $(echo "$final_rate > 50" | bc -l) )); then
        echo "✓✓ GOOD"
        echo "Reasonable acceptance for speculative decoding"
        echo "Expected speedup: 1.5-2x"
    elif (( $(echo "$final_rate > 20" | bc -l) )); then
        echo "✓ MODERATE"
        echo "Speculative decoding possible but marginal benefit"
    else
        echo "✗ POOR"
        echo "Models too divergent for practical speculation"
    fi
else
    echo "ERROR: No tokens generated"
fi
