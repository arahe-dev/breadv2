#!/bin/bash

# GEMMA 4 SPECULATIVE DECODING BENCHMARK (using ollama API)
# Properly measures tokens via API responses, not CLI output

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}GEMMA 4 SPECULATIVE DECODING BENCHMARK (API-BASED)${NC}"
echo -e "${BLUE}Target: gemma4:26b (26B MoE, 4B active)${NC}"
echo -e "${BLUE}Draft:  gemma4:e2b (Effective 2B)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

BENCHMARK_PROMPT="You are a senior software engineer reviewing buggy code. Identify all bugs, root causes, fixes with examples, and performance implications in this binary search implementation:\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1"

TARGET_MODEL="gemma4:26b"
DRAFT_MODEL="gemma4:e2b"
NUM_TOKENS=150
NUM_RUNS=2

echo -e "\n${YELLOW}BENCHMARK PARAMETERS:${NC}"
echo "  Token limit per run: $NUM_TOKENS"
echo "  Number of runs: $NUM_RUNS"
echo "  API endpoint: http://localhost:11434/api/generate"

# Test if ollama API is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Ollama API not responding at http://localhost:11434${NC}"
    echo "Make sure ollama is running (ollama serve)"
    exit 1
fi

echo -e "\n${BLUE}─────────────────────────────────────────────────────────────${NC}"
echo -e "${BLUE}PHASE 1: TARGET MODEL (gemma4:26b)${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

target_total_ms=0
target_total_tokens=0

for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n${YELLOW}Run $run/$NUM_RUNS${NC}"

    start_time=$(date +%s%N)

    # Call ollama API with explicit token limit
    response=$(curl -s http://localhost:11434/api/generate \
        -d "{
            \"model\": \"${TARGET_MODEL}\",
            \"prompt\": \"${BENCHMARK_PROMPT}\",
            \"num_predict\": ${NUM_TOKENS},
            \"stream\": false
        }")

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Extract actual token count from response
    token_count=$(echo "$response" | grep -o '"eval_count":[0-9]*' | head -1 | grep -o '[0-9]*$' || echo "0")

    if [ -z "$token_count" ] || [ "$token_count" -eq 0 ]; then
        # Fallback: count tokens in response text
        text=$(echo "$response" | grep -o '"response":"[^"]*' | sed 's/"response":"//' | tr -cd '[:print:]' | wc -c)
        token_count=$((text / 4))
    fi

    target_total_ms=$((target_total_ms + elapsed_ms))
    target_total_tokens=$((target_total_tokens + token_count))

    elapsed_s=$((elapsed_ms / 1000))

    if [ $elapsed_ms -gt 0 ]; then
        throughput=$((token_count * 1000 / elapsed_ms))
    else
        throughput=0
    fi

    echo -e "  Time: ${GREEN}${elapsed_s}s${NC}"
    echo -e "  Tokens: ${GREEN}${token_count}${NC}"
    echo -e "  Throughput: ${GREEN}${throughput} tok/s${NC}"
done

target_avg_ms=$((target_total_ms / NUM_RUNS))
target_avg_tokens=$((target_total_tokens / NUM_RUNS))
target_throughput=$((target_total_tokens * 1000 / target_total_ms))

echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}TARGET MODEL RESULTS (gemma4:26b):${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "  Avg time: ${BLUE}$((target_avg_ms / 1000))s${NC}"
echo -e "  Avg tokens: ${BLUE}${target_avg_tokens}${NC}"
echo -e "  Throughput: ${BLUE}${target_throughput} tok/s${NC}"

echo -e "\n${BLUE}─────────────────────────────────────────────────────────────${NC}"
echo -e "${BLUE}PHASE 2: DRAFT MODEL (gemma4:e2b)${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

draft_total_ms=0
draft_total_tokens=0

for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n${YELLOW}Run $run/$NUM_RUNS${NC}"

    start_time=$(date +%s%N)

    response=$(curl -s http://localhost:11434/api/generate \
        -d "{
            \"model\": \"${DRAFT_MODEL}\",
            \"prompt\": \"${BENCHMARK_PROMPT}\",
            \"num_predict\": ${NUM_TOKENS},
            \"stream\": false
        }")

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    token_count=$(echo "$response" | grep -o '"eval_count":[0-9]*' | head -1 | grep -o '[0-9]*$' || echo "0")

    if [ -z "$token_count" ] || [ "$token_count" -eq 0 ]; then
        text=$(echo "$response" | grep -o '"response":"[^"]*' | sed 's/"response":"//' | tr -cd '[:print:]' | wc -c)
        token_count=$((text / 4))
    fi

    draft_total_ms=$((draft_total_ms + elapsed_ms))
    draft_total_tokens=$((draft_total_tokens + token_count))

    elapsed_s=$((elapsed_ms / 1000))

    if [ $elapsed_ms -gt 0 ]; then
        throughput=$((token_count * 1000 / elapsed_ms))
    else
        throughput=0
    fi

    echo -e "  Time: ${GREEN}${elapsed_s}s${NC}"
    echo -e "  Tokens: ${GREEN}${token_count}${NC}"
    echo -e "  Throughput: ${GREEN}${throughput} tok/s${NC}"
done

draft_avg_ms=$((draft_total_ms / NUM_RUNS))
draft_avg_tokens=$((draft_total_tokens / NUM_RUNS))
draft_throughput=$((draft_total_tokens * 1000 / draft_total_ms))

echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}DRAFT MODEL RESULTS (gemma4:e2b):${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "  Avg time: ${BLUE}$((draft_avg_ms / 1000))s${NC}"
echo -e "  Avg tokens: ${BLUE}${draft_avg_tokens}${NC}"
echo -e "  Throughput: ${BLUE}${draft_throughput} tok/s${NC}"

echo -e "\n${BLUE}─────────────────────────────────────────────────────────────${NC}"
echo -e "${BLUE}COMPARATIVE ANALYSIS${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

if [ $draft_throughput -gt 0 ] && [ $target_throughput -gt 0 ]; then
    if [ $draft_throughput -gt $target_throughput ]; then
        speedup=$((draft_throughput / target_throughput))
        echo -e "  ${GREEN}✓ Draft is ${speedup}x FASTER${NC}"
        echo -e "    E2B: ${BLUE}${draft_throughput} tok/s${NC} vs 26B: ${BLUE}${target_throughput} tok/s${NC}"
    else
        ratio=$((target_throughput / draft_throughput))
        echo -e "  ${RED}✗ Draft is ${ratio}x SLOWER${NC}"
    fi
fi

echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}FINAL SUMMARY${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "Target (26B MoE): ${BLUE}${target_throughput} tok/s${NC}"
echo -e "Draft (E2B):     ${BLUE}${draft_throughput} tok/s${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
