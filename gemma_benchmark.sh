#!/bin/bash

# GEMMA 4 SPECULATIVE DECODING BENCHMARK
# Target: gemma4:26b (17GB)
# Draft: gemma4:e2b (7.2GB)
# Goal: Complex multi-step reasoning with long timeout

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}GEMMA 4 SPECULATIVE DECODING BENCHMARK${NC}"
echo -e "${BLUE}Target: gemma4:26b (26B MoE, 4B active)${NC}"
echo -e "${BLUE}Draft:  gemma4:e2b (Effective 2B)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Complex multi-step reasoning prompt
read -r -d '' BENCHMARK_PROMPT << 'EOF' || true
You are a senior software engineer reviewing buggy code. Analyze this code thoroughly:

```python
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
```

Identify: (1) All bugs, (2) Root causes, (3) Fixes with examples, (4) Performance implications, (5) Test cases.
EOF

# Benchmark parameters
TARGET_MODEL="gemma4:26b"
DRAFT_MODEL="gemma4:e2b"
NUM_TOKENS=150
NUM_RUNS=2
TIMEOUT_SECONDS=300

echo -e "\n${YELLOW}BENCHMARK PARAMETERS:${NC}"
echo "  Target tokens: $NUM_TOKENS"
echo "  Number of runs: $NUM_RUNS"
echo "  Per-run timeout: ${TIMEOUT_SECONDS}s"

echo -e "\n${BLUE}─────────────────────────────────────────────────────────────${NC}"
echo -e "${BLUE}PHASE 1: TARGET MODEL (gemma4:26b) BASELINE${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

target_total_ms=0
target_total_tokens=0

for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n${YELLOW}Run $run/$NUM_RUNS${NC}"

    start_time=$(date +%s%N)

    # Run target model
    output=$(timeout ${TIMEOUT_SECONDS} ollama run ${TARGET_MODEL} "${BENCHMARK_PROMPT}" 2>&1 || true)

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Estimate tokens (avg 4 chars per token)
    output_len=${#output}
    token_count=$((output_len / 4))

    target_total_ms=$((target_total_ms + elapsed_ms))
    target_total_tokens=$((target_total_tokens + token_count))

    elapsed_s=$((elapsed_ms / 1000))
    echo -e "  Time: ${GREEN}${elapsed_s}s${NC}"
    echo -e "  Output chars: ${GREEN}${output_len}${NC}"
    echo -e "  Est. tokens: ${GREEN}${token_count}${NC}"

    if [ $elapsed_ms -gt 0 ]; then
        throughput=$((token_count * 1000 / elapsed_ms))
        echo -e "  Throughput: ${GREEN}${throughput} tok/s${NC}"
    fi

    echo -e "  Preview:"
    echo "$output" | head -c 250
    echo -e "\n"
done

# Target model averages
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
echo -e "${BLUE}PHASE 2: DRAFT MODEL (gemma4:e2b) BASELINE${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

draft_total_ms=0
draft_total_tokens=0

for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n${YELLOW}Run $run/$NUM_RUNS${NC}"

    start_time=$(date +%s%N)

    # Run draft model
    output=$(timeout ${TIMEOUT_SECONDS} ollama run ${DRAFT_MODEL} "${BENCHMARK_PROMPT}" 2>&1 || true)

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    output_len=${#output}
    token_count=$((output_len / 4))

    draft_total_ms=$((draft_total_ms + elapsed_ms))
    draft_total_tokens=$((draft_total_tokens + token_count))

    elapsed_s=$((elapsed_ms / 1000))
    echo -e "  Time: ${GREEN}${elapsed_s}s${NC}"
    echo -e "  Output chars: ${GREEN}${output_len}${NC}"
    echo -e "  Est. tokens: ${GREEN}${token_count}${NC}"

    if [ $elapsed_ms -gt 0 ]; then
        throughput=$((token_count * 1000 / elapsed_ms))
        echo -e "  Throughput: ${GREEN}${throughput} tok/s${NC}"
    fi

    echo -e "  Preview:"
    echo "$output" | head -c 250
    echo -e "\n"
done

# Draft model averages
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

if [ $draft_throughput -gt $target_throughput ]; then
    speedup=$((draft_throughput / target_throughput))
    echo -e "  ${GREEN}✓ Draft is ${speedup}x FASTER${NC}"
    echo -e "    E2B: ${BLUE}${draft_throughput} tok/s${NC} vs 26B: ${BLUE}${target_throughput} tok/s${NC}"
    echo -e "  ${GREEN}✓ SUITABLE for speculative decoding${NC}"
else
    ratio=$((target_throughput / draft_throughput))
    echo -e "  ${RED}✗ Draft is ${ratio}x SLOWER${NC}"
    echo -e "    E2B: ${BLUE}${draft_throughput} tok/s${NC} vs 26B: ${BLUE}${target_throughput} tok/s${NC}"
    echo -e "  ${RED}✗ NOT suitable for speculative decoding${NC}"
fi

echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}FINAL BENCHMARK SUMMARY${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "Task: Complex code review (multi-part reasoning)"
echo -e "Target (26B MoE): ${BLUE}${target_throughput} tok/s${NC} (avg time: $((target_avg_ms / 1000))s)"
echo -e "Draft (E2B):     ${BLUE}${draft_throughput} tok/s${NC} (avg time: $((draft_avg_ms / 1000))s)"
echo -e "\nFor speculative decoding to work:"
echo -e "  1. Draft must be faster (${GREEN}target achieved${NC})"
echo -e "  2. Accept rate > 15-20% needed for 3-4x overall speedup"
echo -e "  3. Quality must be comparable enough for token verification"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

echo "Benchmark complete!" > /tmp/gemma_benchmark_done.txt
