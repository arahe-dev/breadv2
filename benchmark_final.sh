#!/bin/bash
# BREAD Performance Baseline Benchmark (Final)
# Properly measures throughput with corrected prompt format

echo "========================================"
echo "BREAD Performance Baseline Benchmark"
echo "========================================"
echo ""

cd /c/bread_v2

benchmark_tokens() {
    local tokens=$1
    local output_file="bench_final_${tokens}.log"

    echo "[TEST] Running with $tokens tokens..."

    # Simplified prompt (no thinking block, direct generation)
    cat > /tmp/test_final_prompt.txt << 'EOF'
<|im_start|>system
You are a helpful assistant. Provide detailed, thorough answers.
<|im_end|>
<|im_start|>user
Explain in detail how neural networks work. Include information about layers, weights, and backpropagation.
<|im_end|>
<|im_start|>assistant
EOF

    # Run in single-mode (NOT server mode) to get timing from BREAD
    # Use bash time measurement for wall-clock
    (time cat /tmp/test_final_prompt.txt | ./bread.exe --tokens $tokens) > "$output_file" 2>&1

    # Extract results
    echo "  Output saved to: $output_file"

    # Count generated tokens (rough word count * 1.3)
    words=$(sed '1,/--- Generated output/d' "$output_file" | \
            sed '/^=/d; /tok\/s/d; /ms/d' | \
            tr ' ' '\n' | grep -v '^$' | wc -l)
    tokens_gen=$((words * 130 / 100))

    # Get actual throughput from BREAD's output
    tok_per_sec=$(grep -oE "[0-9]+\.[0-9]+ tok/s" "$output_file" | awk '{print $1}')

    if [ -n "$tok_per_sec" ]; then
        echo "  Measured throughput: $tok_per_sec tok/s"
    fi

    echo ""
}

# Run benchmarks
benchmark_tokens 50
benchmark_tokens 100
benchmark_tokens 200

echo "========================================"
echo "Summary"
echo "========================================"
echo "Check the output files for detailed results with BREAD's timing measurements"
echo "========================================"
