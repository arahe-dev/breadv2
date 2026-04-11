#!/bin/bash
# BREAD Performance Baseline Benchmark (Corrected)
# Removes thinking block to properly measure throughput
# Usage: bash benchmark_corrected.sh

echo "========================================"
echo "BREAD Performance Baseline Benchmark (Fixed)"
echo "========================================"
echo ""
echo "NOTE: Thinking blocks removed from prompt format"
echo "      to avoid premature EOS token generation"
echo ""

cd /c/bread_v2

benchmark_tokens() {
    local tokens=$1
    local output_file="bench_corrected_${tokens}.log"

    echo "[TEST] Running with $tokens tokens..."

    # Create test prompt WITHOUT thinking block
    cat > /tmp/test_prompt_corrected.txt << 'EOF'
<|im_start|>system
You are a helpful assistant. Provide detailed, thorough explanations.
<|im_end|>
<|im_start|>user
Explain how binary addition works. Include examples and discuss why it matters in computer science.
<|im_end|>
<|im_start|>assistant
EOF

    # Run and time
    time_start=$(date +%s%N)

    ./bread.exe --server --tokens $tokens < /tmp/test_prompt_corrected.txt > "$output_file" 2>&1

    time_end=$(date +%s%N)

    # Calculate elapsed time in seconds
    elapsed_ms=$(( (time_end - time_start) / 1000000 ))
    elapsed_sec=$(echo "scale=3; $elapsed_ms / 1000" | bc)

    # Count tokens generated (excluding debug/sentinel lines)
    tokens_generated=$(grep -c "^" "$output_file" || echo "0")

    # Check for BREAD_END
    if grep -q "BREAD_END" "$output_file"; then
        echo "  ✓ Generation completed"
        echo "  Time: ${elapsed_sec}s"

        # Extract actual token count from output
        # Count visible tokens (rough estimate: words * 1.3)
        output_text=$(sed '/^BREAD_/d; /^loader:/d; /^gguf:/d; /^weight_cache:/d; /^tokenizer:/d' "$output_file" | tr -s ' ' '\n' | wc -l)
        echo "  Requested: $tokens tokens"
        echo "  Output file: $output_file"
    else
        echo "  ✗ BREAD_END not found - generation may have failed"
    fi

    echo ""
}

# Run benchmarks with different token limits
benchmark_tokens 50
benchmark_tokens 100
benchmark_tokens 200

echo "========================================"
echo "Benchmark complete!"
echo "Check output files for detailed results:"
echo "  bench_corrected_50.log"
echo "  bench_corrected_100.log"
echo "  bench_corrected_200.log"
echo ""
echo "NOTE: Output includes loader/cache initialization time"
echo "      Each test reloads the 23.87 GB model (~20s)"
echo "========================================"
