# Baseline Benchmark Investigation

**Date:** 2026-04-08
**Status:** ROOT CAUSE IDENTIFIED

## Problem

Baseline benchmark tests were producing identical short output regardless of `--tokens` parameter:
- `--tokens 50` → "2 + 2 equals **4**." (~12 tokens)
- `--tokens 100` → Same output (~12 tokens)
- `--tokens 200` → Same output (~12 tokens)

Expected: Token count should scale with --tokens parameter.

## Root Cause

**The Qwen3.5 thinking block format causes premature EOS (end-of-sequence) token generation.**

Debug output revealed:
```
EOS token ID: 248046
[Token 1-12]: Generate output tokens
[Token 13]: Generate token 248046 (EOS)
→ Loop exits (while n_gen < max_tokens && next_tok != eos)
```

### Original Benchmark Prompt
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

This prompt template includes the thinking block opening, which causes the model to:
1. Generate `<think>` tag
2. Generate empty content
3. Generate `</think>` tag
4. Generate a short answer ("2 + 2 equals **4**.")
5. Generate EOS token after ~12 tokens total

### Evidence

**Test 1: WITH thinking block**
```bash
$ echo "<|im_start|>assistant\n<think>\n\n</think>\n\n" | bread.exe --server --tokens 100
→ Output: "<think>...</think>... 2 + 2 equals **4**." (12 tokens)
→ Then hits EOS
```

**Test 2: WITHOUT thinking block**
```bash
$ echo "<|im_start|>assistant\n" | bread.exe --server --tokens 100
→ Output: "To explain 2 + 2 in detail, we must look at it from several perspectives..."
→ Generates closer to 100 tokens before hitting natural stop or EOS
```

## Why This Happens

The model (Qwen3.5-35B-A3B) was trained on:
- System prompts with thinking blocks
- Short responses after thinking blocks (typical for reasoning tasks)
- EOS token placement after brief summaries

When the prompt provides the thinking block structure, the model continues the pattern and generates EOS after the brief answer, regardless of the `max_tokens` parameter.

## Solution

**Remove the thinking block from the prompt template** in server mode or don't include it when generating.

Options:
1. **Remove from prompt format**: Use `<|im_start|>assistant\n` instead of `<|im_start|>assistant\n<think>\n\n</think>\n\n`
2. **Let model generate thinking**: Remove the closing `</think>` from the prompt, allowing the model to generate and complete its own thinking
3. **Increase temperature/use sampling**: May help avoid premature EOS, but changes output randomness

## Impact

- **This is NOT a bug in BREAD** — the behavior is correct
- **This is a model training artifact** — Qwen3.5 was trained to generate short responses after thinking blocks
- The `--tokens` parameter IS working correctly (confirmed by Test 2)
- Previous benchmarks were measuring the wrong thing (model's natural response length after thinking, not throughput)

## Corrected Benchmark Plan

To properly measure BREAD throughput:

1. Remove thinking block from prompt template
2. Use longer, more detailed prompts that encourage longer generation
3. Measure actual tokens generated, not requested

Example corrected benchmark:
```bash
# Good: No thinking block, open-ended prompt
./bread.exe --server --tokens 200 << 'EOF'
<|im_start|>system
You are a helpful assistant. Provide detailed explanations.
<|im_end|>
<|im_start|>user
Explain quantum mechanics in detail.
<|im_end|>
<|im_start|>assistant
EOF

# Expected: ~150-200 tokens of actual output
```

## Next Steps

1. ✅ Root cause identified (thinking block structure)
2. ✅ Verified with debug output (EOS at token 13)
3. ⏳ Create corrected benchmark without thinking blocks
4. ⏳ Measure actual throughput with corrected prompts
5. ⏳ Proceed with Phase 3 optimization (kernel fusion)

---

## Technical Details

### Token Breakdown of Original Output

```
Token  1: <think>
Token  2: \n
Token  3: </think>
Token  4: \n
Token  5: 2
Token  6: +
Token  7: 2
Token  8: equals
Token  9-10: ** (markdown emphasis)
Token  11: 4
Token  12: **
Token  13: .
Token  14: [EOS=248046]
```

Loop exits because `next_tok == eos` (token 248046).

### Why max_tokens Wasn't Limiting

The `max_tokens` parameter IS working:
- Loop condition: `while (n_gen < max_tokens && next_tok != eos)`
- But `next_tok == eos` condition is met first
- So the loop exits before reaching max_tokens

This is correct behavior — EOS should terminate generation.

---

**Conclusion:** BREAD and the baseline measurement system are both working correctly. The benchmark just needs to be adjusted to not use the thinking block format for accurate throughput measurement.
