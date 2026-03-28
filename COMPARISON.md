# BREAD vs llama.cpp: Side-by-Side Comparison

This document compares the inference pipelines of BREAD and llama.cpp for the Qwen3.5-35B-A3B model.

## Quick Reference

| Aspect | BREAD | llama.cpp |
|--------|-------|-----------|
| **Language** | CUDA/C | C/CUDA |
| **Full Attention** | ✓ (10 layers) | ✓ (10 layers) |
| **SSM/GatedDeltaNet** | ✗ (stubbed) | ✓ (fully implemented) |
| **MoE Experts** | ✓ (256 experts, K=8) | ✗ (not in reference) |
| **RoPE** | Partial (32 dims, 3 sections) | Full (128+ dims, 4 sections) |
| **KV Cache** | Host-side (float) | GPU integrated |
| **Quantization** | Q4_K, Q6_K custom kernels | ggml ops (may differ) |
| **Mode** | Normal + Minimal debug | Single inference path |

---

## Detailed Comparison by Phase

### 1. Model Loading

**BREAD** (`main.cu` → `loader_init()`)
```
Load GGUF header
Parse 1959 tensors into index
Allocate 23.87 GB host RAM
Stream all tensors from disk
Extract metadata: hidden_dim=2048, num_layers=40
Extract MoE: num_experts=256, top_k=8
```

**llama.cpp** (`llama.cpp` → `llama_model_load_from_file()`)
```
Load GGUF header
Parse GGUF metadata keys (qwen35.*)
Read hyperparameters:
  - rope_sections = [4 values, not 3]
  - layer type pattern (attention vs SSM)
  - embedding dimensions
Allocate tensors progressively
Map GGUF tensors to llm_layer structures
```

**Key Difference**: llama.cpp expects `rope_sections` to have **4 elements**, but the model has **3**. This causes the load error you encountered.

---

### 2. Tokenization

**BREAD** (`main.cu` → `tokenizer_encode()`)
```
Load tokenizer vocab (248,320 tokens)
Load BPE merges
Apply Qwen35 pre-tokenizer wrapping:
  "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
BPE encode → token IDs
Result: 39 tokens for "The capital of France is"
```

**llama.cpp** (`llama.cpp` → `llama_tokenize()`)
```
Apply Qwen-specific pre-tokenization
BPE encoding with merge table
Apply special token handling (BOS/EOS)
Result: Same token sequence
```

**Difference**: Both should produce identical token IDs if pre-tokenizer logic is correct.

---

### 3. Inference Loop

**BREAD**
```
FOR each prompt token:
  embed_token() → d_hidden [2048] half on VRAM
  FOR each of 40 layers:
    one_layer_forward(d_hidden) → modifies in-place
  CUDA sync

After prompt (n_prompt=39):
  apply_output_norm() → RMSNorm
  compute_logits() → [vocab_size] logits
  greedy_sample() → argmax

FOR each generated token:
  embed_token(next_tok) → d_hidden
  FOR each of 40 layers:
    one_layer_forward(d_hidden)
  Output norm + logits + sample
```

**llama.cpp**
```
Build computation graph for entire context (39 tokens)
Parallel prefill phase (batched):
  All 39 tokens through all layers simultaneously
  KV caches filled for all positions

After prefill:
  One step autoregressive generation:
  embed_token() → hidden
  one_layer() for all 40 layers
  Output norm + logits + sample
```

**Difference**: llama.cpp batches the prefill (39 tokens in parallel), BREAD does them sequentially (token by token). This is a performance difference, not correctness.

---

### 4. Single Layer Forward Pass

#### 4.1 Layer Type Dispatch

**BREAD**
```cpp
is_full = (layer_idx % 4 == 3);
if (is_full) {
  // Full attention (layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
} else {
  // SSM/GatedDeltaNet (layers 0-2, 4-6, 8-10, ...)
  // [CURRENTLY STUBBED - SSM output ≈ 0]
}
```

**llama.cpp**
```cpp
if (llm_layer->is_attention) {
  // Full attention path
} else if (llm_layer->is_recurrent) {
  // SSM / Gated Delta Net path
  // [FULLY IMPLEMENTED]
}
```

**Difference**: BREAD's SSM is not implemented (output = 0). This is why the model produces garbage — 30 out of 40 layers contribute nothing.

---

#### 4.2 Full Attention Path (10 layers)

**BREAD** (`one_layer.cu` lines 773-827)
```
Pre-attn norm: RMSNorm(d_hidden, attn_norm.weight)

Q proj:  normed @matmul attn_q.weight     → [8192]
K proj:  normed @matmul attn_k.weight     → [512]
V proj:  normed @matmul attn_v.weight     → [512]

Split & norm:
  Q: [16 heads, 512 dims] → [16 heads, 256 score + 256 gate]
  K,V: Per-head RMSNorm

RoPE (rope_base=1e7):
  Apply rotary embeddings to Q, K
  rope_sections=[11,11,10] → only 32 dims rotated
  rope_interleaved=True

Attention scoring:
  FOR head h:
    FOR position p in KV cache:
      score[p] = dot(Q[h], K[p]) / sqrt(head_dim_qk)
    softmax(scores) → weights
    output[h] = sum(weights[p] * V[p])
    multiply by sigmoid(gate[h])

O proj: attn_out @matmul attn_output.weight → [2048]
```

**llama.cpp** (`llama.cpp` → `llm_build_moe()` or `llm_build_attn()`)
```
Pre-attn norm: RMSNorm(hidden, attn_norm.weight, eps)

Q,K,V proj: matrix multiplies (fused in CUDA)

RoPE (rope_base=10000):
  Multi-dimensional RoPE with rope_sections=[4 values]
  rope_interleaved=True (applies to full rope_dim, not partial)
  Frequency scaling per section

Attention:
  GQA (grouped query attention): 16 Q heads × (2 KV heads)
  Fused attention kernel (optimized)
  Causal masking (only attend to past)

O proj + gate
```

**Critical Differences**:

1. **RoPE Dimensions**:
   - BREAD: Only 32 dims out of 2048 (1.5%) get rotary embeddings
   - llama.cpp: Full 128 dims per head (or more) get proper RoPE
   - **This is likely a major bug in BREAD**

2. **Attention Implementation**:
   - BREAD: Reference scalar implementation (loop over positions)
   - llama.cpp: Fused kernel (faster but same math)

3. **GQA Handling**:
   - BREAD: Hardcoded 16→2 head expansion
   - llama.cpp: Generic GQA with parameterized expansion

---

#### 4.3 SSM / Gated Delta Net (30 layers)

**BREAD** (`one_layer.cu` lines 828-900)
```
[NOT IMPLEMENTED]
is_full == False → SSM branch
Currently: ssm_out ≈ 0 (returned to residual stream)

What should happen:
- Project to QKV [8192×3]
- Causal convolution (kernel=4, over context history)
- Scan with A/B/C matrices (state update)
- Project Z gate
- Apply gated output
```

**llama.cpp** (`llama.cpp` → `llm_build_moe()` for SSM layers)
```
Full SSM/Gated Delta Net implementation:
1. Project input to QKV [ssm_qkv_dim = 8192]
2. Chunk processing (16-token chunks for efficiency)
3. L2 normalization of Q,K
4. Delta Net attention:
   - Compute deltas from timesteps
   - Exponential decay masking
   - Recurrence relation for state
5. Gate projection [ssm_z_dim = 4096]
6. SiLU gate × output
7. Project back to hidden [2048]
```

**Critical Difference**:
- BREAD outputs zero for 30/40 layers (SSM stubbed)
- llama.cpp fully implements the SSM
- **This is the PRIMARY bug in BREAD's garbage output**

---

#### 4.4 MoE Expert Routing & Combination

**BREAD** (`one_layer.cu` lines 943-1000)
```
Router: normed2 @matmul ffn_router.weight → logits [256]
Top-K:  argmax-8 → expert_indices [8]
Weights: softmax(selected_logits) → [8]

Shared expert (always used):
  gate/up/down projections
  SwiGLU activation
  weighted accumulation

For each of 8 selected experts:
  Load tensors (host RAM or VRAM cache)
  gate/up/down projections
  SwiGLU activation
  weighted accumulation (weight[k] * output[k])

Final: residual += shared + sum(experts)
```

**llama.cpp** (`llama.cpp` → `llm_build_moe()`)
```
[Not explicitly shown for Qwen35 in basic inference]
This model has MoE but llama.cpp may handle it differently
or llama.cpp may use a simplified MoE path
```

**Difference**: BREAD has explicit MoE logic. Qwen35 is a MoE model, but MoE structure may differ between implementations.

---

### 5. Output & Generation

**BREAD**
```
After layer 39:
  Apply output norm: RMSNorm(d_hidden, output_norm.weight)
  Compute logits: normed @matmul output.weight → [vocab_size]
  Argmax: token_id = argmax(logits)
  Decode: tokenizer_decode(token_id) → text
```

**llama.cpp**
```
Same:
  output norm + lm_head
  Sampling (default: argmax, or temperature/top-k/top-p)
  Decode output tokens
```

---

## Summary of Bug Hypotheses

### Why BREAD Outputs Garbage

1. **PRIMARY: SSM Not Implemented** (30/40 layers produce zero output)
   - Each of these layers adds nothing to the residual stream
   - Equivalent to removing 75% of the model
   - **This alone explains the garbage output**

2. **SECONDARY: RoPE Incomplete** (only 32/2048 dims rotated)
   - The 10 full-attention layers can't attend correctly
   - Query/key misalignment due to improper rotation
   - Could cause wrong token predictions

3. **TERTIARY: Other Math Bugs** (quantization, precision, etc.)
   - Less likely if normal & minimal modes agree

### Why Both Normal & Minimal Modes Produce Same Garbage

Both modes share the same `one_layer_forward()` logic, just different precision (fp16 vs fp32). If the bug is in `one_layer_forward()` itself (SSM stubbed, RoPE wrong), both modes will fail identically.

---

## Next Steps to Debug

1. **Enable SSM in BREAD** (copy from llama.cpp or llama-rs)
   - Implement convolution + scan
   - This alone will likely fix most issues

2. **Fix RoPE**
   - Use all 128 dims per head (not just 32)
   - Handle rope_sections properly (4 values, not 3)
   - Ensure interleaved flag applies correctly

3. **Compare Per-Layer RMS**
   - Run: `ollama run qwen3.5:35b-a3b "The capital of France is"`
   - Capture intermediate activations (if ollama supports it)
   - Or: Build a debug llama.cpp with RMS dumps and compare

4. **Validate Against llama.cpp**
   - Once you fix the rope loading issue in llama.cpp
   - Run both with same prompt
   - Compare per-layer RMS values
   - Verify final output matches

---

## File References

- **BREAD Analysis**: `breadflowchartanalysis.md` (816 lines)
- **llama.cpp Analysis**: `llamacppflowchartanalysis.md` (816 lines)
- **This Comparison**: `COMPARISON.md` (this file)

Read all three together to understand the gap between the two implementations.
