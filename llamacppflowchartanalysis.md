# llama.cpp Inference Engine - Qwen3.5-35B-A3B Flowchart Analysis

## Overview

This document traces how llama.cpp (open-source reference implementation) processes the Qwen3.5-35B-A3B model, which uses a hybrid architecture with:
- **Full-Attention Layers**: Standard transformer with Q/K/V and RoPE
- **Linear Attention Layers (SSM/Gated Delta Net)**: Recurrent state-based attention for efficient processing

---

## 1. INITIALIZATION & MODEL LOADING

### 1.1 Model File Loading Process

**Entry Point**: `llama_model_load_from_file()` → `llama_model::load_tensors()`

**Steps**:
1. **Open GGUF File**: Load model from `.gguf` file format using `llama_model_loader`
2. **Load Architecture**: Read `general.architecture` key from GGUF metadata
   - For Qwen3.5: `architecture = "qwen35"`
   - Maps to `LLM_ARCH_QWEN35` enum
3. **Load HParams**: Parse model hyperparameters from GGUF keys:
   - `qwen35.vocab_size`
   - `qwen35.context_length`
   - `qwen35.embedding_length` (n_embd)
   - `qwen35.block_count` (n_layer)
   - `qwen35.feed_forward_length`
   - `qwen35.rope_sections` - array of 4 values defining MRoPE dimensions
   - Recurrent layer pattern (which layers are SSM vs attention)

### 1.2 Architecture Detection: Layer Type Identification

**Function**: `llama_hparams::is_recurrent(uint32_t il)`

```cpp
// Determines if layer 'il' is a recurrent/linear attention layer
bool is_recurrent(il) {
    // Check if layer uses SSM pattern
    // Returns true for SSM/GatedDeltaNet layers
    // Returns false for full attention layers
}
```

**For Qwen3.5-35B-A3B**:
- Pattern: Alternating layers or specific recurrent layer indices
- When `is_recurrent(il) == true`: Build SSM/GatedDeltaNet path
- When `is_recurrent(il) == false`: Build full-attention path

### 1.3 Tensor Loading (1959 tensors total)

**Files**: Located in `/c/Users/arahe/llama.cpp/src/llama-model.cpp` (load_tensors function)

**Tensor Categories**:

1. **Embedding Layer**:
   - `tok_embd`: Token embeddings [vocab_size, n_embd]

2. **Per-Layer Tensors** (repeated for each of 64 layers):
   - **Attention Norm**: `layers[il].attn_norm` (RMSNorm weight)
   - **Attention Post-Norm**: `layers[il].attn_post_norm` (RMSNorm weight)
   - **Full Attention Layers**:
     - `layers[il].wq`: Q projection [n_embd, n_embd * (1 + gate_factor)]
     - `layers[il].wk`: K projection [n_embd, n_embd_kv]
     - `layers[il].wv`: V projection [n_embd, n_embd_kv]
     - `layers[il].wo`: Output projection [n_embd, n_embd]
     - `layers[il].attn_q_norm`: Q layer norm
     - `layers[il].attn_k_norm`: K layer norm

   - **Linear Attention / SSM Layers**:
     - `layers[il].wqkv`: QKV mixed projection [n_embd, d_inner * (2 + num_heads + num_heads)]
     - `layers[il].wqkv_gate`: Gate projection
     - `layers[il].ssm_conv1d`: Convolution kernel [conv_kernel_size, channels]
     - `layers[il].ssm_alpha`: Alpha parameter
     - `layers[il].ssm_beta`: Beta parameter
     - `layers[il].ssm_dt`: DT (delta-time) parameter
     - `layers[il].ssm_a`: State matrix A
     - `layers[il].ssm_norm`: Gated normalization weight
     - `layers[il].ssm_out`: Output projection

3. **FFN (Feed-Forward Network)**:
   - `layers[il].attn_post_norm`: Pre-FFN norm
   - `layers[il].ffn_up`: Up projection
   - `layers[il].ffn_gate`: Gate projection (SwiGLU)
   - `layers[il].ffn_down`: Down projection

4. **Output Layer**:
   - `output_norm`: Final RMSNorm weight
   - `output`: LM head weights [vocab_size, n_embd]

**Total Parameters**: ~35 billion (35B)

---

## 2. TOKENIZATION

**Module**: `llama-vocab.cpp` with `llm_tokenizer` interface

### 2.1 Tokenizer Selection

For Qwen models, llama.cpp uses:
- **Bytepair Encoding (BPE)** tokenizer implementation
- **Pre-tokenization**: Qwen-specific regex pattern handling
- **Merging**: Greedy algorithm combining high-frequency token pairs

### 2.2 Tokenization Process

```
Input Text
    ↓
[Pre-tokenization] - Split on whitespace, punctuation, regex patterns
    ↓
[Token Lookup] - Look up each symbol in vocabulary
    ↓
[Merge Phase] - Iteratively merge highest-score bigrams
    ↓
[Output] - Token IDs array
```

### 2.3 Prompt Wrapping

- Format specifier per model: typically `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- Encoded as token sequence
- Special tokens handled separately (not merged)

### 2.4 Batch Processing

- Tokens passed to `llama_batch` structure
- Each token marked with position, sequence ID, and output flag
- Support for multiple sequences in parallel

---

## 3. INFERENCE LOOP (MAIN COMPUTATION)

**Entry Point**: `llama_decode(llama_context*, llama_batch)`

### 3.1 Graph Construction Phase

**Function**: `llm_graph_context::build_graph()`

Creates computation graph with two main phases:

#### Phase 1: Prefill (Context Encoding)
```
Process all prompt tokens in parallel:
  for each token in batch:
    - Embed token
    - Apply all 64 layers sequentially
    - Skip final LM head (intermediate representations only)
    - Store KV cache values in recurrent state memory
```

#### Phase 2: Autoregressive Generation
```
Generate one token at a time:
  repeat until EOS token:
    - Get last token embedding
    - Apply all 64 layers (using cached KV/state from prefill)
    - Apply final LM head
    - Sample next token from logits
    - Add token to batch
    - Update position cache and KV cache
```

### 3.2 Graph Execution

**Backend**: GGML with optional GPU offloading
- Operations scheduled on available devices (CPU, CUDA, Metal, etc.)
- KV cache managed by `llama_kv_cache_context`
- Recurrent state managed by `llama_memory_recurrent_context`

---

## 4. SINGLE LAYER FORWARD PASS

**File**: `/c/Users/arahe/llama.cpp/src/models/qwen35.cpp`

### 4.1 Layer Type Detection & Routing

```cpp
// In llm_build_qwen35::constructor loop over n_layer
for (int il = 0; il < n_layer; ++il) {
    // 1. Pre-attention norm
    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);

    // 2. Route based on layer type
    if (hparams.is_recurrent(il)) {
        // SSM/Gated Delta Net path
        cur = build_layer_attn_linear(inp->get_recr(), cur, il);
    } else {
        // Full attention path
        cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
    }

    // 3. Residual connection
    cur = ggml_add(ctx0, cur, inpSA);

    // 4. Post-attention norm
    cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);

    // 5. FFN
    cur = build_layer_ffn(cur, il);

    // 6. Residual connection
    cur = ggml_add(ctx0, cur, ffn_residual);
}
```

### 4.2 FULL ATTENTION IMPLEMENTATION

**Function**: `build_layer_attn()`

#### Step 1: QKV Projection with Gating
```cpp
// Q with gate
Qcur_full = build_lora_mm(model.layers[il].wq, cur);  // [n_embd_head*2, n_head, n_tokens]
Qcur = extract_view(Qcur_full, 0, n_embd_head);       // [n_embd_head, n_head, n_tokens]
gate = extract_view(Qcur_full, n_embd_head, n_embd_head); // [n_embd_head, n_head, n_tokens]

// K and V
Kcur = build_lora_mm(model.layers[il].wk, cur);       // [n_embd_head, n_head_kv, n_tokens]
Vcur = build_lora_mm(model.layers[il].wv, cur);       // [n_embd_head, n_head_kv, n_tokens]
```

#### Step 2: Layer Normalization
```cpp
// Q-norm and K-norm (QueryKey Norm)
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
// V not normalized in standard attention
```

#### Step 3: RoPE (Rotary Position Embedding) - Multi-dimensional
```cpp
// MRoPE with 4 rope_sections
Qcur = ggml_rope_multi(
    ctx0, Qcur,                          // input
    inp_pos,                             // position indices
    nullptr,                             // mask
    n_rot,                               // number of rotation dimensions
    sections,                            // [n_rot_1, n_rot_2, n_rot_3, n_rot_4]
    rope_type,                           // LLAMA_ROPE_TYPE_NORM
    n_ctx_orig,                          // context length (training)
    freq_base,                           // base frequency (default 10000.0)
    freq_scale,                          // scaling factor
    ext_factor,                          // extrapolation factor
    attn_factor,                         // attention scaling
    beta_fast, beta_slow                 // yarn parameters
);

Kcur = ggml_rope_multi(...same parameters...);
```

**RoPE Parameters for Qwen3.5**:
- `freq_base`: 10000.0 (standard)
- `freq_scale`: 1.0 (no scaling)
- `rope_sections`: [128, 128, ...] (distributed across multiple frequencies)
- `rope_type`: LLAMA_ROPE_TYPE_NORM (non-interleaved)

#### Step 4: Attention Computation
```cpp
// Attention scale: 1.0 / sqrt(n_embd_head)
kq_scale = 1.0f / sqrtf(float(n_embd_head));

// Core attention: Q @ K^T
kq = ggml_mul_mat(K, Q);                // [n_tokens, n_tokens, n_head, n_seqs]
kq = ggml_scale(kq, kq_scale);          // Scale by 1/sqrt(d)

// Attention mask (causal)
kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, ...);  // [n_tokens, n_tokens, n_head, n_seqs]

// Value attention: attention_weights @ V
kqv = ggml_mul_mat(V, kq);              // [n_embd_head, n_tokens, n_head, n_seqs]
```

#### Step 5: Gate Application
```cpp
// Apply sigmoid gate to attention output
gate_sigmoid = ggml_sigmoid(ctx0, gate);
cur = ggml_mul(ctx0, attn_output, gate_sigmoid);
```

#### Step 6: Output Projection
```cpp
cur = build_lora_mm(model.layers[il].wo, cur);  // Project back to n_embd
```

**Full Attention Parameters**:
- `n_embd_head`: Dimension per head (e.g., 128)
- `n_head`: Number of heads (e.g., 40)
- `n_head_kv`: Number of KV heads (for grouped query attention, typically same as n_head)
- `n_tokens`: Batch size of tokens
- `kq_scale`: Attention softmax scaling (1/sqrt(d_head))

### 4.3 SSM / GATED DELTA NET IMPLEMENTATION

**Function**: `build_layer_attn_linear()`

#### Step 1: Input Projections
```cpp
// Combined QKV projection with mixed output
qkv_mixed, z = build_qkvz(cur, il);
  // qkv_mixed = build_lora_mm(wqkv, cur) -> [d_inner + 2*ssm_d_state, n_tokens]
  // z = build_lora_mm(wqkv_gate, cur)     -> [d_inner, n_tokens]

// Additional parameters
beta = ggml_sigmoid(build_lora_mm(ssm_beta, cur));     // Decay bias
alpha = build_lora_mm(ssm_alpha, cur);                 // Alpha parameter
alpha_biased = ggml_add(ctx0, alpha, ssm_dt);          // Add DT bias
alpha_softplus = ggml_softplus(ctx0, alpha_biased);    // Softplus activation
gate = ggml_mul(ctx0, alpha_softplus, ssm_a);          // Apply state matrix
```

**Parameters**:
- `d_inner`: SSM inner dimension (2304)
- `ssm_d_state`: State dimension (64)
- `ssm_n_group`: Group heads for KV
- `ssm_dt_rank`: DT rank

#### Step 2: Convolution Operation
```cpp
// 1D depthwise convolution along token sequence
conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);  // Prepend state history

// SSM convolution: fold previous states into input
conv_output = ggml_ssm_conv(ctx0, conv_input, model.layers[il].ssm_conv1d);
conv_output_silu = ggml_silu(ctx0, conv_output);            // SiLU activation

// Extract Q, K, V from convolved output
q_conv = extract_4d(conv_output_silu, ...);  // [head_k_dim, num_k_heads, n_tokens, n_seqs]
k_conv = extract_4d(conv_output_silu, ...);  // [head_k_dim, num_k_heads, n_tokens, n_seqs]
v_conv = extract_4d(conv_output_silu, ...);  // [head_v_dim, num_v_heads, n_tokens, n_seqs]
```

**Convolution Kernel**:
- `conv_kernel_size`: Typically 4 (look-back window)
- `channels`: d_inner + 2*ssm_d_state*ssm_n_group

#### Step 3: Query/Key Normalization
```cpp
// L2 normalization instead of LayerNorm
q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);  // eps = 1e-6
k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
```

#### Step 4: Chunk-based Delta Net Attention
```cpp
// Build delta net with chunked computation (efficient scanless recurrence)
auto [attn_out, new_state] = build_delta_net(
    q_conv, k_conv, v_conv,    // Projected inputs
    gate,                        // Decay gate
    beta,                        // Decay bias
    state,                       // Previous recurrent state
    il
);

// Delta Net performs:
// - Chunk-wise attention computation with exponential decay
// - Implicit recurrence via state update
// - O(n) complexity instead of O(n^2) for full attention
```

**Delta Net Internal Computation** (from `delta-net-base.cpp`):
```cpp
// 1. Scale Q by 1/sqrt(head_dim)
q = ggml_scale(ctx0, q, 1.0f / sqrtf(head_k_dim));

// 2. Compute cumulative decay: exp(cumsum(gate))
g_cs = ggml_cumsum(ctx0, ggml_transpose(ctx0, gate));

// 3. Compute decay masks for causal attention
// decay_mask = exp(g_cs[j] - g_cs[i]) for i <= j (lower triangular)

// 4. Per-chunk attention with state update
for each chunk:
    // Intra-chunk attention
    kq_chunk = K @ Q_chunk (with decay masking)

    // Cross-chunk state contribution
    v_chunk = state @ K_cumulative_decay

    // Output: attention + state contribution
    output = (attention_output + state_contribution) * beta

    // State update: s_new = s_old * g_last_exp + K_gdiff @ V_new
    state = ggml_mul(state, g_last_exp_t) + kg_t @ v_t_new
```

#### Step 5: Gated Normalization
```cpp
// Apply gated norm after delta net
attn_out_norm = build_norm_gated(
    attn_output,
    model.layers[il].ssm_norm,
    z_2d,
    il
);
// Internally: norm(attn_out) * SiLU(z)
```

#### Step 6: Output Projection
```cpp
cur = build_lora_mm(model.layers[il].ssm_out, attn_out_norm);  // Project back to n_embd
```

**State Management**:
- Previous state: `mctx_cur->get_s_l(il)` [head_v_dim, head_v_dim, num_v_heads, n_seqs]
- Conv state: `mctx_cur->get_r_l(il)` [conv_kernel_size-1, conv_channels, n_seqs]
- Updated after each layer for next sequence token

---

## 5. KEY OPERATIONS WITH EXACT PARAMETERS

### 5.1 RMSNorm (Pre & Post Layer Norms)

**Function**: `ggml_rms_norm(ctx0, cur, eps)`

```cpp
// RMS Normalization formula:
// output[i] = input[i] / sqrt(mean(input[i]^2) + eps) * weight[i]

Parameters:
  - eps (epsilon): hparams.f_norm_rms_eps (typically 1e-5 or 1e-6)
  - weight: model.layers[il].attn_norm (loaded from GGUF)
  - Input shape: [n_embd, n_tokens]
  - Output shape: [n_embd, n_tokens]
```

**Applied at**:
- Pre-attention norm: `attn_norm`
- Pre-FFN norm: `attn_post_norm`
- Final output norm: `output_norm`

### 5.2 RoPE (Rotary Position Embeddings) - Multi-dimensional

**Function**: `ggml_rope_multi(ctx0, input, pos, mask, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)`

```cpp
// MRoPE applies rotations to multiple dimension groups
// Each rope_section[i] gets its own frequency base: base^(2*i / sum(sections))

rope_sections = [128, 128, ...];  // 4 sections for Qwen3.5
n_rot_full = 256 (sum of sections)

// For each position p and dimension group k:
// freq_k = freq_base ^ (2*k / n_rot)
// theta_k = p * freq_k
// Apply rotation matrix: [cos(theta_k), -sin(theta_k); sin(theta_k), cos(theta_k)]

Parameters:
  - freq_base: 10000.0 (default for Qwen)
  - freq_scale: 1.0 (no frequency scaling)
  - ext_factor: 1.0 (extrapolation factor for context extension)
  - attn_factor: 1.0 (attention scaling factor)
  - beta_fast: 32.0 (YaRN parameter)
  - beta_slow: 1.0 (YaRN parameter)
  - rope_type: LLAMA_ROPE_TYPE_NORM (non-interleaved)
  - rope_sections: [128, 128, 0, 0] (4 values from GGUF)

Input shapes:
  - Q: [n_embd_head, n_head, n_tokens, n_seqs]
  - K: [n_embd_head, n_head_kv, n_tokens, n_seqs]
```

### 5.3 Attention (Full Attention Layers Only)

**Core Computation**:
```cpp
// 1. Query-Key similarity
kq = ggml_mul_mat(K, Q);           // [n_kv, n_tokens, n_head, n_seqs]
kq = ggml_scale(kq, kq_scale);     // Scale by 1/sqrt(d_head)

// 2. Causal mask application
kq_mask[i,j] = 0 if i > j else -inf  // Prevent attending to future tokens

// 3. Softmax with scale
kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, f_max_alibi_bias);

// 4. Weighted sum over values
output = ggml_mul_mat(V, softmax(kq));  // [n_embd_head, n_tokens, n_head, n_seqs]

Parameters:
  - kq_scale: 1.0 / sqrt(n_embd_head)
  - f_max_alibi_bias: Max ALiBi bias (typically 0.0 for standard attention)
  - f_attn_logit_softcapping: Softcap for logits (if enabled)
  - flash_attn: GPU-optimized implementation (optional)
```

### 5.4 SSM / Gated Delta Net Parameters

**Convolution**:
```cpp
// Depthwise 1D convolution along token dimension
conv_kernel_size: [4, channels] (typically 4-step history)
conv_stride: 1 (sequential processing)
conv_padding: 0 (manual padding with previous states)

State matrices:
  - ssm_a: [head_v_dim, head_v_dim, num_v_heads] (recurrent state matrix)
  - ssm_alpha: [num_v_heads, n_tokens] (alpha parameter per head)
  - ssm_beta: [num_v_heads, n_tokens] (decay/forget gate)
  - ssm_dt: [num_v_heads] (delta-time bias)

Decay computation:
  - gate = alpha_softplus * -A
  - cumsum(gate) -> exponential decay factor exp(cumsum)
  - Applied as multiplicative decay in state update
```

**Delta Net Specific**:
```cpp
// Chunk-based computation
chunk_size: 16 (for KDA variant) or 64 (standard)
Total chunks: ceil(n_tokens / chunk_size)

// Per-chunk recurrence
s_new = s_old * exp(g_last) + K_gdiff @ V_new

Where:
  - g_last: Last cumulative decay in chunk
  - K_gdiff: K values weighted by decay difference
  - V_new: Updated values after chunk processing
```

### 5.5 MoE (If Present)

**Note**: Qwen3.5-35B-A3B uses standard FFN, NOT MoE. The assertion confirms this:
```cpp
GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);  // No MoE
```

---

## 6. OUTPUT GENERATION

### 6.1 Final Normalization

```cpp
// After all 64 layers
cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
// RMSNorm with eps = hparams.f_norm_rms_eps
```

### 6.2 LM Head Projection

```cpp
// Linear projection to vocabulary
cur = build_lora_mm(model.output, cur);
// Matrix: [vocab_size, n_embd]
// Input: [n_embd, n_tokens]
// Output: [vocab_size, n_tokens] -> logits
```

### 6.3 Sampling

**Post-processing** (handled by llama_sampler):
```cpp
logits = cur;  // [vocab_size]

// Temperature scaling
scaled_logits = logits / temperature;

// Top-k filtering
filtered = keep_top_k(scaled_logits, k=40);  // default

// Top-p (nucleus) filtering
filtered = filter_top_p(filtered, p=0.9);    // default

// Apply logit processors (optional)
// - Repeat penalty
// - Frequency penalty
// - Presence penalty

// Final softmax
probs = softmax(filtered);

// Sample token
next_token = sample(probs);  // or argmax for greedy
```

**Sampling Parameters** (typically from GGUF or defaults):
```
temperature: 0.8 (default)
top_k: 40
top_p: 0.9
repeat_penalty: 1.1
```

---

## 7. KEY DIFFERENCES FROM BREAD

### 7.1 Layer Type Detection

**llama.cpp**:
```cpp
if (hparams.is_recurrent(il)) { /* SSM */ }
else { /* Full Attention */ }
```

**vs BREAD** (assumed):
- Should check GGUF layer type metadata
- Must match llama.cpp's logic exactly

### 7.2 RoPE Implementation

**llama.cpp**: Uses `ggml_rope_multi` with `rope_sections`
- Multi-dimensional rope with independent frequency bases per section
- Handles MRoPE correctly for Qwen3.5

**BREAD Should**:
- Parse `rope_sections` array from GGUF
- Implement multi-rope rotation per section
- Handle both interleaved and non-interleaved variants

### 7.3 Attention Computation

**llama.cpp**:
```cpp
// Uses optimized kernels
kq = ggml_mul_mat(K, Q);      // [batch, n_kv, n_tokens, n_head]
kq = ggml_soft_max_ext(...);  // Softmax with optional flash attention
output = ggml_mul_mat(V, kq); // [batch, d_head, n_tokens, n_head]
```

**BREAD Should**:
- Verify tensor layout/strides match llama.cpp
- Implement efficient softmax (may need custom kernel)
- Handle KV cache indexing correctly

### 7.4 Delta Net / SSM Attention

**llama.cpp**: Chunk-based computation with explicit state updates
- `build_delta_net_chunking()`: Processes in 16/64 token chunks
- Explicit recurrent state storage and update
- L2 normalization for Q/K

**BREAD Should**:
- Match chunking strategy exactly
- Implement scan operation with proper recurrence
- Correctly apply decay masks (exponential, lower triangular)
- Update state in cache after each layer

### 7.5 FFN Implementation

**llama.cpp**:
```cpp
// For Qwen3.5 (no MoE)
cur = build_lora_mm(ffn_up, cur);
cur = ggml_silu(cur);
cur = ggml_mul(cur, gate_output);  // SwiGLU
cur = build_lora_mm(ffn_down, cur);
```

**BREAD Should**:
- Verify hidden dimension (4x or 8/3x n_embd)
- Use SwiGLU gating (not GELU)
- Apply post-norm before FFN (already done via attn_post_norm)

### 7.6 Tensor Orientation & Strides

**llama.cpp**:
- Q/K/V shaped as [head_dim, n_heads, n_tokens, n_seqs]
- Attention operates along specific dimensions
- Permute/reshape operations carefully ordered

**BREAD** (potential issues):
- Verify tensor axis definitions match
- Check row-major vs column-major memory layout
- Stride calculations for efficient data access

### 7.7 KV Cache Management

**llama.cpp**:
- Separate KV cache context for full attention layers
- Recurrent memory context for SSM layers
- Cache indexed by layer and sequence position

**BREAD Should**:
- Manage two separate cache types
- Update KV cache after each full-attention layer
- Update recurrent state after each SSM layer
- Handle variable-length sequences

### 7.8 Tokenization

**llama.cpp**: Bytepair encoding with pre-tokenization

**BREAD Should**:
- Use same tokenizer implementation
- Match special token handling
- Verify prompt wrapping format

---

## 8. EXECUTION FLOW SUMMARY

```
┌─────────────────────────────────────┐
│ 1. INITIALIZATION                   │
├─────────────────────────────────────┤
│ Load GGUF → Parse HParams           │
│ Allocate 1959 tensors               │
│ Detect layer types (attn vs SSM)    │
│ Initialize KV + recurrent caches    │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 2. TOKENIZATION                     │
├─────────────────────────────────────┤
│ Encode prompt → Token IDs           │
│ Wrap with special tokens            │
│ Batch tokens with metadata          │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 3. PREFILL (Parallel Processing)    │
├─────────────────────────────────────┤
│ Embed tokens [n_embd, n_prompt]     │
│ Loop 64 layers:                     │
│   - Layer norm                      │
│   - If full attention: Q/K/V + RoPE │
│   - If SSM: Conv + Delta Net        │
│   - FFN + residual                  │
│ Store KV/state cache                │
│ Return hidden states (no logits)    │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 4. AUTOREGRESSIVE DECODING          │
├─────────────────────────────────────┤
│ while not EOS and tokens < max_len: │
│   - Embed last token                │
│   - Apply 64 layers (use cache)     │
│   - Final norm + LM head            │
│   - Sample next token               │
│   - Update position + cache         │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 5. OUTPUT                           │
├─────────────────────────────────────┤
│ Token IDs → Decode to text          │
│ Stop at EOS or max length           │
└─────────────────────────────────────┘
```

---

## 9. CRITICAL IMPLEMENTATION DETAILS

### 9.1 Layer Ordering in Block

1. **Pre-Norm**: attn_norm (RMSNorm)
2. **Attention** (choice based on is_recurrent):
   - Full: Q/K/V projection, RoPE, attention, gate, output projection
   - SSM: Conv, delta net, gated norm, output projection
3. **Residual**: cur += inpSA
4. **Pre-FFN Norm**: attn_post_norm (RMSNorm)
5. **FFN**: up → SiLU + gate → down
6. **Residual**: cur += ffn_residual

### 9.2 Causal Masking

- Applied in softmax step for full attention only
- Implemented as -inf mask for future positions
- Prevents information leakage from future tokens

### 9.3 State Persistence

**Prefill Phase**:
- Build complete KV caches
- Accumulate recurrent states through layers
- Save final states

**Generation Phase**:
- Load previous cache/states
- Append new KV values
- Update recurrent states
- Discard old cache entries beyond context window

### 9.4 Batch Handling

- Support multiple sequences in parallel
- Each sequence maintains separate position and cache
- Padding for variable lengths (with masking)

---

## 10. RESOURCE UTILIZATION

### Memory Requirements
- **Model**: ~70-80 GB (int8 quantized from fp32)
- **KV Cache**: ~n_layer * 2 * n_tokens * (2 * n_head_kv * n_embd_head) * 2 bytes
  - Example for n_tokens=2048: ~100 MB per full attention layer
  - SSM layers use recurrent state: much smaller
- **Activation**: ~2-4 GB for intermediate tensors

### Compute Patterns
- **Prefill**: Memory-bound (large batch, cache write)
- **Generation**: Compute-bound (small batch, cache read)
- **Attention**: O(n²) for full attention, O(n) for SSM (per chunk)

---

## References

- llama.cpp repository: `/c/Users/arahe/llama.cpp/src/`
- Key files:
  - `models/qwen35.cpp`: Qwen3.5 architecture
  - `models/delta-net-base.cpp`: SSM/delta net implementation
  - `llama-graph.cpp`: Attention and FFN kernels
  - `llama-hparams.cpp`: Parameter management
  - `llama-vocab.cpp`: Tokenization
