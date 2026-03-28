# BREAD Inference Engine Flowchart Analysis

## Overview

BREAD is a custom CUDA/C inference engine for Qwen3.5-35B-A3B (MoE model). This document traces the exact computational flow from prompt input through token generation, including all data transformations, tensor operations, and layer-specific logic.

---

## 1. Initialization & Model Loading

```
┌─────────────────────────────────────────┐
│ main.cu: main()                         │
│ Entry point, parse CLI args             │
│ Args: --prompt, --tokens, --debug,      │
│       --minimal, --model                │
└─────────────────────┬───────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
    [Load Model]             [Load Tokenizer]
    loader_init()            tokenizer_load()
        │                           │
        ├─ Read GGUF header         ├─ Load vocab
        ├─ Parse tensor table       ├─ Parse BPE merges
        ├─ Allocate 23.87 GB        └─ Get special tokens
        │  host RAM (pinned)           (BOS=-1, EOS=248046)
        └─ Stream all 1959
           tensors from disk

RESULT: loader_t *L, gguf_ctx_t *g, tokenizer_t *tok
```

**Key parameters from GGUF metadata:**
- `hidden_dim = 2048`
- `num_layers = 40`
- `vocab_size = 248320`
- `num_q_heads = 16`
- `num_kv_heads = 2`
- `full_attention_interval = 4` (every 4th layer is full attention)
- `rope_sections = [11, 11, 10]` (RoPE dimension split across 3 sections)

---

## 2. Tokenization & Prompt Encoding

```
┌─────────────────────────────────────────┐
│ main.cu: format_prompt_for_model()      │
│ Qwen3.5 specific wrapper                │
└─────────────────────┬───────────────────┘
                      │
Input: "The capital of France is"
        │
        ▼
┌──────────────────────────────────────────────┐
│ QWEN3.5 PROMPT WRAPPING                      │
│ <|im_start|>user                             │
│ The capital of France is<|im_end|>           │
│ <|im_start|>assistant                        │
│ <think>                                      │
│                                              │
│ </think>                                     │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │ tokenizer_encode()          │
        │ BPE tokenization            │
        └─────────────────────┬───────┘
                              │
RESULT: token_buf[0..38] = [27, 91, 316, 62, 2388, 91, 29, 846, ...]
        n_prompt = 39 tokens
        eos_token = 248046
```

---

## 3. Prompt Prefill Loop

```
┌────────────────────────────────────────────────────┐
│ main.cu: Prompt prefill (lines 477-485)           │
│ FOR EACH of 39 prompt tokens:                      │
└─────────────────┬────────────────────────────────┬─┘
                  │                                  │
         FOR p = 0 TO n_prompt-1:              FOR each token p:
                  │                                  │
                  ▼                                  ▼
         ┌──────────────────────┐         ┌──────────────────────┐
         │ embed_token(          │         │ FOR layer = 0 TO 39: │
         │   token_buf[p],       │         │   one_layer_forward()│
         │   ...                 │         │ CUDA Synchronize     │
         │ )                     │         └──────────────────────┘
         └──────────┬───────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
    [Token Embedding]      [Transform Through]
    Load row from           All 40 Layers
    token_embd.weight       (details below)
    Q4_K dequant to fp16
        │
        ▼
    d_hidden ← [2048] half
    (on VRAM)
```

**Prefill timing (39 tokens):** ~13-20 seconds (356 ms/token)

---

## 4. One Layer Forward Pass (The Core Computation)

This is where the bug likely is. For each of 40 layers:

### Layer Dispatch

```
one_layer_forward(d_hidden, layer_idx, pos, L, g, stream_a)

INPUT:  d_hidden [2048] half on VRAM (from prev layer or embedding)
OUTPUT: d_hidden [2048] half on VRAM (modified in-place)

┌────────────────────────────────────────────────────┐
│ Check layer type:                                  │
│   is_full = (layer_idx % 4 == 3) ?                │
│   → Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39   │
│     are FULL ATTENTION (10 layers)                 │
│   → All others are SSM/GatedDeltaNet (30 layers)   │
└─────────────┬──────────────────────┬───────────────┘
              │ is_full == True       │ is_full == False
              │                       │
              ▼                       ▼
        ┌──────────────┐        ┌──────────────┐
        │ FULL ATTN    │        │ SSM/GATED    │
        │ PATH         │        │ DELTANET     │
        │ (10 layers)  │        │ PATH         │
        └──────┬───────┘        │ (30 layers)  │
               │                └──────┬───────┘
               │                       │
               └───────────┬───────────┘
                           │
           ┌───────────────┴────────────────┐
           │                                │
           ▼                                ▼
    ┌─────────────────────────┐   ┌─────────────────────────┐
    │ PRE-ATTENTION RMSNORM   │   │ PRE-SSM RMSNORM         │
    │ h_normed = RMSNorm(     │   │ h_normed = RMSNorm(     │
    │   d_hidden,             │   │   d_hidden,             │
    │   attn_norm.weight      │   │   attn_norm.weight      │
    │ )                       │   │ )                       │
    └──────────┬──────────────┘   └──────────┬──────────────┘
               │                              │
               ▼                              ▼
        ┌──────────────────────────┐  ┌──────────────────────┐
        │ FULL ATTENTION COMPUTE   │  │ SSM/GATED DELTANET   │
        │ (Details in 4.1)         │  │ (Details in 4.2)     │
        └──────────┬───────────────┘  └──────────┬───────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────┴───────────────┐
                   │                              │
                   ▼                              ▼
            ┌─────────────────────────────┐
            │ POST-ATTENTION/SSM RMSNORM  │
            │ h_normed2 = RMSNorm(        │
            │   h_hidden + attn/ssm_out,  │
            │   mlp_norm.weight           │
            │ )                           │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────────────────────────┐
            │ MoE EXPERT ROUTING & FFN (on all layers)        │
            │ (Details in 4.3)                                │
            └──────────────┬──────────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ RESIDUAL CONNECTION          │
            │ h_hidden = d_hidden +        │
            │   (attn_out | ssm_out) +     │
            │   shared_expert_out +        │
            │   moe_experts_out            │
            └──────────────┬───────────────┘
                           │
                           ▼
        RESULT: d_hidden [2048] half (modified)
```

---

### 4.1 Full Attention Path (10 layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)

```
INPUT: h_normed [2048] float (after attn_norm)

┌────────────────────────────────────┐
│ PROJECT Q, K, V                    │
├────────────────────────────────────┤
│ Q: h_normed [2048]                 │
│    @matmul attn_q.weight [8192×2048]
│    → h_q_full [8192]               │
│                                    │
│ K: h_normed [2048]                 │
│    @matmul attn_k.weight [512×2048]│
│    → h_kv_k [512]                  │
│                                    │
│ V: h_normed [2048]                 │
│    @matmul attn_v.weight [512×2048]│
│    → h_kv_v [512]                  │
└─────────────────┬──────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
    SPLIT & NORM       K/V NORM
    Q: [8192] →        Per-head RMSNorm
       heads_per_kv=8:     k_norm.weight
       [16, 512]           v_norm.weight
       extract:
       q_score: [16, 256]  →  h_kv_k [512]
       q_gate:  [16, 256]     h_kv_v [512]
                │              │
                ▼              ▼
                RMSNorm    Per-head RMSNorm
                q_norm.weight    k_norm.weight
                               v_norm.weight
                │              │
                ▼              ▼
            ┌─────────────────────────────────────┐
            │ APPLY ROTARY EMBEDDINGS (RoPE)      │
            │ θ = 10000^(-2i/64)                  │
            │ rope_sections=[11,11,10] → dim=32   │
            │ rope_base = 1e7                     │
            │ rope_interleaved = True             │
            │                                     │
            │ For each head h:                    │
            │   For pos in context:               │
            │     q_score[h] ⊗ rotate(pos)       │
            │     k_cache[h] ⊗ rotate(pos)       │
            └────────────┬────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────────────┐
            │ STORE IN KV CACHE (host RAM)     │
            │ kv_k_cache[layer][pos] ← h_kv_k │
            │ kv_v_cache[layer][pos] ← h_kv_v │
            │ Cache capacity: 8192 positions   │
            └────────────┬─────────────────────┘
                         │
                         ▼
            ┌───────────────────────────────────────┐
            │ ATTENTION SCORES & SOFTMAX            │
            │ FOR each query head h:                │
            │   FOR each cached position p:         │
            │     score[p] = dot(q[h], k[p]) /     │
            │                 sqrt(head_dim_qk)     │
            │   softmax(score) → attn_weights       │
            │                                       │
            │ FOR each output position h:           │
            │   FOR each cached position p:         │
            │     out[h] += attn_weights[p] *       │
            │               v_cache[p]              │
            │   Multiply by sigmoid gate            │
            └────────────┬────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────────────┐
            │ O_PROJ                           │
            │ h_attn_out [4096] ←              │
            │   @matmul attn_output.weight     │
            │   h_o_cpu [2048]                 │
            └────────────┬─────────────────────┘
                         │
            RESULT: h_o_cpu [2048] float
            (attention output, added to residual)
```

---

### 4.2 SSM/GatedDeltaNet Path (30 layers: all except 3,7,11,15,19,23,27,31,35,39)

```
INPUT: h_normed [2048] float (after attn_norm)
NOTE: This is a STUB/PLACEHOLDER in current BREAD
      SSM output is set to zero, not actually computed

┌────────────────────────────────────────────┐
│ SSM CONFIG                                 │
│ num_k_heads = 16                           │
│ num_v_heads = 32                           │
│ head_dim = 128                             │
│ ssm_qkv_dim = 8192                         │
│ ssm_z_dim = 4096                           │
│ conv_kernel = 4                            │
│ state_size = 128 × 128 × 32 (per layer)    │
│ conv_state_size = 3 × 8192 (history)       │
└────────────────────────────────────────────┘

[APPROXIMATION IN CURRENT BREAD]
ssm_out ≈ 0 (not fully implemented)

What SHOULD happen (reference):
1. Project to QKV [8192×3]
2. Causal convolution over context (kernel=4)
3. Scan (sequential state update) A, B, C matrices
4. Project Z gate
5. Multiply by Z gate (SiLU)
6. Project to hidden [2048]

This is where the ~2x divergence vs ollama likely originates.
```

---

### 4.3 Shared Expert + MoE Routing (ALL 40 layers)

```
┌──────────────────────────────────────────────────┐
│ SHARED EXPERT FORWARD                            │
│ (Simple SwiGLU, not routed)                      │
└────────────┬─────────────────────────────────────┘
             │
INPUT: h_normed2 [2048] float

│            ├─ SHARED GATE: h_normed2 @matmul      │
             │              ffn_gate_shexp.weight    │
             │              [512×2048] → h_sg [512] │
             │                                      │
             ├─ SHARED UP: h_normed2 @matmul        │
             │             ffn_up_shexp.weight      │
             │             [512×2048] → h_su [512]  │
             │                                      │
             ├─ SILU(h_sg) * h_su (element-wise)    │
             │                                      │
             └─ SHARED DOWN: result @matmul         │
                            ffn_down_shexp.weight   │
                            [2048×512]              │
                            × sigmoid(shared_score) │
                            → d_sh_out [2048]       │
                            (weighted with gate)    │

            RESULT: d_sh_out [2048] (shared expert contribution)
```

```
┌──────────────────────────────────────────────────┐
│ ROUTER: Select top-K experts per token          │
└────────────┬─────────────────────────────────────┘
             │
INPUT: h_normed2 [2048] float

│            ├─ ROUTER: h_normed2 @matmul          │
             │           ffn_router.weight          │
             │           [256×2048] → logits [256]  │
             │                                      │
             ├─ top_k=8: argmax-8 logits            │
             │           → expert_indices [8]       │
             │                                      │
             └─ Softmax(selected_logits)            │
                        → expert_weights [8]

            RESULT: expert_indices [8], weights [8]
            (which experts to use + how much)
```

```
┌──────────────────────────────────────────────────┐
│ TOP-K EXPERT FORWARD (K=8 active experts)       │
└────────────┬─────────────────────────────────────┘
             │
FOR each expert_idx in expert_indices:
    │
    ├─ Load expert tensors (host RAM or VRAM cache)
    │  ffn_gate_exps.weight[expert_idx]
    │  ffn_up_exps.weight[expert_idx]
    │  ffn_down_exps.weight[expert_idx]
    │
    ├─ EXPERT GATE: h_normed2 @matmul
    │              ffn_gate_exps.weight[expert_idx]
    │              [512×2048] → h_eg_cpu [512]
    │
    ├─ EXPERT UP: h_normed2 @matmul
    │            ffn_up_exps.weight[expert_idx]
    │            [512×2048] → h_eu_cpu [512]
    │
    ├─ SILU(h_eg) * h_eu (element-wise)
    │
    ├─ EXPERT DOWN: result @matmul
    │              ffn_down_exps.weight[expert_idx]
    │              [2048×512] → h_eo_cpu [2048]
    │
    └─ ACCUMULATE with weight:
       h_hidden += expert_weights[k] * h_eo_cpu

RESULT: h_hidden += sum of 8 weighted expert outputs
```

---

## 5. Output Norm & Logits

```
┌──────────────────────────────────────────────┐
│ After all 40 layers:                         │
│ d_hidden [2048] half (final hidden state)    │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ OUTPUT RMSNORM                 │
    │ h_normed = RMSNorm(            │
    │   d_hidden,                    │
    │   output_norm.weight           │
    │ )                              │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ LM HEAD: Logits                │
    │ logits [248320] ←              │
    │   h_normed @matmul output.weight
    │   (tied with token_embd)       │
    │   Q6_K quantized [248320×2048] │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ ARGMAX SAMPLING                │
    │ token_id = argmax(logits)      │
    │ (greedy, no temperature)       │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ TOKENIZER DECODE               │
    │ text = tokenizer_decode(       │
    │   token_id                     │
    │ )                              │
    └────────────────────────────────┘
```

---

## 6. Generation Loop (Autoregressive)

```
┌──────────────────────────────────────────────┐
│ WHILE n_gen < max_tokens AND token ≠ EOS:   │
│   FOR each generated token:                  │
└────────────┬────────────────────────────────┘
             │
    ┌────────┴──────────┐
    │                   │
    ▼                   ▼
Embed token         Run through
(same as            40 layers
prefill)            (same as
    │               prefill)
    │                   │
    └────────┬──────────┘
             │
             ▼
    Apply output norm
    + lm_head → logits
             │
             ▼
    Argmax sample → next_token
             │
             ▼
    Print token
    n_gen++
```

---

## 7. Minimal Mode (Debugging Path)

When `--minimal` flag is used:

```
KEY DIFFERENCE:
- Normal mode: All computation on CUDA (GPU fp16)
- Minimal mode: CPU float computation, only sync to VRAM on layer 39

FLOW:
┌────────────────────────────────────────────┐
│ one_layer_forward() with bread_boring_mode │
└────────────┬───────────────────────────────┘
             │
             ├─ Copy d_hidden (VRAM fp16)
             │  → h_hidden (CPU fp32) [once per pos]
             │
             ├─ All layer compute on CPU:
             │  - RMSNorm (float)
             │  - Q/K/V projections (float matvec)
             │  - Attention (float)
             │  - Softmax (float)
             │  - MoE routing/experts (float)
             │
             └─ Only copy back to VRAM
                if layer == 39 (last layer)

RESULT: Same algorithm, different precision
        → detects if GPU quantization is the bug
        → but both modes output garbage (same first token)
```

---

## 8. Current RMS Observations

When run with `--debug` flag:

### Normal Mode (GPU path)
```
Embed  : rms=0.010701
Layer  0: rms=0.144010
Layer  1: rms=0.206197
...
Layer 39: rms=0.545473
Output: "人 Coldicheyer违反"
```

### Minimal Mode (CPU path)
```
Embed  : rms=0.013345
Layer  0: rms=0.072764
Layer  1: rms=0.102676
...
Layer 39: rms=0.913563
Output: "关于..."
```

**Observation:** Both differ from layer 0, suggesting embedding differences (fp16 vs fp32 upcast), but the real divergence from correct values happens in the layers themselves.

---

## 9. Known Issues & Suspects

### High Confidence Bugs

1. **SSM/GatedDeltaNet (30 layers)** — STUB implementation
   - Current code sets output to ~0
   - Should compute full causal scan with state matrices
   - This alone accounts for ~30/40 of the model being garbage

2. **RoPE Implementation** — May not match Qwen35 spec
   - rope_sections = [11, 11, 10] (non-standard split)
   - rope_interleaved = True (applies to first 32 dims only)
   - RoPE frequency base = 1e7
   - Dimension only 32 (not full 64)

### Medium Confidence
3. **Expert routing logic** — Could have subtle bugs
4. **Shared expert scaling** — Sigmoid gate application
5. **Residual connection math** — Potential precision issues

### Lower Confidence
6. **Quantization boundaries** (fp16↔Q4K/Q6K)
7. **Tokenization** (verified working)
8. **Final layer norm** (verified in isolation)

---

## 10. What Needs Comparison vs Reference

To isolate the bug, we need per-layer RMS from **ollama/llama.cpp**:

```
Layer -1 (Embedding): Should be ~0.01-0.015 (RMS of init embedding)
Layer  0 (SSM):       Should diverge from current ~0.07-0.14
Layer  3 (Attention): Should show different pattern
...
Layer 39 (Final):     Should be ~0.5-0.6 typically
```

Once we have reference RMS values, we can:
1. Find exact layer of first divergence
2. Determine if it's SSM, attention, or shared expert bug
3. Fix that layer in isolation
4. Verify it matches reference

---

## Summary

**BREAD's inference pipeline is complete but produces garbage output because:**

1. SSM path is stubbed (30/40 layers do nothing)
2. RoPE may not match Qwen35's non-standard layout
3. Expert routing/combining may have subtle math bugs

**Next step:** Compare per-layer RMS against a known-good reference (ollama/llama.cpp) to find the exact divergence point.
