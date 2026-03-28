# BREAD SSM Workflow Map Investigation

Date: 2026-03-28
Workspace: `C:\bread_v2`
Scope: Current BREAD workflow only. This document maps what is loaded where, what the main recurrent-layer inputs and outputs are, and how data flows through the runtime. It is intentionally not a correctness judgement.

## Sources Read

- `C:\bread_v2\main.cu`
- `C:\bread_v2\bread.c`
- `C:\bread_v2\one_layer.cu`
- `C:\bread_v2\loader.c`
- `C:\bread_v2\tokenizer.c`

## Investigation Posture

- Treat BREAD as a black-box-to-gray-box system.
- Focus first on control flow, state residency, inputs, outputs, and modifiers.
- Avoid deciding whether a step is correct until the dedicated comparison document.

## Entry Points And Runtime Controls

The executable entry point is `main()` in `main.cu`.

Relevant CLI modifiers observed:

- `--prompt "..."`
- `--tokens N`
- `--model PATH`
- `--minimal`
- `--boring`
- `--debug`
- `--force-ssm-zero`
- `--disable-rope`

These flags feed into global runtime switches via `bread_set_*()` in `bread.c`.

Observed switch roles:

- `--minimal` / `--boring`
  - chooses a CPU-float-heavy minimal core path for layer math
- `--debug`
  - enables trace output such as per-layer hidden RMS, branch RMS, and top logits
- `--force-ssm-zero`
  - forces the SSM branch output to zero after the recurrent computation finishes
- `--disable-rope`
  - disables the RoPE helper on the attention side

## Model Contract And Configuration Flow

Configuration is assembled in two stages in `bread.c`:

1. Defaults from `bread_cfg_defaults()`
2. Overrides from GGUF metadata and tensor-shape discovery in `bread_model_config_init()`

Observed GGUF-derived fields used by the current runtime:

- `qwen35moe.block_count`
- `qwen35moe.embedding_length`
- `qwen35moe.attention.head_count`
- `qwen35moe.attention.head_count_kv`
- `qwen35moe.attention.key_length`
- `qwen35moe.attention.value_length`
- `qwen35moe.full_attention_interval`
- `qwen35moe.expert_feed_forward_length`
- `qwen35moe.expert_shared_feed_forward_length`
- `qwen35moe.expert_count`
- `qwen35moe.expert_used_count`
- `qwen35moe.rope.dimension_count`
- `qwen35moe.rope.dimension_sections`
- `qwen35moe.attention.layer_norm_rms_epsilon`
- `qwen35moe.rope.freq_base`
- `qwen35moe.ssm.conv_kernel`
- `qwen35moe.ssm.inner_size`
- `qwen35moe.ssm.state_size`
- `qwen35moe.ssm.time_step_rank`
- `qwen35moe.ssm.group_count`
- `qwen35moe.ssm.v_head_reordered`
- `qwen35moe.rope.mrope_interleaved`

Observed derived SSM fields:

- `ssm_head_dim = ssm_state_size`
- `ssm_num_k_heads = ssm_group_count`
- `ssm_num_v_heads = ssm_time_step_rank`
- `ssm_z_dim = ssm_inner_size`
- `ssm_qkv_dim = ssm_z_dim + 2 * ssm_num_k_heads * ssm_head_dim`

Layer typing is determined by:

- `bread_layer_is_recurrent(layer_idx)`
- `bread_layer_is_full_attention(layer_idx)`

The decision comes from `full_attention_interval`, not from a hardcoded modulus inside `one_layer_forward()`.

## Data Residency Map

### Model Blob

The full GGUF blob is loaded by `loader_init()` into:

- `L->pinned_data`

Despite the name, it is currently normal host RAM allocated by `malloc()`, not a giant `cudaMallocHost()` allocation.

### Expert Cache

The loader allocates:

- 18 expert slots in VRAM
- each slot stores `[gate | up | down]` bytes for one expert

Expert base offsets and slice sizes are computed from GGUF tensor metadata.

### Hidden State

Per-token hidden state lives in:

- `d_hidden` on device
- mirrored into `g_h_hidden` host float scratch in minimal mode

### Recurrent State

Per recurrent layer, the following host-side persistent states are allocated on first use in `one_layer_forward()`:

- `ssm_conv_state[layer]`
  - size: `(ssm_conv_kernel - 1) * ssm_qkv_dim`
- `ssm_state[layer]`
  - size: `ssm_num_v_heads * ssm_head_dim * ssm_head_dim`

### Attention Cache

Per full-attention layer, host-side persistent cache arrays are allocated:

- `kv_k_cache[layer]`
- `kv_v_cache[layer]`
- `kv_cache_len[layer]`

### Reused Scratch

`one_layer_forward()` allocates and reuses a mixed set of device-half and host-float buffers:

- device buffers for hidden, projections, and FFN scratch
- host float buffers for CPU inspection and CPU-side layer math
- separate recurrent buffers for:
  - `h_qkv`
  - `h_z`
  - `h_alpha`
  - `h_beta`
  - `h_conv_out`
  - `h_attn_out`
  - `h_head_tmp`

## High-Level Per-Token Flow

Observed token workflow in `main.cu`:

1. Load model with `loader_init()`
2. Parse GGUF for config with `bread_model_config_init()`
3. Load tokenizer from model metadata
4. Upload `output_norm.weight` and `output.weight` to VRAM
5. Format prompt with `format_prompt_for_model()`
6. Tokenize prompt
7. For each prompt token:
   - embed token with `embed_token()`
   - run `one_layer_forward()` for all layers
8. Apply output norm
9. Compute logits
10. Greedy sample next token
11. Repeat generation loop for requested token count

## BREAD Layer Skeleton

Each layer in `one_layer_forward()` follows this broad structure:

1. Pre-attention RMSNorm
2. Branch:
   - full attention
   - recurrent SSM
3. Branch output projection back to hidden size
4. Residual add into hidden
5. Post-attention RMSNorm
6. Shared-expert FFN path
7. MoE router and top-k expert selection
8. Selected expert FFN accumulation
9. Residual accumulation remains in hidden for next layer

## Recurrent Layer Workflow In BREAD

### Shared Inputs To The Recurrent Branch

Input to the recurrent branch:

- normalized hidden vector `h_normed` or device equivalent

Projection tensors read or projected:

- `blk.%d.attn_qkv.weight`
- `blk.%d.attn_gate.weight`
- `blk.%d.ssm_alpha.weight`
- `blk.%d.ssm_beta.weight`
- `blk.%d.ssm_conv1d.weight`
- `blk.%d.ssm_a`
- `blk.%d.ssm_dt`
- `blk.%d.ssm_norm.weight`
- `blk.%d.ssm_out.weight`

### Minimal Mode Recurrent Flow

Observed minimal-mode recurrent flow:

1. Project `h_normed` into:
   - `h_qkv`
   - `h_z`
   - `h_alpha`
   - `h_beta`
2. Fetch scalar/tensor parameters from host RAM:
   - `conv_w`
   - `ssm_a`
   - `ssm_dt`
   - `ssm_norm_w`
3. Read persistent states:
   - `conv_state`
   - `layer_state`
4. Run `cpu_conv1d_step(conv_state, h_qkv, conv_w, h_conv_out, ...)`
5. Shift and update `conv_state`
6. Apply `cpu_silu_inplace(h_conv_out, ssm_qkv_dim)`
7. Split convolved buffer into:
   - `lin_q`
   - `lin_k`
   - `lin_v`
8. Normalize q and k with `cpu_l2_norm_bare`
9. Scale q by `1 / sqrt(key_dim)`
10. For each v-head:
    - choose a k-head via `ssm_k_head_for_v_head()`
    - compute `gate = softplus(alpha + dt) * ssm_a`
    - compute `decay = exp(gate)`
    - compute `beta_gate = sigmoid(beta)`
    - decay each row of the per-head state matrix
    - compute a rowwise memory dot against `k_h`
    - compute `delta = (v_h - kv_mem) * beta_gate`
    - update the state row with `delta * k_h`
    - read out `out = dot(row, q_h)`
    - apply `cpu_gated_rms_norm(out, z_slice, ssm_norm_w, ...)`
11. Project concatenated recurrent output with `ssm_out.weight`
12. Optionally zero the branch output if `--force-ssm-zero` is active

### Orchestrated Mode Recurrent Flow

Observed orchestrated-mode recurrent flow:

1. Pre-attn norm and major projections happen through GPU matvecs into half buffers
2. Projection outputs are copied back to host floats
3. The conv, recurrent state update, gated norm, and final recurrent output assembly happen on CPU
4. The recurrent output is copied back to device half
5. `ssm_out.weight` projection happens via GPU matvec

So the orchestrated path and minimal path share the same recurrent semantic structure, but differ in where matrix-vector products run and when values cross the CPU/GPU boundary.

## Inputs / Outputs / Modifiers Summary

| Item | Observed In BREAD | Notes |
| --- | --- | --- |
| Main recurrent input | Pre-attn RMS-normalized hidden | `h_normed` in minimal mode |
| Persistent recurrent state | `ssm_state[layer]` | Host float, one matrix per v-head |
| Persistent conv history | `ssm_conv_state[layer]` | Host float sliding window |
| Projection outputs | `qkv`, `z`, `alpha`, `beta` | Derived from current hidden |
| Static modifiers | `ssm_a`, `ssm_dt`, `ssm_norm.weight`, `ssm_conv1d.weight` | Read from model blob |
| Runtime modifiers | `--minimal`, `--debug`, `--force-ssm-zero`, `--disable-rope` | Affect path or observability |
| Recurrent branch output | `h_o_cpu` or device equivalent | Added back into hidden as residual |
| Final layer output | hidden after MoE residual | Fed into next layer |

## Recent Runtime Observations

Recent first-token observations collected from the current build:

- baseline minimal debug output emitted `ucho`
- `--force-ssm-zero` collapsed toward blank / whitespace-like output
- `--disable-rope` preserved `ucho`

Recent per-layer trace pattern:

- SSM branch RMS remains non-zero across recurrent layers
- many attention-layer branch RMS values stay very small until late layers

These are observations only; interpretation is deferred to later documents.

## Open Questions Left Intentionally Open

- Whether the current recurrent update is mathematically equivalent to the reference implementation
- Whether the state matrix orientation is the same as the reference implementation
- Whether attention weakness is an independent issue or a downstream consequence of recurrent-state drift
- Whether tokenizer / prompt-wrap parity is materially affecting the same test prompts
