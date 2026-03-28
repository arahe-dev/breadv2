# llama.cpp Qwen35MoE SSM Deep Mapping Investigation

Date: 2026-03-28
Reference workspace: `C:\Users\arahe\llama.cpp`
Scope: Deep mapping of the Qwen3.5 MoE recurrent / SSM workflow in `llama.cpp`, with emphasis on variable meanings, tensor shapes, modifiers, and state transitions.

## Sources Read

- `C:\Users\arahe\llama.cpp\src\models\qwen35moe.cpp`
- `C:\Users\arahe\llama.cpp\src\models\delta-net-base.cpp`

## Investigation Posture

- Treat `llama.cpp` as the semantic reference for the Qwen35MoE recurrent path.
- Record the flow and variable contract before comparing it to BREAD.
- Note conditional behavior, especially around fused vs non-fused delta-net execution.

## Layer Wrapper Structure

Inside `llm_build_qwen35moe`, each layer follows this pattern:

1. Pre-attention RMSNorm
2. Branch:
   - recurrent linear attention via `build_layer_attn_linear()`
   - full attention via `build_layer_attn()`
3. Residual add
4. Post-attention RMSNorm
5. MoE FFN
6. FFN residual add

For recurrent layers, `build_layer_attn_linear()` is the relevant entry point.

## Recurrent Hyperparameter Contract

Inside `build_layer_attn_linear()` the recurrent branch derives:

- `d_inner = hparams.ssm_d_inner`
- `head_k_dim = hparams.ssm_d_state`
- `num_k_heads = hparams.ssm_n_group`
- `num_v_heads = hparams.ssm_dt_rank`
- `head_v_dim = d_inner / num_v_heads`
- `n_seq_tokens = ubatch.n_seq_tokens`
- `n_seqs = ubatch.n_seqs`

Observed implied contract:

- q and k live on `num_k_heads`
- v, beta, gate, and recurrent state live on `num_v_heads`
- the recurrent state is shaped by `head_v_dim x head_v_dim x num_v_heads`

## Recurrent Input Projections

The branch begins with these projections:

- `qkv_mixed = wqkv(cur)`
- `z = wqkv_gate(cur)`
- `beta = sigmoid(ssm_beta(cur))`
- `alpha = ssm_alpha(cur)`

Then:

- `alpha_biased = alpha + ssm_dt`
- `alpha_softplus = softplus(alpha_biased)`
- `gate = alpha_softplus * ssm_a`

Observed meaning split:

- `qkv_mixed`
  - shared projection that later feeds conv and then splits into q, k, v
- `z`
  - later used by gated normalization
- `beta`
  - sigmoid-applied per-v-head scalar tensor
- `gate`
  - exp-applied recurrent decay/update control tensor

## Recurrent State Inputs

Two persistent state banks are pulled from recurrent memory:

- `conv_states_all = mctx_cur->get_r_l(il)`
- `ssm_states_all = mctx_cur->get_s_l(il)`

They are then reshaped into:

- `conv_states`
  - shape: `[conv_kernel_size - 1, conv_channels, n_seqs]`
- `state`
  - shape: `[head_v_dim, head_v_dim, num_v_heads, n_seqs]`

Where:

- `conv_kernel_size = ssm_conv1d.ne[0]`
- `conv_channels = d_inner + 2 * ssm_n_group * ssm_d_state`

## Convolution Preparation And State Update

Observed preprocessing:

1. `qkv_mixed` is transposed
2. `conv_input = concat(conv_states, qkv_mixed, axis=0)`
3. The last `conv_kernel_size - 1` time steps of `conv_input` are copied back into recurrent conv cache

This means the conv history cache is updated before the convolved output is consumed by the delta-net path.

## Convolution And Activation

The current-token recurrent content path is:

1. `conv_output_raw = ggml_ssm_conv(conv_input, conv_kernel)`
2. `conv_output_silu = silu(conv_output_raw)`

The branch uses the SiLU-applied output for the downstream q/k/v split.

## Post-Conv Q/K/V Split

`conv_output_silu` is treated as one packed buffer and then viewed into:

- `q_conv`
  - `[head_k_dim, num_k_heads, n_seq_tokens, n_seqs]`
- `k_conv`
  - `[head_k_dim, num_k_heads, n_seq_tokens, n_seqs]`
- `v_conv`
  - `[head_v_dim, num_v_heads, n_seq_tokens, n_seqs]`

The packed dimension is:

- `qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads`

## Post-Conv Normalization And Head Broadcast

Observed operations:

- `q_conv = l2_norm(q_conv, eps_norm)`
- `k_conv = l2_norm(k_conv, eps_norm)`

Then, when:

- `num_k_heads != num_v_heads`
- and fused GDN is not active

`llama.cpp` explicitly repeats q and k to the v-head count:

- `q_conv = ggml_repeat_4d(q_conv, ..., num_v_heads, ...)`
- `k_conv = ggml_repeat_4d(k_conv, ..., num_v_heads, ...)`

This creates matching shapes for the downstream delta-net operator.

## Delta-Net Dispatch Conditions

The recurrent output is built through:

- `build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il)`

The important mode split observed in `delta-net-base.cpp`:

- autoregressive path for `n_tokens == 1`
- fused path when fused kernels are available
- non-fused / explicit tensor-graph path otherwise

The audit target for single-token generation is `build_delta_net_autoregressive()`.

## Autoregressive Delta-Net Workflow

For `n_tokens == 1`, the reference path does the following:

1. Assert tensor compatibility:
   - `S_k == S_v`
   - `H_v % H_k == 0`
2. Compute:
   - `scale = 1 / sqrt(S_k)`
3. Apply:
   - `q = scale(q)`
4. Permute:
   - q, k, v into shapes suitable for rowwise recurrent math
5. Reshape:
   - `g` into `[1 or S_v, 1 or S_v, H_v, n_seqs]`
   - `b` into `[1, 1, H_v, n_seqs]`
6. Decay state:
   - `g = exp(g)`
   - `s = s * g`
7. Compute memory summary:
   - `sk = sum_rows(s * k)`
8. Compute update delta:
   - `d = (v - transpose(sk)) * b`
9. Broadcast k over state shape:
   - `k = repeat(k, s)`
10. Form update:
    - `kd = k * transpose(d)`
11. Update state:
    - `s = s + kd`
12. Read out:
    - `s_q = s * q`
    - `o = sum_rows(s_q)`
13. Permute output back to:
    - `[S_v, H_v, n_tokens, n_seqs]`

## Post Delta-Net Processing

After delta-net returns:

- `output`
- `new_state`

the runtime:

1. writes `new_state` back into recurrent memory
2. reshapes `z` to align with the output head structure
3. applies gated norm:
   - `normalized = RMSNorm(output, ssm_norm)`
   - `gated = normalized * SiLU(z)`
4. reshapes to flattened hidden-channel order
5. projects through `ssm_out`

## Inputs / Outputs / Modifiers Summary

| Item | Observed Role In llama.cpp |
| --- | --- |
| `cur` | recurrent branch input after pre-attn RMSNorm |
| `wqkv` | packed recurrent projection feeding conv |
| `wqkv_gate` | projection producing `z` for gated normalization |
| `ssm_beta` | per-v-head beta logits, sigmoid-applied |
| `ssm_alpha` | per-v-head alpha logits |
| `ssm_dt` | additive learned offset applied to alpha before softplus |
| `ssm_a` | multiplicative learned factor applied after softplus |
| `ssm_conv1d` | depthwise recurrent conv kernel |
| `conv_states_all` | persistent conv history cache |
| `ssm_states_all` | persistent recurrent state cache |
| `q_conv` / `k_conv` | L2-normalized post-conv tensors on k-head geometry |
| `v_conv` | post-conv value tensor on v-head geometry |
| `gate` | pre-exp recurrent decay/update control |
| `beta` | per-v-head update gate |
| `build_delta_net_*` | recurrent state decay, correction, update, readout |
| `ssm_norm` + `z` | gated normalization stage |
| `ssm_out` | projection back to hidden dimension |

## Modifiers And Conditions

The recurrent branch behavior is modified by:

- layer type: recurrent vs full-attention
- `n_tokens == 1` selecting autoregressive path
- fused GDN availability
- `num_k_heads != num_v_heads` triggering explicit repeat when not fused

Observed important modifiers:

- `ssm_dt` and `ssm_a` alter the recurrent gate before `exp()`
- `beta` is sigmoid-applied before the delta-net update
- `z` is only applied at the gated norm stage, not inside the recurrent update

## Observations Recorded Without Comparison

- The recurrent branch has two distinct state systems:
  - conv history
  - delta-net recurrent state
- The conv output is activated with SiLU before q/k/v are split
- q and k are explicitly L2-normalized
- q receives a single `1 / sqrt(S_k)` scale in the autoregressive path
- head broadcast to v-head count is explicit in the non-fused path
- gated normalization is `RMSNorm(output) * SiLU(z)`

## Open Questions Left Intentionally Open

- Which exact fused/non-fused path is active in the most comparable reference runtime for the same prompt
- Whether the recurrent state update is numerically sensitive to layout details beyond the symbolic tensor graph
- Whether additional upstream tokenizer or prompt-format parity is needed before a one-to-one numeric trace becomes fully comparable
