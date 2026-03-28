# BREAD Re-Audit Against llama.cpp Qwen35MoE SSM Investigation

Date: 2026-03-28
Workspace: `C:\bread_v2`
Reference: `C:\Users\arahe\llama.cpp`
Scope: Return to BREAD after mapping `llama.cpp`, re-express the BREAD recurrent path in the same semantic slots, and record convergences, divergences, contradictions, agreements, and open questions.

## Sources Used

- `C:\bread_v2\bread_ssm_workflow_map_investigation_2026-03-28.md`
- `C:\bread_v2\llamacpp_qwen35moe_ssm_deepmap_investigation_2026-03-28.md`
- `C:\bread_v2\bread.c`
- `C:\bread_v2\one_layer.cu`
- `C:\Users\arahe\llama.cpp\src\models\qwen35moe.cpp`
- `C:\Users\arahe\llama.cpp\src\models\delta-net-base.cpp`

## Re-Audit Method

- Take the semantic slots identified in `llama.cpp`
- Locate the corresponding BREAD values, buffers, and steps
- Mark each slot as:
  - agreement
  - partial agreement
  - divergence
  - contradiction
  - unresolved

## Stepwise Mapping

| Semantic Slot | llama.cpp Reference Shape / Meaning | BREAD Current Mapping | Relationship | Notes |
| --- | --- | --- | --- | --- |
| Layer selection | recurrent decided by model hparams | recurrent decided by `bread_layer_is_recurrent()` from metadata | agreement | Both use interval-driven layer typing |
| Branch input | pre-attn RMS-normalized hidden | `h_normed` / `d_normed` after `attn_norm.weight` | agreement | Same broad entry point |
| Packed recurrent projection | `wqkv(cur) -> qkv_mixed` | `attn_qkv.weight * h_normed -> h_qkv` | agreement | Same high-level slot |
| Gated norm side projection | `wqkv_gate(cur) -> z` | `attn_gate.weight * h_normed -> h_z` | agreement | Same broad role |
| Beta projection | `ssm_beta(cur)` then sigmoid | `ssm_beta.weight * h_normed -> h_beta`, then `sigmoid` per v-head | agreement | Same role, scalarized in BREAD |
| Alpha projection | `ssm_alpha(cur)` | `ssm_alpha.weight * h_normed -> h_alpha` | agreement | Same role |
| Alpha bias | `alpha + ssm_dt` | `h_alpha[vh] + ssm_dt[vh]` | agreement | Same visible contract |
| Softplus stage | `softplus(alpha + dt)` | `cpu_softplus(h_alpha[vh] + ssm_dt[vh])` | agreement | Same visible contract |
| Recurrent gate factor | `gate = alpha_softplus * ssm_a` | `gate = softplus(alpha + dt) * ssm_a[vh]` | agreement | BREAD now matches the visible formula more closely |
| Conv cache source | recurrent memory bank | `ssm_conv_state[layer_idx]` host float array | partial agreement | Same conceptual slot, different storage/backend |
| State cache source | recurrent memory bank | `ssm_state[layer_idx]` host float array | partial agreement | Same conceptual slot, different storage/backend |
| Conv input assembly | concat(previous conv states, current qkv_mixed) | `cpu_conv1d_step(conv_state, h_qkv, ...)` plus manual shift/writeback | partial agreement | Equivalent intent, implemented implicitly rather than via explicit concatenated tensor |
| Conv activation | `SiLU(conv_output_raw)` | `cpu_silu_inplace(h_conv_out)` | agreement | This was aligned in the recent fix |
| Q split | first packed segment of post-conv buffer | `lin_q = h_conv_out` | agreement | Same segment role |
| K split | second packed segment of post-conv buffer | `lin_k = h_conv_out + num_k_heads * key_dim` | agreement | Same segment role |
| V split | third packed segment of post-conv buffer | `lin_v = h_conv_out + 2 * num_k_heads * key_dim` | agreement | Same segment role |
| Q normalization | L2 norm | `cpu_l2_norm_bare(qh, ...)` | agreement | Recent fix aligned this |
| K normalization | L2 norm | `cpu_l2_norm_bare(kh, ...)` | agreement | Recent fix aligned this |
| Q scaling | `q *= 1 / sqrt(S_k)` | `qh[d] *= 1 / sqrt(key_dim)` | agreement | Same broad scaling contract |
| K scaling | no extra scaling after norm in visible reference path | none after norm | agreement | Recent fix aligned this |
| q/k repeat to v-head count | explicit `ggml_repeat_4d` in non-fused path | `ssm_k_head_for_v_head(vh, num_k_heads, num_v_heads)` | partial agreement | Same broadcast intent, scalarized via index mapping instead of materialized repeat |
| `v_head_reordered` usage | not visibly used in forward code | parsed in config, not currently active in head mapping helper | unresolved | Metadata exists, forward usage is not evident in current reference path |
| State tensor shape | `[head_v_dim, head_v_dim, num_v_heads, n_seqs]` | `num_v_heads * value_dim * key_dim` | agreement | Size and logical slot align for this model where `value_dim == key_dim` |
| State decay | `s = s * exp(g)` | rowwise `row[ki] *= decay` | partial agreement | Same intent, scalar-loop implementation |
| Memory summary | `sk = sum_rows(s * k)` | `kv_mem = dot(row, k_h)` per output row | partial agreement | Similar algebraic role, different formulation |
| Delta computation | `d = (v - transpose(sk)) * beta` | `delta = (v_h[vi] - kv_mem) * beta_gate` | partial agreement | Similar symbolic role, scalarized form |
| State update | `s = s + k * d^T` | `row[ki] += delta * k_h[ki]` | partial agreement | Similar outer-product intent, manual rowwise update |
| Output readout | `o = sum_rows(s * q)` | `out += row[ki] * q_h[ki]` | partial agreement | Same role, scalarized |
| Gated norm | `RMSNorm(output, ssm_norm) * SiLU(z)` | `cpu_gated_rms_norm(h_head_tmp, z_slice, ssm_norm_w, ...)` | agreement | Same visible stage |
| Output projection | `ssm_out(attn_out_norm)` | `ssm_out.weight * h_attn_out` | agreement | Same slot |
| State writeback | recurrent memory updated through graph copy | host arrays mutated in place | partial agreement | Same outcome slot, different mechanism |

## Agreements

The following areas now line up cleanly at the workflow-contract level:

- layer-type dispatch based on model contract
- recurrent input projections
- `alpha + dt -> softplus -> * ssm_a`
- post-conv SiLU
- q/k/v split ordering
- L2 normalization of q and k
- q scaling by `1 / sqrt(key_dim)`
- gated normalization with `SiLU(z)`
- output projection via `ssm_out`

## Partial Agreements

These areas occupy the same semantic role but are implemented differently:

- conv state handling
- q/k repeat vs index-based broadcast
- recurrent state decay
- memory summary computation
- delta computation
- state update
- output readout
- recurrent state writeback

These are not automatically wrong, but they are not literal transcriptions of the reference tensor algebra.

## Divergences

The main remaining divergences observed after the recent fixes are:

1. BREAD still uses an explicit nested-loop formulation instead of the reference tensor-graph formulation for the delta-net update.
2. BREAD keeps the recurrent path split across:
   - GPU matvec
   - host float conversion
   - CPU recurrent math
   - GPU projection again
3. BREAD expresses head repeat semantics through an index helper instead of explicit repeated q/k tensors.

## Contradictions

No direct contradiction remains in the previously corrected areas:

- conv activation is now present
- q/k normalization mode is now L2-based
- q scaling is now single-scale
- `alpha/dt/ssm_a` are now composed in the same visible order

The earlier direct contradictions were reduced by those fixes.

## Unresolved Items

1. Literal numeric equivalence of the scalarized BREAD update with the reference autoregressive tensor algebra
2. Whether the scalarized state orientation matches the reference in all head/value dimensions
3. Whether mixed CPU/GPU boundary effects are materially changing the same symbols even when the symbolic workflow matches
4. Whether weak attention-layer branch RMS is an independent second issue
5. Whether tokenizer / prompt-wrap parity is still affecting the same prompt tests

## Runtime Observations Relevant To The Re-Audit

Recent ablations from the current BREAD build:

- baseline minimal debug first token: `ucho`
- `--force-ssm-zero`: collapses output and logits materially
- `--disable-rope`: leaves first-token behavior close to baseline

These observations make the recurrent branch the primary active subsystem for the investigated prompt.

## Observed Convergence Trend

After the recent recurrent fixes, BREAD appears closer to the reference contract than before in these slots:

- conv activation
- gate construction
- q/k normalization
- q scaling
- q/k broadcast intent

The remaining unresolved space is increasingly concentrated in the literal delta-net update semantics rather than in missing preprocessing stages.

## Re-Audit Summary

From a workflow-contract perspective:

- BREAD now resembles the reference more strongly than earlier versions
- the current disagreements are more concentrated in how the delta-net algebra is executed than in which high-level steps exist
- the remaining mismatch surface is narrower, but still central to recurrent correctness
