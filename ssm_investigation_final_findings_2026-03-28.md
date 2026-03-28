# SSM Investigation Final Findings

Date: 2026-03-28
Workspace: `C:\bread_v2`
Investigation set:

- `C:\bread_v2\bread_ssm_workflow_map_investigation_2026-03-28.md`
- `C:\bread_v2\llamacpp_qwen35moe_ssm_deepmap_investigation_2026-03-28.md`
- `C:\bread_v2\bread_reaudit_against_llamacpp_ssm_investigation_2026-03-28.md`

## Purpose

This document consolidates the current investigation set into a single findings summary. It is based on the workflow maps and re-audit, plus the recent runtime ablations already observed in the current BREAD build.

## What Is Established

### 1. The recurrent / SSM path is active in the current BREAD build

This is supported by recent runtime behavior:

- baseline minimal debug first token produced `ucho`
- `--force-ssm-zero` materially changed output, hidden RMS progression, and logits

So the current BREAD build is not behaving like a pure SSM stub.

### 2. The SSM path is the dominant subsystem for the investigated first-token prompt

Recent ablations indicate:

- forcing SSM to zero collapses behavior strongly
- disabling RoPE leaves first-token behavior close to baseline

This does not prove RoPE is correct. It does indicate that, for the tested prompt and first generated token, the recurrent path is currently the primary driver.

### 3. Several direct recurrent-path mismatches have already been corrected

The following issues were previously identified and then fixed:

- missing SiLU after recurrent conv
- wrong `ssm_a` / decay composition
- wrong q/k normalization mode
- wrong extra q/k scaling pattern

These fixes changed the generated first token from `关于` to `ucho`, which is strong evidence that they were real sources of drift.

### 4. The current BREAD recurrent workflow is structurally closer to the reference than before

At the slot-mapping level, BREAD now aligns with the reference in:

- recurrent input projections
- `alpha + dt -> softplus -> * ssm_a`
- post-conv SiLU
- q/k/v split order
- q/k L2 normalization
- q scaling
- gated norm with `SiLU(z)`
- final `ssm_out` projection

## What Is Not Yet Established

### 1. Numeric equivalence of BREAD’s scalarized delta-net update to the reference

This remains unproven.

BREAD still implements the recurrent update as explicit nested loops over:

- v-heads
- output rows
- key dimensions

while the reference expresses the same operation as a tensor-graph delta-net update.

The symbolic roles are now closer, but exact equivalence is still not demonstrated.

### 2. Whether the remaining garbage output is entirely due to the recurrent block

The evidence points there first, but does not fully exclude:

- weak full-attention contribution
- tokenizer / prompt-wrap parity issues
- other second-order effects at CPU/GPU boundaries

### 3. Whether the `v_head_reordered` metadata should affect the forward path directly

The current BREAD config parses it.
The visible `llama.cpp` Qwen35MoE forward code does not expose a direct use of it in the recurrent branch.
So its runtime role remains unresolved from the current source read.

## Highest-Confidence Current Finding

The remaining high-value gap is not “missing recurrent stages” anymore. It is:

- the literal execution semantics of the delta-net state update and readout

In other words:

- BREAD now has most of the right boxes in the right order
- the likely remaining error is in how the recurrent state algebra is executed inside those boxes

## Secondary Finding

Attention still looks weaker than expected in current traces.

Observed trace pattern:

- many attention-layer branch RMS values remain near zero until late layers

This makes attention a real secondary investigation target, but not the first one for the tested prompt.

## Working Interpretation Of The Current State

The investigation set supports the following current interpretation:

1. BREAD’s recurrent branch is live and dominant for the investigated prompt.
2. Earlier recurrent mismatches were real and materially changed output when fixed.
3. The remaining drift is now concentrated in the exact delta-net execution semantics, not in whether the branch exists at all.
4. Attention weakness remains an open second issue.

## Current Priorities

### Priority 1

Do a literal numeric audit of the recurrent substeps, not just a symbolic workflow comparison.

The most valuable next measurements are:

- post-conv `q`
- post-conv `k`
- post-conv `v`
- `gate`
- `beta`
- state before decay
- state after decay
- state after update
- per-head readout before gated norm
- output after gated norm

### Priority 2

Investigate why attention-layer branch RMS is still so weak in the current debug traces.

### Priority 3

Verify tokenizer / prompt-wrap parity with a trusted reference for the exact same prompt.

## Bottom Line

The investigation no longer supports the old story that BREAD is failing because the recurrent path is absent.

The current story is narrower and more actionable:

- BREAD now contains the main recurrent stages
- several important recurrent mismatches have already been fixed
- the remaining error surface is concentrated in the exact delta-net state-update semantics and possibly a secondary attention weakness

That is the present best-supported explanation for why the output is still garbage even though the recurrent path is no longer obviously stubbed.
