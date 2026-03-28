# BREAD Handoff Context - 2026-03-28

## Scope
This document captures the current debugging state of BREAD output-quality work as of 2026-03-28. It is meant to let the next session continue without repeating the last several days of investigation.

## Current High-Level Status
- Loader stability is fixed.
- Full-model pinned-host allocation was removed; model now loads into normal RAM.
- Tokenizer special/control token handling was fixed and then transplanted toward `llama.cpp` behavior.
- Prompt tokenization for wrapped Qwen prompts now matches `llama-tokenize.exe`.
- SSM is no longer treated as "stubbed" in the current build; it is active and dominant.
- Several SSM semantic mismatches versus `llama.cpp` were fixed.
- RoPE was transplanted to use metadata-driven sections/interleaving, but this did not materially change outputs for text prompts.
- The newest concrete finding is on the attention side: `V` is coming out as all zeros in the first full-attention layer investigated.

## Recent Output Evolution
- Earlier broken outputs included:
  - `关于`
  - `ucho`
  - `helsing`
- After tokenizer transplant, outputs changed to:
  - prompt `The capital of France is` -> `zadek`
  - prompt `What is 2 + 2?` -> `seter`
- These are still wrong, but they confirm that tokenizer and SSM fixes materially changed model behavior.

## What Was Proven / Ruled Out

### 1. Scheduler / expert orchestration is not the main bug
- `--minimal` and normal/orchestrated mode produced the same bad outputs in prior tests.
- This strongly suggests the shared forward semantics are the issue, not the expert-cache orchestration layer.

### 2. Tokenizer mismatch was real and major
- BREAD was previously byte/BPE-splitting Qwen special tokens like `<|im_start|>`.
- After the tokenizer work, wrapped prompt token IDs now match `llama-tokenize.exe`.
- This removed a major upstream mismatch.

### 3. SSM is active, not absent
- `--force-ssm-zero` drastically changes the model state and output.
- Therefore the old "SSM is stubbed" explanation does not fit the current source tree.

### 4. RoPE is not the primary first-token failure
- `--disable-rope` barely changed first-token traces.
- The more complete RoPE transplant using `rope_sections` and `rope_mrope_interleaved` did not materially change output.

## Important SSM Findings Already Incorporated
The following SSM-side mismatches were identified against `llama.cpp` and fixed:
- Added `SiLU` after `ssm_conv1d`
- Changed `ssm_a` / decay composition
- Switched q/k normalization toward L2-style behavior
- Changed q scaling
- Reworked the autoregressive delta-net update to be more literal

These changes affected outputs, but did not solve correctness.

## Most Important New Concrete Finding

### Attention layer 3 dies because V is zero
In the minimal CPU-float path, layer 3 was instrumented with detailed traces.

For prompt `The capital of France is`, layer 3 showed:
- `q_score_head0`: non-zero
- `k_head0`: non-zero
- `attn_scores_softmax`: non-zero
- `q_gate_sigmoid_head0`: small but non-zero
- `attn_pre_gate`: exactly zero
- `attn_post_gate`: exactly zero
- `o_proj_out`: exactly zero

The decisive trace values were:
- `v_head0`: exactly zero
- `v_cache_current`: exactly zero

Interpretation:
- query path is alive
- key path is alive
- attention weights are alive
- gate is not the source of total silence
- value projection / cache path is dead

This is currently the strongest concrete bug in the attention path.

## Exact Trace Snippet
From `trace_capital_attn_seq.txt`:

```text
TRACE bread.layer3.q_score_head0: n=256 rms=1.327721 min=-4.706113 max=2.922463
TRACE bread.layer3.k_head0: n=256 rms=1.303173 min=-5.076401 max=3.446739
TRACE bread.layer3.v_head0: n=256 rms=0.000000 min=0.000000 max=0.000000
TRACE bread.layer3.q_gate_raw_head0: n=256 rms=4.066732 min=-7.044468 max=-1.692869
TRACE bread.layer3.q_gate_sigmoid_head0: n=256 rms=0.040811 min=0.000871 max=0.155399
TRACE bread.layer3.head0.attn_scores_softmax: n=17 rms=0.118631 min=0.000000 max=0.438576
TRACE bread.layer3.head0.v_cache_current: n=256 rms=0.000000 min=0.000000 max=0.000000
TRACE bread.layer3.head0.attn_pre_gate: n=256 rms=0.000000 min=0.000000 max=0.000000
TRACE bread.layer3.head0.attn_post_gate: n=256 rms=0.000000 min=0.000000 max=0.000000
TRACE bread.layer3.attn_out_all_heads_post_gate: n=4096 rms=0.000000 min=0.000000 max=0.000000
TRACE bread.layer3.o_proj_out: n=2048 rms=0.000000 min=0.000000 max=0.000000
Layer  3 [attn]: hidden_rms=0.442447 branch_rms=0.000000
```

## Likely Root Causes For The Zero-V Bug
In order of confidence:
1. BREAD CPU `Q6_K` matvec/dequant path is wrong for real model tensors
2. `attn_v.weight` is being interpreted with the wrong layout/orientation
3. A shared `Q6_K` handling bug exists in both CPU and GPU paths

Why these are the leading suspects:
- `attn_v.weight` is `Q6_K`
- `attn_k.weight` is quantized too, but K is non-zero
- attention scores are non-zero
- only the value path is fully zero

## Files Most Relevant Right Now
- [one_layer.cu](C:/bread_v2/one_layer.cu)
- [tokenizer.c](C:/bread_v2/tokenizer.c)
- [bread.c](C:/bread_v2/bread.c)
- [main.cu](C:/bread_v2/main.cu)
- `C:\Users\arahe\llama.cpp\src\models\qwen35moe.cpp`
- `C:\Users\arahe\llama.cpp\src\models\delta-net-base.cpp`
- `C:\Users\arahe\llama.cpp\ggml\src\ggml-cpu\ops.cpp`

## Current Investigation Artifacts
- `trace_capital_attn_seq.txt`
- `trace_math_attn_seq.txt`
- several investigation markdown files already created earlier in the repo

## Recommended Next Steps

### Immediate next step
Audit the `attn_v.weight` path end-to-end.

Suggested order:
1. Trace `cpu_named_matvec()` for `blk.3.attn_v.weight`
2. Verify the raw tensor type and shape again from GGUF
3. Compare BREAD CPU `Q6_K` matvec output for one known input vector against:
   - BREAD GPU `Q6_K` path
   - `llama.cpp` / ggml reference behavior if possible
4. Confirm whether the zero output is:
   - a dequant bug
   - a matvec bug
   - a tensor layout/orientation bug

### After that
- If `V` becomes non-zero, rerun the same attention trace before touching anything else.
- Only then re-evaluate whether the remaining main bug is still SSM-dominant.

## Important Operational Notes
- Do not run two `bread.exe` prompts in parallel.
- There are static caches/state buffers inside `one_layer.cu`; parallel runs can contaminate traces.
- Run one prompt at a time in a fresh process when collecting debug traces.

## Build / Run Command That Produced The Key Finding
```powershell
.\bread.exe --minimal --debug --prompt 'The capital of France is' --tokens 1 2> trace_capital_attn_seq.txt
```

## Summary
The current best lead is no longer "SSM missing" or "RoPE wrong." The strongest concrete bug is that the first inspected full-attention layer has a dead value path: `V` is exactly zero while `Q`, `K`, and attention weights are not. Fixing the `attn_v.weight` / `Q6_K` handling is the next highest-confidence move.
