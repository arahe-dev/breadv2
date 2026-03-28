# BREAD — CLAUDE.md

## What BREAD Is

BREAD is a custom CUDA/C inference engine for large MoE models on consumer NVIDIA hardware.

The project goal is still the same:
- keep model semantics inside a custom engine
- schedule MoE inference across memory tiers
- eventually stream experts across VRAM, RAM, and SSD

The current target model is:
- `Qwen3.5-35B-A3B`
- raw GGUF blob at `C:\Users\arahe\.ollama\models\blobs\sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a`

This repo is text-only right now even though the GGUF also contains vision tensors.

## Project Direction

The important clarification is that BREAD is no longer blocked on "missing plumbing."

The codebase now has:
- end-to-end model loading
- tokenizer
- embedding
- layer loop
- output norm + lm head
- a real full-attention path
- a real SSM/GatedDeltaNet path
- a host-side KV cache path for full-attention layers
- a metadata-driven runtime config path
- an orchestrated expert cache path
- a minimal correctness-first path that bypasses the expert cache orchestration

So the current problem is not "SSM and KV cache do not exist."

The current problem is:
- output quality is still wrong
- performance is still poor
- the remaining bug is likely in forward semantics, not missing infrastructure
- the most suspicious remaining gap is now exact Qwen3.5 semantics, not missing subsystems

## Hardware Context

- GPU: NVIDIA RTX 4060 Laptop (8 GB VRAM)
- CPU: Intel i7-13650HX
- RAM: 48 GB DDR5
- OS: Windows 11
- CUDA: 13.2

## Repo Structure

```text
C:\bread_v2\
├─ CLAUDE.md
├─ main.cu
├─ one_layer.cu
├─ bread.h / bread.c
├─ loader.h / loader.c
├─ kernels.cu
├─ gguf.h / gguf.c
├─ tokenizer.h / tokenizer.c
├─ dequant_q4k_cpu.c
├─ bread_info.c
├─ config_reader.c
├─ infer_ref.m
└─ build_*.bat
```

## What The Core Files Do

- `main.cu`
  - CLI entry point
  - prompt encode / generation loop
  - `--minimal` mode for correctness isolation

- `one_layer.cu`
  - per-layer forward pass
  - full-attention branch
  - SSM / GatedDeltaNet branch
  - MoE routing + expert combine

- `bread.h` / `bread.c`
  - runtime model config derived from GGUF metadata
  - layer typing helpers
  - boring/minimal mode control

- `loader.h` / `loader.c`
  - full model resident in host RAM
  - VRAM expert slot cache
  - async expert orchestration path

- `kernels.cu`
  - Q4_K and Q6_K matvec kernels

- `gguf.h` / `gguf.c`
  - GGUF parser and tensor metadata index

## Current Architecture Notes

### Runtime config

The active path is now more metadata-driven than before.

`bread.c` now parses or carries:
- hidden dim
- layer count
- vocab size
- q/k/v dims
- RoPE base
- RoPE dimension count
- RoPE sections
- RoPE mrope interleaved flag
- full-attention interval
- SSM config fields
- SSM v-head reorder flag
- MoE config fields

For the current target GGUF, the important verified values are:
- `qwen35moe.full_attention_interval = 4`
- `qwen35moe.rope.dimension_count = 64`
- `qwen35moe.rope.dimension_sections = [11, 11, 10]`
- `qwen35moe.ssm.conv_kernel = 4`
- `qwen35moe.ssm.inner_size = 4096`
- `qwen35moe.ssm.state_size = 128`
- `qwen35moe.ssm.time_step_rank = 32`
- `qwen35moe.ssm.group_count = 16`
- `qwen35moe.ssm.v_head_reordered = 1`
- `qwen35moe.rope.mrope_interleaved = 1`
- `tokenizer.ggml.pre = qwen35`

This was added to move BREAD closer to `llama.cpp`'s model-contract style instead of relying on old hardcoded assumptions.

### Minimal mode

`bread.exe --minimal ...`

This mode exists specifically to answer:
"Is the custom expert-cache orchestration causing the garbage output?"

In minimal mode:
- BREAD still uses its own forward pass
- it keeps activations in float across the layer stack for correctness-first testing
- it uses a more reference-like CPU-float layer path instead of the mixed quantized GPU matvec plus CPU post-processing path
- it bypasses the normal expert-cache path
- selected expert tensors are copied directly from host GGUF memory into temporary VRAM buffers

This mode is slower, but it is now the main semantic-debugging baseline.

### Loader behavior

Important update:
- the full model is no longer loaded with a giant `cudaMallocHost`
- it now lives in normal host RAM for stability on Windows/WDDM

Reason:
- the old giant pinned allocation caused CUDA/WDDM virtual address exhaustion during startup

This means:
- startup is now stable
- async transfer behavior is less ideal than the original design
- the loader architecture still needs a future "small pinned staging buffers" pass

## What Has Been Verified

### Working

- GGUF parsing works
- Tensor table inspection works (`bread_info.exe`)
- Runtime config extraction works
- Q4_K CPU reference dequant works
- Q6_K CPU reference dequant works
- Q4_K CUDA matvec self-test passes
- Q6_K CUDA matvec self-test passes
- Tokenizer loads and encodes prompts
- Qwen35 prompt wrapping is now wired into the CLI path
- End-to-end generation loop runs
- Loader startup OOM/WDDM issue is fixed
- Minimal mode and orchestrated mode both run
- Qwen35 metadata flags that were previously missing are now parsed

### Important empirical result

For the current broken prompt behavior, minimal mode and orchestrated mode produced the same text on the same prompt.

This is a big result:
- the custom expert-cache orchestration does not currently look like the root cause of the garbage output

It may still affect speed, but it is not the first place to blame for correctness.

### Important secondary result

The old config inspection tools were under-reporting useful metadata.

Direct GGUF header inspection confirmed several Qwen35-specific fields that matter for correctness and were easy to miss when relying only on the repo helper tools.

## Current Known Behavior

### Loader stability

The old startup failure:
- CUDA out of memory during loader setup after pinning the full model

Current status:
- fixed by keeping the full GGUF in normal host RAM instead of pinning the whole blob

### Output quality

**Output is now correct as of 2026-03-28.**

The model correctly answers factual questions, generates coherent multi-sentence responses, and produces well-formed Qwen3 chat format output including `<think>` blocks.

Example verified outputs:
- `"The capital of France is"` → `"The capital of France is **Paris**. Located"`
- `"what is 2+2"` → `"2 + 2 = 4."`

#### Root cause of all previous garbage output

The bug was in `fp16_to_f32_host()` in `one_layer.cu` (and the identical `fp16_to_fp32_cpu()` in `kernels.cu` and `h2f_host()` in `main.cu`):

```c
// OLD (buggy):
if (exponent == 0) bits = sign;  // returns ±0.0 for ALL subnormal fp16 values
```

The `attn_v.weight` tensors for all full-attention layers have Q6_K block scales (`d`) that are subnormal fp16 values (exponent field = 0, mantissa ≠ 0). The old code silently returned 0.0 for these, making every V projection output exactly zero across all 10 full-attention layers.

Fix applied:
```c
if (exponent == 0) {
    float val = (float)mantissa * (1.0f / 16777216.0f);
    return (h >> 15) ? -val : val;
}
```

Why it was hard to find:
- Q4_K (used by attn_k and attn_q) happens to have normal-range fp16 `d` values — only Q6_K attn_v was affected
- No NaN, no crash, no error — just silent zero output from all attention layers
- The Q6_K self-test uses synthetic `d = 1.0` (0x3C00, normal fp16) so the bug was never caught in testing
- The GPU kernel uses hardware `__half2float` which handles subnormals correctly — only the CPU path was broken

### Performance

Performance is still dominated by avoidable overhead.

The biggest known speed issue remains:
- `one_layer_forward()` repeatedly allocates, uploads, synchronizes, and frees non-expert tensors every layer call

Minimal mode is slower than orchestrated mode, as expected.

## Current Status (2026-03-28)

**Correctness: SOLVED.** The model produces correct outputs.

**Performance: next priority.** Minimal mode runs at ~0.6 tok/s due to per-layer `cudaMalloc/cudaFree` overhead. GPU (non-minimal) mode is faster but not yet benchmarked post-fix.

## What Has Been Ruled Out Or Resolved

- "SSM is missing" — SSM is active and working
- "KV cache is missing" — KV cache is active and working
- "the expert-cache orchestration novelty is the root cause" — not the issue
- "Q4_K / Q6_K kernels are blatantly broken" — kernels are correct
- "attention output is zero" — **FIXED** (fp16 subnormal bug)
- "output is garbage" — **FIXED**

## Relationship To llama.cpp

The current strategy is:
- keep BREAD as a custom engine
- but align its model semantics much more closely with `llama.cpp`
- use `llama.cpp` as the semantic reference, not as something to blindly transplant whole

That means:
- copy architecture contract ideas
- copy forward semantics where needed
- keep BREAD's novel parts around scheduling, residency, and memory-tier orchestration

The intended split is:
- inner loop: boring, faithful, debuggable
- outer system: novel, scheduled, optimized

## What BREAD Is Not Doing Yet

- non-expert weights cached permanently in VRAM
- SSD streaming for this 35B model
- robust validation against a trusted reference at intermediate tensor checkpoints
- numerically proven parity with `llama.cpp`
- polished multimodal support
- exact tokenizer parity with a trusted Qwen35 runtime

## Recommended Next Steps

### Performance (now the priority)

1. **Cache non-expert weights in VRAM at startup**
   - biggest single win
   - currently every layer call allocates, uploads, and frees these tensors
   - caching them eliminates most of the `cudaMalloc` overhead

2. **Reduce `cudaMalloc/cudaFree` per layer call**
   - preallocate activation buffers once at startup
   - reuse across layers

3. **Reintroduce smarter pinned staging for transfers**
   - async expert uploads while compute runs on previous layer

### Correctness verification (now secondary)

4. Numerical comparison against llama.cpp per-layer hidden_rms
   - confirm SSM and attention values match to within float precision
   - catch any remaining semantic mismatches

5. Test more prompt types: coding, math, multi-turn

## Quick Commands

Build (from bash terminal — PowerShell cannot find cl.exe without PATH setup):

```bash
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64"
cd C:/bread_v2
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c -I. -o bread.exe
```

Run (PowerShell — use `.\` prefix and chat template):

```powershell
.\bread.exe --prompt "<|im_start|>user
YOUR PROMPT<|im_end|>
<|im_start|>assistant
<think>

</think>

" --tokens 200
```

Run minimal mode (slower, CPU-float, correctness baseline):

```powershell
.\bread.exe --minimal --prompt "<|im_start|>user
YOUR PROMPT<|im_end|>
<|im_start|>assistant
<think>

</think>

" --tokens 200
```

Inspect tensors:

```cmd
bread_info.exe C:\Users\arahe\.ollama\models\blobs\sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a
```

Run kernel self-test:

```cmd
kernels_test.exe
```

## Bottom Line

BREAD is a working custom inference engine.

It has:
- real loader behavior
- real layer execution
- real quantized matvec kernels (Q4_K, Q6_K)
- real MoE routing and expert dispatch
- real full-attention with RoPE, GQA, gating, KV cache
- real SSM / GatedDeltaNet path
- correct outputs verified on factual and math prompts

The central task now is performance — getting from ~0.6 tok/s to something usable.

---

## Phase 2: Detailed Analysis (March 2026)

### Root Cause Identified

After two weeks of debugging and comparing against llama.cpp reference implementation, the primary bugs are:

1. **SSM/GatedDeltaNet Not Implemented** (CRITICAL)
   - 30 out of 40 layers are stubbed (output ≈ 0)
   - This alone deletes 75% of the model's computation
   - Both normal and minimal modes fail identically because they share the same flawed `one_layer_forward()`
   - **This is why output is garbage**

2. **RoPE Incomplete** (HIGH PRIORITY)
   - Only 32 out of 2048 dimensions receive rotary embeddings
   - Should use full 128+ dimensions per head
   - rope_sections=[11,11,10] from GGUF but BREAD treats it as [64,64,64,...]
   - Attention heads can't align properly without full RoPE

3. **RoPE Dimension Section Mismatch** (BLOCKING)
   - Model has 3 rope sections, llama.cpp expects 4
   - This is why llama.cpp fails to load: `expected 4, got 3`
   - Needs one-line fix in llama.cpp source to compare against

### Validation Findings

Created per-layer RMS dumping system (`--debug` flag):
- **Normal mode (GPU fp16)**: Layer RMS ranges 0.14–0.55, gradually increases
- **Minimal mode (CPU float)**: Layer RMS ranges 0.07–0.91, different values but same pattern
- **Both modes produce identical garbage tokens** (first token differs from ollama)
- **Embedding matches between modes** (fp16 vs fp32 precision difference only)
- **First divergence**: Layer 0, suggesting SSM/RoPE bug starts immediately

### Documentation Created

Three comprehensive flowchart analysis documents:
1. **breadflowchartanalysis.md** (816 lines) — Complete BREAD inference pipeline with exact parameters
2. **llamacppflowchartanalysis.md** (816 lines) — Complete llama.cpp reference pipeline
3. **COMPARISON.md** (250 lines) — Side-by-side comparison, shows exactly where they differ

### Next Steps (Priority Order)

**IMMEDIATE (Days 1-3)**
1. Implement SSM/GatedDeltaNet in `one_layer.cu`
   - Copy reference implementation from llama.cpp or llama-rs
   - Convolution + state scan + gate projection
   - This will likely fix 80% of the garbage output issue

2. Fix RoPE to use full dimensions
   - Parse rope_sections correctly (3 elements for this model)
   - Apply to all 128+ dims per head, not just 32
   - Verify against llama.cpp RoPE output

**MEDIUM (Days 3-5)**
3. Fix llama.cpp rope dimension check (one-line fix)
4. Build debug llama.cpp with per-layer RMS dumps
5. Compare BREAD vs llama.cpp per-layer RMS values
6. Iterate on layer-specific fixes based on divergence

**LONG TERM**
7. Cache non-expert weights in VRAM at startup (performance)
8. Reduce cudaMalloc/free overhead (performance)
9. Polished tokenizer parity verification

### Current Status (updated 2026-03-28)

- ✅ Full inference pipeline working (end-to-end)
- ✅ Loader, tokenizer, embedding, all 40 layers, output norm+lm_head
- ✅ Validation harness and per-layer RMS dumping
- ✅ SSM / GatedDeltaNet implemented and active
- ✅ RoPE implemented with metadata-driven sections
- ✅ fp16 subnormal bug fixed — all attention layers now produce correct output
- ✅ Correct answers verified: "Paris", "2+2=4", coherent multi-sentence responses
- ⏳ Performance: ~0.6 tok/s (minimal), dominated by per-layer cudaMalloc overhead
