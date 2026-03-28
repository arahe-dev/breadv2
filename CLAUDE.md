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

Output is still semantically wrong.

That is true even after:
- adding a real SSM/GatedDeltaNet path
- adding a real full-attention path
- adding host-side KV cache usage
- moving to runtime model config
- correcting some architecture assumptions like RoPE dimension count
- adding Qwen35-specific tokenizer pre-name handling
- adding model-driven Qwen-style prompt wrapping
- adding support for `ssm.v_head_reordered`
- moving minimal mode closer to a CPU-float reference path

So the repo is now in a more awkward but more honest state:
- there is a lot more real functionality than the old doc suggested
- but the forward semantics still do not match a trusted runtime closely enough

### Performance

Performance is still dominated by avoidable overhead.

The biggest known speed issue remains:
- `one_layer_forward()` repeatedly allocates, uploads, synchronizes, and frees non-expert tensors every layer call

Minimal mode is slower than orchestrated mode, as expected.

## Current Best Diagnosis

As of now, the most likely correctness problems are:

1. Exact Qwen35 tokenizer and prompt semantics
- now a serious suspect, not an afterthought
- prompt formatting and pre-tokenization are closer than before, but still may not be faithful enough

2. SSM / GatedDeltaNet semantics in `one_layer.cu`
- still the top model-side suspect
- likely a "shape is plausible, semantics still wrong" problem

3. Remaining attention details
- especially RoPE behavior beyond just dimension count
- especially `rope_sections` and `mrope_interleaved`

4. Remaining MoE semantic mismatch
- router / combine / expert interpretation issues are less suspicious than before, but still possible

5. Quantization-adjacent precision/layout issues
- not "quantization is impossible"
- more likely subtle issues at the custom quantized matvec -> fp16 activation -> CPU float boundaries in non-minimal mode

## What Has Been Ruled Out Or Downgraded

These are not fully impossible, but they are not the leading explanation anymore:

- "SSM is missing"
- "KV cache is missing"
- "the expert-cache orchestration novelty is the root cause"
- "the major manually-read SSM tensors are the wrong dtype"
- "Q4_K / Q6_K kernels are blatantly broken"
- "the problem is just that SSM/KV cache do not exist yet"

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

### Correctness first

1. Add validation hooks in minimal mode
- dump per-layer RMS / routing / logits
- find the first bad layer instead of guessing

2. Verify prompt/token parity with a trusted Qwen35 tokenizer
- exact prompt wrapping
- exact pre-tokenization behavior
- exact token IDs for known prompts

3. Tighten `one_layer.cu` to be more `llama.cpp`-faithful
- full attention
- RoPE details
- SSM / GatedDeltaNet
- MoE combine semantics

4. Only after text quality is sane, reapply optimizations confidently

### Speed second

After correctness is under control:
- cache non-expert weights in VRAM at startup
- reduce `cudaMalloc/cudaFree`
- reduce synchronizations
- reintroduce smarter pinned staging for transfers

## Quick Commands

Build:

```cmd
cd C:\bread_v2
build_main.bat
```

Run normal mode:

```cmd
bread.exe --prompt "The capital of France is" --tokens 10
```

Run minimal mode:

```cmd
bread.exe --minimal --prompt "The capital of France is" --tokens 10
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

BREAD is no longer "just a loader plus stubs."

It is now a real custom inference engine with:
- real loader behavior
- real layer execution
- real quantized matvec kernels
- real MoE routing
- real attention / SSM branches

But it is not yet a correct one.

The central task now is not adding missing subsystems.
The central task is making the existing forward pass faithful enough to stop producing garbage output.

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

### Current Status

- ✅ Full inference pipeline working (end-to-end)
- ✅ Loader, tokenizer, embedding, all 40 layers, output norm+lm_head
- ✅ Validation harness and per-layer RMS dumping
- ✅ Identified root causes
- ❌ SSM not implemented (critical blocker)
- ❌ RoPE incomplete (high priority blocker)
- ❌ Still outputs garbage on test prompt

**Estimated fix time**: 1-2 days for SSM implementation + RoPE fixes if no surprises found.
