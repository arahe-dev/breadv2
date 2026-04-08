# BREAD: Custom CUDA/C Inference Engine for MoE Models

BREAD is a high-performance custom inference engine for large Mixture-of-Experts (MoE) models on consumer NVIDIA hardware. It targets the **Qwen3.5-35B-A3B** model, running on an RTX 4060 Laptop (8 GB VRAM) with production-quality outputs and measurable performance optimization.

## Status

**Correctness:** ✅ Verified — outputs correct answers ("Paris", "2+2=4", coherent multi-turn)
**Performance:** 5.5–6.6 tok/s (baseline) → targeting 12–16 tok/s via optimization phases
**Architecture:** Custom C/CUDA engine with novel expert scheduling and memory-tier orchestration

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 13.2+
- Windows 11
- GGUF model: `Qwen3.5-35B-A3B` (23.87 GB)

### Build

```bash
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64"
cd /c/bread_v2
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c hooks.c progress_tracking.c buffer_pool.c -I. -o bread.exe
```

### Run

```powershell
# Standard inference
.\bread.exe --prompt "<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
<think>

</think>

" --tokens 50

# Minimal mode (correctness baseline, slower)
.\bread.exe --minimal --prompt "..." --tokens 50

# Server mode (for AGENCY tool-calling agent)
.\bread.exe --server --tokens 256
```

## Architecture

### Core Components

| File | Purpose |
|------|---------|
| `main.cu` | CLI entry point, prompt encoding, generation loop, server mode |
| `one_layer.cu` | Per-layer forward pass: attention, SSM, MoE routing, expert combination |
| `kernels.cu` | Q4_K and Q6_K quantized matvec CUDA kernels |
| `loader.h/c` | GGUF model loading, expert slot caching (18 VRAM slots) |
| `bread.h/c` | Runtime config metadata extraction, layer typing |
| `tokenizer.h/c` | Qwen3.5 tokenizer (BPE, vocab 248K) |
| `agency/` | Rust-based tool-calling agent (optional, for interactive use) |

### Performance Phases

| Phase | Focus | Status | Gain |
|-------|-------|--------|------|
| Phase 3 | Baseline optimization | ✅ Complete | Baseline: 5.5 tok/s |
| Phase 4 | Sync removal + DMA overlap | ✅ Complete | +6% (5.88 tok/s) |
| Phase 5 | Full weight caching | ✅ Complete | +12% (6.62 tok/s on 5 tokens) |
| Phase 6 | FMA kernel reformulation | ⏳ Pending build | +10-12% expected |
| Phase 7 | AGENCY tool-calling | ✅ Complete | Hermes format support |
| Phase 8 | FloE expert compression | 📋 Research phase | +2-4 tok/s target |

### Optimization Roadmap

**Next (Week 1-2):** Complete Phase 6 (FMA), implement SSM AVX2 vectorization, CPU expert execution
**Target:** 8–10 tok/s
**Then:** Evaluate FloE compression for 12–16 tok/s ceiling

See [PAPER_ANALYSIS_REPORT.md](PAPER_ANALYSIS_REPORT.md) for detailed research on next-generation optimizations.

## Implementation Highlights

### What's Working

✅ End-to-end inference pipeline (embedding → 40 layers → output norm → lm_head)
✅ Full-attention layers with RoPE, GQA, gating (10 out of 40 layers)
✅ SSM/GatedDeltaNet layers (30 out of 40 layers)
✅ MoE routing + expert selection (K=8 experts per token)
✅ Quantized kernels (Q4_K, Q6_K) for both GPU and CPU paths
✅ Host-side KV cache for full-attention layers
✅ Expert weight caching in VRAM (18 LRU slots → all 256 experts pre-loaded)
✅ Metadata-driven runtime config (parsed from GGUF headers)
✅ AGENCY tool-calling agent with Hermes format

### Known Bottlenecks

1. **GPU kernel efficiency** (Phase 6: FMA dequant)
   - Current: Separate dequant-then-multiply, separate instructions
   - Target: Fused multiply-add chains, fewer dependent instructions
   - Expected gain: +10-12% on GPU matvec

2. **SSM recurrence (30 layers)**
   - Current: Scalar unoptimized CPU loop, 15-30 ms/token
   - Target: AVX2 SIMD vectorization
   - Expected gain: 2–3× speedup, +0.5-1 tok/s

3. **Expert DMA latency**
   - Current: 8 experts × 40 layers × ~1-2 ms per expert = 160 ms/token
   - Target: CPU-side expert execution (arithmetic intensity argument) OR FloE compression
   - Expected gain: +1-4 tok/s

## Hardware & Configuration

**Target Hardware:**
- GPU: NVIDIA RTX 4060 Laptop (8 GB GDDR6, 4 GB/s peak, 128-bit bus)
- CPU: Intel i7-13650HX (10 P-cores, 48 GB DDR5)
- OS: Windows 11

**Model:** Qwen3.5-35B-A3B (23.87 GB GGUF)
- 40 layers (alternating full-attention + SSM)
- 3968 hidden dim
- 96 attention heads, GQA with 8 KV heads
- 256 MoE experts, K=8 routing
- Q4_K_M + Q6_K quantization

## Validation & Correctness

Comprehensive per-layer RMS validation against llama.cpp reference implementation. Test prompts validated:

```
"The capital of France is" → "Paris"
"What is 2+2?" → "4"
Multi-turn conversation support with thinking blocks (<think></think>)
```

## Code Organization

```
C:\bread_v2\
├─ CLAUDE.md                      # Project instructions & architecture decisions
├─ PAPER_ANALYSIS_REPORT.md       # Research on FloE, MoE-Gen, FlashAttention-3
├─ PHASE2_METRICS_AND_OBSERVATIONS.md  # Analysis of Phase 2 (deprecated optimization)
├─ main.cu                        # CLI + generation loop
├─ one_layer.cu                   # Per-layer forward pass
├─ kernels.cu                     # CUDA quantized matvec kernels
├─ bread.{h,c}                    # Runtime config
├─ loader.{h,c}                   # GGUF loader + expert caching
├─ tokenizer.{h,c}                # Qwen3.5 tokenizer
├─ gguf.{h,c}                     # GGUF format parser
├─ agency/                        # Rust tool-calling agent
│  ├─ main.rs
│  ├─ bread.rs
│  ├─ hermes.rs
│  ├─ tools.rs
│  └─ Cargo.toml
├─ build_main.bat                 # Windows build script
└─ benchmark.sh                   # Performance measurement
```

## Performance Benchmarks

**Baseline (Phase 5 + weight caching):**
```
Prompt: 16 tokens
Generated: 45 tokens
Prefill: 2045 ms (127.8 ms/tok)
Time to 1st: 58.7 ms
Decode: 8091 ms (5.56 tok/s)
Total: ~63 seconds (includes 20s model load)
```

**Latest (Phase 5):**
- 5 tokens: 6.62 tok/s
- 50 tokens: 5.71 tok/s
- 100 tokens: 5.73 tok/s

See benchmark logs in `bench_*.log` files for detailed runs.

## Next Steps

1. **Benchmark Phase 6 FMA kernel** — verify +10-12% gain (currently marked "implemented, pending build")
2. **Implement SSM AVX2 vectorization** — expected +0.5-1 tok/s in 6 hours
3. **Add CPU-side expert execution** — expected +1-2 tok/s in 10 hours
4. **Evaluate FloE compression** — only if above reach 8-10 tok/s plateau

## References

- [CLAUDE.md](CLAUDE.md) — Full architecture documentation
- [PAPER_ANALYSIS_REPORT.md](PAPER_ANALYSIS_REPORT.md) — Research on recent MoE optimization papers
- [Qwen3.5-35B-A3B](https://github.com/QwenLM/Qwen) — Target model
- [GGUF format](https://github.com/ggerganov/ggml) — Model quantization standard

## Contributing

BREAD is a custom research engine. Contributions should align with:
- Maintaining C/CUDA-only codebase (no Python, minimal Rust)
- Correctness-first, then optimization
- Measured performance gains with per-layer profiling
- Testing against llama.cpp reference outputs

## License

MIT (pending formalization)

---

**Last updated:** 2026-04-08
**Maintainer:** [BREAD Project](https://github.com/yourusername/bread_v2)
