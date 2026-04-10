# Path D Decision Memo

Date: 2026-04-11  
Scope: Decide whether BREAD should pursue CPU experts, SSD streaming, or continue relying on full expert preload

## Summary

Three sequential measurements were run to determine what is actually driving BREAD performance on the RTX 4060 Laptop.

The original question was:

> Is the current baseline fast because it truly fits, because it spills into shared memory, or because GPU experts are inherently better than CPU experts on this machine?

The answer is now much clearer:

1. The all-experts preload baseline is heavily using shared GPU memory under WDDM.
2. The fixed SSD-streaming path matches or slightly beats the spilled preload baseline while staying within clean memory limits.
3. The isolated expert-block benchmark now shows CPU experts beating the current serial GPU expert block on this machine.

This means the project no longer needs to treat the spilled preload baseline as the gold standard.

## Measurements

### 1. Spilled all-expert preload baseline

Command:

```powershell
.\bread.exe --prompt "The capital of France is" --tokens 30
```

Observed result:

- Throughput: `5.34 tok/s`
- Peak 4060 dedicated memory: `7954.9 MB`
- Peak 4060 shared memory: `14140.2 MB`

Interpretation:

- This path is not fully resident in the 4060’s dedicated VRAM.
- It fills the GPU and then spills heavily into Windows shared GPU memory.
- Therefore, this baseline is faster than expected partly because it relies on memory oversubscription behavior.

### 2. Fixed clean SSD-streaming path

Command:

```powershell
.\bread.exe --ssd-streaming --prompt "The capital of France is" --tokens 30
```

Observed result after fixing pipelined routing-buffer validity:

- Throughput: `5.48 tok/s`
- Peak 4060 dedicated memory: `1437.1 MB`
- Peak 4060 shared memory: `150.2 MB`
- Correct output preserved

Interpretation:

- The out-of-range prefetch/routing bug is fixed.
- The clean-memory streaming path is now effectively tied with, and slightly faster than, the spilled preload baseline.
- This strongly validates BREAD’s original memory-tier architecture direction.

### 3. Expert block benchmark

Command:

```powershell
.\bread.exe --bench-experts
```

Observed result:

- Layer: `0`
- Experts: `8`
- CPU path: `5.408 ms/block`
- GPU path: `10.775 ms/block`
- CPU/GPU ratio: `0.50x`
- Output MSE: `1.072712e-12`
- Output max abs: `4.902948e-06`

Interpretation:

- On this machine, for the isolated routed-expert block, CPU parallel execution is now about `2x` faster than the current serial GPU expert block.
- Numerical parity with the GPU path is excellent.
- This does **not** mean full decode will automatically become `2x` faster, but it does move CPU experts from “speculative idea” to “serious next experiment.”

## What Changed the Decision

Before these measurements, the architecture choice was ambiguous because the current baseline mixed together:

- full expert preload
- WDDM shared-memory spill
- serial GPU expert execution
- a buggy SSD-streaming path

After the measurements:

- The preload baseline is exposed as spill-heavy and therefore not a clean baseline.
- SSD streaming is now trustworthy and competitive.
- CPU experts are now promising again because the isolated block is favorable.

## Decision

### What is now ruled out

- Treating the spilled all-experts preload path as the “true” baseline
- Assuming SSD streaming is inherently too slow
- Assuming CPU experts are inherently slower on this hardware

### What is now supported by evidence

1. **Path C remains valid**
   - The fixed SSD-streaming path already reaches `5.48 tok/s`
   - It avoids the huge shared-memory spill
   - It aligns with BREAD’s original memory-tier identity

2. **Path D is now worth implementing experimentally**
   - The expert-block benchmark favors CPU experts on this machine
   - The right next step is a real `--cpu-experts` decode path
   - The target should be modest and benchmark-driven, not based on optimistic paper extrapolation

## Updated Throughput Expectations

Based on the current measurements:

- Current spilled preload baseline: `5.34 tok/s`
- Current fixed clean SSD-streaming: `5.48 tok/s`
- Predicted first working `--cpu-experts` decode path: `6.5–8.5 tok/s`
- Predicted stronger CPU-expert path after cleanup and overlap work: `8.5–10.5 tok/s`
- Stretch target after further CPU kernel work and careful overlap: `10–12+ tok/s`

## Recommended Next Step

Implement a real experimental `--cpu-experts` decode path on top of the now-trustworthy clean baseline.

That should be done as a benchmarked alternative mode, not as a replacement for the current path.

The key reason is simple:

- Path C is now clean and competitive
- Path D now has enough local evidence to justify implementation

So the project no longer has to choose blindly.

## Practical Takeaway

The investigation did not just produce another benchmark. It changed the architecture decision:

- **Path C is real**
- **Path D is now worth building**
- **The spilled preload baseline should no longer be treated as the target design**
