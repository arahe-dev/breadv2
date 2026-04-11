# BREAD Autoresearch Setup - Active

**Status:** 🟢 RUNNING
**Started:** 2026-03-31 19:20 UTC
**Duration:** 60 minutes (until usage limit)
**Results Location:** `C:\bread_v2\autoresearch_results\`

---

## Active Profiling Jobs

### 1. Layer Performance Profiling
**Command:** `bread.exe --prompt "The capital of France is" --tokens 20 --hooks-debug`
**Frequency:** Every 5 minutes
**Captures:** Per-layer forward pass timing (all 40 layers)
**Output:** `layer_timing_live.log`
**Purpose:** Identify slowest layers and layer type (attention vs SSM) performance

**Sample Output:**
```
[LAYER_TIMING] Per-layer forward times:
  Layer  0: 10.90 ms  (first layer, likely includes expert load)
  Layer  1:  5.12 ms  (SSM)
  Layer  2:  3.53 ms  (SSM)
  Layer  3:  2.97 ms  (full-attention)
  ...
  Layer 39:  4.50 ms
  Total:   180.45 ms
```

**Insights from data:**
- If Layer 0 >> others: Expert loading bottleneck
- If attention layers >> SSM: GQA/RoPE opportunity
- If last layers slow: Output norm/lm_head issue

---

### 2. GPU Memory Monitoring
**Command:** `nvidia-smi --query-gpu=memory.used,memory.total`
**Frequency:** Every 10 minutes
**Captures:** VRAM usage (idle, startup, mid-inference, peak)
**Output:** `vram_usage.log`
**Purpose:** Track memory pressure and peak utilization

**Current Baseline:**
```
2026-03-31 19:20:57 VRAM: 570MB/8188MB (7.0%)
```

**What it reveals:**
- Peak VRAM: Know if close to 8GB limit
- Startup overhead: Model loading memory impact
- Sustained usage: Typical working set
- Fragmentation: Memory efficiency

---

### 3. Throughput Scaling Test
**Command:** Tests with tokens=10, 50, 100 sequences
**Frequency:** Every 15 minutes
**Captures:** Tokens per second at different sequence lengths
**Output:** `throughput_scaling.log`
**Purpose:** Identify memory pressure and batch efficiency

**What it reveals:**
- Linear scaling = compute-bound (good)
- Sublinear = memory-bound (optimization opportunity)
- Cliff at N tokens = KV cache size limit or expert cache eviction

---

### 4. Minimal Mode Variance
**Command:** `bread.exe --minimal --prompt "..." --tokens 30`
**Frequency:** Every 20 minutes
**Captures:** Minimal mode performance and stability
**Output:** `minimal_mode_variance.log`
**Purpose:** Baseline for correctness-first comparison, variance measurement

---

## Comprehensive Baseline (background)

Script `autoresearch_profile.sh` is running 5 iterations of layer profiling to collect:
- `layer_profiles_[timestamp].csv` — Averaged layer times across 5 runs
- `memory_usage_[timestamp].csv` — Full VRAM timeline during inference
- `throughput_[timestamp].csv` — Throughput at 5, 10, 20, 50, 100 tokens
- `initial_run.log` — Full profiling summary with slowest layer analysis

---

## How to Use Results

### Real-time Monitoring
```bash
# Watch layer timing as it comes in
tail -f C:\bread_v2\autoresearch_results\layer_timing_live.log

# Watch VRAM pressure
tail -f C:\bread_v2\autoresearch_results\vram_usage.log

# Watch throughput scaling
tail -f C:\bread_v2\autoresearch_results\throughput_scaling.log
```

### Analysis After Collection
```bash
# Find slowest layers
awk -F: '{print $2, $3}' layer_profiles_*.csv | sort -k2 -rn | head -10

# Average VRAM usage
awk '{sum+=$3; count++} END {print "Avg VRAM:", sum/count, "MB"}' memory_usage_*.csv

# Throughput vs sequence length
awk '{print "Tokens:", $1, "→", $2, "tok/s"}' throughput_*.csv
```

---

## Optimization Targets This Reveals

**If Layer 0 is 3-5x slower:**
- → Expert loading bottleneck
- → Solution: Batch expert loads, prefetch async

**If attention layers 2-3x slower than SSM:**
- → GQA matmul or RoPE computation heavy
- → Solution: Optimize attention kernel, reduce RoPE computation

**If throughput drops >20% from 5→100 tokens:**
- → Memory pressure (KV cache, expert cache eviction)
- → Solution: Adjust cache sizes, streaming

**If peak VRAM > 7.5GB:**
- → Close to limit, risky on consumer hardware
- → Solution: More aggressive expert cache eviction

---

## Expected Timeline

**0-5 min:** Model loading, VRAM ramp-up
**5-15 min:** Stable inference, baseline layer times established
**15-30 min:** Throughput scaling test results visible
**30-45 min:** Memory pressure patterns emerge
**45-60 min:** Minimal mode variance, statistical significance

---

## Next Steps (After 60 min)

With complete profiling data, we can:

1. **Identify Top 3 Bottlenecks**
   - From layer timing: Which layer(s) slow
   - From memory: Is it VRAM, transfer, or compute
   - From throughput: Is scaling sublinear (memory bound)

2. **Propose Targeted Fixes**
   - Layer 0 slow? Profile expert loading
   - Throughput degrades? Test KV cache size
   - VRAM pressure? Tune eviction policy

3. **Run Ablation Studies**
   - Test Phase 5 vs Phase 6 benefit
   - Measure FMA dequant improvement
   - Profile speculative decoding gain

4. **Create Optimization Roadmap**
   - Priority-ordered list of improvements
   - Expected speedup for each
   - Implementation effort estimate

---

## Cron Jobs Active

| Job ID | Command | Frequency | Active |
|--------|---------|-----------|--------|
| 07d196c4 | Layer profiling | Every 5 min | ✅ |
| 454a5cef | VRAM monitoring | Every 10 min | ✅ |
| 09ce7351 | Throughput test | Every 15 min | ✅ |
| 4fe3dae0 | Minimal mode | Every 20 min | ✅ |

All jobs are session-only and will auto-expire after 7 days.

---

## Results Location

```
C:\bread_v2\autoresearch_results\
├── layer_timing_live.log              (live, updates every 5 min)
├── vram_usage.log                     (live, updates every 10 min)
├── throughput_scaling.log             (live, updates every 15 min)
├── minimal_mode_variance.log          (live, updates every 20 min)
├── layer_profiles_[timestamp].csv     (baseline, 5 iterations)
├── memory_usage_[timestamp].csv       (baseline, full timeline)
├── throughput_[timestamp].csv         (baseline, multiple sequence lengths)
├── initial_run.log                    (comprehensive baseline summary)
└── AUTORESEARCH_SETUP.md              (this file)
```

---

**Autoresearch actively collecting data. Check logs as they update!**
