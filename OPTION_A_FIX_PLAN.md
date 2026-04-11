# Option A: Find & Fix the Heisenbug - Simple Plan

## Overview
Use AddressSanitizer to pinpoint the exact memory corruption, then fix the root cause.

**Total Time: ~2 hours**

---

## Step 1: Compile with AddressSanitizer (10 min)

```bash
cd /c/bread_v2
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64"

# Compile with ASAN flags
nvcc -fsanitize=address -O1 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c progress_tracking.c buffer_pool.c hooks.c layer_ops.cu -I. -o bread_asan.exe
```

**Why `-O1`?**
- Enables ASAN with better debugging info
- `-O2` might optimize away the corruption

---

## Step 2: Run with AddressSanitizer (5 min)

```bash
# Set environment variables
set ASAN_OPTIONS=detect_leaks=1:verbosity=2:halt_on_error=1

# Run test
timeout 120 bread_asan.exe --server --tokens 100 < /tmp/test_single_query.txt 2>&1 | tee asan_output.txt
```

**What to expect:**
- ASAN will detect memory corruption
- Program will halt and print detailed error report
- Output saved to `asan_output.txt`

---

## Step 3: Analyze ASAN Output (15-30 min)

Look for messages like:
```
ERROR: AddressSanitizer: buffer-overflow on address 0x...
    READ of size X at 0x... thread T0
    #0 0x... in [FUNCTION_NAME] [FILE]:[LINE]
    #1 0x... in [CALLER]
```

**Key info to extract:**
- **Line number**: Where the corruption happens
- **Variable name**: What's being corrupted
- **Function**: Which function is the culprit
- **Stack trace**: How we got there

---

## Step 4: Fix the Bug (1-2 hours)

Based on ASAN output, apply the appropriate fix:

### If Buffer Overflow
- Increase buffer allocation size
- OR reduce number of writes to buffer
- OR add bounds checking before access

### If Use-After-Free
- Initialize pointer to NULL at declaration
- OR check pointer is valid before use
- OR defer deallocation until later

### If Stack Overflow
- Move large arrays from stack to heap
- OR reduce array sizes
- OR split computation into smaller chunks

### If Race Condition
- Add proper CUDA synchronization
- OR use atomic operations
- OR add memory barriers

---

## Step 5: Rebuild & Test (5 min)

```bash
# Rebuild clean version (without ASAN)
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c progress_tracking.c buffer_pool.c hooks.c layer_ops.cu -I. -o bread.exe

# Quick test for BREAD_END sentinel
timeout 60 ./bread.exe --server --tokens 200 < /tmp/test_single_query.txt 2>&1 | grep "BREAD_END"
```

**Expected result:**
```
BREAD_END
```

---

## Step 6: Verify AGENCY Works (5 min)

```bash
cd agency
cargo build --release
.\target\release\agency.exe --bread ..\bread.exe --tokens 256
```

In the REPL, test:
```
>>> What is 2+2?
[Should complete with full answer]

>>> What is 3+3?
[Should work on second query]
```

---

## Fallback Options

### If ASAN Doesn't Compile
Use Valgrind instead:
```bash
valgrind --leak-check=full --show-leak-kinds=all ./bread.exe --server --tokens 50
```

### If Valgrind Not Available
Use GDB with watchpoints on suspected buffers:
```bash
gdb ./bread.exe
(gdb) break one_layer_forward
(gdb) run --server --tokens 50
(gdb) watch ssm_state
(gdb) continue
```

---

## Success Criteria

✅ ASAN identifies corruption with exact line number
✅ Fix applied to identified variable/buffer
✅ Rebuild succeeds without warnings
✅ `bread.exe --server` generates BREAD_END sentinel
✅ AGENCY completes multi-turn conversation

---

## Files to Monitor

| File | Purpose |
|------|---------|
| `bread_asan.exe` | ASAN-enabled binary |
| `asan_output.txt` | Error report |
| `one_layer.cu` | Likely location of bug |
| `bread.exe` | Final fixed binary |

---

## Quick Reference: ASAN Flags

```
-fsanitize=address          # Enable AddressSanitizer
-O1                         # Optimization level (preserves debug info)
detect_leaks=1              # Detect memory leaks
verbosity=2                 # Detailed output
halt_on_error=1             # Stop on first error
```

---

**Status:** Ready to execute
**Priority:** CRITICAL - Unblocks AGENCY
**Success Probability:** 95% (ASAN is very good at finding memory bugs)
