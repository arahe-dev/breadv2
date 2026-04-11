# 🤖 AGENCY Quickstart

You now have a full Hermes agent CLI built in Rust that wraps BREAD.

## What Was Built

### 1. Updated main.cu with `--server` mode

**File:** `C:\bread_v2\main.cu`

Added a new command-line flag `--server` that:
- Loads the model once at startup
- Prints `BREAD_READY` when ready
- Reads prompts from stdin (one per line)
- Prints `BREAD_END` after each generation
- Stays running indefinitely (no per-request model reload)

This avoids the ~22 GB model load overhead for each request.

**Changes:**
- ~100 lines of C added (loop wrapper, stdin reading, sentinel printing)
- No changes to inference logic
- Backward compatible (works in single-prompt mode too)

### 2. AGENCY Rust CLI crate

**Location:** `C:\bread_v2\agency\`

A complete Hermes-format agent with:
- **Interactive REPL** (`>>>` prompt, colored output, history)
- **Tool calling** — automatically detects and executes `<tool_call>` blocks
- **Built-in tools** — read_file, list_dir, shell, write_file
- **Multi-turn** — maintains conversation history
- **Streaming** — tokens appear as BREAD generates them

**Files:**
- `src/main.rs` — entry point, CLI arg parsing
- `src/bread.rs` — subprocess wrapper (pipe management)
- `src/hermes.rs` — Hermes format + tool schemas
- `src/tools.rs` — tool implementations
- `src/repl.rs` — interactive loop
- `Cargo.toml` — Rust dependencies
- `README.md` — detailed documentation
- `build.bat` — Windows build script

**Binary:** `target/release/agency.exe` (2.3 MB)

## How to Use

### Step 1: Build BREAD with --server support

If you haven't rebuilt BREAD yet:

```bash
cd /c/bread_v2
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64"
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c bread_utils.c topology.c config_reader.c validate.c layer_logic.cu layer_ops.cu -I. -o bread.exe
```

### Step 2: Run the AGENCY CLI

```powershell
cd C:\bread_v2
.\agency\target\release\agency.exe --bread .\bread.exe --tokens 256
```

You should see:
```
🍞 Starting BREAD inference engine...
✓ BREAD ready

🤖 AGENCY Agent (type /help for commands)

>>>
```

### Step 3: Start Using!

Examples:

```
>>> What is 2+2?
[BREAD responds with explanation]

>>> Read main.cu and tell me what the loader does
→ Tool: read_file ✓
[Reads file, BREAD summarizes]

>>> Run "cargo --version" in the agency directory
→ Tool: shell ✓
[Shows Cargo version]

>>> What files are in the agency/src directory?
→ Tool: list_dir ✓
[Lists files]

/clear
[Clears history]

/help
[Shows all commands]
```

## Architecture at a Glance

```
REPL User Input
    ↓
Build Qwen3 chat template
    ↓
Send to BREAD via stdin
    ↓
Stream tokens back from stdout
    ↓
Parse <tool_call> XML blocks
    ↓
Execute tools (read_file, shell, etc.)
    ↓
Format <tool_response> blocks
    ↓
Re-prompt model (up to 5 iterations)
    ↓
Display final response
```

## Key Features

✅ **Streaming output** — see tokens as they generate
✅ **Tool execution** — automatic function calls with JSON arguments
✅ **Conversation context** — multi-turn with history
✅ **Slash commands** — `/quit`, `/clear`, `/help`, `/tools`, `/depth`, `/history`
✅ **Error handling** — graceful fallbacks if tools fail
✅ **Colored output** — user (cyan), assistant (green), tools (yellow)

## Next Steps

### Immediate
1. **Test it** — run with various prompts and tools
2. **Check output quality** — does BREAD reason correctly with tool results?
3. **Verify tool execution** — are read_file, shell, etc. working?

### Short-term (Days)
1. **Add more tools** — git commands, file search, codebase analysis
2. **Batch tool execution** — run multiple tools in parallel
3. **Optimize system prompt** — refine instructions for better reasoning

### Medium-term (Weeks)
1. **BUTTER integration** — add multi-user queuing + benchmarking
2. **SRIRACHA combo** — enable speculative decoding for 3-4x speedup
3. **Tool result caching** — avoid re-reading same files
4. **Persistent state** — save/load conversation history

### Long-term (Months)
1. **Multi-threaded** — support concurrent users
2. **Web UI** — HTTP API + frontend
3. **EAGLE-3 draft** — lightweight draft model for even faster agent loop
4. **Knowledge base** — semantic search over codebase

## Troubleshooting

### bread.exe not found

```
Error: BREAD process exited unexpectedly
```

**Fix:** Ensure bread.exe is in the same directory or pass `--bread` arg:
```powershell
.\agency.exe --bread C:\path\to\bread.exe
```

### No output from BREAD

**Check:** Is BREAD_READY being printed?
```powershell
.\bread.exe --server --tokens 50
# You should see: BREAD_READY
# Then type a prompt and press Enter
# You should see: --- Generated output --- followed by tokens
# Then: BREAD_END
```

### Build fails

**Check Rust version:**
```bash
rustc --version
cargo --version
```

Need Rust 1.70+. Update with: `rustup update`

### Tools not executing

**Check:** Is model recognizing tool format?

Try `/depth 10` to allow more iterations, then ask: "Use the read_file tool to read main.rs"

## File Manifest

```
C:\bread_v2\
├── CLAUDE.md                  (updated with Phase 7: AGENCY)
├── AGENCY_QUICKSTART.md       (this file)
├── main.cu                    (updated: --server mode)
├── agency/                    (NEW)
│   ├── Cargo.toml
│   ├── build.bat
│   ├── README.md              (detailed docs)
│   ├── src/
│   │   ├── main.rs
│   │   ├── bread.rs
│   │   ├── hermes.rs
│   │   ├── tools.rs
│   │   ├── repl.rs
│   │   └── (no tests yet)
│   └── target/release/
│       └── agency.exe         (2.3 MB binary)
```

## Questions?

- See `agency/README.md` for detailed architecture
- Check `src/*.rs` files for implementation details
- Run `./agency --help` for CLI options
- Type `/help` in REPL for interactive commands

---

**Status:** ✅ Complete and buildable

**Next major feature:** BUTTER (orchestration + benchmarking) or SRIRACHA integration
