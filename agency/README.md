# AGENCY — Hermes Agent CLI for BREAD

A Rust-based interactive CLI that wraps BREAD (the custom CUDA inference engine) with Hermes tool-calling support for agentic tasks.

## Features

- **Interactive REPL** with history and line editing (rustyline)
- **Hermes XML tool-calling** format for structured function calls
- **Built-in tools**: `read_file`, `list_dir`, `shell`, `write_file`
- **Multi-turn conversations** with Qwen3.5 chat template formatting
- **Automatic tool execution** with result feedback to model
- **Streaming output** from BREAD inference

## Quick Start

### Prerequisites

- **Rust** 1.70+ (for building)
- **BREAD** executable with `--server` mode (requires updated main.cu)
- Windows 11 or similar (built for Windows x86-64)

### Build

```bash
cd C:\bread_v2\agency
cargo build --release
```

Binary: `target/release/agency.exe`

### Run

First, ensure bread.exe has been built with the latest main.cu (which includes --server mode):

```bash
# Build BREAD with server mode support
cd C:\bread_v2
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c bread_utils.c topology.c config_reader.c validate.c layer_logic.cu layer_ops.cu -I. -o bread.exe
```

Then run the agency CLI:

```bash
.\agency\target\release\agency.exe --bread ..\bread.exe --tokens 256
```

Or with defaults (looks for `.\bread.exe`):

```bash
.\agency\target\release\agency.exe
```

## Usage

Once in the REPL, you can:

### User Input

Type your message and press Enter. The agent will:
1. Reason about your request
2. Call tools if needed (automatically detected)
3. Execute tools and feed results back
4. Generate a final response

### Slash Commands

- `/quit` or `/exit` — Exit the REPL
- `/clear` — Clear conversation history
- `/help` — Show available commands
- `/tools` — List available tools
- `/depth N` — Set max tool loop iterations (default: 5)
- `/history` — Show conversation history

## Example Interactions

### Reading Files

```
>>> Read the first 100 lines of main.cu and summarize it
→ Tool: read_file ✓
[BREAD summarizes the file contents]
```

### Shell Commands

```
>>> What files are in the bread_v2 directory? Use the shell if needed.
→ Tool: shell ✓
[Lists directory contents and summarizes]
```

### Multi-turn Conversation

```
>>> What is 2+2?
[BREAD responds with "4"]

>>> Can you explain why?
[Continues conversation with context]
```

## Architecture

### Modules

- **`main.rs`** — CLI entry point and argument parsing
- **`bread.rs`** — `BreadServer` subprocess wrapper
  - Spawns `bread.exe --server`
  - Pipes prompts to stdin
  - Reads streamed tokens from stdout
  - Handles `BREAD_READY` and `BREAD_END` sentinels
- **`hermes.rs`** — Hermes format handler
  - System prompt with tool definitions
  - Chat template formatting (Qwen3.5 `<|im_start|>` style)
  - Tool call parsing (`<tool_call>` XML extraction)
  - Tool response formatting
- **`tools.rs`** — Built-in tool implementations
  - `read_file(path)` — Read file contents (8KB limit)
  - `list_dir(path)` — List directory
  - `shell(cmd)` — Run shell command (4KB output limit)
  - `write_file(path, content)` — Write file
- **`repl.rs`** — Interactive loop
  - `run_repl()` — Main loop with history tracking
  - `generate_with_tools()` — Agentic generation loop (up to N iterations)
  - `handle_slash_command()` — Slash command dispatcher

### Data Flow

```
User Input
    ↓
History + System Prompt + User Message
    ↓
build_prompt() → Qwen3.5 chat format
    ↓
BreadServer.generate() → BREAD stdin
    ↓
Token streaming from BREAD stdout
    ↓
extract_tool_calls() → Find <tool_call> blocks
    ↓
execute_tool() → Run each tool
    ↓
format_tool_response() → Add <tool_response> blocks
    ↓
(loop if more tools or max iterations)
    ↓
Add to history, display to user
```

## Configuration

### Command-line Arguments

- `--bread PATH` — Path to bread.exe (default: `.\bread.exe`)
- `--tokens N` — Max tokens per response (default: 256, passed to bread.exe)
- `--help` — Show help

### System Prompt

The system prompt is embedded in `hermes::system_prompt()` and includes:
- Instructions for tool-calling format
- Tool definitions (JSON schema style)
- Tool response format expectations

## Limitations

- **Single-threaded** — One request at a time (BREAD loads model once and stays warm)
- **Model loads once** — ~20–30 seconds startup time, then requests are fast (~0.5–2s per token)
- **Output limits** — Tools truncate large outputs (8KB files, 4KB shell output) to avoid overwhelming the model
- **No sampling** — BREAD uses greedy argmax; no temperature/top-p control
- **Max iterations** — Tool loop limited to 5 iterations to prevent infinite loops

## Future Enhancements

- **Streaming tool execution** — Print tool results as they complete
- **More tools** — git, http, SQL, etc.
- **Prompt templates** — User-definable system prompts
- **Caching** — Cache tool results across requests
- **Multi-threaded** — Support multiple concurrent users with connection pooling
- **EAGLE-3 draft** — Combine with SRIRACHA speculative decoding for 3-4x speedup

## Building from Source

```bash
cd C:\bread_v2\agency
cargo build --release
```

Requires Rust 1.70+. Dependencies are in `Cargo.toml`:
- `rustyline` — Readline with history
- `colored` — Colored terminal output
- `serde_json` — JSON tool argument parsing
- `regex` — `<tool_call>` block extraction
- `anyhow`, `thiserror` — Error handling
