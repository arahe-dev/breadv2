# AGENCY Fixes (Priority Order)

## Fix 1: Tool Call Extraction Regex ⭐ CRITICAL

**File:** `agency/src/hermes.rs`
**Lines:** 123
**Impact:** Enables 100% of tool calls (currently 0% work)
**Time:** 1 minute

### Change

```diff
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
-   let re = Regex::new(r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#).unwrap();
+   let re = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();
    re.captures_iter(text)
        .filter_map(|cap| {
            cap.get(1).and_then(|m| ToolCall::parse(m.as_str()))
        })
        .collect()
}
```

### Verification

```bash
cargo test extract_tool_calls
cargo run --example test_regex_bug --release
```

Expected output:
```
✅ Fixed regex: WORKING
   Tool name: "read_file"
   Arguments: {"path":"main.cu"}
```

---

## Fix 2: BreadServer Output Parsing 🔴 HIGH

**File:** `agency/src/bread.rs`
**Lines:** 52-94 (the `generate` method)
**Impact:** Empty responses when BREAD runs in --server mode
**Time:** 15 minutes
**Root Cause:** Code expects "--- Generated output ---" banner which BREAD doesn't print in server mode

### Current Code (Broken)

```rust
pub fn generate(&mut self, prompt: &str) -> Result<String> {
    write!(self.stdin, "{}\n\n", prompt)?;
    self.stdin.flush()?;

    let mut response = String::new();
    let mut line = String::new();
    let mut started = false;
    let mut in_generation = false;

    loop {
        line.clear();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 { break; }

        let trimmed = line.trim_end();

        // Skip BREAD's header noise until we see "--- Generated output ---"
        // ❌ PROBLEM: This banner is never printed in server mode!
        if !started {
            if trimmed == "--- Generated output ---" {
                started = true;
                in_generation = true;
            }
            continue;  // Skip everything until banner is found
        }

        // Stop at BREAD_END sentinel
        if trimmed == "BREAD_END" { break; }

        // Collect generated tokens
        if in_generation && !trimmed.is_empty() {
            response.push_str(&line);
        }
    }

    Ok(response.trim_end().to_string())
}
```

### Fixed Code

```rust
pub fn generate(&mut self, prompt: &str) -> Result<String> {
    write!(self.stdin, "{}\n\n", prompt)?;
    self.stdin.flush()?;

    let mut response = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 { break; }

        let trimmed = line.trim_end();

        // Stop at BREAD_END sentinel
        if trimmed == "BREAD_END" { break; }

        // Collect all non-empty lines (no banner required)
        if !trimmed.is_empty() {
            response.push_str(&line);
        }
    }

    Ok(response.trim_end().to_string())
}
```

### Rationale

The banner is printed in single-mode (non-server) to separate input from output. In server mode (`--server` flag), BREAD doesn't print this banner. AGENCY runs BREAD with `--server`, so the banner is never present. The code waits forever for a marker that never comes, then timeouts or returns empty string.

**Simple fix:** Collect all output until BREAD_END, don't wait for a banner.

### Verification

```bash
# Test that BREAD's server mode output is captured
./bread.exe --server --tokens 10 --no-progress
# Send: "Hello\n\n"
# BREAD outputs tokens directly, then BREAD_END
# AGENCY should capture tokens, not wait for banner
```

---

## Fix 3: Conversation History Truncation 🟡 MEDIUM

**File:** `agency/src/repl.rs`
**Lines:** 60-131 (the `generate_with_tools` function)
**Impact:** Token overflow after ~5 tool interactions
**Time:** 45 minutes
**Root Cause:** Conversation history grows unbounded; each tool result adds ~2000 tokens to context

### Problem

Each tool call adds to conversation:
- User message: ~20 tokens
- Tool result (e.g., 8KB file): ~2000 tokens
- Assistant response: ~256 tokens
- Total: ~2276 tokens per iteration

With max_tokens=256, after 2-3 iterations the prompt becomes:
```
prompt (2000 tokens) + history (5000 tokens) + new response = 7000 tokens
```

BREAD can't fit this in a 256-token window. Generation fails or becomes incoherent.

### Solution

Implement sliding window: Keep removing oldest messages when history is too large.

```rust
const MAX_HISTORY_TOKENS: usize = 2000; // 8x max_tokens to keep room

fn estimate_prompt_tokens(history: &[Message], system: &str) -> usize {
    // Rough estimate: 1 token per 4 chars
    let system_tokens = system.len() / 4;
    let history_tokens: usize = history.iter()
        .map(|m| m.content.len() / 4)
        .sum();
    system_tokens + history_tokens + 100 // fudge factor
}

fn truncate_history_if_needed(history: &mut Vec<Message>, system: &str, max_tokens: usize) {
    while estimate_prompt_tokens(history, system) > max_tokens / 2 {
        // Remove oldest assistant+user pair (index 1 and 2)
        if history.len() > 2 {
            history.remove(1); // Remove assistant response
            if history.len() > 1 {
                history.remove(1); // Remove user query
            }
        } else {
            break;
        }
    }
}

// In generate_with_tools(), before building prompt:
fn generate_with_tools(
    server: &mut BreadServer,
    history: &mut Vec<Message>,
    max_iterations: usize,
    max_tokens: u32,  // Add this parameter
) -> Result<()> {
    let system = hermes::system_prompt();

    truncate_history_if_needed(history, &system, max_tokens as usize);

    for iteration in 0..max_iterations {
        // ... rest of code
    }

    Ok(())
}

// In run_repl(), pass max_tokens:
if let Err(e) = generate_with_tools(server, &mut history, max_tool_iterations, server.max_tokens) {
    // ...
}
```

### Verification

```bash
# Have AGENCY read 5 large files in sequence
>>> Read file1.cu
>>> Read file2.cu
>>> Read file3.cu
>>> Read file4.cu
>>> Read file5.cu
# Should not timeout or fail (currently fails after 2-3)
```

---

## Fix 4: stderr Suppression ⏱️ MEDIUM

**File:** `agency/src/bread.rs`
**Lines:** 20
**Impact:** User can't see why BREAD is slow (`--hooks-debug` shows nothing)
**Time:** 60 minutes (depends on approach)
**Root Cause:** stderr is redirected to null to suppress model loading messages

### Problem

BREAD writes to stderr:
- Model loading progress
- Per-layer timing (from `--hooks-debug`)
- Error messages

AGENCY suppresses all of it with `Stdio::null()`.

### Simple Fix (15 min)

Show stderr unconditionally:

```diff
- .stderr(Stdio::null())
+ .stderr(Stdio::inherit())
```

**Trade-off:** BREAD's startup noise appears in user's terminal.

### Better Fix (60 min)

Read stderr in a separate thread, filter for important messages:

```rust
use std::io::BufRead;
use std::thread;

pub struct BreadServer {
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    // stderr reader runs in background thread
}

impl BreadServer {
    pub fn start(bread_exe: &Path, max_tokens: u32) -> Result<Self> {
        let mut child = Command::new(bread_exe)
            .arg("--server")
            .arg("--tokens").arg(max_tokens.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())  // Capture stderr
            .spawn()?;

        let stderr = child.stderr.take().ok_or_else(|| anyhow!("No stderr"))?;
        let stderr_reader = BufReader::new(stderr);

        // Spawn thread to read and filter stderr
        thread::spawn(move || {
            for line in stderr_reader.lines() {
                if let Ok(line) = line {
                    // Only print important messages
                    if line.contains("[LAYER_TIMING]")
                        || line.contains("[HOOKS]")
                        || line.contains("[PROGRESS]")
                        || line.contains("error")
                        || line.contains("Error") {
                        eprintln!("{}", line);
                    }
                }
            }
        });

        // ... rest of BreadServer::start
        Ok(server)
    }
}
```

**Trade-off:** More complex, but selective output. Better user experience.

### Verification

```bash
./agency.exe --bread "./bread.exe --hooks-debug" --tokens 256
>>> Hello
# Should see:
# [HOOKS] Enabled layer timing
# [LAYER_TIMING] Per-layer forward times: ...
```

---

## Fix 5: bread.exe Path Validation ✅ LOW

**File:** `agency/src/main.rs`
**Lines:** 30-33
**Impact:** Confusing error if user provides wrong path
**Time:** 15 minutes

### Current Code

```rust
"--bread" => {
    if let Some(path) = args.next() {
        config.bread_exe = PathBuf::from(path);  // No validation!
    }
}
```

### Fixed Code

```rust
use std::path::Path;

"--bread" => {
    if let Some(path) = args.next() {
        let pb = PathBuf::from(&path);
        if !pb.exists() {
            eprintln!("Error: bread.exe not found at: {}", path);
            eprintln!("Please check the path and try again.");
            return Ok(());
        }
        if !pb.is_file() {
            eprintln!("Error: {} is not a file", path);
            return Ok(());
        }
        config.bread_exe = pb;
    }
}
```

### Verification

```bash
./agency.exe --bread ./nonexistent.exe
# Error: bread.exe not found at: ./nonexistent.exe
```

---

## Fix 6: Silent Token Parse Failure ✅ LOW

**File:** `agency/src/main.rs`
**Lines:** 37
**Impact:** Unexpected behavior if user types bad token count
**Time:** 15 minutes

### Current Code

```rust
config.max_tokens = tokens.parse().unwrap_or(256);  // Silent fallback!
```

### Fixed Code

```rust
config.max_tokens = match tokens.parse::<u32>() {
    Ok(n) if n > 0 => n,
    _ => {
        eprintln!("Warning: invalid token count '{}', using default 256", tokens);
        256
    }
};
```

### Verification

```bash
./agency.exe --tokens abc
# Warning: invalid token count 'abc', using default 256
```

---

## Recommended Application Order

### Day 1 (30 min, gets it working)
1. ✅ Fix 1 (regex)
2. ✅ Fix 2 (output parsing)

### Day 2 (45 min, makes it robust)
3. ✅ Fix 3 (history truncation)

### Week 1 (optional improvements)
4. ⏱️ Fix 4 (stderr filtering)
5. ✅ Fix 5 (validation)
6. ✅ Fix 6 (error messages)

## Testing Checklist

- [ ] Fix 1: `cargo test extract_tool_calls` passes
- [ ] Fix 1: `cargo run --example test_regex_bug` shows ✅
- [ ] Fix 2: Tool calls are extracted from BREAD output
- [ ] Fix 2: Multiple tool calls in one response work
- [ ] Fix 3: 10+ tool interactions don't fail
- [ ] Fix 4: Layer timing visible when `--hooks-debug` used
- [ ] Fix 5: Invalid bread.exe path gives helpful error
- [ ] Fix 6: Invalid token count shows warning

---

## All Fixes Combined: Final Diff

```bash
# File: agency/src/hermes.rs, line 123
# Change 1 character:
- let re = Regex::new(r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#).unwrap();
+ let re = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();

# File: agency/src/bread.rs, lines 52-94
# Simplify output parsing (see Fix 2)

# File: agency/src/repl.rs, line 60-131
# Add history truncation (see Fix 3)

# And 3 small validation/error message improvements
```

---

Total effort to make AGENCY fully functional: **~3 hours**
- Fix 1-2: 15 minutes (get it working)
- Fix 3: 45 minutes (handle long conversations)
- Fix 4: 60 minutes (improve debugging experience)
- Fix 5-6: 30 minutes (nice error messages)

**Minimum viable:** Just Fix 1 (regex). Takes 1 minute. AGENCY works.
