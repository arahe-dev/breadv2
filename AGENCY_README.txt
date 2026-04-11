================================================================================
                        AGENCY INVESTIGATION RESULTS
================================================================================

BREAD AGENCY is a tool-calling agent built in Rust that wraps the BREAD
inference engine (via --server mode) with Hermes-format prompt handling.

STATUS: 🔴 BROKEN (but 1-line fixable)

KEY FINDING: Critical regex bug prevents ALL tool calls from being extracted.
The regex pattern cannot match nested JSON objects, which is what all
Hermes tool calls contain.

================================================================================
                           CRITICAL BUG LOCATION
================================================================================

File: C:\bread_v2\agency\src\hermes.rs
Line: 123

Current (BROKEN):
  let re = Regex::new(r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#).unwrap();

Fixed (WORKING):
  let re = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();

The pattern \{[^}]*\} stops at the FIRST closing brace, so it cannot match
nested JSON. All tool calls have nested arguments, so 0% of calls are extracted.

PROOF:
  Run: cargo run --example test_regex_bug --release
  
  Output shows:
    ❌ Current regex: BROKEN (found 0 of 1 tool calls)
    ✅ Fixed regex:  WORKING (found 1 of 1 tool calls)

================================================================================
                           DOCUMENTATION PROVIDED
================================================================================

1. AGENCY_INVESTIGATION.md (2000+ lines)
   - Complete codebase analysis
   - All 6 issues identified and explained
   - Severity ratings and test cases
   - Root causes and impact analysis

2. AGENCY_FIXES.md (500+ lines)
   - Detailed fixes for all 6 issues
   - Code diffs showing exact changes
   - Verification procedures
   - Time estimates for each fix

3. AGENCY_SUMMARY.md (200 lines)
   - Quick overview of the bug
   - Impact analysis
   - 1-line fix with proof
   - Deployment path

4. agency/examples/test_regex_bug.rs
   - Runnable demonstration of the bug
   - Proof that fixed regex works
   - Compile and run with: cargo run --example test_regex_bug --release

================================================================================
                          ISSUES FOUND (6 TOTAL)
================================================================================

1. ⭐ CRITICAL: Tool call regex bug
   File: hermes.rs:123
   Impact: 0% of tool calls work (currently broken for all use cases)
   Fix time: 1 minute
   
2. 🔴 HIGH: Output parsing fragility
   File: bread.rs:74-90
   Impact: Empty responses from BREAD in server mode
   Fix time: 15 minutes
   
3. 🟡 MEDIUM: No history truncation
   File: repl.rs:11
   Impact: Token overflow after 5+ tool interactions
   Fix time: 45 minutes
   
4. 🟡 MEDIUM: stderr suppression
   File: bread.rs:20
   Impact: Can't see layer timing with --hooks-debug
   Fix time: 60 minutes
   
5. ✅ LOW: No bread.exe path validation
   File: main.rs:30-33
   Impact: Confusing errors when path is wrong
   Fix time: 15 minutes
   
6. ✅ LOW: Silent token parse failure
   File: main.rs:37
   Impact: Unexpected behavior with bad input
   Fix time: 15 minutes

================================================================================
                            QUICK START FIX
================================================================================

To get AGENCY working TODAY:

1. Edit: C:\bread_v2\agency\src\hermes.rs
2. Line: 123
3. Change: r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#
4. To:     r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#
5. Build:  cargo build --release
6. Run:    ./target/release/agency.exe --bread ../bread.exe --tokens 256

Then you can:
  >>> Read main.cu and summarize it
  → Tool: read_file ✓
  [File content received and analyzed]

================================================================================
                        RECOMMENDED FIX ORDER
================================================================================

PRIORITY 1 (15 min, enables tool calling):
  - Fix #1: Regex bug (1 min)
  - Fix #2: Output parsing (15 min)
  ➜ Result: AGENCY works for simple queries

PRIORITY 2 (45 min, handles multi-turn):
  - Fix #3: History truncation (45 min)
  ➜ Result: AGENCY handles long conversations

PRIORITY 3 (75 min, polish):
  - Fix #4: stderr filtering (60 min)
  - Fix #5: Path validation (15 min)
  ➜ Result: Better debugging and error messages

PRIORITY 4 (15 min, nice-to-have):
  - Fix #6: Token parse error handling (15 min)

Total for full fix: ~3 hours
Minimum viable fix: 1 line (Fix #1)

================================================================================
                       ARCHITECTURE OVERVIEW
================================================================================

main.rs
  └─ Start BreadServer (--server mode)
  └─ Run REPL loop
      └─ Await user input
      └─ Call generate_with_tools()
          ├─ Build Qwen3.5 chat prompt
          ├─ Send to BREAD via stdin
          ├─ Read tokens from stdout until BREAD_END
          ├─ Extract tool calls from response  ← BUG HERE (regex)
          ├─ Execute tools (read_file, list_dir, shell, write_file)
          ├─ Format tool responses
          ├─ Loop up to max_iterations times
          └─ Return final response

Files:
  - main.rs         Entry point, CLI args
  - bread.rs        BreadServer subprocess management
  - repl.rs         REPL loop, generate_with_tools()
  - hermes.rs       Qwen3.5 format, tool call extraction ← BUG
  - tools.rs        Tool implementations

================================================================================
                            TESTING PROOF
================================================================================

To verify the bug and fix work, run:

  cd C:\bread_v2\agency
  cargo run --example test_regex_bug --release

Output demonstrates:
  ❌ Current regex: BROKEN (cannot match nested JSON)
  ❌ Test 1 fails: Tool call with nested arguments
  ❌ Test 2 fails: Minimal nested object
  ✅ Fixed regex: WORKING (matches complete JSON)
  ✅ Fixed regex extracts full JSON correctly

This proves:
  1. Current code is broken (0% of tool calls extracted)
  2. 1-line fix solves the problem completely
  3. Fixed regex extracts JSON correctly

================================================================================
                          NEXT STEPS
================================================================================

1. Read AGENCY_INVESTIGATION.md for complete analysis
2. Read AGENCY_FIXES.md for detailed code changes
3. Run: cargo run --example test_regex_bug --release (to see proof)
4. Apply Fix #1 (1-line change to hermes.rs)
5. Apply Fix #2 (simplify bread.rs output parsing)
6. Test with: >>> Read main.cu
7. Apply remaining fixes as time permits

================================================================================
                            SUMMARY
================================================================================

AGENCY is a well-designed Rust tool-calling agent that integrates nicely with
BREAD's inference engine. Unfortunately, it has a critical regex bug that makes
it completely non-functional.

The bug is a SINGLE REGEX PATTERN that cannot match nested JSON objects.
This affects 100% of tool calls and makes the entire tool-calling feature
unusable.

The fix is ONE LINE of code.

Once fixed, AGENCY enables BREAD to:
  • Read files and analyze code
  • List directories
  • Run shell commands
  • Write files
  • Reason about results and iterate

Total effort to fully fix: ~3 hours
Minimum viable fix: ~1 minute (one regex change)

======================== Investigation Complete =============================
