/// Test demonstrating the critical tool call extraction regex bug
///
/// This test shows that the current regex in hermes.rs cannot extract
/// tool calls with nested JSON objects (which is ALL tool calls).

use regex::Regex;
use serde_json::Value;

fn main() {
    println!("===== AGENCY REGEX BUG DEMONSTRATION =====\n");

    // The CURRENT regex from hermes.rs:123
    let broken_regex = Regex::new(r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#).unwrap();

    // Example 1: Simple nested object (what the model actually outputs)
    let example_1 = r#"The file is a CUDA program. Let me read it.
<tool_call>
{"name": "read_file", "arguments": {"path": "main.cu"}}
</tool_call>
I can see it defines..."#;

    println!("Test 1: Tool call with nested arguments");
    println!("Input:\n{}\n", example_1);

    let captures: Vec<_> = broken_regex.captures_iter(example_1).collect();
    println!("Regex captures found: {}", captures.len());

    if captures.is_empty() {
        println!("❌ FAILED: Regex found 0 tool calls (expected 1)\n");
    } else {
        for cap in &captures {
            if let Some(m) = cap.get(1) {
                println!("Captured: {}", m.as_str());
                match serde_json::from_str::<Value>(m.as_str()) {
                    Ok(json) => {
                        println!("  Parsed as JSON: OK");
                        if let Some(name) = json.get("name") {
                            println!("  Tool name: {}", name);
                        }
                    }
                    Err(e) => {
                        println!("  Parse error: {} ❌", e);
                    }
                }
            }
        }
    }

    println!("\n---\n");

    // Example 2: Even simpler - one level of nesting
    let example_2 = r#"<tool_call>
{"name": "test", "args": {"x": 1}}
</tool_call>"#;

    println!("Test 2: Minimal nested object");
    println!("Input: {}\n", example_2);

    let captures: Vec<_> = broken_regex.captures_iter(example_2).collect();
    println!("Regex captures found: {}", captures.len());

    if let Some(cap) = captures.first() {
        if let Some(m) = cap.get(1) {
            let captured = m.as_str();
            println!("Captured text: '{}'", captured);
            println!("Expected text: '{{\n  \"name\": \"test\",\n  \"args\": {{\"x\": 1}}\n}}'");
            println!("Match: {}", if captured.contains("args") { "❌ PARTIAL" } else { "❌ FAILED" });
        }
    } else {
        println!("❌ FAILED: No match found");
    }

    println!("\n---\n");

    // Now show the FIXED regex
    println!("FIXED REGEX TEST:\n");
    let fixed_regex = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();

    let example_3 = r#"Let me read the file.
<tool_call>
{"name": "read_file", "arguments": {"path": "main.cu"}}
</tool_call>
This is a CUDA file because..."#;

    println!("Input:\n{}\n", example_3);

    let captures: Vec<_> = fixed_regex.captures_iter(example_3).collect();
    println!("Fixed regex captures found: {}", captures.len());

    if !captures.is_empty() {
        for cap in &captures {
            if let Some(m) = cap.get(1) {
                let json_str = m.as_str();
                println!("Captured: {}", json_str);
                match serde_json::from_str::<Value>(json_str) {
                    Ok(json) => {
                        println!("✅ Parsed successfully as JSON");
                        if let Some(name) = json.get("name") {
                            println!("   Tool name: {}", name);
                        }
                        if let Some(args) = json.get("arguments") {
                            println!("   Arguments: {}", args);
                        }
                    }
                    Err(e) => {
                        println!("❌ Parse error: {}", e);
                    }
                }
            }
        }
    } else {
        println!("❌ Fixed regex also failed!");
    }

    println!("\n===== SUMMARY =====");
    println!("Current regex: BROKEN (cannot match nested JSON)");
    println!("Fixed regex:  WORKING (matches complete JSON objects)");
}
