use std::fs;
use std::process::Command;
use anyhow::Result;
use regex::Regex;
use serde_json::json;

/// Represents a parsed tool call from BREAD output
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(name: String, arguments: serde_json::Value) -> Self {
        Self { name, arguments }
    }
}

pub fn read_file(path: &str) -> Result<String> {
    let content = fs::read_to_string(path)?;
    // Limit to first 8KB to avoid overwhelming the model
    Ok(content.chars().take(8192).collect())
}

pub fn list_dir(path: &str) -> Result<String> {
    let path = if path.is_empty() { "." } else { path };
    let entries = fs::read_dir(path)?;
    let mut result = String::new();

    for entry in entries {
        if let Ok(entry) = entry {
            if let Ok(metadata) = entry.metadata() {
                let is_dir = metadata.is_dir();
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                let marker = if is_dir { "/" } else { "" };
                result.push_str(&format!("{}{}\n", name_str, marker));
            }
        }
    }

    Ok(result)
}

pub fn shell(command: &str) -> Result<String> {
    let output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(&["/C", command])
            .output()?
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()?
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let combined = if stderr.is_empty() {
        stdout.to_string()
    } else {
        format!("{}\n{}", stdout, stderr)
    };

    // Limit output
    Ok(combined.chars().take(4096).collect())
}

pub fn write_file(path: &str, content: &str) -> Result<String> {
    fs::write(path, content)?;
    Ok(format!("Written {} bytes to {}", content.len(), path))
}

pub fn execute_tool(name: &str, arguments: &serde_json::Value) -> Result<String> {
    match name {
        "read_file" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'path' argument"))?;
            read_file(path)
        }
        "list_dir" => {
            let path = arguments["path"].as_str().unwrap_or("");
            list_dir(path)
        }
        "shell" => {
            let command = arguments["command"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;
            shell(command)
        }
        "write_file" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'path' argument"))?;
            let content = arguments["content"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'content' argument"))?;
            write_file(path, content)
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}

/// Extract a single tool call from text
/// Looks for: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
pub fn extract_tool_call(text: &str) -> Option<ToolCall> {
    // Regex pattern for tool_call blocks
    let re = Regex::new(r#"<tool_call>\s*(\{.*?\})\s*</tool_call>"#)
        .expect("Failed to compile regex");

    if let Some(cap) = re.captures(text) {
        let json_str = cap.get(1)?.as_str();

        // Try to parse the JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = json["name"].as_str()?.to_string();
            let arguments = json.get("arguments")
                .cloned()
                .unwrap_or_else(|| json! ({}));

            return Some(ToolCall { name, arguments });
        }
    }

    None
}

/// Extract all tool calls from text (handles multiple tool_call blocks)
pub fn extract_all_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut tool_calls = Vec::new();

    // Use a non-greedy regex to find all tool_call blocks
    let re = Regex::new(r#"<tool_call>\s*(\{.*?\})\s*</tool_call>"#)
        .expect("Failed to compile regex");

    for cap in re.captures_iter(text) {
        if let Some(json_str) = cap.get(1) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str.as_str()) {
                if let Some(name) = json["name"].as_str() {
                    let arguments = json.get("arguments")
                        .cloned()
                        .unwrap_or_else(|| json! ({}));

                    tool_calls.push(ToolCall {
                        name: name.to_string(),
                        arguments,
                    });
                }
            }
        }
    }

    tool_calls
}

/// Remove tool call blocks from text (for display purposes)
pub fn remove_tool_calls_from_text(text: &mut String) {
    let re = Regex::new(r#"<tool_call>\s*\{.*?\}\s*</tool_call>"#)
        .expect("Failed to compile regex");
    *text = re.replace_all(text, "").to_string();
}

/// Check if text contains any tool call blocks
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>") && text.contains("</tool_call>")
}

/// Format a tool result for feeding back to BREAD
pub fn format_tool_result(tool_name: &str, result: &str) -> String {
    format!(
        "<tool_result>\n{{\n  \"name\": \"{}\",\n  \"content\": {}\n}}\n</tool_result>",
        tool_name,
        serde_json::to_string(&result).unwrap_or_else(|_| format!("\"{}\"", result))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_call() {
        let text = r#"I'll read that file for you.
<tool_call>
{"name": "read_file", "arguments": {"path": "bread.h"}}
</tool_call>
Let me continue..."#;

        let tool_call = extract_tool_call(text).expect("Failed to extract tool call");
        assert_eq!(tool_call.name, "read_file");
        assert_eq!(tool_call.arguments["path"].as_str().unwrap(), "bread.h");
    }

    #[test]
    fn test_extract_all_tool_calls() {
        let text = r#"
<tool_call>{"name": "read_file", "arguments": {"path": "file1.txt"}}</tool_call>
Some text here.
<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
"#;

        let calls = extract_all_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "shell");
    }

    #[test]
    fn test_remove_tool_calls() {
        let mut text = "Before <tool_call>{\"name\": \"test\", \"arguments\": {}}</tool_call> after".to_string();
        remove_tool_calls_from_text(&mut text);
        assert_eq!(text, "Before  after");
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("<tool_call>{}</tool_call>"));
        assert!(!has_tool_calls("no tool calls here"));
    }

    #[test]
    fn test_format_tool_result() {
        let result = format_tool_result("read_file", "file contents here");
        assert!(result.contains("\"name\": \"read_file\""));
        assert!(result.contains("file contents here"));
    }
}
