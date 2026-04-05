use std::fs;
use std::process::Command;
use anyhow::Result;

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
