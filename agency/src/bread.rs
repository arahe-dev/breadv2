use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Child, Stdio};
use anyhow::{Result, anyhow};

pub struct BreadServer {
    _child: Child,  // Keep child alive; prefix with _ to silence unused warning
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    pub max_tokens: u32,
}

impl BreadServer {
    /// Start a BREAD server process and wait for BREAD_READY signal
    pub fn start(bread_exe: &Path, max_tokens: u32) -> Result<Self> {
        let mut child = Command::new(bread_exe)
            .arg("--server")
            .arg("--tokens")
            .arg(max_tokens.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())  // Suppress debug output
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to open BREAD stdin"))?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| anyhow!("Failed to open BREAD stdout"))?,
        );

        let mut server = Self { _child: child, stdin, stdout, max_tokens };

        // Wait for BREAD_READY signal
        let mut line = String::new();
        loop {
            line.clear();
            let n = server.stdout.read_line(&mut line)?;
            if n == 0 {
                return Err(anyhow!("BREAD process exited unexpectedly"));
            }
            if line.trim() == "BREAD_READY" {
                break;
            }
        }

        Ok(server)
    }

    /// Send a prompt and get streamed response tokens
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        // Send prompt to BREAD stdin, followed by double newline to signal end
        write!(self.stdin, "{}\n\n", prompt)?;
        self.stdin.flush()?;

        // Read tokens until BREAD_END sentinel
        let mut response = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            match self.stdout.read_line(&mut line) {
                Ok(0) => break,
                Ok(_n) => {
                    let trimmed = line.trim_end();

                    // Stop at BREAD_END sentinel
                    if trimmed == "BREAD_END" {
                        break;
                    }

                    // Collect all non-empty lines as generated output
                    if !trimmed.is_empty() {
                        response.push_str(&line);
                    }
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Failed to read from BREAD: {}", e));
                }
            }
        }

        Ok(response.trim_end().to_string())
    }
}
