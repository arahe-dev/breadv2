#!/usr/bin/env python3
"""
Test gemma4:e2b as draft model for BREAD speculative decoding.
Measures acceptance rate (how often gemma4 token matches BREAD's argmax).
"""

import subprocess
import time
import json
import sys
import requests
from threading import Thread

# Configuration
BREAD_EXE = r".\bread.exe"
BREAD_TOKENS = 50
BREAD_PROMPT = """<|im_start|>user
What is the capital of France? Please provide a detailed response.
<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

OLLAMA_MODEL = "gemma4:e2b"
OLLAMA_API = "http://localhost:11434/api/generate"

def start_bread_server():
    """Start BREAD in --server mode."""
    print("[BREAD] Starting server mode...")
    proc = subprocess.Popen(
        [BREAD_EXE, "--server", "--tokens", str(BREAD_TOKENS)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Wait for BREAD_READY signal
    for line in proc.stdout:
        print(f"[BREAD] {line.rstrip()}")
        if "BREAD_READY" in line:
            print("[BREAD] Server ready!")
            return proc

    raise RuntimeError("BREAD server failed to start")

def get_draft_token(context, ollama_api=OLLAMA_API):
    """Call ollama API to get draft token from gemma4:e2b."""
    try:
        response = requests.post(
            ollama_api,
            json={
                "model": OLLAMA_MODEL,
                "prompt": context,
                "stream": False,
                "num_predict": 1,
            },
            timeout=10
        )
        response.raise_for_status()

        # Extract last token from response
        text = response.json()["response"].strip()
        if text:
            return text.split()[-1]  # crude but works
        return None
    except Exception as e:
        print(f"[OLLAMA] Error: {e}")
        return None

def run_test():
    """Run speculative decoding test."""
    # Start BREAD server
    bread_proc = start_bread_server()
    time.sleep(0.5)

    try:
        print("\n[TEST] Starting speculative decoding test...")
        print(f"[TEST] Target: {BREAD_TOKENS} tokens")
        print(f"[TEST] Draft model: {OLLAMA_MODEL}\n")

        # Send prompt to BREAD
        bread_proc.stdin.write(BREAD_PROMPT + "\n")
        bread_proc.stdin.flush()

        context = BREAD_PROMPT
        accepted = 0
        rejected = 0
        ollama_errors = 0
        start_time = time.time()

        for token_idx in range(BREAD_TOKENS):
            # Read BREAD's token (one token at a time)
            bread_output = bread_proc.stdout.readline().rstrip()

            if "BREAD_END" in bread_output:
                print(f"\n[BREAD] Generation complete at token {token_idx}")
                break

            if not bread_output or bread_output.startswith("["):
                continue

            bread_token = bread_output.split()[-1] if bread_output.split() else ""

            if not bread_token:
                continue

            # Get draft from ollama
            draft_token = get_draft_token(context)

            if draft_token is None:
                ollama_errors += 1
                print(f"[{token_idx}] BREAD='{bread_token}' | OLLAMA=ERROR")
                context += " " + bread_token
                continue

            # Check acceptance
            match = (bread_token.lower() == draft_token.lower())
            accepted += match
            rejected += not match

            if match:
                status = "✓ ACCEPT"
            else:
                status = "✗ REJECT"

            print(f"[{token_idx}] BREAD='{bread_token}' | DRAFT='{draft_token}' | {status}")

            # Always advance with BREAD's token (verified token)
            context += " " + bread_token

        elapsed = time.time() - start_time
        total = accepted + rejected

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        if total > 0:
            acceptance_rate = (accepted / total) * 100
            print(f"Acceptance rate: {acceptance_rate:.1f}% ({accepted}/{total})")
        else:
            print("No tokens generated")
        print(f"Ollama errors: {ollama_errors}")
        print(f"Elapsed time: {elapsed:.2f}s")
        print(f"Throughput: {total/elapsed:.2f} tok/s (BREAD only, no verification overhead)")
        print("="*60)

        if total > 0:
            if acceptance_rate > 20:
                print("✓ Gemma4:e2b shows promise (>20%). Worth implementing GGUF loader.")
            elif acceptance_rate > 10:
                print("◐ Gemma4:e2b is marginal (10-20%). SRIRACHA comparable.")
            else:
                print("✗ Gemma4:e2b too weak (<10%). Stick with SRIRACHA.")

    finally:
        # Kill BREAD server
        bread_proc.terminate()
        try:
            bread_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            bread_proc.kill()
        print("\n[BREAD] Server stopped")

if __name__ == "__main__":
    # Check if ollama is running
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        print(f"✓ Ollama running (found {len(resp.json().get('models', []))} models)")
    except:
        print("✗ Ollama not running on localhost:11434")
        print("  Start ollama: ollama serve")
        sys.exit(1)

    run_test()
