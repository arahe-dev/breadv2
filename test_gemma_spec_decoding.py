#!/usr/bin/env python3
"""
Speculative decoding test: Gemma4:e2b (draft) vs Gemma4:26b (target)
Measures token alignment and acceptance rate for same-family models.
"""

import requests
import time
import json
import sys
from typing import Tuple, List

OLLAMA_API = "http://localhost:11434/api/generate"
TARGET_MODEL = "gemma4:26b"
DRAFT_MODEL = "gemma4:e2b"

# Test prompts with expected length
# NOTE: Use FACTUAL/DETERMINISTIC prompts only
# (Speculative decoding requires models to give same answer, not creative variation)
PROMPTS = [
    ("What is the capital of France?", 10),
    ("What is 2 plus 2?", 8),
    ("Who wrote Romeo and Juliet?", 10),
]

def get_next_token(model: str, context: str) -> Tuple[str, float]:
    """
    Call Ollama API to get next token from a model.
    Returns: (token, response_time_ms)
    """
    start = time.time()
    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": context,
                "stream": False,
                "num_predict": 1,  # Single token
                "temperature": 0,  # Greedy for reproducibility
            },
            timeout=None  # No timeout
        )
        elapsed = (time.time() - start) * 1000

        if response.status_code != 200:
            return None, elapsed

        data = response.json()
        text = data.get("response", "").strip()

        # Extract just the first word/token
        if text:
            # Split on whitespace and punctuation
            token = text.split()[0] if text.split() else text[0]
            return token, elapsed
        return None, elapsed
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, 0

def test_speculative_decoding(prompt: str, target_tokens: int):
    """
    Run speculative decoding test:
    1. Draft model (e2b) predicts next token
    2. Target model (26b) generates next token
    3. Compare: acceptance = draft matches target
    """
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt[:60]}")
    print(f"Target: {target_tokens} tokens")
    print(f"{'='*70}\n")

    context = prompt
    accepted = 0
    rejected = 0
    draft_errors = 0
    target_errors = 0

    draft_times = []
    target_times = []

    start_total = time.time()

    for pos in range(target_tokens):
        # Step 1: Draft model predicts
        draft_token, draft_time = get_next_token(DRAFT_MODEL, context)
        draft_times.append(draft_time)

        if draft_token is None:
            draft_errors += 1
            print(f"[{pos}] DRAFT ERROR")
            # Continue with target only
            target_token, target_time = get_next_token(TARGET_MODEL, context)
            target_times.append(target_time)
            if target_token:
                context += " " + target_token
            rejected += 1
            continue

        # Step 2: Target model generates
        target_token, target_time = get_next_token(TARGET_MODEL, context)
        target_times.append(target_time)

        if target_token is None:
            target_errors += 1
            print(f"[{pos}] TARGET ERROR")
            context += " " + draft_token  # Use draft as fallback
            rejected += 1
            continue

        # Step 3: Check acceptance
        match = (draft_token.lower() == target_token.lower())

        if match:
            accepted += 1
            status = "✓"
        else:
            rejected += 1
            status = "✗"

        print(f"[{pos:2d}] Draft: '{draft_token:12s}' | Target: '{target_token:12s}' | {status} "
              f"({draft_time:.0f}ms | {target_time:.0f}ms)")

        # Always use target token (verified correct)
        context += " " + target_token

    total_time = time.time() - start_total
    total = accepted + rejected

    print(f"\n{'-'*70}")
    print(f"RESULTS")
    print(f"{'-'*70}")

    if total > 0:
        acceptance_rate = (accepted / total) * 100
        print(f"Acceptance rate: {acceptance_rate:.1f}% ({accepted}/{total})")
    else:
        print("No tokens generated")
        return

    if draft_errors > 0:
        print(f"Draft model errors: {draft_errors}")
    if target_errors > 0:
        print(f"Target model errors: {target_errors}")

    avg_draft_time = sum(draft_times) / len(draft_times) if draft_times else 0
    avg_target_time = sum(target_times) / len(target_times) if target_times else 0

    print(f"\nTiming:")
    print(f"  Avg draft latency:  {avg_draft_time:.0f}ms")
    print(f"  Avg target latency: {avg_target_time:.0f}ms")
    print(f"  Total time:         {total_time:.1f}s")
    print(f"  Throughput:         {total/total_time:.2f} tok/s (sequential)")

    # Speculative speedup calculation
    if acceptance_rate > 0:
        # With spec-decoding, draft token accepted immediately
        # Rejected draft costs extra target call
        # Net: reduce target calls by acceptance_rate
        theoretical_speedup = 1.0 / (1.0 + (100 - acceptance_rate) / 100)
        print(f"\n  Theoretical speedup with speculation:")
        print(f"    (would reduce target calls by {acceptance_rate:.0f}%)")
        print(f"    Speedup factor: ~{theoretical_speedup:.2f}x (if draft is free)")

    return acceptance_rate

def main():
    print("="*70)
    print("GEMMA4 SPECULATIVE DECODING TEST")
    print("="*70)
    print(f"\nTarget model: {TARGET_MODEL} (17 GB, large)")
    print(f"Draft model:  {DRAFT_MODEL} (7.2 GB, small)")
    print(f"\nBoth same family → expect high acceptance rate\n")

    # Check ollama is running
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = {m['name'] for m in resp.json().get('models', [])}
        if TARGET_MODEL not in models:
            print(f"ERROR: {TARGET_MODEL} not found in Ollama")
            print(f"Available: {models}")
            sys.exit(1)
        if DRAFT_MODEL not in models:
            print(f"ERROR: {DRAFT_MODEL} not found in Ollama")
            sys.exit(1)
        print(f"✓ Both models available in Ollama\n")
    except Exception as e:
        print(f"ERROR: Can't connect to Ollama: {e}")
        sys.exit(1)

    # Run tests
    results = []
    for prompt, target_len in PROMPTS:
        try:
            acc_rate = test_speculative_decoding(prompt, target_len)
            if acc_rate is not None:
                results.append((prompt[:40], acc_rate))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            continue

    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        for prompt, acc_rate in results:
            print(f"  {prompt:40s} → {acc_rate:5.1f}% acceptance")

        avg_acceptance = sum(r[1] for r in results) / len(results)
        print(f"\n  Average acceptance rate: {avg_acceptance:.1f}%")

        if avg_acceptance > 50:
            print(f"\n  ✓ EXCELLENT: Speculative decoding is viable!")
            print(f"    Same-family models align well on tokens.")
        elif avg_acceptance > 20:
            print(f"\n  ◐ GOOD: Reasonable acceptance for speculation.")
        elif avg_acceptance > 5:
            print(f"\n  ✗ POOR: Acceptance rate too low for practical benefit.")
        else:
            print(f"\n  ✗ UNUSABLE: Models don't align sufficiently.")

if __name__ == "__main__":
    main()
