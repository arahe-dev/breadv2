#!/bin/bash

echo "=== Test 1: 50 tokens with simple prompt ==="
(
  echo "Hello world"
  echo ""
) | timeout 120 ./bread.exe --server --tokens 50 2>&1 | head -20

echo ""
echo "=== Test 2: 100 tokens with simple prompt ==="
(
  echo "Explain 2+2"
  echo ""
) | timeout 120 ./bread.exe --server --tokens 100 2>&1 | head -30

echo ""
echo "=== Test 3: Check that --tokens is being recognized ==="
./bread.exe --help 2>&1 | grep -i token || echo "(no help, checking code manually)"

