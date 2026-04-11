#!/bin/bash
(
  sleep 30  # Wait for model load
  echo "hello"
  echo ""
) | timeout 60 ./bread.exe --server --tokens 256 2>&1 | head -200
