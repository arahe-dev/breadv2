#!/bin/bash
(
  sleep 40  # Wait for model load
  echo "hello"
  echo ""
  sleep 5
  echo "/quit"
) | timeout 120 ./agency/target/release/agency.exe --bread ./bread.exe --tokens 50 2>&1
