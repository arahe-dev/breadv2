#!/bin/bash
(
  sleep 35  # Wait for BREAD to load (about 30s)
  echo "hello"
  sleep 10
  echo "/quit"
) | ./agency/target/release/agency.exe --bread ./bread.exe --tokens 256
