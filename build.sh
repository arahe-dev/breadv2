#!/bin/bash
# Setup MSVC environment and build BREAD

VS_SETUP="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"

# Call vcvarsall to setup environment, then build
cmd.exe /c "\"$VS_SETUP\" x64 && cd /d C:\bread_v2 && nvcc -O2 -x cu main.cu one_layer.cu layer_ops.cu kernels.cu loader.c gguf.c tokenizer.c bread.c -I. -o bread.exe"
