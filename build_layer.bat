@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cd /d C:\bread_v2

echo === Building one_layer_test.exe ===

nvcc -DSELFTEST_MAIN ^
     -O2 ^
     -x cu ^
     one_layer.cu kernels.cu loader.c gguf.c ^
     -I. ^
     -o one_layer_test.exe ^
     -Xcompiler /W3

if errorlevel 1 (
    echo FAILED: compile error
    exit /b 1
)

echo Build OK.
echo.
echo NOTE: Running will load 22GB model (~30s). Press Ctrl+C to abort.
echo Run with: one_layer_test.exe
