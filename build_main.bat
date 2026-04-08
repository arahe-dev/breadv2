@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cd /d C:\bread_v2

echo === Building bread.exe ===

nvcc -O2 ^
     -x cu ^
     main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c ^
     kernel_tasks.c error_classification.c progress_tracking.c hooks.c buffer_pool.c ^
     -I. ^
     -o bread.exe ^
     -Xcompiler /W3

if errorlevel 1 (
    echo FAILED: compile error
    exit /b 1
)

echo Build OK.
echo.
echo Usage:
echo   bread.exe --prompt "Hello" --tokens 50
echo   bread.exe --prompt "Explain quantum computing" --tokens 20
