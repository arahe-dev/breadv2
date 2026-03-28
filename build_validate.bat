@echo off
REM Build validate.exe with MSVC
REM Depends on: bread.c, loader.c, gguf.c, tokenizer.c, dequant_q4k_cpu.c

echo Building validate.exe...

cl /O2 /W4 /nologo ^
   validate.c bread.c loader.c gguf.c tokenizer.c dequant_q4k_cpu.c ^
   /link ws2_32.lib /OUT:validate.exe

if errorlevel 1 (
    echo Build failed
    exit /b 1
) else (
    echo Build succeeded: validate.exe
)
