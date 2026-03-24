@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cd /d C:\bread_v2

echo === Building gguf.obj ===
cl /nologo /W3 /O2 /c /Fogguf.obj gguf.c
if errorlevel 1 ( echo FAILED: gguf.c & exit /b 1 )

echo === Building bread_info ===
cl /nologo /W3 /O2 /c /Fobread_info.obj bread_info.c
if errorlevel 1 ( echo FAILED: bread_info.c & exit /b 1 )
cl /nologo /W3 /O2 /Febread_info.exe bread_info.obj gguf.obj
if errorlevel 1 ( echo FAILED: bread_info link & exit /b 1 )

echo === Building topology ===
nvcc -Xcompiler "/nologo /W3 /O2" -c topology.c -o topology.obj
if errorlevel 1 ( echo FAILED: topology.c & exit /b 1 )
nvcc -Xcompiler "/nologo /W3 /O2" topology.obj gguf.obj -o topology.exe
if errorlevel 1 ( echo FAILED: topology link & exit /b 1 )

echo.
echo BUILD OK
echo   bread_info.exe  -- tensor inspector
echo   topology.exe    -- hardware bandwidth prober
