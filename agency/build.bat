@echo off
REM Build script for AGENCY Hermes CLI

echo Building AGENCY CLI...
cargo build --release

if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Build Successful ===
    echo Binary: target\release\agency.exe
    echo.
    echo Run with:
    echo   .\target\release\agency.exe --bread ..\bread.exe
) else (
    echo.
    echo === Build Failed ===
    echo Check the error messages above
    exit /b 1
)
