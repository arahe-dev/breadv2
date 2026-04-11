@echo off
REM BREAD Performance Benchmark Script
REM Measures tokens/sec and timing breakdown

setlocal enabledelayedexpansion

echo.
echo ========================================
echo BREAD Performance Benchmark
echo ========================================
echo.

cd /d C:\bread_v2

REM Run 1: 100 tokens
echo [RUN 1] Generating 100 tokens...
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set start_time=%%a:%%b)
echo Start time: !start_time!

bread.exe --server --tokens 100 < \tmp\test_single_query.txt > bench_100.txt 2>&1

for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set end_time=%%a:%%b)
echo End time: !end_time!

type bench_100.txt | findstr "BREAD_END" >nul
if errorlevel 1 (
    echo [FAIL] BREAD_END not found - generation failed
) else (
    echo [PASS] Generation completed successfully
)

echo.
echo [RUN 2] Generating 200 tokens...
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set start_time=%%a:%%b)

bread.exe --server --tokens 200 < \tmp\test_single_query.txt > bench_200.txt 2>&1

for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set end_time=%%a:%%b)

type bench_200.txt | findstr "BREAD_END" >nul
if errorlevel 1 (
    echo [FAIL] BREAD_END not found
) else (
    echo [PASS] Generation completed
)

echo.
echo [RUN 3] Generating 50 tokens (quick test)...
bread.exe --server --tokens 50 < \tmp\test_single_query.txt > bench_50.txt 2>&1

type bench_50.txt | findstr "BREAD_END" >nul
if errorlevel 1 (
    echo [FAIL] BREAD_END not found
) else (
    echo [PASS] Generation completed
)

echo.
echo ========================================
echo Output files created:
echo  - bench_50.txt (50 tokens)
echo  - bench_100.txt (100 tokens)
echo  - bench_200.txt (200 tokens)
echo.
echo To measure throughput, look for timing info in outputs
echo or use: findstr /i "BREAD_READY\|BREAD_END" bench_*.txt
echo ========================================

pause
