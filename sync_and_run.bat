@echo off
setlocal

echo ==========================================
echo   TREND GENERATOR - SYNC AND RUN
echo ==========================================
echo.

echo [1/3] Updating project from GitHub...
git pull --rebase origin main

if errorlevel 1 (
    echo.
    echo ERROR: Git pull failed.
    echo Generator will NOT start.
    pause
    exit /b 1
)

echo.
echo [2/3] Starting generator...
python generate.py

if errorlevel 1 (
    echo.
    echo ERROR: Generator failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Generator finished successfully.
echo GitHub push is handled by generate.py.
echo.
pause