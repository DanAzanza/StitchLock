@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title StitchLocker (garment-fix-pipeline) - One-Click Setup and Self-Test

echo ===========================================================================
echo   StitchLocker - Automated Environment Setup and Pixel-Lock Self-Test
echo ===========================================================================
echo.

set "SYSTEM_PYTHON=python"
%SYSTEM_PYTHON% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('%SYSTEM_PYTHON% --version') do set "PY_VER=%%v"
echo [OK] Detected Python %PY_VER%

if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    %SYSTEM_PYTHON% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" set "VENV_PYTHON=python"

echo [INFO] Upgrading pip and installing dependencies...
%VENV_PYTHON% -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

%VENV_PYTHON% -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo [INFO] Checking test_cases directory...
%VENV_PYTHON% setup_test_cases.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test case initialization failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Running automated Quality Gate and Pixel-Lock test suite...
%VENV_PYTHON% scripts/run_ci_locally.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Quality Gate reported errors.
    pause
    exit /b 1
)

echo.
echo ===========================================================================
echo   [SUCCESS] StitchLocker Setup and Verification Completed!
echo   Double-click 'run_pipeline.bat' to launch the Artist Studio UI,
echo   or run 'python run_batch.py' for headless batch processing.
echo ===========================================================================
if "%~1"=="" pause
exit /b 0
