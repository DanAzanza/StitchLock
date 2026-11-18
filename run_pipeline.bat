@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title StitchLocker - Garment Detail Correction Web UI

set "PYTHON_EXE=python"
if defined LOCALAPPDATA (
    if exist "%LOCALAPPDATA%\StitchLocker\.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\StitchLocker\.venv\Scripts\python.exe"
    )
)
if "%PYTHON_EXE%"=="python" (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    ) else if exist "venv\Scripts\python.exe" (
        set "PYTHON_EXE=venv\Scripts\python.exe"
    )
)

"%PYTHON_EXE%" --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH or virtual environment directories.
    echo Please run install.bat first or install Python 3.11+.
    pause
    exit /b 1
)

echo ==============================================================================
echo   StitchLocker - Launching Web UI (http://127.0.0.1:7860)
echo   Closing the browser tab will automatically shut down ComfyUI and the UI.
echo ==============================================================================
"%PYTHON_EXE%" -u app.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% NEQ 0 (
    echo.
    echo [ERROR] StitchLocker exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
