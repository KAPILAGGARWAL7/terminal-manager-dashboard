@echo off
REM AI Backend Installation Script
REM Auto Python environment setup

setlocal enabledelayedexpansion

echo ========================================
echo AI Backend Setup
echo ========================================
echo.

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is not installed
    exit /b 1
)

REM Check pip
where pip >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    echo [SUCCESS] Virtual environment created!
) else (
    echo [INFO] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [INFO] Installing Python dependencies...
pip install -r requirements.txt

REM Install package in development mode
if exist "setup.py" (
    echo [INFO] Installing AI backend package...
    pip install -e .
)

echo [SUCCESS] AI Backend setup complete!

REM Check Ollama
echo.
echo [INFO] Checking Ollama connection...
python -c "from config import Config; status = Config.validate_ollama_connection(); print('Connected!' if status['connected'] else 'Not connected: ' + status.get('error', 'Unknown'))"

call deactivate
echo.
echo [SUCCESS] AI Backend is ready!

endlocal
