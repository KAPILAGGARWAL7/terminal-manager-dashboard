@echo off
setlocal enabledelayedexpansion

REM Self Service Dashboard AI - Complete Setup Script
REM This script sets up the entire development environment

echo [INFO] Starting Self Service Dashboard AI Setup...

REM Check if command exists function
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed. Please install Node.js 16+ and try again.
    echo [INFO] Download from: https://nodejs.org/
    exit /b 1
)

where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] npm is not installed. Please install npm and try again.
    exit /b 1
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is not installed. Please install Python 3.8+ and try again.
    echo [INFO] Download from: https://python.org/
    exit /b 1
)

where pip >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed. Please install pip and try again.
    exit /b 1
)

echo [SUCCESS] All prerequisites found!

REM Step 2: Copy environment file
echo [INFO] Setting up environment configuration...
if not exist ".env" (
    copy .env.example .env >nul 2>nul
    echo [SUCCESS] Created .env file from template
) else (
    echo [WARNING] .env file already exists, skipping...
)

REM Step 3: Install root dependencies
echo [INFO] Installing root dependencies...
npm install concurrently --save-dev
if %errorlevel% equ 0 (
    echo [SUCCESS] Root dependencies installed!
) else (
    echo [ERROR] Failed to install root dependencies
    exit /b 1
)

REM Step 4: Setup AI Backend
echo [INFO] Setting up AI Backend...
cd ai-backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    echo [SUCCESS] Virtual environment created!
)

REM Activate virtual environment and install dependencies
echo [INFO] Installing Python dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
call venv\Scripts\deactivate.bat
echo [SUCCESS] AI Backend setup complete!

cd ..

REM Step 5: Setup Frontend
echo [INFO] Setting up Frontend...
cd frontend
npm install
if %errorlevel% equ 0 (
    echo [SUCCESS] Frontend dependencies installed!
) else (
    echo [ERROR] Failed to install frontend dependencies
    exit /b 1
)
cd ..

REM Step 6: Setup Simple Backend (if exists)
if exist "simple-backend" (
    echo [INFO] Setting up Simple Backend...
    cd simple-backend
    if exist "package.json" (
        npm install
        if %errorlevel% equ 0 (
            echo [SUCCESS] Simple Backend dependencies installed!
        )
    )
    cd ..
)

REM Step 7: Create necessary directories
echo [INFO] Creating runtime directories...
if not exist "generated-dashboards" mkdir generated-dashboards
if not exist "logs" mkdir logs
echo [SUCCESS] Runtime directories created!

REM Step 8: Check Ollama
echo [INFO] Checking Ollama installation...
where ollama >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is installed!
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
    if %errorlevel% equ 0 (
        echo [SUCCESS] Ollama is running!
    ) else (
        echo [WARNING] Ollama is installed but not running.
        echo [INFO] Start Ollama with: ollama serve
    )
) else (
    echo [WARNING] Ollama is not installed.
    echo [INFO] To install Ollama:
    echo [INFO]   Windows: Download from https://ollama.ai/
    echo [INFO] After installation, run: ollama pull llama3
)

REM Step 9: Final checks
echo [INFO] Running health checks...
if exist "scripts\health-check.bat" (
    call scripts\health-check.bat --setup-mode
)

REM Success message
echo.
echo [SUCCESS] Setup Complete!
echo.
echo [INFO] Next steps:
echo [INFO] 1. Install Ollama if not already installed (see instructions above)
echo [INFO] 2. Start Ollama: ollama serve
echo [INFO] 3. Pull a model: ollama pull llama3
echo [INFO] 4. Start the development environment: scripts\start-development.bat
echo [INFO] 5. Open http://localhost:3000 in your browser
echo.
echo [INFO] For help, see docs\SETUP.md or run: scripts\health-check.bat

pause