@echo off
setlocal enabledelayedexpansion

REM Ollama Installation Helper Script
REM Provides instructions and automated installation for Ollama

echo [OLLAMA] Ollama Installation Helper for Windows

REM Check if Ollama is already installed
where ollama >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is already installed!
    goto check_running
) else (
    echo [INFO] Ollama is not installed
    goto install_ollama
)

:install_ollama
echo [OLLAMA] Installing Ollama for Windows...
echo.
echo [INFO] Windows detected. Manual installation required:
echo.
echo 1. Open your web browser
echo 2. Go to: https://ollama.ai/
echo 3. Download Ollama for Windows
echo 4. Run the installer (ollama-windows-amd64.exe)
echo 5. Follow the installation wizard
echo 6. Restart your terminal after installation
echo 7. Run this script again to verify installation
echo.
echo [WARNING] Automated installation not available for Windows
echo [INFO] Press any key to open Ollama download page...
pause >nul
start https://ollama.ai/
goto end

:check_running
echo [INFO] Checking if Ollama is running...

REM Check if Ollama service is running
curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is running!
    goto check_models
) else (
    echo [WARNING] Ollama is not running
    goto start_ollama
)

:start_ollama
echo [INFO] Starting Ollama service...
echo [INFO] You can start Ollama in several ways:
echo.
echo Method 1 - Command line:
echo   ollama serve
echo.
echo Method 2 - Background service:
echo   Start Ollama from Start Menu or Desktop shortcut
echo.
echo [INFO] Starting Ollama now...
start /B ollama serve
timeout /t 3 >nul

REM Wait and check again
echo [INFO] Waiting for Ollama to start...
timeout /t 5 >nul

curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama started successfully!
    goto check_models
) else (
    echo [ERROR] Failed to start Ollama automatically
    echo [INFO] Please start Ollama manually: ollama serve
    goto end
)

:check_models
echo [INFO] Checking for required models...

REM Check for Llama3 model
curl -s http://localhost:11434/api/tags | find "llama3" >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Llama3 model is available!
    goto success
) else (
    echo [WARNING] Llama3 model not found
    goto install_model
)

:install_model
echo [INFO] Installing Llama3 model...
echo [INFO] This may take several minutes depending on your internet connection...

ollama pull llama3
if %errorlevel% equ 0 (
    echo [SUCCESS] Llama3 model installed successfully!
    goto success
) else (
    echo [ERROR] Failed to install Llama3 model
    echo [INFO] You can install it manually later with: ollama pull llama3
    goto end
)

:success
echo.
echo [SUCCESS] ✅ Ollama Setup Complete!
echo.
echo [INFO] Available commands:
echo   ollama serve          - Start Ollama service
echo   ollama pull llama3    - Install Llama3 model
echo   ollama list           - List installed models
echo   ollama run llama3     - Test chat with Llama3
echo.
echo [INFO] Ollama is ready for the AI Dashboard!

:end
pause