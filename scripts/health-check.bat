@echo off
setlocal enabledelayedexpansion

REM Comprehensive Health Check Script
REM Verifies system requirements and service health

set SETUP_MODE=false
set DEV_MODE=false

REM Parse arguments
:parse_args
if "%1"=="--setup-mode" (
    set SETUP_MODE=true
    shift
    goto parse_args
)
if "%1"=="--dev-mode" (
    set DEV_MODE=true
    shift
    goto parse_args
)
if "%1" neq "" (
    echo Unknown option: %1
    exit /b 1
)

echo [HEALTH] Starting comprehensive health check...

REM Test functions
:test_command
set cmd=%1
set name=%2
set required=%3

echo [🔍] Checking %name%...

where %cmd% >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('%cmd% --version 2^>^&1') do set version=%%i
    echo [✅] %name% is installed (!version!)
    goto :eof
) else (
    if "%required%"=="true" (
        echo [❌] %name% is not installed (required)
        exit /b 1
    ) else (
        echo [⚠️] %name% is not installed (optional)
        goto :eof
    )
)

REM Test ports
:test_port
set port=%1
set service=%2
set should_be_free=%3

echo [🔍] Checking port %port% for %service%...

netstat -an | find ":%port%" | find "LISTENING" >nul
if %errorlevel% equ 0 (
    if "%should_be_free%"=="true" (
        echo [⚠️] Port %port% is in use (should be free for %service%)
        echo    💡 To free: netstat -ano | findstr :%port% ^& taskkill /PID [PID] /F
        exit /b 1
    ) else (
        echo [✅] Port %port% is in use (%service% running)
        goto :eof
    )
) else (
    if "%should_be_free%"=="true" (
        echo [✅] Port %port% is free (%service% can start)
        goto :eof
    ) else (
        echo [⚠️] Port %port% is free (%service% not running)
        exit /b 1
    )
)

REM Test URL
:test_url
set url=%1
set service=%2
set timeout=%3
if "%timeout%"=="" set timeout=5

echo [🔍] Testing %service% at %url%...

curl -s --max-time %timeout% "%url%" >nul 2>nul
if %errorlevel% equ 0 (
    echo [✅] %service% is responding
    goto :eof
) else (
    echo [⚠️] %service% is not responding at %url%
    exit /b 1
)

REM Test directory
:test_directory
set dir=%1
set name=%2
set should_exist=%3

echo [🔍] Checking %name% directory...

if exist "%dir%" (
    if "%should_exist%"=="true" (
        echo [✅] %name% directory exists: %dir%
        goto :eof
    ) else (
        echo [⚠️] %name% directory exists but shouldn't: %dir%
        exit /b 1
    )
) else (
    if "%should_exist%"=="true" (
        echo [❌] %name% directory missing: %dir%
        exit /b 1
    ) else (
        echo [✅] %name% directory doesn't exist (as expected): %dir%
        goto :eof
    )
)

REM Test Ollama
:test_ollama
echo [🔍] Testing Ollama installation and connectivity...

where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo [⚠️] Ollama is not installed
    echo    💡 Install: Download from https://ollama.ai/
    exit /b 1
)

echo [✅] Ollama is installed

REM Test if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% equ 0 (
    echo [✅] Ollama is running
    
    REM Check for required model
    curl -s http://localhost:11434/api/tags | find "llama3" >nul
    if %errorlevel% equ 0 (
        echo [✅] Llama3 model is available
        goto :eof
    ) else (
        echo [⚠️] Llama3 model not found
        echo    💡 Install: ollama pull llama3
        exit /b 1
    )
) else (
    echo [⚠️] Ollama is not running
    echo    💡 Start: ollama serve
    exit /b 1
)

REM Main health check routine
echo [HEALTH] 🏥 Self Service Dashboard AI - Health Check
echo.

REM Check system requirements
echo [HEALTH] System Requirements:
call :test_command node "Node.js" true
call :test_command npm "npm" true
call :test_command python "Python 3" true
call :test_command pip "pip" true
call :test_command git "Git" false

echo.

REM Check project structure
echo [HEALTH] Project Structure:
call :test_directory "frontend" "Frontend" true
call :test_directory "ai-backend" "AI Backend" true
call :test_directory "scripts" "Scripts" true
call :test_directory "logs" "Logs" false

echo.

REM Check ports (in setup mode, ports should be free)
echo [HEALTH] Port Availability:
if "%SETUP_MODE%"=="true" (
    call :test_port 3000 "React Frontend" true
    call :test_port 5247 "AI Backend" true
    call :test_port 11434 "Ollama" false
) else (
    call :test_port 3000 "React Frontend" false
    call :test_port 5247 "AI Backend" false
    call :test_port 11434 "Ollama" false
)

echo.

REM Check Ollama
echo [HEALTH] AI Services:
call :test_ollama

echo.
echo [✅] Health check completed!

pause