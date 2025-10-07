@echo off
setlocal enabledelayedexpansion

REM Development Environment Startup Script
REM Starts all services in development mode with hot reload

echo [DEV] Starting Self Service Dashboard - Development Environment

REM Check if setup has been run
if not exist "ai-backend\venv" (
    echo [ERROR] AI Backend not set up. Please run setup.bat first.
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [ERROR] Frontend not set up. Please run setup.bat first.
    exit /b 1
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Function to check if a port is in use
:check_port
set port=%1
netstat -an | find ":%port%" | find "LISTENING" >nul
if %errorlevel% equ 0 (
    exit /b 0
) else (
    exit /b 1
)

REM Function to start a service and track its PID
:start_service
set name=%1
set command=%2
set port=%3
set log_file=logs\%name%.log

echo [DEV] Starting %name% on port %port%...

REM Create log file
type nul > "%log_file%"

REM Start service in background and capture PID
start /B %command% > "%log_file%" 2>&1
REM Get the PID of the last started process (approximation)
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID"') do (
    echo %%i > "logs\%name%.pid"
    set pid=%%i
)

echo [SUCCESS] %name% started (PID: %pid%)
echo [INFO] Logs: type logs\%name%.log
goto :eof

REM Function to cleanup on exit
:cleanup
echo [WARNING] Shutting down development environment...

REM Kill all tracked services
for %%f in (logs\*.pid) do (
    if exist "%%f" (
        set /p pid=<"%%f"
        echo [INFO] Stopping service (PID: !pid!)
        taskkill /F /PID !pid! >nul 2>nul
        del "%%f" >nul 2>nul
    )
)

REM Additional cleanup for any remaining processes
taskkill /F /IM python.exe /T >nul 2>nul
taskkill /F /IM node.exe /T >nul 2>nul

echo [SUCCESS] Development environment stopped
exit /b 0

REM Set up cleanup trap (Windows doesn't have native signal handling, so we'll use a different approach)
REM We'll create a cleanup script that can be called

REM Main execution
echo [DEV] 🚀 Initializing development environment...

REM Load environment variables
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
)

REM Set development environment
set NODE_ENV=development
set FLASK_ENV=development
set FLASK_DEBUG=1

REM Check port availability
echo [DEV] 🔍 Checking port availability...

call :check_port 3000
if %errorlevel% equ 0 (
    echo [WARNING] Port 3000 is already in use. Frontend may not start properly.
)

call :check_port 5247
if %errorlevel% equ 0 (
    echo [WARNING] Port 5247 is already in use. AI Backend may not start properly.
)

call :check_port 5000
if %errorlevel% equ 0 (
    echo [WARNING] Port 5000 is already in use. Simple Backend may not start properly.
)

echo [SUCCESS] Port check completed

REM Check Ollama
echo [DEV] 🦙 Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is running
) else (
    echo [WARNING] Ollama is not running. AI features may not work.
    echo [INFO] Start Ollama with: ollama serve
)

REM Start AI Backend
echo [DEV] 🧠 Starting AI Backend...
cd ai-backend
call venv\Scripts\activate.bat
start /B python app.py > ..\logs\ai-backend.log 2>&1
REM Approximate PID capture for AI Backend
timeout /t 2 >nul
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID"') do (
    echo %%i > ..\logs\ai-backend.pid
    set ai_backend_pid=%%i
)
call venv\Scripts\deactivate.bat
cd ..
echo [SUCCESS] AI Backend started (PID: %ai_backend_pid%)

REM Start Simple Backend (if exists)
if exist "simple-backend\package.json" (
    echo [DEV] 🔧 Starting Simple Backend...
    cd simple-backend
    start /B npm start > ..\logs\simple-backend.log 2>&1
    timeout /t 2 >nul
    for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq node.exe" /FO LIST ^| find "PID"') do (
        echo %%i > ..\logs\simple-backend.pid
        set simple_backend_pid=%%i
    )
    cd ..
    echo [SUCCESS] Simple Backend started (PID: %simple_backend_pid%)
)

REM Start Frontend (React)
echo [DEV] ⚛️ Starting React Frontend...
cd frontend
start /B npm start > ..\logs\frontend.log 2>&1
timeout /t 3 >nul
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq node.exe" /FO LIST ^| find "PID"') do (
    echo %%i > ..\logs\frontend.pid
    set frontend_pid=%%i
)
cd ..
echo [SUCCESS] React Frontend started (PID: %frontend_pid%)

REM Wait for services to start
echo [DEV] ⏳ Waiting for services to initialize...
timeout /t 10 >nul

REM Health check
echo [DEV] 🏥 Running health checks...

REM Check AI Backend
curl -s http://localhost:5247/health >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] AI Backend is responding
) else (
    echo [WARNING] AI Backend may not be ready yet
)

REM Check Frontend (this will typically redirect, so we just check if the port is listening)
call :check_port 3000
if %errorlevel% equ 0 (
    echo [SUCCESS] Frontend is running
) else (
    echo [WARNING] Frontend may not be ready yet
)

echo.
echo [SUCCESS] ✅ Development Environment Started!
echo.
echo [INFO] 📊 Services Status:
echo [INFO] - AI Backend:     http://localhost:5247
echo [INFO] - React Frontend: http://localhost:3000
if defined simple_backend_pid (
    echo [INFO] - Simple Backend: http://localhost:5000
)
echo.
echo [INFO] 📝 Logs:
echo [INFO] - AI Backend:     logs\ai-backend.log
echo [INFO] - Frontend:       logs\frontend.log
if exist "logs\simple-backend.log" (
    echo [INFO] - Simple Backend: logs\simple-backend.log
)
echo.
echo [INFO] 🔧 Development Commands:
echo [INFO] - View logs:      type logs\[service].log
echo [INFO] - Health check:   scripts\health-check.bat --dev-mode
echo [INFO] - Stop services:  Press Ctrl+C or close this window
echo.
echo [INFO] 🌐 Open in browser: http://localhost:3000
echo.

REM Keep the script running and monitor
echo [INFO] Development environment is running. Press Ctrl+C to stop all services.
echo [INFO] Monitoring services...

:monitor_loop
timeout /t 30 >nul
REM Check if services are still running
for %%f in (logs\*.pid) do (
    if exist "%%f" (
        set /p pid=<"%%f"
        tasklist /PID !pid! >nul 2>nul
        if %errorlevel% neq 0 (
            echo [WARNING] Service with PID !pid! has stopped unexpectedly
        )
    )
)
goto monitor_loop