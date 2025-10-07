@echo off
setlocal enabledelayedexpansion

REM Production Environment Startup Script
REM Builds and starts all services in production mode

echo [PROD] Starting Self Service Dashboard - Production Mode

REM Load environment variables
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
)

REM Set production environment
set NODE_ENV=production
set FLASK_ENV=production

REM Function to start a service
:start_service
set name=%1
set command=%2
set port=%3
set log_file=logs\%name%-prod.log

echo [PROD] Starting %name% on port %port%...

if not exist "logs" mkdir logs
type nul > "%log_file%"

REM Start service in background
start /B %command% > "%log_file%" 2>&1
timeout /t 2 >nul

REM Get PID (approximation)
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID"') do (
    echo %%i > "logs\%name%-prod.pid"
    set pid=%%i
)

echo [SUCCESS] %name% started (PID: %pid%)
goto :eof

REM Cleanup function
:cleanup
echo [PROD] Stopping production services...

for %%f in (logs\*-prod.pid) do (
    if exist "%%f" (
        set /p pid=<"%%f"
        echo [INFO] Stopping service (PID: !pid!)
        taskkill /F /PID !pid! >nul 2>nul
        del "%%f" >nul 2>nul
    )
)

echo [SUCCESS] Production services stopped
exit /b 0

echo [PROD] 🚀 Initializing production environment...

REM Build frontend
echo [PROD] 🔨 Building frontend for production...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo [ERROR] Frontend build failed
    exit /b 1
)
cd ..
echo [SUCCESS] Frontend built successfully!

REM Start AI Backend in production mode
echo [PROD] 🧠 Starting AI Backend (Production)...
cd ai-backend
call venv\Scripts\activate.bat
start /B python app.py > ..\logs\ai-backend-prod.log 2>&1
timeout /t 3 >nul
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID"') do (
    echo %%i > ..\logs\ai-backend-prod.pid
    set ai_backend_pid=%%i
)
call venv\Scripts\deactivate.bat
cd ..
echo [SUCCESS] AI Backend started (PID: %ai_backend_pid%)

REM Start Simple Backend in production mode (if exists)
if exist "simple-backend\package.json" (
    echo [PROD] 🔧 Starting Simple Backend (Production)...
    cd simple-backend
    start /B npm run start:prod > ..\logs\simple-backend-prod.log 2>&1
    timeout /t 3 >nul
    for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq node.exe" /FO LIST ^| find "PID"') do (
        echo %%i > ..\logs\simple-backend-prod.pid
        set simple_backend_pid=%%i
    )
    cd ..
    echo [SUCCESS] Simple Backend started (PID: %simple_backend_pid%)
)

REM Serve frontend build (using a simple HTTP server)
echo [PROD] 🌐 Starting Static File Server for Frontend...
cd frontend\build
start /B python -m http.server 3000 > ..\..\logs\frontend-prod.log 2>&1
timeout /t 2 >nul
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID"') do (
    echo %%i > ..\..\logs\frontend-prod.pid
    set frontend_pid=%%i
)
cd ..\..
echo [SUCCESS] Static server started (PID: %frontend_pid%)

REM Wait for services to stabilize
echo [PROD] ⏳ Waiting for services to initialize...
timeout /t 15 >nul

REM Health checks
echo [PROD] 🏥 Running production health checks...

REM Check AI Backend
curl -s http://localhost:5247/health >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] AI Backend is responding
) else (
    echo [WARNING] AI Backend health check failed
)

REM Check if frontend is serving
netstat -an | find ":3000" | find "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Frontend is serving
) else (
    echo [WARNING] Frontend server check failed
)

REM Check Ollama connection
curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is connected
) else (
    echo [WARNING] Ollama connection failed - AI features may not work
)

echo.
echo [SUCCESS] ✅ Production Environment Started!
echo.
echo [INFO] 🏭 Production Services:
echo [INFO] - Frontend:       http://localhost:3000 (Static)
echo [INFO] - AI Backend:     http://localhost:5247
if defined simple_backend_pid (
    echo [INFO] - Simple Backend: http://localhost:5000
)
echo.
echo [INFO] 📊 Service Status:
for %%f in (logs\*-prod.pid) do (
    if exist "%%f" (
        set /p pid=<"%%f"
        set service=%%~nf
        set service=!service:-prod=!
        tasklist /PID !pid! >nul 2>nul
        if %errorlevel% equ 0 (
            echo [INFO] - !service!: Running (PID: !pid!)
        ) else (
            echo [WARNING] - !service!: Not running
        )
    )
)
echo.
echo [INFO] 📝 Production Logs:
for %%f in (logs\*-prod.log) do (
    echo [INFO] - %%~nf: %%f
)
echo.
echo [INFO] 🔧 Management Commands:
echo [INFO] - View logs:      type logs\[service]-prod.log  
echo [INFO] - Stop services:  Press Ctrl+C or run stop-production.bat
echo [INFO] - Health check:   scripts\health-check.bat
echo.
echo [INFO] 🌐 Application URL: http://localhost:3000
echo.

REM Monitor production services
echo [INFO] Production environment is running. Press Ctrl+C to stop all services.
echo [INFO] Monitoring production services...

:monitor_loop
timeout /t 60 >nul
echo [INFO] %date% %time% - Production health check...

REM Check if all services are still running
set all_running=true
for %%f in (logs\*-prod.pid) do (
    if exist "%%f" (
        set /p pid=<"%%f"
        tasklist /PID !pid! >nul 2>nul
        if %errorlevel% neq 0 (
            set service=%%~nf
            set service=!service:-prod=!
            echo [ERROR] !service! service has stopped unexpectedly (was PID: !pid!)
            set all_running=false
        )
    )
)

if "!all_running!"=="true" (
    echo [SUCCESS] All production services are running
) else (
    echo [WARNING] Some services have failed. Check logs for details.
)

goto monitor_loop