@echo off
setlocal enabledelayedexpansion

REM Project Reset Script
REM Clean restart of the entire development environment

echo [RESET] Project Reset Script - Clean Development Environment

REM Function to confirm action
echo [WARNING] This will stop all services and clean the development environment
set /p confirm="Are you sure? (y/N): "
if /i not "%confirm%"=="y" (
    echo [INFO] Operation cancelled
    exit /b 0
)

REM Kill running processes
echo [RESET] Stopping all running services...

REM Stop Python processes
taskkill /F /IM python.exe /T >nul 2>nul
taskkill /F /IM pythonw.exe /T >nul 2>nul

REM Stop Node.js processes
taskkill /F /IM node.exe /T >nul 2>nul

REM Stop processes by name patterns
for /f "tokens=2" %%i in ('tasklist /FI "WINDOWTITLE eq *streamlit*" /FO CSV ^| find /V "PID"') do taskkill /F /PID %%i >nul 2>nul
for /f "tokens=2" %%i in ('tasklist /FI "WINDOWTITLE eq *react-scripts*" /FO CSV ^| find /V "PID"') do taskkill /F /PID %%i >nul 2>nul

REM Kill processes by PID files
if exist "logs" (
    for %%f in (logs\*.pid) do (
        if exist "%%f" (
            set /p pid=<"%%f"
            taskkill /F /PID !pid! >nul 2>nul
            del "%%f" >nul 2>nul
        )
    )
)

echo [SUCCESS] All processes stopped

REM Clean logs
echo [RESET] Cleaning logs...
if exist "logs" (
    del /Q logs\* >nul 2>nul
    echo [SUCCESS] Logs cleaned
) else (
    echo [INFO] No logs directory found
)

REM Clean generated files
echo [RESET] Cleaning generated files...

REM Generated dashboards
if exist "generated-dashboards" (
    del /Q generated-dashboards\* >nul 2>nul
    echo [SUCCESS] Generated dashboards cleaned
)

REM Python cache files
for /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /S /Q "%%d" >nul 2>nul
    )
)

REM Temporary files
del /S /Q *.pyc >nul 2>nul
del /S /Q .DS_Store >nul 2>nul

echo [SUCCESS] Generated files cleaned

REM Reset Python environment
echo [RESET] Resetting Python environment...
if exist "ai-backend\venv" (
    echo [INFO] Removing Python virtual environment...
    rmdir /S /Q ai-backend\venv
    echo [SUCCESS] Python virtual environment removed
    
    echo [INFO] Recreating Python virtual environment...
    cd ai-backend
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    call venv\Scripts\deactivate.bat
    cd ..
    echo [SUCCESS] Python environment recreated
) else (
    echo [INFO] No Python virtual environment found
)

REM Reset Node.js environments
echo [RESET] Resetting Node.js environments...

REM Frontend reset
if exist "frontend\node_modules" (
    echo [INFO] Removing frontend node_modules...
    rmdir /S /Q frontend\node_modules >nul 2>nul
    echo [SUCCESS] Frontend node_modules removed
)

if exist "frontend\package.json" (
    echo [INFO] Reinstalling frontend dependencies...
    cd frontend
    npm install
    if %errorlevel% equ 0 (
        echo [SUCCESS] Frontend dependencies reinstalled
    ) else (
        echo [ERROR] Failed to reinstall frontend dependencies
    )
    cd ..
)

REM Simple backend reset
if exist "simple-backend\node_modules" (
    echo [INFO] Removing simple-backend node_modules...
    rmdir /S /Q simple-backend\node_modules >nul 2>nul
    echo [SUCCESS] Simple-backend node_modules removed
)

if exist "simple-backend\package.json" (
    echo [INFO] Reinstalling simple-backend dependencies...
    cd simple-backend
    npm install
    if %errorlevel% equ 0 (
        echo [SUCCESS] Simple-backend dependencies reinstalled
    ) else (
        echo [ERROR] Failed to reinstall simple-backend dependencies
    )
    cd ..
)

REM Root dependencies reset
if exist "node_modules" (
    echo [INFO] Removing root node_modules...
    rmdir /S /Q node_modules >nul 2>nul
    echo [SUCCESS] Root node_modules removed
)

if exist "package.json" (
    echo [INFO] Reinstalling root dependencies...
    npm install
    if %errorlevel% equ 0 (
        echo [SUCCESS] Root dependencies reinstalled
    ) else (
        echo [ERROR] Failed to reinstall root dependencies
    )
)

REM Clean uploaded files and database
echo [RESET] Cleaning data directories...

if exist "simple-backend\uploads" (
    del /Q simple-backend\uploads\* >nul 2>nul
    echo [SUCCESS] Upload directory cleaned
)

if exist "data" (
    REM Keep sample files but remove generated data
    for %%f in (data\*.generated.*) do del "%%f" >nul 2>nul
    echo [SUCCESS] Data directory cleaned (keeping sample files)
)

REM Reset configuration
echo [RESET] Resetting configuration...

if exist ".env.backup" (
    copy .env.backup .env >nul 2>nul
    echo [SUCCESS] Configuration restored from backup
) else if exist ".env.example" (
    copy .env.example .env >nul 2>nul
    echo [SUCCESS] Configuration reset to example
) else (
    echo [WARNING] No configuration backup or example found
)

REM Recreate necessary directories
echo [RESET] Recreating runtime directories...
if not exist "generated-dashboards" mkdir generated-dashboards
if not exist "logs" mkdir logs
if not exist "data" mkdir data
echo [SUCCESS] Runtime directories recreated

REM Final cleanup
echo [RESET] Final cleanup...
del /Q *.log >nul 2>nul
del /Q *.tmp >nul 2>nul

echo.
echo [SUCCESS] ✅ Project Reset Complete!
echo.
echo [INFO] The development environment has been completely reset.
echo [INFO] Next steps:
echo [INFO] 1. Run setup.bat to reinitialize the project
echo [INFO] 2. Start development with scripts\start-development.bat
echo [INFO] 3. Verify everything works with scripts\health-check.bat
echo.

pause