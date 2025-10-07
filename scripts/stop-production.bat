@echo off
setlocal enabledelayedexpansion

REM Stop Production Services Script
REM Cleanly stops all production services

echo [PROD] Stopping Production Services...

REM Kill services by PID files
if exist "logs" (
    for %%f in (logs\*-prod.pid) do (
        if exist "%%f" (
            set /p pid=<"%%f"
            set service=%%~nf
            set service=!service:-prod=!
            echo [INFO] Stopping !service! (PID: !pid!)
            taskkill /F /PID !pid! >nul 2>nul
            if %errorlevel% equ 0 (
                echo [SUCCESS] !service! stopped
            ) else (
                echo [WARNING] Could not stop !service! - may already be stopped
            )
            del "%%f" >nul 2>nul
        )
    )
)

REM Additional cleanup for common production processes
echo [INFO] Cleaning up any remaining processes...

REM Stop Python processes (AI Backend)
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST 2^>nul ^| find "PID"') do (
    tasklist /PID %%i /FI "WINDOWTITLE eq*app.py*" >nul 2>nul
    if %errorlevel% equ 0 (
        echo [INFO] Stopping Python AI Backend (PID: %%i)
        taskkill /F /PID %%i >nul 2>nul
    )
)

REM Stop Node.js processes (Simple Backend)
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq node.exe" /FO LIST 2^>nul ^| find "PID"') do (
    tasklist /PID %%i /FI "WINDOWTITLE eq*start:prod*" >nul 2>nul
    if %errorlevel% equ 0 (
        echo [INFO] Stopping Node.js Backend (PID: %%i)
        taskkill /F /PID %%i >nul 2>nul
    )
)

REM Stop HTTP server processes (Frontend)
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST 2^>nul ^| find "PID"') do (
    tasklist /PID %%i /FI "WINDOWTITLE eq*http.server*" >nul 2>nul
    if %errorlevel% equ 0 (
        echo [INFO] Stopping HTTP Server (PID: %%i)
        taskkill /F /PID %%i >nul 2>nul
    )
)

echo [SUCCESS] Production services cleanup completed!

pause