@echo off
setlocal enabledelayedexpansion

echo Starting Terminal Manager AI Dashboard System...

REM Check if Python backend dependencies are installed
if not exist "ai-backend\venv" (
    echo Setting up Python environment...
    cd ai-backend
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    call venv\Scripts\deactivate.bat
    cd ..
)

REM Start Python AI backend
echo Starting AI Dashboard Backend on port 5247...
cd ai-backend
call venv\Scripts\activate.bat
start /B python app.py
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID"') do set AI_BACKEND_PID=%%a
cd ..

echo AI Backend started with PID: %AI_BACKEND_PID%
echo.
echo Setup Complete!
echo Your existing React frontend is unchanged and ready
echo AI Dashboard Generator is now available in your app
echo.
echo Next steps:
echo 1. Make sure Ollama is running: ollama serve
echo 2. Install Llama 3: ollama pull llama3
echo 3. Start your React frontend: npm start (in frontend folder)
echo 4. Your app will now have both GraphicWalker AND AI dashboards!
echo.
echo To stop AI backend later: taskkill /PID %AI_BACKEND_PID% /F

pause