@echo off
echo Starting Complete AI Dashboard Workflow Test
echo ============================================
echo.

:: Check if Ollama is running
echo 1. Checking Ollama status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Ollama is not running!
    echo Please start Ollama with: ollama serve
    echo Then run: ollama pull llama3
    echo.
    pause
    exit /b 1
) else (
    echo ✓ Ollama is running
)

:: Start AI Backend
echo.
echo 2. Starting AI Backend (Python)...
cd ai-backend
start "AI Backend" cmd /k "python app.py"
cd ..

:: Wait for AI Backend to start
timeout /t 3 /nobreak >nul

:: Start Simple Backend
echo.
echo 3. Starting Simple Backend (Node.js)...
cd simple-backend
start "Simple Backend" cmd /k "node server.js"
cd ..

:: Wait for Simple Backend to start
timeout /t 3 /nobreak >nul

:: Start Frontend
echo.
echo 4. Starting React Frontend...
cd frontend
start "React Frontend" cmd /k "npm start"
cd ..

echo.
echo ============================================
echo All services are starting up...
echo.
echo Services will be available at:
echo   - React Frontend: http://localhost:3000
echo   - Simple Backend: http://localhost:5246
echo   - AI Backend: http://localhost:5247
echo   - Dashboard Generator: dashboard-generator.html
echo.
echo To test the workflow:
echo 1. Upload an Excel file through React frontend
echo 2. Enter a prompt in the AI Dashboard Generator section
echo 3. Watch the AI create a Streamlit dashboard
echo.
pause