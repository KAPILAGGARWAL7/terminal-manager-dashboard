#!/bin/bash

echo "🚀 Starting Terminal Manager AI Dashboard System..."

# Check if Python backend dependencies are installed
if [ ! -d "ai-backend/venv" ]; then
    echo "📦 Setting up Python environment..."
    cd ai-backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# Start Python AI backend
echo "🐍 Starting AI Dashboard Backend on port 5247..."
cd ai-backend
source venv/bin/activate
python app.py &
AI_BACKEND_PID=$!
cd ..

echo "✅ AI Backend started with PID: $AI_BACKEND_PID"
echo ""
echo "🎯 Setup Complete!"
echo "📊 Your existing React frontend is unchanged and ready"
echo "🤖 AI Dashboard Generator is now available in your app"
echo ""
echo "Next steps:"
echo "1. Make sure Ollama is running: ollama serve"
echo "2. Install Llama 3: ollama pull llama3"
echo "3. Start your React frontend: npm start (in frontend folder)"
echo "4. Your app will now have both GraphicWalker AND AI dashboards!"
echo ""
echo "To stop AI backend later: kill $AI_BACKEND_PID"

# Keep script running so we can see the backend logs
wait $AI_BACKEND_PID