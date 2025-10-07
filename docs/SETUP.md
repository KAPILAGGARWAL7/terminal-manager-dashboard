# Setup Guide

> **Complete step-by-step setup guide for Self Service Dashboard AI**

## 🎯 Overview

This guide will walk you through setting up the complete AI-powered dashboard system from scratch. The process typically takes 10-15 minutes.

## 📋 Prerequisites

### System Requirements

| Component | Requirement | Check Command | Install Guide |
|-----------|-------------|---------------|---------------|
| **Node.js** | 16.0.0+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| **npm** | 8.0.0+ | `npm --version` | Included with Node.js |
| **Python** | 3.8+ | `python3 --version` | [python.org](https://python.org/) |
| **pip** | Latest | `pip3 --version` | Included with Python |
| **Git** | Any recent | `git --version` | [git-scm.com](https://git-scm.com/) |

### Optional but Recommended
- **Ollama** - For AI dashboard generation ([ollama.ai](https://ollama.ai/))
- **curl** - For API testing
- **lsof** - For port management

## 🚀 Quick Setup (Recommended)

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd SelfServiceDashboard-TM
```

### 2. Run One-Command Setup
```bash
./setup.sh
```

This script will:
- ✅ Verify all prerequisites
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Configure environment variables
- ✅ Create necessary directories
- ✅ Run health checks

### 3. Install Ollama (Interactive)
```bash
./scripts/install-ollama.sh
```

Follow the prompts to:
- Install Ollama for your OS
- Download the Llama3 model
- Test the installation

### 4. Start Development Environment
```bash
./scripts/start-development.sh
```

### 5. Open Application
Navigate to [http://localhost:3000](http://localhost:3000)

## 📋 Manual Setup (Step by Step)

If you prefer manual control or need to troubleshoot:

### Step 1: Verify Prerequisites
```bash
# Check Node.js and npm
node --version    # Should be 16.0.0+
npm --version     # Should be 8.0.0+

# Check Python and pip
python3 --version # Should be 3.8+
pip3 --version    # Should be latest

# Check Git
git --version     # Any recent version
```

### Step 2: Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit if needed (optional)
nano .env
```

**Environment Variables:**
```bash
# AI Configuration
OLLAMA_URL=http://localhost:11434
AI_BACKEND_PORT=5247
OLLAMA_MODEL=llama3

# Frontend Configuration
REACT_APP_AI_BACKEND_URL=http://localhost:5247
REACT_APP_SIMPLE_BACKEND_URL=http://localhost:5246

# Development Configuration
NODE_ENV=development
LOG_LEVEL=info
```

### Step 3: Install Root Dependencies
```bash
npm install
```

### Step 4: Setup AI Backend
```bash
cd ai-backend

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Test installation
python test_connection.py

cd ..
```

### Step 5: Setup Frontend
```bash
cd frontend
npm install
cd ..
```

### Step 6: Setup Simple Backend (if exists)
```bash
cd simple-backend
npm install
cd ..
```

### Step 7: Install Ollama

#### macOS/Linux (Automated)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
1. Download installer from [ollama.ai](https://ollama.ai/)
2. Run the installer
3. Restart terminal

#### Start Ollama
```bash
ollama serve
```

#### Download Model
```bash
ollama pull llama3
```

### Step 8: Create Directories
```bash
mkdir -p generated-dashboards
mkdir -p logs
mkdir -p data
```

## ✅ Verification

### Health Check
```bash
./scripts/health-check.sh
```

**Expected Output:**
```
🔍 Self Service Dashboard - System Health Check

📋 System Requirements:
✅ Node.js is installed (v18.17.0)
✅ npm is installed (9.6.7)
✅ Python 3 is installed (Python 3.9.6)
✅ pip3 is installed (pip 23.2.1)

🛠️  Optional Tools:
✅ Git is installed (git version 2.39.2)
✅ curl is installed (curl 8.1.2)
✅ lsof is installed

📁 Project Structure:
✅ Environment config: .env
✅ Root package.json: package.json
✅ Frontend: frontend
✅ AI Backend: ai-backend

🔧 Environment Setup:
✅ Python virtual environment exists
✅ Python dependencies appear to be installed
✅ Frontend dependencies are installed

🦙 Ollama Setup:
✅ Ollama is installed
✅ Ollama is running
✅ Llama3 model is available

🎉 All critical health checks passed!
```

### Test Individual Components

#### Test AI Backend
```bash
cd ai-backend
source venv/bin/activate
python test_connection.py
```

#### Test Frontend
```bash
cd frontend
npm start
# Should open http://localhost:3000
```

#### Test Ollama
```bash
ollama run llama3 "Hello, are you working?"
```

## 🚀 Starting the Application

### Development Mode
```bash
# Start all services with hot reload
./scripts/start-development.sh
```

**Services Started:**
- 🌐 Frontend: http://localhost:3000
- 🧠 AI Backend: http://localhost:5247
- 🔧 Simple Backend: http://localhost:5246
- 🦙 Ollama: http://localhost:11434

### Production Mode
```bash
# Build and start optimized version
./scripts/start-production.sh
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3` | Model for AI generation |
| `AI_BACKEND_PORT` | `5247` | AI backend port |
| `SIMPLE_BACKEND_PORT` | `5246` | Simple backend port |
| `NODE_ENV` | `development` | Environment mode |
| `LOG_LEVEL` | `info` | Logging verbosity |

### Customizing Ports

If you need to change ports:

1. **Edit `.env` file:**
```bash
AI_BACKEND_PORT=5248
SIMPLE_BACKEND_PORT=5247
```

2. **Update frontend configuration:**
```bash
REACT_APP_AI_BACKEND_URL=http://localhost:5248
REACT_APP_SIMPLE_BACKEND_URL=http://localhost:5247
```

3. **Restart services:**
```bash
./scripts/reset-project.sh --quick
./scripts/start-development.sh
```

## 🐳 Docker Setup (Optional)

### Development with Docker
```bash
# Start all services
docker-compose up

# Start with hot reload
docker-compose -f docker-compose.dev.yml up
```

### Production with Docker
```bash
# Build and start production
docker-compose -f docker-compose.prod.yml up --build
```

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :3000

# Kill process
kill -9 <PID>

# Or use our helper
./scripts/reset-project.sh --quick
```

#### Python Virtual Environment Issues
```bash
# Remove and recreate
cd ai-backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running
ollama serve

# If model missing
ollama pull llama3
```

#### Frontend Build Errors
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

1. **Run Health Check:**
```bash
./scripts/health-check.sh
```

2. **Check Logs:**
```bash
# Development logs
tail -f logs/frontend.log
tail -f logs/ai-backend.log

# Or view all logs
ls -la logs/
```

3. **Reset Everything:**
```bash
./scripts/reset-project.sh --full
```

## 🎯 Next Steps

After successful setup:

1. **Upload Data**: Try uploading an Excel file or use the sample data
2. **Test AI Generation**: Use the AI dashboard generator with natural language
3. **Explore PygWalker**: Try the drag-and-drop analytics
4. **Customize**: Modify components in `frontend/src/components/`
5. **Deploy**: Use production scripts for deployment

## 📚 Additional Resources

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[API Documentation](API.md)** - AI backend API reference
- **[Project Structure](../README.md#-project-structure)** - Detailed component overview

---

**🎉 You're all set! Open [http://localhost:3000](http://localhost:3000) and start exploring your AI-powered dashboard system.**