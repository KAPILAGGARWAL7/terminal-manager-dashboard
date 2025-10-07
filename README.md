# Self Service Dashboard AI

> **AI-powered Self Service Dashboard with PygWalker integration and Ollama-based intelligent dashboard generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-16%2B-green)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Ready-purple)](https://ollama.ai/)

## 🚀 Quick Start

### One-Command Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd SelfServiceDashboard-TM

# One-command setup
./setup.sh

# Install Ollama (interactive)
./scripts/install-ollama.sh

# Start development environment
./scripts/start-development.sh

# Open http://localhost:3000
```

## ✨ Features

### 🤖 AI-Powered Dashboard Generation
- **Natural Language to Dashboard**: Describe what you want, AI creates it
- **Ollama Integration**: Local LLM for privacy and speed
- **Streamlit Dashboards**: Interactive, professional dashboards
- **Real-time Generation**: Watch your ideas come to life

### 📊 Interactive Analytics
- **PygWalker Integration**: Drag-and-drop data visualization
- **Multiple Data Sources**: Excel files, REST APIs, databases
- **Terminal Manager Optimized**: Built for operational data
- **Real-time Performance**: No timeouts, smooth interactions

### 🔧 Developer Friendly
- **Zero Configuration**: Works out of the box
- **Hot Reload**: All services support development mode
- **Comprehensive Scripts**: Health checks, deployment, reset
- **Cross-Platform**: macOS, Linux, Windows (WSL)

## 📁 Project Structure

```
SelfServiceDashboard-TM/
├── setup.sh                     # 🎯 One-command setup
├── package.json                 # 📦 Workspace configuration
├── .env.example                 # 🔧 Environment template
├── docker-compose.yml           # 🐳 Optional Docker setup
│
├── frontend/                    # ⚛️ React Frontend
│   ├── src/components/
│   │   ├── AIDashboardGenerator.js  # 🤖 AI dashboard component
│   │   └── LoadingStates.js         # 🔄 Shared loading components
│   └── src/StandaloneChart.js       # 📊 Main visualization component
│
├── ai-backend/                  # 🧠 AI Backend (Python/Flask)
│   ├── app.py                   # 🌐 Flask server
│   ├── config.py                # ⚙️ Configuration management
│   ├── requirements.txt         # 📋 Python dependencies
│   ├── install.sh              # 🔧 Auto setup script
│   └── test_connection.py      # 🔍 Ollama connection test
│
├── simple-backend/              # 🔧 Simple Backend (Node.js)
│   └── (existing backend)       # 📡 REST API endpoints
│
├── scripts/                     # 🛠️ Management Scripts
│   ├── start-development.sh     # 🚀 Start dev environment
│   ├── start-production.sh      # 🏭 Start production mode
│   ├── health-check.sh          # 🔍 System health verification
│   ├── install-ollama.sh        # 🦙 Ollama installation helper
│   └── reset-project.sh         # 🔄 Clean project reset
│
├── docs/                        # 📚 Documentation
│   ├── SETUP.md                # 📋 Detailed setup guide
│   ├── TROUBLESHOOTING.md       # 🔧 Common issues & solutions
│   └── API.md                  # 📖 AI backend API docs
│
└── generated-dashboards/        # 📊 Runtime generated dashboards
```

## 🔧 Requirements

### System Requirements
- **Node.js** 16+ and npm 8+
- **Python** 3.8+ and pip3
- **Git** (recommended)
- **Ollama** (for AI features)

### Ports Used
- `3000` - Frontend (React)
- `5246` - Simple Backend (Node.js)
- `5247` - AI Backend (Python/Flask)
- `8501+` - Generated Streamlit dashboards
- `11434` - Ollama (if installed)

## 📖 Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API Documentation](docs/API.md)** - AI backend API reference

## 🛠️ Available Scripts

### Development
```bash
npm run start              # Start development environment
npm run health            # Run health checks
./scripts/health-check.sh  # Detailed system health check
```

### Production
```bash
npm run start:prod        # Start production environment
npm run build:frontend    # Build frontend for production
```

### Maintenance
```bash
npm run reset             # Reset project (interactive)
./scripts/reset-project.sh --full    # Full reset
./scripts/reset-project.sh --quick   # Quick reset
```

### AI Backend
```bash
cd ai-backend
./install.sh             # Setup Python environment
./test_connection.py     # Test Ollama connection
python app.py            # Start AI backend directly
```

## 🤖 AI Dashboard Examples

### Natural Language Prompts
```
"Create a sales performance dashboard with regional comparisons"
"Build an operational efficiency dashboard with KPIs and uptime metrics"
"Generate a financial overview with profit margins and cost analysis"
"Show fuel volume analysis with environmental impact metrics"
```

### Generated Features
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Data**: Live connection to your data sources
- **Professional Styling**: Terminal Manager branding
- **Export Options**: Multiple formats supported
- **Responsive Design**: Works on all devices

## 🔍 Health Monitoring

The project includes comprehensive health monitoring:

```bash
# Quick health check
./scripts/health-check.sh

# Development mode check
./scripts/health-check.sh --dev-mode

# Setup mode check (during installation)
./scripts/health-check.sh --setup-mode
```

**Monitors:**
- ✅ System requirements (Node.js, Python, etc.)
- ✅ Project structure and files
- ✅ Environment setup (virtual envs, dependencies)
- ✅ Ollama installation and connectivity
- ✅ Port availability
- ✅ Service health (in dev/prod mode)

## 🐳 Docker Support

Optional Docker setup for containerized deployment:

```bash
# Start with Docker Compose
docker-compose up

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

## 🔒 Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# AI Backend Configuration
OLLAMA_URL=http://localhost:11434
AI_BACKEND_PORT=5247
OLLAMA_MODEL=llama3

# Frontend Configuration
REACT_APP_AI_BACKEND_URL=http://localhost:5247

# Development Configuration
NODE_ENV=development
LOG_LEVEL=info
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `./scripts/health-check.sh`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs or request features in [GitHub Issues](../../issues)
- **Documentation**: Check [docs/](docs/) for detailed guides
- **Health Check**: Run `./scripts/health-check.sh` for diagnostics

## 🙏 Acknowledgments

- **PygWalker** - Drag-and-drop visualization engine
- **Ollama** - Local LLM infrastructure
- **Streamlit** - Dashboard generation framework
- **React** - Frontend framework
- **Flask** - Backend API framework

---

**🎯 Built for Terminal Manager Operations | 🚀 Powered by AI | 💻 Ready for Production**