# 🏭 AI Manufacturing Dashboard Generator - Setup Guide

## Overview

This system generates professional manufacturing dashboards using AI (Ollama) based on natural language prompts. It creates dynamic Streamlit dashboards with real-time KPIs, OEE metrics, lane/bay performance, and advanced analytics.

## 🎯 Supported Dashboard Types

### Manufacturing Dashboards
1. **Lane/Bay Performance Dashboard** - Real-time throughput, utilization, and performance metrics
2. **Schedule vs Actual Adherence Dashboard** - Production planning vs actual execution
3. **Throughput and Capacity Dashboard** - Production capacity analysis and optimization
4. **Product Mix and Performance Dashboard** - Product-specific performance metrics
5. **Overall Equipment Effectiveness (OEE)** - Availability, Performance, Quality breakdown

### Professional Features
- **Real-time KPI Cards** with status indicators
- **Interactive Charts** using Plotly (gauges, heatmaps, trend lines)
- **Professional Styling** with manufacturing-focused color schemes
- **Responsive Design** for desktop and mobile viewing
- **Data Export** capabilities and filtering options

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Ollama installed and running
- Node.js (for frontend development, optional)

### 2. Install Ollama
```bash
# Run the provided installation script
./scripts/install-ollama.sh

# Or manually:
# Windows: Download from https://ollama.ai/
# Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3
```

### 3. Setup AI Backend
```bash
# Navigate to AI backend directory
cd ai-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the AI backend
python app.py
```

### 4. Start Dashboard Generator
```bash
# Open the web interface
# Option 1: Open dashboard-generator.html in your browser
start dashboard-generator.html

# Option 2: Use a local server
python -m http.server 8080
# Then visit: http://localhost:8080/dashboard-generator.html
```

## 📊 Usage Flow

### Step 1: Enter Natural Language Prompt
Examples:
- "Create a lane performance dashboard showing throughput, OEE, and bay utilization"
- "Build a schedule vs actual adherence dashboard with production targets"
- "Generate a throughput and capacity dashboard for manufacturing lines"

### Step 2: AI Processing
- Ollama analyzes your prompt
- Determines dashboard type (manufacturing/operational/etc.)
- Generates specialized Streamlit code
- Creates professional styling and KPIs

### Step 3: Dashboard Generation
- Streamlit dashboard is created and launched
- Accessible via web browser
- Real-time data visualization
- Interactive filtering and controls

## 🏗️ Architecture

```
User Prompt → Flask AI Backend → Ollama LLM → Generated Code → Streamlit Dashboard
     ↓              ↓                ↓              ↓              ↓
  Web UI    →   Port 5247    →   Port 11434   →   Python    →   Port 8501+
```

### Components:
1. **Web Interface** (`dashboard-generator.html`) - User-friendly prompt input
2. **AI Backend** (`ai-backend/app.py`) - Flask API with Ollama integration
3. **Dashboard Templates** - Specialized manufacturing dashboard code
4. **Streamlit Dashboards** - Generated interactive dashboards

## 🎛️ Configuration

### Environment Variables
```bash
# AI Backend Configuration
AI_BACKEND_PORT=5247
AI_BACKEND_HOST=localhost
AI_DEBUG=false

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=60

# Dashboard Configuration
STREAMLIT_BASE_PORT=8501
MAX_DASHBOARD_SIZE=10MB
DASHBOARD_TIMEOUT=30000
```

### Custom Models
To use different Ollama models:
```bash
# Pull alternative models
ollama pull codellama
ollama pull mistral

# Update configuration
export OLLAMA_MODEL=codellama
```

## 📈 Dashboard Features

### Manufacturing KPIs
- **Overall Equipment Effectiveness (OEE)** - Industry standard metric
- **Schedule Adherence** - Production vs. planned targets
- **Throughput Analysis** - Units per hour/shift/day
- **Quality Metrics** - First Pass Yield, defect rates
- **Downtime Tracking** - Planned vs. unplanned stops
- **Energy Efficiency** - kWh per unit produced

### Visual Components
- **Real-time Gauges** for live metrics
- **Heatmaps** for lane/bay performance
- **Trend Lines** for historical analysis
- **Bar Charts** for comparisons
- **Pie Charts** for distributions
- **Gantt Charts** for schedules

### Interactive Features
- **Date Range Selection**
- **Lane/Bay Filtering**
- **Shift Analysis**
- **Product Mix Views**
- **Export Capabilities**
- **Drill-down Analytics**

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check available models
ollama list
```

#### 2. AI Backend Not Responding
```bash
# Check if backend is running
curl http://localhost:5247/health

# Restart backend
cd ai-backend
python app.py
```

#### 3. Dashboard Generation Fails
- Verify Ollama model is available: `ollama list`
- Check AI backend logs for errors
- Ensure sufficient disk space for generated files
- Try with simpler prompts first

#### 4. Streamlit Port Conflicts
- Multiple dashboards use sequential ports (8501, 8502, etc.)
- Check running processes: `netstat -an | grep 850`
- Kill processes if needed: `pkill -f streamlit`

### Performance Optimization

#### 1. Model Selection
- **llama3** - Balanced performance and speed
- **codellama** - Better for code generation
- **mistral** - Faster inference

#### 2. System Resources
- Minimum 8GB RAM for Ollama
- SSD storage recommended
- GPU acceleration optional but beneficial

## 🛠️ Development

### Adding Custom Dashboard Types
1. Add new type detection in `analyze_dashboard_type()`
2. Create requirements in `get_dashboard_requirements()`
3. Implement template in `get_*_template()`
4. Test with appropriate prompts

### Extending Templates
- Modify existing templates in `ai-backend/app.py`
- Add new chart types using Plotly
- Implement additional KPIs and metrics
- Customize styling and layouts

### API Endpoints
- `GET /health` - System health check
- `GET /api/status` - Detailed system status
- `POST /api/dashboard/generate` - Generate dashboard
- `GET /api/dashboard/list` - List running dashboards
- `POST /api/dashboard/stop/<id>` - Stop specific dashboard

## 📝 Example Prompts

### Manufacturing Focused
```
"Create a comprehensive OEE dashboard showing availability, performance, and quality metrics for 4 production lanes with real-time alerts"

"Build a shift performance comparison dashboard with throughput, downtime analysis, and energy consumption by bay"

"Generate a product mix optimization dashboard showing cycle times, changeover efficiency, and capacity utilization"

"Design a predictive maintenance dashboard with equipment health scores, failure predictions, and maintenance schedules"
```

### Advanced Analytics
```
"Create a statistical process control dashboard with control charts, capability analysis, and quality trends"

"Build a lean manufacturing dashboard with waste identification, value stream mapping, and continuous improvement metrics"

"Generate a cost optimization dashboard showing labor efficiency, material utilization, and overhead allocation"
```

## 🔐 Security Notes

- AI backend runs on localhost by default
- No external API keys required (local Ollama)
- Generated dashboards contain sample data unless real data provided
- Consider network security for production deployments

## 📞 Support

For issues or enhancements:
1. Check logs in `ai-backend/` directory
2. Verify system status via web interface
3. Test individual components (Ollama, Flask, Streamlit)
4. Review this documentation for configuration options

## 🚀 Next Steps

1. **Test Basic Functionality** - Generate a simple dashboard
2. **Upload Real Data** - Replace sample data with actual manufacturing data
3. **Customize Templates** - Modify dashboards for specific needs
4. **Scale Deployment** - Consider production hosting options
5. **Integrate Systems** - Connect to existing MES/ERP systems