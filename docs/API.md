# AI Backend API Documentation

> **Complete API reference for the AI-powered dashboard generation backend**

## 🚀 Overview

The AI Backend provides REST API endpoints for generating interactive dashboards using natural language prompts and Ollama-powered LLM processing.

**Base URL:** `http://localhost:5247`
**Content-Type:** `application/json`
**CORS:** Enabled for `http://localhost:3000`

## 📊 Quick Start

### Basic Dashboard Generation
```bash
curl -X POST http://localhost:5247/api/dashboard/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a sales dashboard with regional analysis",
    "excel_path": "/path/to/data.xlsx"
  }'
```

### Check System Health
```bash
curl http://localhost:5247/health
```

## 🔗 API Endpoints

### Health & Status

#### `GET /health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "AI Dashboard Backend Running"
}
```

#### `GET /api/status`
Comprehensive system status including Ollama connectivity.

**Response:**
```json
{
  "ollama": {
    "connected": true,
    "models": ["llama3", "codellama"],
    "has_required_model": true
  },
  "directories": {
    "dashboard_dir": true,
    "data_dir": true,
    "logs_dir": true
  },
  "config": {
    "port": 5247,
    "debug": false,
    "model": "llama3"
  }
}
```

#### `GET /api/config`
Current backend configuration.

**Response:**
```json
{
  "ollama_url": "http://localhost:11434",
  "model": "llama3",
  "port": 5247,
  "debug": false,
  "streamlit_base_port": 8501
}
```

### Dashboard Generation

#### `POST /api/dashboard/generate`
Generate an interactive Streamlit dashboard from natural language prompt.

**Request Body:**
```json
{
  "prompt": "string (required)",
  "excel_path": "string (optional)",
  "data": "object (optional)",
  "fileName": "string (optional)"
}
```

**Parameters:**
- `prompt` **(required)**: Natural language description of desired dashboard
- `excel_path` **(optional)**: Path to Excel file to use as data source
- `data` **(optional)**: JSON data array to use instead of Excel file
- `fileName` **(optional)**: Display name for the data source

**Example Request:**
```json
{
  "prompt": "Create a comprehensive sales dashboard with regional breakdowns, fuel type analysis, and performance metrics with interactive filters",
  "excel_path": "/path/to/sales-data.xlsx"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "dashboard_id": "1698765432_1234",
  "dashboard_url": "http://localhost:8501",
  "embed_url": "http://localhost:8501/?embed=true",
  "message": "Dashboard generated successfully"
}
```

**Error Response (400/500):**
```json
{
  "error": "Error message describing what went wrong"
}
```

#### `GET /api/dashboard/list`
List all currently running dashboards.

**Response:**
```json
{
  "dashboards": [
    {
      "id": "1698765432_1234",
      "port": 8501,
      "created_at": "2024-01-15T10:30:00",
      "prompt": "Sales dashboard with regional analysis",
      "url": "http://localhost:8501"
    }
  ]
}
```

#### `POST /api/dashboard/stop/{dashboard_id}`
Stop a running dashboard by ID.

**Path Parameters:**
- `dashboard_id`: The unique identifier of the dashboard to stop

**Success Response (200):**
```json
{
  "success": true,
  "message": "Dashboard stopped"
}
```

**Error Response (404):**
```json
{
  "error": "Dashboard not found"
}
```

## 🤖 Natural Language Prompts

### Prompt Guidelines

The AI backend accepts natural language prompts and converts them into interactive dashboards. Here are examples of effective prompts:

#### Sales Analysis
```json
{
  "prompt": "Create a sales performance dashboard with regional comparisons and trend analysis"
}
```

#### Operational Efficiency
```json
{
  "prompt": "Build an operational efficiency dashboard with KPIs, uptime metrics, and fuel volume analysis"
}
```

#### Financial Overview
```json
{
  "prompt": "Generate a financial overview with profit margins, cost analysis, and revenue trends by terminal"
}
```

#### Custom Visualization
```json
{
  "prompt": "Show fuel volume trends over time with environmental impact metrics and regional breakdowns"
}
```

### Prompt Best Practices

1. **Be Specific**: Include specific metrics and dimensions you want to see
2. **Mention Chart Types**: Reference charts like "bar chart", "line graph", "pie chart"
3. **Include Filters**: Request interactive filters for better user experience
4. **Specify Grouping**: Mention how to group data (by region, time, category)
5. **Request Context**: Ask for relevant business context and insights

### Generated Dashboard Features

The AI backend automatically includes:
- **Terminal Manager Branding**: Professional blue/navy theme
- **Interactive Charts**: Plotly-powered visualizations
- **Data Validation**: Handles null values gracefully
- **Responsive Design**: Works on all screen sizes
- **Export Options**: Built-in sharing and export capabilities

## 📁 Data Processing

### Supported Data Sources

#### Excel Files
- **Formats**: `.xlsx`, `.xls`
- **Size Limit**: 10MB (configurable)
- **Requirements**: First row must contain headers

#### JSON Data
- **Format**: Array of objects
- **Schema**: Flexible, auto-detected
- **Size Limit**: 10MB in memory

### Data Processing Pipeline

1. **Data Ingestion**: Excel → SQLite conversion or JSON processing
2. **Schema Detection**: Automatic field type inference
3. **Data Cleaning**: Null value handling, type conversion
4. **Context Building**: Statistical analysis for LLM context
5. **Dashboard Generation**: LLM prompt → Streamlit code
6. **Deployment**: Auto-deployment on available port

### Data Schema Example

```json
{
  "table_name": "sales_data",
  "columns": ["region", "fuel_type", "volume", "revenue"],
  "sample_data": [
    {
      "region": "North",
      "fuel_type": "Diesel",
      "volume": 1500,
      "revenue": 4500
    }
  ]
}
```

## ⚙️ Configuration

### Environment Variables

The backend reads configuration from environment variables:

```bash
# Core Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
AI_BACKEND_PORT=5247
AI_BACKEND_HOST=localhost

# Performance
OLLAMA_TIMEOUT=60
DASHBOARD_TIMEOUT=30000
MAX_DASHBOARD_SIZE=10MB

# Development
DEBUG=false
LOG_LEVEL=info
```

### Runtime Configuration

Update configuration at runtime via environment or config files:

```python
from config import Config

# Check current config
status = Config.get_status()

# Validate Ollama connection
ollama_status = Config.validate_ollama_connection()
```

## 🔒 Security & CORS

### CORS Policy
- **Allowed Origins**: Configurable via `CORS_ORIGINS` environment variable
- **Default**: `http://localhost:3000`
- **Methods**: GET, POST, OPTIONS
- **Headers**: Content-Type, Authorization

### Security Headers
- Content-Type validation
- Request size limits
- Path traversal protection
- Input sanitization

### Rate Limiting
- **Dashboard Generation**: 10 requests per minute per IP
- **Health Checks**: Unlimited
- **Configuration**: Adjustable via environment variables

## 🧪 Testing

### Health Check Endpoints
```bash
# Basic health
curl http://localhost:5247/health

# Detailed status
curl http://localhost:5247/api/status

# Configuration
curl http://localhost:5247/api/config
```

### Dashboard Generation Test
```bash
# Test with sample data
curl -X POST http://localhost:5247/api/dashboard/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a simple test dashboard",
    "data": [
      {"name": "Product A", "sales": 100},
      {"name": "Product B", "sales": 200}
    ]
  }'
```

### Python Test Script
```python
#!/usr/bin/env python3
import requests
import json

# Test health
response = requests.get('http://localhost:5247/health')
print(f"Health: {response.json()}")

# Test dashboard generation
data = {
    "prompt": "Show sales by product",
    "data": [
        {"product": "A", "sales": 100},
        {"product": "B", "sales": 200}
    ]
}

response = requests.post(
    'http://localhost:5247/api/dashboard/generate',
    json=data
)
print(f"Dashboard: {response.json()}")
```

## 🚨 Error Handling

### Common Error Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Missing prompt, invalid data format |
| 404 | Not Found | Dashboard ID not found |
| 500 | Internal Server Error | Ollama connection failed, LLM error |
| 503 | Service Unavailable | Ollama not running |

### Error Response Format
```json
{
  "error": "Descriptive error message",
  "code": "ERROR_CODE",
  "details": {
    "component": "ollama|llm|data|dashboard",
    "suggestion": "Helpful suggestion for fixing the issue"
  }
}
```

### Debugging

#### Enable Debug Mode
```bash
export AI_DEBUG=true
export LOG_LEVEL=debug
```

#### Check Logs
```bash
# View AI backend logs
tail -f logs/ai-backend.log

# Check for specific errors
grep -i "error" logs/ai-backend.log
```

## 📊 Performance

### Response Times
- **Health Check**: < 50ms
- **Status Check**: < 200ms
- **Dashboard Generation**: 10-30 seconds (depends on data size and LLM response time)

### Resource Usage
- **Memory**: ~500MB base + ~100MB per dashboard
- **CPU**: Moderate during generation, low during serving
- **Disk**: ~1MB per generated dashboard

### Optimization Tips
- Use smaller datasets for faster generation
- Keep Ollama model loaded (first request is slower)
- Monitor system resources during heavy usage
- Consider horizontal scaling for production

## 🔧 Development

### Local Development Setup
```bash
cd ai-backend
source venv/bin/activate
export FLASK_ENV=development
export AI_DEBUG=true
python app.py
```

### Adding New Endpoints
```python
@app.route('/api/custom', methods=['POST'])
def custom_endpoint():
    try:
        data = request.get_json()
        # Your logic here
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### Testing Changes
```bash
# Run connection test
python test_connection.py

# Test specific functionality
python -c "
from app import app
with app.test_client() as client:
    response = client.get('/health')
    print(response.json)
"
```

---

**🚀 For more examples and advanced usage, see the [Setup Guide](SETUP.md) and [Troubleshooting Guide](TROUBLESHOOTING.md).**