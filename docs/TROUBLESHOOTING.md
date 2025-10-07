# Troubleshooting Guide

> **Common issues and solutions for Self Service Dashboard AI**

## 🔍 Quick Diagnostics

### Health Check First
Always start with the health check to identify issues:

```bash
./scripts/health-check.sh
```

This will identify most common problems and suggest solutions.

## 🚨 Common Issues

### 1. Setup Issues

#### ❌ "Command not found: ./setup.sh"
**Cause:** Script is not executable
**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

#### ❌ "Node.js/Python not found"
**Cause:** Prerequisites not installed
**Solution:**
```bash
# Install Node.js
# macOS: brew install node
# Ubuntu: sudo apt install nodejs npm
# Windows: Download from nodejs.org

# Install Python
# macOS: brew install python
# Ubuntu: sudo apt install python3 python3-pip
# Windows: Download from python.org
```

#### ❌ "Permission denied" during setup
**Cause:** Insufficient permissions
**Solution:**
```bash
# Don't use sudo with the setup script
# Instead fix permissions:
chmod +x setup.sh scripts/*.sh ai-backend/*.sh
```

### 2. Port Conflicts

#### ❌ "Port 3000 already in use"
**Cause:** Another service is using the port
**Solution:**
```bash
# Find what's using the port
lsof -i :3000

# Kill the process
kill -9 <PID>

# Or use our helper
./scripts/reset-project.sh --quick
```

#### ❌ "AI Backend port conflict"
**Cause:** Multiple instances running
**Solution:**
```bash
# Check AI backend port
lsof -i :5247

# Kill all Python processes
pkill -f "python.*app.py"

# Or reset everything
./scripts/reset-project.sh --quick
```

### 3. Python Environment Issues

#### ❌ "Virtual environment not found"
**Cause:** venv not created or corrupted
**Solution:**
```bash
cd ai-backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### ❌ "ModuleNotFoundError: No module named 'flask'"
**Cause:** Dependencies not installed in venv
**Solution:**
```bash
cd ai-backend
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### ❌ "Python version too old"
**Cause:** Python 3.7 or older
**Solution:**
```bash
# Check version
python3 --version

# Install Python 3.8+
# macOS: brew install python@3.9
# Ubuntu: sudo apt install python3.9
# Windows: Download from python.org
```

### 4. Ollama Issues

#### ❌ "Ollama not connected"
**Cause:** Ollama not installed or not running
**Solution:**
```bash
# Install Ollama
./scripts/install-ollama.sh

# Or manually:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

#### ❌ "Model 'llama3' not found"
**Cause:** Model not downloaded
**Solution:**
```bash
# Pull the model (this may take time)
ollama pull llama3

# Verify model is available
ollama list
```

#### ❌ "Ollama connection timeout"
**Cause:** Ollama taking too long to respond
**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Test simple query
ollama run llama3 "Hello"
```

### 5. Frontend Issues

#### ❌ "npm install fails"
**Cause:** Node modules cache corruption
**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### ❌ "React scripts not found"
**Cause:** Dependencies not properly installed
**Solution:**
```bash
cd frontend
npm install react-scripts --save
npm start
```

#### ❌ "Frontend build fails"
**Cause:** Memory issues or dependency conflicts
**Solution:**
```bash
# Increase Node memory
export NODE_OPTIONS="--max-old-space-size=4096"

# Clean install
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### 6. AI Backend Issues

#### ❌ "AI Backend not responding"
**Cause:** Backend crashed or not started
**Solution:**
```bash
# Check if running
curl http://localhost:5247/health

# Check logs
tail -f logs/ai-backend.log

# Restart manually
cd ai-backend
source venv/bin/activate
python app.py
```

#### ❌ "Dashboard generation fails"
**Cause:** Various issues with data or LLM
**Solution:**
```bash
# Test AI backend connection
cd ai-backend
source venv/bin/activate
python test_connection.py

# Check Ollama is responding
curl http://localhost:11434/api/tags

# Verify data is loaded
# Check in frontend console for errors
```

### 7. Data Loading Issues

#### ❌ "Excel file upload fails"
**Cause:** File format or size issues
**Solution:**
- Ensure file is .xlsx or .xls format
- Check file size (< 10MB recommended)
- Verify file is not corrupted
- Try with a simple test file

#### ❌ "Data processing errors"
**Cause:** Invalid data format
**Solution:**
- Ensure first row contains headers
- Remove empty rows/columns
- Check for special characters in headers
- Convert dates to standard format

### 8. Development Environment Issues

#### ❌ "Services won't start together"
**Cause:** Resource conflicts or dependency issues
**Solution:**
```bash
# Full reset
./scripts/reset-project.sh --full

# Setup again
./setup.sh

# Start development
./scripts/start-development.sh
```

#### ❌ "Hot reload not working"
**Cause:** File watching issues
**Solution:**
```bash
# Restart development environment
Ctrl+C  # Stop current processes
./scripts/start-development.sh
```

## 🛠️ Advanced Troubleshooting

### Debug Mode

#### Enable Verbose Logging
```bash
# Edit .env file
LOG_LEVEL=debug
AI_DEBUG=true

# Restart services
./scripts/reset-project.sh --quick
./scripts/start-development.sh
```

#### Check All Logs
```bash
# View all log files
ls -la logs/

# Tail multiple logs
tail -f logs/*.log

# Search for errors
grep -i error logs/*.log
```

### Manual Component Testing

#### Test AI Backend Directly
```bash
cd ai-backend
source venv/bin/activate
python app.py

# In another terminal
curl http://localhost:5247/health
curl http://localhost:5247/api/status
```

#### Test Frontend Directly
```bash
cd frontend
npm start

# Should open http://localhost:3000
```

#### Test Ollama Directly
```bash
# Basic test
ollama run llama3 "Test response"

# API test
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "prompt": "Hello", "stream": false}'
```

### Network Issues

#### Check Port Availability
```bash
# Check all required ports
for port in 3000 5246 5247 8501 11434; do
  echo "Port $port:"
  lsof -i :$port || echo "  Available"
done
```

#### Test Internal Connectivity
```bash
# Test frontend -> AI backend
curl http://localhost:5247/health

# Test AI backend -> Ollama
curl http://localhost:11434/api/tags
```

## 🔄 Reset Strategies

### Quick Reset (Processes Only)
```bash
./scripts/reset-project.sh --quick
```
- Stops all processes
- Cleans logs
- Removes generated files
- Keeps dependencies

### Full Reset (Everything)
```bash
./scripts/reset-project.sh --full
```
- Everything in quick reset
- Removes virtual environments
- Reinstalls all dependencies
- Resets configuration

### Targeted Resets
```bash
# Reset only Python environment
cd ai-backend
rm -rf venv
./install.sh

# Reset only frontend
cd frontend
rm -rf node_modules package-lock.json
npm install

# Reset only configuration
rm .env
cp .env.example .env
```

## 📊 Performance Issues

### Slow Dashboard Generation
**Symptoms:** AI dashboard takes too long
**Solutions:**
- Check Ollama model is running locally
- Ensure sufficient system memory (8GB+ recommended)
- Try smaller datasets
- Check network connectivity to Ollama

### High Memory Usage
**Symptoms:** System slows down during operation
**Solutions:**
```bash
# Monitor memory usage
top
htop

# Reduce Node.js memory if needed
export NODE_OPTIONS="--max-old-space-size=2048"

# Restart services periodically
./scripts/reset-project.sh --quick
```

### Slow File Uploads
**Symptoms:** Excel files take long to process
**Solutions:**
- Reduce file size (< 5MB recommended)
- Remove unnecessary columns/rows
- Use CSV format for large datasets
- Check disk space availability

## 🆘 Getting Additional Help

### Collect Diagnostic Information
```bash
# Run comprehensive health check
./scripts/health-check.sh > health-report.txt

# Collect system information
echo "=== System Info ===" >> health-report.txt
uname -a >> health-report.txt
node --version >> health-report.txt
python3 --version >> health-report.txt

# Collect logs
echo "=== Logs ===" >> health-report.txt
tail -100 logs/*.log >> health-report.txt 2>/dev/null
```

### Before Reporting Issues
1. ✅ Run health check: `./scripts/health-check.sh`
2. ✅ Try full reset: `./scripts/reset-project.sh --full`
3. ✅ Check this troubleshooting guide
4. ✅ Collect diagnostic information
5. ✅ Include specific error messages

### Contact Support
- **GitHub Issues**: Create detailed issue with diagnostic info
- **Include**: OS, Node version, Python version, error logs
- **Steps**: Clear reproduction steps

## 📚 Prevention Tips

### Regular Maintenance
```bash
# Weekly health check
./scripts/health-check.sh

# Monthly dependency updates
cd frontend && npm update
cd ../ai-backend && pip list --outdated
```

### Best Practices
- Keep system dependencies updated
- Monitor disk space (>5GB free recommended)
- Restart development environment daily
- Use version control for configuration changes
- Backup working configurations

---

**💡 Remember: Most issues can be resolved with `./scripts/health-check.sh` followed by `./scripts/reset-project.sh --full` if needed.**