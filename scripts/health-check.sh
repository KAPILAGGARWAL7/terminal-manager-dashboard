#!/bin/bash

# Comprehensive Health Check Script
# Verifies system requirements and service health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Flags
SETUP_MODE=false
DEV_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-mode)
            SETUP_MODE=true
            shift
            ;;
        --dev-mode)
            DEV_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[HEALTH]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✅]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠️]${NC} $1"
}

log_error() {
    echo -e "${RED}[❌]${NC} $1"
}

log_check() {
    echo -e "${PURPLE}[🔍]${NC} $1"
}

# Test functions
test_command() {
    local cmd=$1
    local name=$2
    local required=$3

    log_check "Checking $name..."

    if command -v "$cmd" >/dev/null 2>&1; then
        local version=$($cmd --version 2>&1 | head -1 || echo "unknown")
        log_success "$name is installed ($version)"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$name is not installed (required)"
            return 1
        else
            log_warning "$name is not installed (optional)"
            return 0
        fi
    fi
}

test_port() {
    local port=$1
    local service=$2
    local should_be_free=$3

    log_check "Checking port $port for $service..."

    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        if [ "$should_be_free" = "true" ]; then
            log_warning "Port $port is in use (should be free for $service)"
            echo "   💡 To free: lsof -ti :$port | xargs kill -9"
            return 1
        else
            log_success "Port $port is in use ($service running)"
            return 0
        fi
    else
        if [ "$should_be_free" = "true" ]; then
            log_success "Port $port is free ($service can start)"
            return 0
        else
            log_warning "Port $port is free ($service not running)"
            return 1
        fi
    fi
}

test_url() {
    local url=$1
    local service=$2
    local timeout=${3:-5}

    log_check "Testing $service at $url..."

    if curl -s --max-time $timeout "$url" >/dev/null 2>&1; then
        log_success "$service is responding"
        return 0
    else
        log_warning "$service is not responding at $url"
        return 1
    fi
}

test_directory() {
    local dir=$1
    local name=$2
    local should_exist=$3

    log_check "Checking $name directory..."

    if [ -d "$dir" ]; then
        if [ "$should_exist" = "true" ]; then
            log_success "$name directory exists: $dir"
            return 0
        else
            log_warning "$name directory exists but shouldn't: $dir"
            return 1
        fi
    else
        if [ "$should_exist" = "true" ]; then
            log_error "$name directory missing: $dir"
            return 1
        else
            log_success "$name directory doesn't exist (as expected): $dir"
            return 0
        fi
    fi
}

test_file() {
    local file=$1
    local name=$2
    local required=$3

    log_check "Checking $name file..."

    if [ -f "$file" ]; then
        log_success "$name exists: $file"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$name missing: $file"
            return 1
        else
            log_warning "$name missing: $file"
            return 0
        fi
    fi
}

test_ollama() {
    log_check "Testing Ollama installation and connectivity..."

    if ! command -v ollama >/dev/null 2>&1; then
        log_warning "Ollama is not installed"
        echo "   💡 Install: curl -fsSL https://ollama.ai/install.sh | sh"
        return 1
    fi

    log_success "Ollama is installed"

    # Test if Ollama is running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        log_success "Ollama is running"

        # Check for required model
        if curl -s http://localhost:11434/api/tags | grep -q "llama3"; then
            log_success "Llama3 model is available"
            return 0
        else
            log_warning "Llama3 model not found"
            echo "   💡 Install: ollama pull llama3"
            return 1
        fi
    else
        log_warning "Ollama is not running"
        echo "   💡 Start: ollama serve"
        return 1
    fi
}

test_python_env() {
    log_check "Testing Python environment..."

    if [ ! -d "ai-backend/venv" ]; then
        log_warning "Python virtual environment not found"
        echo "   💡 Create: cd ai-backend && python3 -m venv venv"
        return 1
    fi

    log_success "Python virtual environment exists"

    # Test if requirements are installed
    if [ -f "ai-backend/venv/lib/python*/site-packages/flask" ] || [ -f "ai-backend/venv/lib/python*/site-packages/Flask*" ]; then
        log_success "Python dependencies appear to be installed"
        return 0
    else
        log_warning "Python dependencies may not be installed"
        echo "   💡 Install: cd ai-backend && source venv/bin/activate && pip install -r requirements.txt"
        return 1
    fi
}

test_node_env() {
    log_check "Testing Node.js environment..."

    if [ ! -d "frontend/node_modules" ]; then
        log_warning "Frontend dependencies not installed"
        echo "   💡 Install: cd frontend && npm install"
        return 1
    fi

    log_success "Frontend dependencies are installed"

    if [ -d "simple-backend" ] && [ ! -d "simple-backend/node_modules" ]; then
        log_warning "Simple backend dependencies not installed"
        echo "   💡 Install: cd simple-backend && npm install"
        return 1
    fi

    return 0
}

# Main health check function
main() {
    echo ""
    log_info "🔍 Self Service Dashboard - System Health Check"
    echo ""

    # Load environment if exists
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '#' | xargs) 2>/dev/null || true
    fi

    # Set defaults
    AI_BACKEND_PORT=${AI_BACKEND_PORT:-5247}
    SIMPLE_BACKEND_PORT=${SIMPLE_BACKEND_PORT:-5246}

    # Track test results
    FAILED_TESTS=0

    # Core system requirements
    echo "📋 System Requirements:"
    test_command "node" "Node.js" "true" || ((FAILED_TESTS++))
    test_command "npm" "npm" "true" || ((FAILED_TESTS++))
    test_command "python3" "Python 3" "true" || ((FAILED_TESTS++))
    test_command "pip3" "pip3" "true" || ((FAILED_TESTS++))

    echo ""

    # Optional tools
    echo "🛠️  Optional Tools:"
    test_command "git" "Git" "false"
    test_command "curl" "curl" "false"
    test_command "lsof" "lsof" "false"

    echo ""

    # Files and directories
    echo "📁 Project Structure:"
    test_file ".env" "Environment config" "false"
    test_file "package.json" "Root package.json" "true" || ((FAILED_TESTS++))
    test_directory "frontend" "Frontend" "true" || ((FAILED_TESTS++))
    test_directory "ai-backend" "AI Backend" "true" || ((FAILED_TESTS++))
    test_directory "generated-dashboards" "Generated dashboards" "true"
    test_directory "logs" "Logs" "true"

    echo ""

    # Environment setup
    echo "🔧 Environment Setup:"
    test_python_env || ((FAILED_TESTS++))
    test_node_env || ((FAILED_TESTS++))

    echo ""

    # Ollama (optional but recommended)
    echo "🦙 Ollama Setup:"
    test_ollama

    echo ""

    # Port availability checks (setup mode)
    if [ "$SETUP_MODE" = "true" ]; then
        echo "🔌 Port Availability (Setup Mode):"
        test_port "3000" "Frontend" "true"
        test_port "$AI_BACKEND_PORT" "AI Backend" "true"
        test_port "$SIMPLE_BACKEND_PORT" "Simple Backend" "true"
        test_port "8501" "Streamlit dashboards" "true"
    fi

    # Service health checks (dev mode)
    if [ "$DEV_MODE" = "true" ]; then
        echo "🌐 Service Health (Development Mode):"
        test_url "http://localhost:3000" "Frontend" 3
        test_url "http://localhost:$AI_BACKEND_PORT/health" "AI Backend" 3
        test_port "3000" "Frontend" "false"
        test_port "$AI_BACKEND_PORT" "AI Backend" "false"
    fi

    echo ""

    # Summary
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "🎉 All critical health checks passed!"
        echo ""
        if [ "$SETUP_MODE" = "false" ] && [ "$DEV_MODE" = "false" ]; then
            echo "Next steps:"
            echo "  1. Run setup: ./setup.sh"
            echo "  2. Start development: ./scripts/start-development.sh"
            echo "  3. Open: http://localhost:3000"
        fi
        exit 0
    else
        log_error "❌ $FAILED_TESTS critical health check(s) failed"
        echo ""
        echo "Please address the issues above and run health check again."
        exit 1
    fi
}

main "$@"