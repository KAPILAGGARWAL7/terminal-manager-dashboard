#!/bin/bash

# Project Reset Script
# Clean restart of the entire development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[RESET]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to confirm action
confirm() {
    local message=$1
    echo -e "${YELLOW}⚠️  $message${NC}"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Operation cancelled"
        exit 0
    fi
}

# Kill running processes
kill_processes() {
    log_info "Stopping all running services..."

    # Stop all development processes
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "python.*app.py" 2>/dev/null || true
    pkill -f "react-scripts start" 2>/dev/null || true
    pkill -f "node.*start" 2>/dev/null || true

    # Kill processes by PID files
    if [ -d "logs" ]; then
        for pid_file in logs/*.pid; do
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file" 2>/dev/null || echo "")
                if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                    log_info "Killing process $pid"
                    kill $pid 2>/dev/null || true
                fi
                rm -f "$pid_file"
            fi
        done
    fi

    log_success "All processes stopped"
}

# Clean logs
clean_logs() {
    log_info "Cleaning logs..."

    if [ -d "logs" ]; then
        rm -rf logs/*
        log_success "Logs cleaned"
    else
        log_info "No logs directory found"
    fi
}

# Clean generated files
clean_generated() {
    log_info "Cleaning generated files..."

    # Generated dashboards
    if [ -d "generated-dashboards" ]; then
        rm -rf generated-dashboards/*
        log_success "Generated dashboards cleaned"
    fi

    # Temporary files
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true

    log_success "Generated files cleaned"
}

# Reset Python environment
reset_python_env() {
    log_info "Resetting Python environment..."

    cd ai-backend

    if [ -d "venv" ]; then
        rm -rf venv
        log_success "Python virtual environment removed"
    fi

    log_info "Creating new Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "Python environment recreated"

    cd ..
}

# Reset Node.js environment
reset_node_env() {
    log_info "Resetting Node.js environment..."

    # Frontend
    if [ -d "frontend/node_modules" ]; then
        cd frontend
        rm -rf node_modules package-lock.json
        npm install
        log_success "Frontend dependencies reinstalled"
        cd ..
    fi

    # Simple backend
    if [ -d "simple-backend" ] && [ -f "simple-backend/package.json" ]; then
        cd simple-backend
        if [ -d "node_modules" ]; then
            rm -rf node_modules package-lock.json
        fi
        npm install
        log_success "Simple backend dependencies reinstalled"
        cd ..
    fi

    # Root
    if [ -f "package.json" ]; then
        if [ -d "node_modules" ]; then
            rm -rf node_modules package-lock.json
        fi
        npm install
        log_success "Root dependencies reinstalled"
    fi
}

# Reset configuration
reset_config() {
    log_info "Resetting configuration..."

    if [ -f ".env" ]; then
        rm -f .env
        log_info "Environment file removed"
    fi

    if [ -f ".env.example" ]; then
        cp .env.example .env
        log_success "Environment file recreated from template"
    fi
}

# Create fresh directories
create_directories() {
    log_info "Creating fresh directories..."

    mkdir -p generated-dashboards
    mkdir -p logs
    mkdir -p data

    log_success "Directories created"
}

# Main function
main() {
    echo ""
    log_info "🔄 Self Service Dashboard - Project Reset"
    echo ""

    # Parse arguments
    FULL_RESET=false
    QUICK_RESET=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                FULL_RESET=true
                shift
                ;;
            --quick)
                QUICK_RESET=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --full    Full reset (dependencies, virtual environments)"
                echo "  --quick   Quick reset (processes, logs, generated files only)"
                echo "  --help    Show this help message"
                echo ""
                echo "Default: Interactive mode (asks what to reset)"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    if [ "$QUICK_RESET" = "true" ]; then
        log_info "Performing quick reset..."
        kill_processes
        clean_logs
        clean_generated
        create_directories
        log_success "🎉 Quick reset complete!"
        exit 0
    fi

    if [ "$FULL_RESET" = "true" ]; then
        confirm "This will completely reset the project environment!"
        kill_processes
        clean_logs
        clean_generated
        reset_config
        reset_python_env
        reset_node_env
        create_directories
        log_success "🎉 Full reset complete!"
        echo ""
        echo "Next steps:"
        echo "  1. ./setup.sh"
        echo "  2. ./scripts/start-development.sh"
        exit 0
    fi

    # Interactive mode
    echo "What would you like to reset?"
    echo ""
    echo "1. Quick reset (processes, logs, generated files)"
    echo "2. Full reset (everything including dependencies)"
    echo "3. Custom reset (choose components)"
    echo "4. Cancel"
    echo ""
    read -p "Choose option (1-4): " -n 1 -r
    echo

    case $REPLY in
        1)
            log_info "Performing quick reset..."
            kill_processes
            clean_logs
            clean_generated
            create_directories
            ;;
        2)
            confirm "This will completely reset the project environment!"
            kill_processes
            clean_logs
            clean_generated
            reset_config
            reset_python_env
            reset_node_env
            create_directories
            ;;
        3)
            echo ""
            echo "Custom reset options:"
            echo ""

            read -p "Stop running processes? (y/N): " -n 1 -r; echo
            [[ $REPLY =~ ^[Yy]$ ]] && kill_processes

            read -p "Clean logs? (y/N): " -n 1 -r; echo
            [[ $REPLY =~ ^[Yy]$ ]] && clean_logs

            read -p "Clean generated files? (y/N): " -n 1 -r; echo
            [[ $REPLY =~ ^[Yy]$ ]] && clean_generated

            read -p "Reset configuration? (y/N): " -n 1 -r; echo
            [[ $REPLY =~ ^[Yy]$ ]] && reset_config

            read -p "Reset Python environment? (y/N): " -n 1 -r; echo
            [[ $REPLY =~ ^[Yy]$ ]] && reset_python_env

            read -p "Reset Node.js environment? (y/N): " -n 1 -r; echo
            [[ $REPLY =~ ^[Yy]$ ]] && reset_node_env

            create_directories
            ;;
        4)
            log_info "Reset cancelled"
            exit 0
            ;;
        *)
            log_error "Invalid option"
            exit 1
            ;;
    esac

    echo ""
    log_success "🎉 Reset complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run health check: ./scripts/health-check.sh"
    echo "  2. Setup if needed: ./setup.sh"
    echo "  3. Start development: ./scripts/start-development.sh"
}

main "$@"