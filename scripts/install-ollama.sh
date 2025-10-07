#!/bin/bash

# Ollama Installation Helper Script
# Provides instructions and automated installation for Ollama

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[OLLAMA]${NC} $1"
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

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check if Ollama is already installed
check_ollama_installed() {
    if command -v ollama >/dev/null 2>&1; then
        return 0  # Installed
    else
        return 1  # Not installed
    fi
}

# Check if Ollama is running
check_ollama_running() {
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Install Ollama
install_ollama() {
    local os=$(detect_os)

    log_info "🦙 Installing Ollama for $os..."

    case $os in
        "linux"|"macos")
            log_info "Downloading and running Ollama installer..."
            curl -fsSL https://ollama.ai/install.sh | sh
            if [ $? -eq 0 ]; then
                log_success "Ollama installed successfully!"
                return 0
            else
                log_error "Ollama installation failed"
                return 1
            fi
            ;;
        "windows")
            log_warning "Windows detected. Please install manually:"
            echo ""
            echo "1. Download Ollama for Windows from: https://ollama.ai/"
            echo "2. Run the installer"
            echo "3. Restart your terminal"
            echo "4. Run this script again"
            return 1
            ;;
        *)
            log_error "Unsupported operating system: $os"
            echo ""
            echo "Please install Ollama manually from: https://ollama.ai/"
            return 1
            ;;
    esac
}

# Start Ollama service
start_ollama() {
    log_info "Starting Ollama service..."

    # Try to start Ollama in background
    ollama serve &
    local ollama_pid=$!

    # Wait a moment for startup
    
    sleep 3

    # Check if it's running
    if check_ollama_running; then
        log_success "Ollama is now running!"
        echo "PID: $ollama_pid"
        return 0
    else
        log_error "Failed to start Ollama"
        return 1
    fi
}

# Pull required model
pull_model() {
    local model=${1:-llama3}

    log_info "Pulling $model model..."
    log_warning "This may take a while (several GB download)..."

    if ollama pull $model; then
        log_success "$model model downloaded successfully!"
        return 0
    else
        log_error "Failed to download $model model"
        return 1
    fi
}

# Test Ollama functionality
test_ollama() {
    log_info "Testing Ollama with a simple query..."

    local test_response=$(ollama run llama3 "Say 'Hello from Ollama!' and nothing else." 2>/dev/null | head -1)

    if [[ "$test_response" == *"Hello from Ollama"* ]]; then
        log_success "Ollama test successful!"
        echo "Response: $test_response"
        return 0
    else
        log_warning "Ollama test gave unexpected response: $test_response"
        return 1
    fi
}

# Main function
main() {
    echo ""
    log_info "🦙 Ollama Installation Helper"
    echo ""

    # Check current status
    if check_ollama_installed; then
        log_success "Ollama is already installed"

        if check_ollama_running; then
            log_success "Ollama is running"

            # Check if model exists
            if ollama list | grep -q "llama3"; then
                log_success "Llama3 model is available"
                echo ""
                log_info "🎉 Ollama is fully set up and ready!"
                echo ""
                echo "You can now:"
                echo "  • Start the AI dashboard: ./scripts/start-development.sh"
                echo "  • Test the connection: ./ai-backend/test_connection.py"
                echo ""
                exit 0
            else
                log_warning "Llama3 model not found"
                read -p "Would you like to download the Llama3 model? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    pull_model "llama3"
                fi
            fi
        else
            log_warning "Ollama is installed but not running"
            read -p "Would you like to start Ollama? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                start_ollama
                if [ $? -eq 0 ]; then
                    pull_model "llama3"
                fi
            else
                echo ""
                log_info "To start Ollama manually, run: ollama serve"
                exit 0
            fi
        fi
    else
        log_info "Ollama is not installed"
        echo ""
        read -p "Would you like to install Ollama now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if install_ollama; then
                log_info "Starting Ollama..."
                start_ollama
                if [ $? -eq 0 ]; then
                    pull_model "llama3"
                    test_ollama
                fi
            fi
        else
            echo ""
            log_info "Manual installation instructions:"
            echo ""
            local os=$(detect_os)
            case $os in
                "linux"|"macos")
                    echo "Run: curl -fsSL https://ollama.ai/install.sh | sh"
                    ;;
                "windows")
                    echo "Download from: https://ollama.ai/"
                    ;;
                *)
                    echo "Visit: https://ollama.ai/ for installation instructions"
                    ;;
            esac
            echo ""
            echo "After installation:"
            echo "1. ollama serve"
            echo "2. ollama pull llama3"
            echo "3. ./scripts/start-development.sh"
            exit 0
        fi
    fi

    echo ""
    log_success "🎉 Ollama setup complete!"
    echo ""
    echo "Next steps:"
    echo "  • Test connection: ./ai-backend/test_connection.py"
    echo "  • Start dashboard: ./scripts/start-development.sh"
    echo ""
}

main "$@"