#!/bin/bash

# AI Backend Installation Script
# Auto Python environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[AI-BACKEND]${NC} $1"
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

main() {
    log_info "🧠 Setting up AI Backend..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        exit 1
    fi

    # Create virtual environment
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created!"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    # Install requirements
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt

    # Install package in development mode
    log_info "Installing AI backend package..."
    pip install -e .

    log_success "AI Backend setup complete!"

    # Check Ollama
    log_info "Checking Ollama connection..."
    python3 -c "
from config import Config
status = Config.validate_ollama_connection()
if status['connected']:
    print('✅ Ollama connected successfully')
    if status.get('has_required_model'):
        print('✅ Required model available')
    else:
        print('⚠️  Required model not found. Run: ollama pull llama3')
else:
    print('❌ Ollama not connected:', status.get('error', 'Unknown error'))
    print('💡 Start Ollama with: ollama serve')
"

    deactivate
    log_success "🎉 AI Backend is ready!"
}

main "$@"