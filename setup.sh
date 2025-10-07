#!/bin/bash

# Self Service Dashboard AI - Complete Setup Script
# This script sets up the entire development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    log_info "🚀 Starting Self Service Dashboard AI Setup..."

    # Step 1: Check prerequisites
    log_info "📋 Checking prerequisites..."

    if ! command_exists node; then
        log_error "Node.js is not installed. Please install Node.js 16+ and try again."
        log_info "Download from: https://nodejs.org/"
        exit 1
    fi

    if ! command_exists npm; then
        log_error "npm is not installed. Please install npm and try again."
        exit 1
    fi

    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.8+ and try again."
        log_info "Download from: https://python.org/"
        exit 1
    fi

    if ! command_exists pip3; then
        log_error "pip3 is not installed. Please install pip3 and try again."
        exit 1
    fi

    log_success "All prerequisites found!"

    # Step 2: Copy environment file
    log_info "📄 Setting up environment configuration..."
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_success "Created .env file from template"
    else
        log_warning ".env file already exists, skipping..."
    fi

    # Step 3: Install root dependencies
    log_info "📦 Installing root dependencies..."
    npm install concurrently --save-dev
    log_success "Root dependencies installed!"

    # Step 4: Setup AI Backend
    log_info "🧠 Setting up AI Backend..."
    cd ai-backend

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created!"
    fi

    # Activate virtual environment and install dependencies
    log_info "Installing Python dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    log_success "AI Backend setup complete!"

    cd ..

    # Step 5: Setup Frontend
    log_info "⚛️ Setting up Frontend..."
    cd frontend
    npm install
    log_success "Frontend dependencies installed!"
    cd ..

    # Step 6: Setup Simple Backend (if exists)
    if [ -d "simple-backend" ]; then
        log_info "🔧 Setting up Simple Backend..."
        cd simple-backend
        if [ -f "package.json" ]; then
            npm install
            log_success "Simple Backend dependencies installed!"
        fi
        cd ..
    fi

    # Step 7: Create necessary directories
    log_info "📁 Creating runtime directories..."
    mkdir -p generated-dashboards
    mkdir -p logs
    log_success "Runtime directories created!"

    # Step 8: Make scripts executable
    log_info "🔐 Making scripts executable..."
    chmod +x scripts/*.sh 2>/dev/null || true
    chmod +x ai-backend/install.sh 2>/dev/null || true
    log_success "Scripts made executable!"

    # Step 9: Check Ollama
    log_info "🦙 Checking Ollama installation..."
    if command_exists ollama; then
        log_success "Ollama is installed!"
        if pgrep -x "ollama" > /dev/null; then
            log_success "Ollama is running!"
        else
            log_warning "Ollama is installed but not running."
            log_info "Start Ollama with: ollama serve"
        fi
    else
        log_warning "Ollama is not installed."
        log_info "To install Ollama:"
        log_info "  macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh"
        log_info "  Windows: Download from https://ollama.ai/"
        log_info "After installation, run: ollama pull llama2"
    fi

    # Step 10: Final checks
    log_info "🔍 Running health checks..."
    ./scripts/health-check.sh --setup-mode || true

    # Success message
    echo ""
    log_success "🎉 Setup Complete!"
    echo ""
    log_info "Next steps:"
    log_info "1. Install Ollama if not already installed (see instructions above)"
    log_info "2. Start Ollama: ollama serve"
    log_info "3. Pull a model: ollama pull llama2"
    log_info "4. Start the development environment: ./scripts/start-development.sh"
    log_info "5. Open http://localhost:3000 in your browser"
    echo ""
    log_info "For help, see docs/SETUP.md or run: ./scripts/health-check.sh"
}

# Run main function
main "$@"