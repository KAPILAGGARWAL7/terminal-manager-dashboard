#!/bin/bash

# Production Environment Startup Script
# Builds and starts all services in production mode

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[PROD]${NC} $1"
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

# Function to start a service
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="logs/${name}-prod.log"

    log_info "Starting $name on port $port..."

    mkdir -p logs
    touch "$log_file"

    # Start service in background
    eval "$command" > "$log_file" 2>&1 &
    local pid=$!

    echo $pid > "logs/${name}-prod.pid"
    log_success "$name started (PID: $pid)"
}

# Cleanup function
cleanup() {
    log_warning "Stopping production services..."

    for pid_file in logs/*-prod.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local service=$(basename "$pid_file" -prod.pid)

            if kill -0 $pid 2>/dev/null; then
                log_info "Stopping $service (PID: $pid)"
                kill $pid
            fi
            rm -f "$pid_file"
        fi
    done

    log_success "Production services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

main() {
    log_info "🚀 Starting Self Service Dashboard - Production Mode"

    # Load environment
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '#' | xargs)
    fi

    # Set production environment
    export NODE_ENV=production
    export FLASK_ENV=production

    # Build frontend
    log_info "🔨 Building frontend for production..."
    cd frontend
    npm run build
    cd ..
    log_success "Frontend built successfully!"

    # Start AI Backend in production mode
    log_info "🧠 Starting AI Backend (Production)..."
    cd ai-backend
    source venv/bin/activate
    start_service "ai-backend" "python app.py" "$AI_BACKEND_PORT"
    cd ..

    # Start Simple Backend if exists
    if [ -d "simple-backend" ] && [ -f "simple-backend/package.json" ]; then
        log_info "🔧 Starting Simple Backend (Production)..."
        cd simple-backend
        start_service "simple-backend" "npm start" "$SIMPLE_BACKEND_PORT"
        cd ..
    fi

    # Serve frontend build (you might want to use nginx in real production)
    log_info "⚛️ Serving Frontend (Production)..."
    cd frontend
    start_service "frontend" "npx serve -s build -l 3000" "3000"
    cd ..

    # Wait for services
    sleep 5

    log_success "🎉 Production environment is running!"
    echo ""
    echo -e "${PURPLE}📱 Application URLs:${NC}"
    echo -e "  🌐 Frontend: ${BLUE}http://localhost:3000${NC}"
    echo -e "  🧠 AI Backend: ${BLUE}http://localhost:$AI_BACKEND_PORT${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

    # Keep running
    while true; do
        sleep 1
    done
}

main "$@"