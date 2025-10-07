#!/bin/bash

# Development Environment Startup Script
# Starts all services in development mode with hot reload

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[DEV]${NC} $1"
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

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start a service and track its PID
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="logs/${name}.log"

    log_info "Starting $name on port $port..."

    # Create log file
    mkdir -p logs
    touch "$log_file"

    # Start service in background
    eval "$command" > "$log_file" 2>&1 &
    local pid=$!

    # Store PID for cleanup
    echo $pid > "logs/${name}.pid"

    log_success "$name started (PID: $pid)"
    log_info "Logs: tail -f $log_file"
}

# Function to cleanup on exit
cleanup() {
    log_warning "Shutting down development environment..."

    # Kill all tracked services
    for pid_file in logs/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local service=$(basename "$pid_file" .pid)

            if kill -0 $pid 2>/dev/null; then
                log_info "Stopping $service (PID: $pid)"
                kill $pid
            fi
            rm -f "$pid_file"
        fi
    done

    # Additional cleanup for any remaining processes
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "python.*app.py" 2>/dev/null || true
    pkill -f "react-scripts start" 2>/dev/null || true
    pkill -f "node.*start" 2>/dev/null || true

    log_success "Development environment stopped"
    exit 0
}

# Set up cleanup trap
trap cleanup SIGINT SIGTERM EXIT

main() {
    log_info "🚀 Starting Self Service Dashboard - Development Environment"

    # Check if setup has been run
    if [ ! -f ".env" ]; then
        log_warning "Environment not configured. Running setup..."
        ./setup.sh
    fi

    # Load environment variables
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '#' | xargs)
    fi

    # Check ports
    PORTS_TO_CHECK=(3000 5246 5247 8501)
    for port in "${PORTS_TO_CHECK[@]}"; do
        if check_port $port; then
            log_warning "Port $port is already in use"
            log_info "Run: lsof -ti :$port | xargs kill -9 (to free the port)"
        fi
    done

    # Health checks
    log_info "🔍 Running health checks..."
    ./scripts/health-check.sh --dev-mode || {
        log_error "Health checks failed. Please review and try again."
        exit 1
    }

    log_success "Health checks passed!"

    # Start AI Backend
    log_info "🧠 Starting AI Backend..."
    cd ai-backend
    if [ ! -d "venv" ]; then
        log_warning "Virtual environment not found. Creating..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi

    start_service "ai-backend" "python app.py" "$AI_BACKEND_PORT"
    cd ..

    # Wait for AI backend to start
    sleep 3

    # Start Simple Backend (if it exists)
    if [ -d "simple-backend" ] && [ -f "simple-backend/package.json" ]; then
        log_info "🔧 Starting Simple Backend..."
        cd simple-backend
        start_service "simple-backend" "npm start" "$SIMPLE_BACKEND_PORT"
        cd ..
        sleep 2
    fi

    # Start Frontend
    log_info "⚛️ Starting Frontend..."
    cd frontend
    start_service "frontend" "npm start" "3000"
    cd ..

    # Wait for services to start
    log_info "⏳ Waiting for services to initialize..."
    sleep 5

    # Service status check
    log_info "📊 Service Status:"
    echo ""

    # Check each service
    services=(
        "Frontend:http://localhost:3000"
        "AI Backend:http://localhost:$AI_BACKEND_PORT/health"
    )

    if [ -d "simple-backend" ]; then
        services+=("Simple Backend:http://localhost:$SIMPLE_BACKEND_PORT")
    fi

    for service_info in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service_info"
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "  ✅ $name: ${GREEN}Running${NC} ($url)"
        else
            echo -e "  ❌ $name: ${RED}Not responding${NC} ($url)"
        fi
    done

    echo ""
    log_success "🎉 Development environment is ready!"
    echo ""
    echo -e "${PURPLE}📱 Application URLs:${NC}"
    echo -e "  🌐 Frontend: ${BLUE}http://localhost:3000${NC}"
    echo -e "  🧠 AI Backend: ${BLUE}http://localhost:$AI_BACKEND_PORT${NC}"
    if [ -d "simple-backend" ]; then
        echo -e "  🔧 Simple Backend: ${BLUE}http://localhost:$SIMPLE_BACKEND_PORT${NC}"
    fi
    echo ""
    echo -e "${PURPLE}📝 Logs:${NC}"
    echo -e "  tail -f logs/frontend.log"
    echo -e "  tail -f logs/ai-backend.log"
    if [ -d "simple-backend" ]; then
        echo -e "  tail -f logs/simple-backend.log"
    fi
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

    # Keep script running
    while true; do
        sleep 1
    done
}

# Run main function
main "$@"