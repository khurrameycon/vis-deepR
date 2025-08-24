#!/bin/bash
# docker/entrypoint.sh
# Production entrypoint script for multi-user browser use API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[ENTRYPOINT]${NC} $1"
}

info() {
    echo -e "${BLUE}[ENTRYPOINT]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[ENTRYPOINT]${NC} $1"
}

error() {
    echo -e "${RED}[ENTRYPOINT]${NC} $1"
}

# Print startup banner
print_banner() {
    cat << 'EOF'
    ____                                          __  __          
   / __ )_________ _      __________  _____      / / / /_______   
  / __  / ___/ __ \ | /| / / ___/ _ \/ ___/_____/ / / / ___/ _ \  
 / /_/ / /  / /_/ / |/ |/ (__  )  __/ /  /_____/ /_/ (__  )  __/  
/_____/_/   \____/|__/|__/____/\___/_/        \____/____/\___/   
                                                                  
           Multi-User Browser Automation API
                 Production Environment
EOF
}

# Validate environment variables
validate_environment() {
    log "Validating environment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "MAX_CONCURRENT_SESSIONS"
        "SESSION_TIMEOUT_MINUTES"
        "HOST"
        "PORT"
        "RESOLUTION_WIDTH"
        "RESOLUTION_HEIGHT"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            error "Required environment variable $var is not set"
            exit 1
        else
            info "$var: ${!var}"
        fi
    done
    
    # Validate numeric values
    if ! [[ "$MAX_CONCURRENT_SESSIONS" =~ ^[0-9]+$ ]] || [ "$MAX_CONCURRENT_SESSIONS" -lt 1 ]; then
        error "MAX_CONCURRENT_SESSIONS must be a positive integer"
        exit 1
    fi
    
    if ! [[ "$SESSION_TIMEOUT_MINUTES" =~ ^[0-9]+$ ]] || [ "$SESSION_TIMEOUT_MINUTES" -lt 1 ]; then
        error "SESSION_TIMEOUT_MINUTES must be a positive integer"
        exit 1
    fi
    
    # Check resource limits
    if [ "$MAX_CONCURRENT_SESSIONS" -gt 50 ]; then
        warn "MAX_CONCURRENT_SESSIONS is set to $MAX_CONCURRENT_SESSIONS (>50). Ensure adequate resources."
    fi
    
    log "Environment validation passed"
}

# Setup directories and permissions
setup_directories() {
    log "Setting up directories and permissions..."
    
    local directories=(
        "/app/data/chrome_data"
        "/app/tmp/downloads"
        "/app/tmp/traces"
        "/app/tmp/recordings"
        "/app/tmp/sessions"
        "/app/logs"
        "/home/browseruse/.vnc"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
        
        # Ensure proper ownership
        if [ "$(stat -c %U "$dir")" != "browseruse" ]; then
            chown -R browseruse:browseruse "$dir" 2>/dev/null || warn "Could not change ownership of $dir"
        fi
    done
    
    log "Directory setup completed"
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check available memory
    local memory_kb
    memory_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local memory_mb=$((memory_kb / 1024))
    
    info "Available memory: ${memory_mb} MB"
    
    if [ "$memory_mb" -lt 2048 ]; then
        warn "Low available memory (${memory_mb} MB). Consider increasing container memory limits."
    fi
    
    # Check disk space
    local disk_available
    disk_available=$(df /app | tail -1 | awk '{print $4}')
    local disk_mb=$((disk_available / 1024))
    
    info "Available disk space: ${disk_mb} MB"
    
    if [ "$disk_mb" -lt 1024 ]; then
        warn "Low available disk space (${disk_mb} MB). Consider increasing container storage."
    fi
    
    # Check CPU cores
    local cpu_cores
    cpu_cores=$(nproc)
    info "CPU cores: $cpu_cores"
    
    if [ "$cpu_cores" -lt 2 ]; then
        warn "Low CPU core count ($cpu_cores). Performance may be limited."
    fi
    
    log "System resource check completed"
}

# Cleanup function for graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Stop supervisord gracefully
    if [ -f /tmp/supervisor.sock ]; then
        supervisorctl -s unix:///tmp/supervisor.sock shutdown 2>/dev/null || true
    fi
    
    # Kill any remaining chrome processes
    pkill -f chrome 2>/dev/null || true
    
    # Clean up temporary files
    rm -f /tmp/supervisor.sock 2>/dev/null || true
    
    log "Cleanup completed"
    exit 0
}

# Setup signal handlers
setup_signal_handlers() {
    trap cleanup SIGTERM SIGINT SIGQUIT
}

# Wait for dependencies
wait_for_dependencies() {
    log "Waiting for system dependencies..."
    
    # Wait for X server to be ready
    local max_wait=30
    local count=0
    
    while [ $count -lt $max_wait ]; do
        if xdpyinfo -display :99 >/dev/null 2>&1; then
            log "X server is ready"
            break
        fi
        
        if [ $count -eq 0 ]; then
            info "Waiting for X server..."
        fi
        
        sleep 1
        count=$((count + 1))
    done
    
    if [ $count -eq $max_wait ]; then
        warn "X server did not start within ${max_wait} seconds"
    fi
}

# Pre-flight checks
run_preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check if required binaries exist
    local required_binaries=(
        "python"
        "supervisord"
        "Xvfb"
        "x11vnc"
        "curl"
        # "jq"
    )
    
    for binary in "${required_binaries[@]}"; do
        if ! command -v "$binary" >/dev/null 2>&1; then
            error "Required binary '$binary' not found"
            exit 1
        fi
    done
    
    # Check if Playwright browser is installed
    if [ ! -d "/ms-playwright/chromium-"* ]; then
        error "Playwright Chromium browser not found"
        exit 1
    fi
    
    # Check Python dependencies
    if ! python -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
        error "Required Python dependencies not installed"
        exit 1
    fi
    
    log "Pre-flight checks passed"
}

# Main execution
main() {
    print_banner
    
    log "Starting Browser Use Multi-User API..."
    info "Version: ${VERSION:-unknown}"
    info "Build Date: ${BUILD_DATE:-unknown}"
    info "Platform: ${TARGETPLATFORM:-unknown}"
    
    # Run initialization steps
    validate_environment
    setup_directories
    check_system_resources
    run_preflight_checks
    setup_signal_handlers
    
    # Export environment variables for supervisor
    export PYTHONPATH="/app:$PYTHONPATH"
    
    log "Initialization completed successfully"
    
    # Start supervisor or run command
    if [ "$1" = "supervisord" ] || [ $# -eq 0 ]; then
        log "Starting supervisord..."
        exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf
    else
        log "Running custom command: $*"
        exec "$@"
    fi
}

# Run main function
main "$@"