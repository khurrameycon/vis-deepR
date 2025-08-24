#!/bin/bash
# docker/healthcheck.sh
# Health check script for the multi-user browser use API

set -e

# Configuration
API_URL="http://localhost:7788"
HEALTH_ENDPOINT="$API_URL/health"
VNC_PORT=5901
BROWSER_DEBUG_PORT=9222

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[HEALTH]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[HEALTH]${NC} $1"
}

error() {
    echo -e "${RED}[HEALTH]${NC} $1"
}

# Check if API server is responding
check_api_health() {
    log "Checking API health endpoint..."
    
    local response
    local http_code
    
    response=$(curl -s -w "\n%{http_code}" "$HEALTH_ENDPOINT" 2>/dev/null || echo "000")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ]; then
        local health_data=$(echo "$response" | head -n -1)
        log "API health check passed"
        echo "$health_data" | jq . 2>/dev/null || echo "$health_data"
        return 0
    else
        error "API health check failed (HTTP $http_code)"
        return 1
    fi
}

# Check if VNC server is running
check_vnc_server() {
    log "Checking VNC server on port $VNC_PORT..."
    
    if nc -z localhost $VNC_PORT 2>/dev/null; then
        log "VNC server is running"
        return 0
    else
        error "VNC server is not accessible"
        return 1
    fi
}

# Check if browser debug port is accessible
check_browser_debug() {
    log "Checking browser debug port $BROWSER_DEBUG_PORT..."
    
    if nc -z localhost $BROWSER_DEBUG_PORT 2>/dev/null; then
        log "Browser debug port is accessible"
        return 0
    else
        warn "Browser debug port is not accessible (this may be normal)"
        return 0  # Don't fail on this as browser might not be started yet
    fi
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check memory usage
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    if [ "${memory_usage%.*}" -gt 90 ]; then
        warn "High memory usage: ${memory_usage}%"
    else
        log "Memory usage: ${memory_usage}%"
    fi
    
    # Check disk space
    local disk_usage
    disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        warn "High disk usage: ${disk_usage}%"
    else
        log "Disk usage: ${disk_usage}%"
    fi
    
    # Check if critical processes are running
    local critical_processes=("supervisord" "python" "Xvfb")
    for process in "${critical_processes[@]}"; do
        if pgrep -f "$process" > /dev/null; then
            log "Process $process is running"
        else
            error "Critical process $process is not running"
            return 1
        fi
    done
    
    return 0
}

# Check log files for errors
check_logs() {
    local log_dir="/app/logs"
    
    if [ ! -d "$log_dir" ]; then
        warn "Log directory $log_dir does not exist"
        return 0
    fi
    
    log "Checking recent log entries for errors..."
    
    # Check for recent errors in API server logs
    local api_log="$log_dir/api_server.error.log"
    if [ -f "$api_log" ]; then
        local recent_errors
        recent_errors=$(tail -n 50 "$api_log" | grep -i "error\|exception\|traceback" | wc -l)
        if [ "$recent_errors" -gt 0 ]; then
            warn "Found $recent_errors recent errors in API server logs"
        else
            log "No recent errors in API server logs"
        fi
    fi
    
    return 0
}

# Test a simple API operation
test_api_functionality() {
    log "Testing basic API functionality..."
    
    # Test session creation
    local session_response
    session_response=$(curl -s -X POST "$API_URL/api/session/create" 2>/dev/null)
    
    if echo "$session_response" | jq -e '.session_id' > /dev/null 2>&1; then
        local session_id
        session_id=$(echo "$session_response" | jq -r '.session_id')
        log "Successfully created test session: $session_id"
        
        # Clean up test session
        curl -s -X DELETE "$API_URL/api/session/$session_id" > /dev/null 2>&1
        log "Cleaned up test session"
        return 0
    else
        error "Failed to create test session"
        return 1
    fi
}

# Main health check function
main() {
    log "Starting comprehensive health check..."
    local exit_code=0
    
    # Run all checks
    check_api_health || exit_code=1
    check_vnc_server || exit_code=1
    check_browser_debug
    check_system_resources || exit_code=1
    check_logs
    test_api_functionality || exit_code=1
    
    if [ $exit_code -eq 0 ]; then
        log "All health checks passed ✓"
    else
        error "Some health checks failed ✗"
    fi
    
    return $exit_code
}

# Run health check
main "$@"