#!/bin/bash
# scripts/entrypoint.sh
# 
# Comprehensive entrypoint script for the trading bot container.
# This script handles all initialization steps in the correct order,
# with proper error handling and logging.

set -e  # Exit immediately if any command fails
set -u  # Exit if undefined variables are used

# ANSI color codes for better logging visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions that make output easy to scan
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_section() {
    echo -e "\n${BLUE}🔷 $1${NC}"
    echo "=================================================="
}

# Function to check if Alembic is already initialized
check_alembic_setup() {
    if [ -d "/workspace/alembic" ] && [ -f "/workspace/alembic.ini" ]; then
        log_success "Alembic migration system already initialized"
        return 0
    else
        log_warning "Alembic not initialized - will set up migration system"
        return 1
    fi
}

# Function to initialize Alembic if needed
setup_alembic() {
    log_section "Setting up Alembic Migration System"
    
    if check_alembic_setup; then
        return 0
    fi
    
    log_info "Creating Alembic migration repository..."
    
    # Run our Python initialization script
    uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, '/workspace')
from src.database.migrations import init_alembic
init_alembic()
    "
    
    if [ $? -eq 0 ]; then
        log_success "Alembic migration system initialized successfully"
    else
        log_error "Failed to initialize Alembic"
        exit 1
    fi
}

# Function to run database initialization
initialize_database() {
    log_section "Initializing Database Schema"
    
    log_info "Running database initialization script..."
    uv run python /workspace/scripts/init_database.py
    
    if [ $? -eq 0 ]; then
        log_success "Database initialization completed successfully"
    else
        log_error "Database initialization failed"
        exit 1
    fi
}

# Function to perform health checks
run_health_checks() {
    log_section "Running System Health Checks"
    
    log_info "Verifying Python environment..."
    uv run python -c "
import sys
print(f'Python version: {sys.version}')

# Check critical imports
try:
    import sqlalchemy
    print(f'SQLAlchemy: {sqlalchemy.__version__}')
    import alpaca
    print('Alpaca SDK: Available')
    import asyncpg
    print('AsyncPG: Available')
    print('✅ All critical dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
    "
    
    log_success "Environment health check passed"
}

# Main initialization function
main() {
    log_section "Trading Bot Container Initialization"
    log_info "Starting comprehensive setup process..."
    
    # Step 1: Environment validation
    run_health_checks
    
    # Step 2: Set up Alembic if needed
    setup_alembic
    
    # Step 3: Initialize database
    # Run as background process to not block VS Code's devcontainer initialization
    {
        initialize_database
        log_section "Initialization Complete"
        log_success "Trading bot infrastructure is ready!"
    } &
    log_info "Container will now stay alive for development work..."
    log_info "You can attach to this container and start developing ETL pipelines"
    
    # Keep container running for development
    log_info "Entering development mode (sleeping infinity)..."
    sleep infinity
}

# Error handling for the entire script
trap 'log_error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"