#!/bin/bash

# cleanup_services.sh
# Removes SysMon and SysMon-Dash services setup by setup_services.sh
# Requirements: sudo privileges

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting SysMon Service Cleanup...${NC}"

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./cleanup_services.sh)"
  exit 1
fi

# Stop and disable services
echo -e "${GREEN}Stopping and disabling services...${NC}"
# Use || true to avoid exit on error if service is already stopped/removed
systemctl stop sysmon.service sysmon-dash.service || true
systemctl disable sysmon.service sysmon-dash.service || true

# Remove service files
echo -e "${GREEN}Removing service files...${NC}"
rm -f /etc/systemd/system/sysmon.service
rm -f /etc/systemd/system/sysmon-dash.service

# Reload systemd to recognize changes
echo -e "${GREEN}Reloading systemd daemon...${NC}"
systemctl daemon-reload

echo -e "${GREEN}Services have been removed.${NC}"

# Cleanup generated artifacts
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
DB_PATH="$SCRIPT_DIR/metrics.db"

echo -e "${GREEN}Cleaning up local artifacts...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo "Removing virtual environment ($VENV_DIR)..."
    rm -rf "$VENV_DIR"
fi

if [ -f "$DB_PATH" ]; then
    echo "Removing metrics database ($DB_PATH)..."
    rm -f "$DB_PATH"
fi

# Also clean pycache
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} +

echo -e "${GREEN}Cleanup Complete! logic environment restored.${NC}"
