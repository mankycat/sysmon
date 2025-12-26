#!/bin/bash

# setup_services.sh
# Automates the setup of SysMon and SysMon-Dash services on Linux.
# Requirements: sudo privileges

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting SysMon Service Setup...${NC}"

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./setup_services.sh)"
  exit 1
fi

# Get current directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
DB_PATH="$SCRIPT_DIR/metrics.db"

echo -e "${GREEN}Working directory: $SCRIPT_DIR${NC}"

# Check for Debian/Ubuntu and install python3-venv if needed
if [ -f /etc/debian_version ]; then
    echo -e "${GREEN}Debian/Ubuntu detected. Installing python3-venv...${NC}"
    # Update package list silently to ensure we can find the package
    apt-get update -qq || true
    apt-get install -y python3-venv
fi

# 1. Setup Virtual Environment
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/pip" ]; then
    echo -e "${GREEN}Virtual environment exists but is missing pip. Recreating...${NC}"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
else
    echo -e "${GREEN}Virtual environment exists.${NC}"
fi

# 2. Install Dependencies
echo -e "${GREEN}Installing/Updating dependencies...${NC}"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install psutil matplotlib nvidia-ml-py dash plotly pandas

# 3. Create Service Files

# --- sysmon.service ---
SYSMON_SERVICE_FILE="/etc/systemd/system/sysmon.service"
echo -e "${GREEN}Generating $SYSMON_SERVICE_FILE...${NC}"

cat > "$SYSMON_SERVICE_FILE" <<EOF
[Unit]
Description=SysMon Collector (CPU/GPU/Disk -> SQLite)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment=SYSMON_DEBUG=0
ExecStart=$VENV_DIR/bin/python $SCRIPT_DIR/sysmon.py collect --db $DB_PATH --interval 2 --gpu-via smi
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# --- sysmon-dash.service ---
DASH_SERVICE_FILE="/etc/systemd/system/sysmon-dash.service"
echo -e "${GREEN}Generating $DASH_SERVICE_FILE...${NC}"

cat > "$DASH_SERVICE_FILE" <<EOF
[Unit]
Description=SysMon Plotly Dashboard
After=network.target sysmon.service

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_DIR/bin/python $SCRIPT_DIR/sysmon_dash.py --db $DB_PATH --host 0.0.0.0 --port 58050
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 4. Reload and Start Services
echo -e "${GREEN}Reloading systemd daemon...${NC}"
systemctl daemon-reload

echo -e "${GREEN}Enabling and starting sysmon.service...${NC}"
systemctl enable sysmon.service
systemctl restart sysmon.service

echo -e "${GREEN}Enabling and starting sysmon-dash.service...${NC}"
systemctl enable sysmon-dash.service
systemctl restart sysmon-dash.service

echo -e "${GREEN}Setup Complete! Services are running.${NC}"
echo "Check status with: systemctl status sysmon sysmon-dash"
