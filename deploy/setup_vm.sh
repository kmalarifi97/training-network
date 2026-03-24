#!/bin/bash
# GPU Network Server - GCP VM Setup Script
# Usage: ssh into your VM, then run: bash setup_vm.sh
#
# Prerequisites: GCP VM with Ubuntu 22.04+, firewall rule allowing TCP port 8000

set -e

echo "=== GPU Network Server Setup ==="

# Install Python if needed
if ! command -v python3 &>/dev/null; then
    echo "[1/5] Installing Python..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
else
    echo "[1/5] Python already installed: $(python3 --version)"
fi

# Create project directory
echo "[2/5] Setting up project..."
PROJECT_DIR=~/gpunetwork
mkdir -p "$PROJECT_DIR"

# Copy server files (assumes you've scp'd them or cloned the repo)
if [ ! -f "$PROJECT_DIR/server/main.py" ]; then
    echo ""
    echo "ERROR: Server files not found at $PROJECT_DIR/server/"
    echo ""
    echo "Copy files from your local machine first:"
    echo "  scp -r server/ user@VM_IP:~/gpunetwork/"
    echo ""
    echo "Or clone your repo:"
    echo "  cd ~/gpunetwork && git clone YOUR_REPO_URL ."
    echo ""
    exit 1
fi

# Create venv and install deps
echo "[3/5] Installing dependencies..."
cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt

# Create systemd service for auto-restart
echo "[4/5] Creating systemd service..."
sudo tee /etc/systemd/system/gpunetwork.service > /dev/null <<EOF
[Unit]
Description=GPU Network Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/server
ExecStart=$PROJECT_DIR/venv/bin/python main.py
Restart=always
RestartSec=5
Environment=PORT=8000

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpunetwork
sudo systemctl start gpunetwork

echo "[5/5] Server started!"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Server running on port 8000"
echo "Check status:  sudo systemctl status gpunetwork"
echo "View logs:     sudo journalctl -u gpunetwork -f"
echo "Restart:       sudo systemctl restart gpunetwork"
echo ""
echo "IMPORTANT: Make sure GCP firewall allows TCP port 8000:"
echo "  gcloud compute firewall-rules create allow-gpunet \\"
echo "    --allow tcp:8000 --target-tags=gpunet-server"
echo ""
echo "Then open: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_VM_IP'):8000"
