# Task: Deploy GPU Network server to GCP Compute Engine

## What This Is
A FastAPI server that orchestrates GPU inference jobs. It needs to run on a GCP VM so remote GPU agents can connect to it via WebSocket.

## Files to Deploy
Only the `server/` folder is needed on the VM:
- server/main.py — FastAPI app (WebSocket hub + REST API + static web UI)
- server/database.py — SQLite persistence layer
- server/requirements.txt — dependencies: fastapi, uvicorn, websockets
- server/static/index.html — web dashboard (served at /)

## Steps

### 1. Create a GCP Compute Engine VM
- Image: Ubuntu 22.04 LTS
- Machine type: e2-small is enough (server is lightweight, no GPU needed)
- Zone: pick closest to Saudi Arabia (me-central1 or europe-west1)
- Allow HTTP traffic (checkbox in console)
- Note the EXTERNAL IP

### 2. Firewall rule — allow port 8000
gcloud compute firewall-rules create allow-gpunet --allow tcp:8000 --direction=INGRESS

### 3. Copy server files to VM
scp -r server/ USERNAME@VM_EXTERNAL_IP:~/gpunetwork/

### 4. SSH into VM and set up
ssh USERNAME@VM_EXTERNAL_IP

sudo apt update && sudo apt install -y python3 python3-pip python3-venv
cd ~/gpunetwork
python3 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
cd server
python3 main.py

### 5. Verify
- Open http://VM_EXTERNAL_IP:8000 in browser — should see the web dashboard
- The dashboard will show "Waiting for agents to connect..."

### 6. Register a demo cafe
curl -X POST http://VM_EXTERNAL_IP:8000/cafes/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Demo Cafe","location":"Riyadh"}'

SAVE THE RESPONSE — it contains cafe_id and api_key needed for the Windows agent.

### 7. (Optional) Run as background service
Create systemd service so it survives SSH disconnect:

sudo tee /etc/systemd/system/gpunetwork.service > /dev/null <<EOF
[Unit]
Description=GPU Network Server
After=network.target
[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/gpunetwork/server
ExecStart=$HOME/gpunetwork/venv/bin/python main.py
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpunetwork
sudo systemctl start gpunetwork

## What to Return
When done, provide:
1. The VM external IP
2. The cafe_id and api_key from the register step
3. Confirmation that http://VM_IP:8000 loads the web UI

## Important Notes
- Server binds to 0.0.0.0:8000 (already configured)
- No TLS/HTTPS needed — plain HTTP for this POC
- SQLite database is created automatically on first run
- No environment variables required (defaults work)
- The repo is NOT on GitHub — files must be scp'd from this machine
