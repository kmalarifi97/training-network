# GPU Network — Windows Agent

A platform that connects idle gaming cafe GPUs in Saudi Arabia to customers who need model fine-tuning and inference. Cafe owners install a lightweight Windows agent that runs AI workloads when PCs are not in use by gamers.

## Architecture Overview

```
Central Server (Linux)
├── REST API (FastAPI)
├── Job Scheduler
├── WebSocket Hub
├── Model Storage (S3/MinIO)
└── Database (PostgreSQL)
        │
        │ internet (WebSocket)
        │
  ┌─────┼─────────────┐
  ▼     ▼             ▼
Agent   Agent         Agent
PC-1    PC-2          PC-N
(Cafe 1)             (Cafe 2)
```

## Windows Agent Components

```
agent/
├── main.py              — Entry point, runs the agent loop
├── gpu_monitor.py       — Reads GPU info via NVIDIA NVML
├── idle_detector.py     — Detects mouse/keyboard inactivity
├── connection.py        — WebSocket connection to central server
├── job_runner.py        — Downloads models, runs inference/fine-tuning
├── resource_limiter.py  — Caps GPU/RAM/disk/bandwidth usage
├── model_cache.py       — Caches downloaded models locally
├── config.py            — Agent settings (cafe ID, server URL, thresholds)
└── tray.py              — System tray icon and UI (Phase 2)
```

---

## Prerequisites (Windows PC)

- Windows 10 or 11
- NVIDIA GPU (RTX 2070 or higher, 8GB+ VRAM recommended)
- NVIDIA drivers installed (verify with `nvidia-smi` in terminal)
- Python 3.11+
- 16GB+ RAM
- 70GB+ free disk space
- Stable internet connection (15 Mbps+ up/down)

---

## Development Setup

### Step 1: Install tools

Open PowerShell as Administrator:

```powershell
winget install Python.Python.3.11
winget install Git.Git
winget install Microsoft.VisualStudioCode
```

Verify NVIDIA drivers:

```powershell
nvidia-smi
```

### Step 2: Clone and set up the project

```powershell
git clone <your-repo-url> C:\Projects\gpu-network
cd C:\Projects\gpu-network

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3: Configure the agent

Copy the example config and edit:

```powershell
copy agent\config.example.json agent\config.json
```

Edit `agent\config.json`:

```json
{
  "server_url": "ws://localhost:8000/agents/connect",
  "cafe_id": "your-cafe-id",
  "api_key": "your-api-key",
  "idle_threshold_minutes": 5,
  "max_gpu_usage_percent": 90,
  "max_ram_usage_percent": 80,
  "max_disk_usage_gb": 50,
  "model_cache_dir": "C:\\GPUNetwork\\models",
  "log_dir": "C:\\GPUNetwork\\logs"
}
```

### Step 4: Run the agent

```powershell
cd C:\Projects\gpu-network
.\venv\Scripts\activate
python agent/main.py
```

---

## Build Plan — Step by Step

### Phase 1: Agent Foundation (Week 1-2)

These run locally, no server needed yet.

- [ ] **1.1 GPU Monitor** (`gpu_monitor.py`)
  - Read GPU name, VRAM total/used/free, temperature, utilization
  - Uses `pynvml` (Python NVIDIA Management Library)
  - Reports info every 30 seconds

- [ ] **1.2 Idle Detector** (`idle_detector.py`)
  - Detect seconds since last mouse/keyboard input (Win32 API)
  - Detect running fullscreen apps / game processes
  - Configurable idle threshold (default: 5 minutes)
  - Must react instantly when user returns (< 3 seconds)

- [ ] **1.3 Agent Loop** (`main.py`)
  - Combines GPU monitor + idle detector
  - State machine: OFFLINE → AVAILABLE → WORKING → BUSY (gamer returned)
  - Logs status changes
  - Heartbeat every 30 seconds

### Phase 2: Server Connection (Week 3-4)

- [ ] **2.1 Central Server API** (`server/`)
  - FastAPI application
  - PostgreSQL database with tables: machines, cafes, jobs, customers, transactions
  - Endpoints:
    - `POST /cafes/register` — cafe owner signs up
    - `POST /agents/register` — register a new machine
    - `WS /agents/connect` — WebSocket for agent communication
    - `GET /agents/status` — list all connected machines

- [ ] **2.2 WebSocket Connection** (`agent/connection.py`)
  - Agent connects to server on startup
  - Sends heartbeat with GPU info + idle status every 30 seconds
  - Receives job assignments from server
  - Auto-reconnects on disconnect
  - Authenticated via API key

- [ ] **2.3 Machine Registry**
  - Server stores each machine: gpu model, vram, cafe, status, last seen
  - Dashboard shows all online machines and their status

### Phase 3: Job Execution (Week 5-6)

- [ ] **3.1 Job Runner** (`agent/job_runner.py`)
  - Receives job spec from server via WebSocket
  - Downloads model from server's storage (if not cached)
  - Runs the job in a subprocess:
    - Inference: load model → process input → return output
    - Fine-tuning: load base model + dataset → run LoRA/QLoRA → upload result
  - Streams progress/logs back to server
  - Kills subprocess immediately when gamer returns

- [ ] **3.2 Model Cache** (`agent/model_cache.py`)
  - Cache downloaded models on local disk
  - LRU eviction when disk limit reached
  - Popular models (Llama 3 7B, Mistral 7B) stay cached
  - Verify model integrity with checksum

- [ ] **3.3 Job Scheduler** (`server/scheduler.py`)
  - Match pending jobs to available machines
  - Filter by: required VRAM, GPU model, location
  - Handle preemption: gamer returns mid-job → save state → reassign to another machine
  - Priority queue: paid > free tier

### Phase 4: Inference Pipeline (Week 7-8)

- [ ] **4.1 Inference Runtime**
  - Support vLLM or llama.cpp on Windows for model serving
  - Load model into VRAM, accept prompts, return completions
  - Support models: Llama 3 7B/8B, Mistral 7B, Qwen 7B (models that fit in consumer GPU VRAM)

- [ ] **4.2 Customer API**
  - `POST /inference` — submit prompt, get completion
  - `POST /inference/stream` — streaming response
  - Compatible with OpenAI API format for easy migration
  - Rate limiting, API key auth

- [ ] **4.3 Load Balancing**
  - Route inference requests across available machines
  - Sticky sessions: keep model loaded, route same model requests to same machine
  - Failover: if machine goes offline, re-route to another

### Phase 5: Fine-Tuning Pipeline (Week 9-10)

- [ ] **5.1 Dataset Upload**
  - Customer uploads training data (JSONL format) via dashboard or API
  - Validation: check format, size limits
  - Storage: upload to S3/MinIO

- [ ] **5.2 Fine-Tuning Job**
  - Support LoRA and QLoRA (fit in consumer GPU VRAM)
  - Base models: Llama 3 7B, Mistral 7B
  - Agent downloads base model + dataset, runs fine-tuning
  - Checkpoint saving every N steps (recovery if interrupted)
  - Upload fine-tuned adapter to server on completion

- [ ] **5.3 One-Click Deploy**
  - Customer fine-tunes → gets a model
  - Click "Deploy" → inference endpoint created
  - Routes to a machine that loads base model + LoRA adapter

### Phase 6: Web Dashboard (Week 11-12)

- [ ] **6.1 Cafe Owner Dashboard**
  - View all registered machines and their status
  - See earnings per machine, per day, per month
  - Control: pause/resume machines, set work hours
  - Download agent installer

- [ ] **6.2 Customer Dashboard**
  - Browse available models
  - Submit inference requests (playground)
  - Upload data and start fine-tuning jobs
  - View deployed models and their endpoints
  - API key management
  - Usage and billing

### Phase 7: Billing (Week 13-14)

- [ ] **7.1 Usage Tracking**
  - Track GPU-seconds per job
  - Track tokens generated (for inference)
  - Track VRAM-hours used

- [ ] **7.2 Pricing**
  - Inference: per token or per request
  - Fine-tuning: per GPU-hour
  - Different rates for different GPU tiers

- [ ] **7.3 Payouts**
  - Calculate cafe owner earnings: percentage of job revenue
  - Monthly payout via bank transfer

### Phase 8: Packaging and Distribution (Week 15-16)

- [ ] **8.1 Windows Installer**
  - Package agent as single `.exe` using PyInstaller
  - NSIS or Inno Setup for proper installer with:
    - Install wizard
    - Auto-start on Windows boot
    - Config input (cafe ID)
    - Uninstaller

- [ ] **8.2 Silent Install**
  - Support `GPUNetwork-Installer.exe /silent /cafe-id=ABC123`
  - For cafe admins deploying to many PCs at once

- [ ] **8.3 Auto-Update**
  - Agent checks for updates on startup
  - Downloads and installs new version silently
  - Rollback if update fails

---

## Agent State Machine

```
            ┌──────────────────────────┐
            ▼                          │
    ┌──────────────┐          ┌────────┴───────┐
    │   OFFLINE    │──start──▶│   AVAILABLE    │
    │ (not running)│          │ (idle, waiting │
    └──────────────┘          │  for work)     │
                              └───────┬────────┘
                                      │
                              server assigns job
                                      │
                              ┌───────▼────────┐
                              │    WORKING     │
                              │ (running a job)│
                              └───────┬────────┘
                                      │
                          ┌───────────┼───────────┐
                          │                       │
                  gamer returns              job completes
                          │                       │
                  ┌───────▼────────┐     ┌────────▼───────┐
                  │     BUSY       │     │   AVAILABLE    │
                  │ (stopped work, │     │  (ready for    │
                  │  gamer active) │     │   next job)    │
                  └───────┬────────┘     └────────────────┘
                          │
                   gamer leaves
                          │
                  ┌───────▼────────┐
                  │   AVAILABLE    │
                  └────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent | Python 3.11, pynvml, psutil, websockets |
| Agent packaging | PyInstaller + Inno Setup |
| Server API | Python, FastAPI, Uvicorn |
| Database | PostgreSQL |
| Job queue | Redis or PostgreSQL-based queue |
| Model storage | MinIO (S3-compatible) |
| Inference runtime | vLLM or llama.cpp (Windows build) |
| Fine-tuning | Hugging Face Transformers + PEFT (LoRA) |
| Web dashboard | Next.js or React |
| Auth | JWT tokens |
| Monitoring | Prometheus + Grafana |

---

## Dependencies

`requirements.txt`:

```
pynvml>=12.0.0
psutil>=5.9.0
websockets>=12.0
httpx>=0.25.0
pydantic>=2.0.0
```

Server dependencies (`server/requirements.txt`):

```
fastapi>=0.110.0
uvicorn>=0.27.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
websockets>=12.0
pydantic>=2.0.0
python-jose>=3.3.0
passlib>=1.7.4
httpx>=0.25.0
redis>=5.0.0
boto3>=1.34.0
```

---

## Environment Variables

Server:

```
DATABASE_URL=postgresql://user:pass@localhost:5432/gpunetwork
REDIS_URL=redis://localhost:6379
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
JWT_SECRET=your-secret-key
```

---

## Running in Development

Terminal 1 — Server:

```powershell
cd server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 — Agent:

```powershell
cd agent
python main.py
```

Terminal 3 — Dashboard (when built):

```powershell
cd dashboard
npm run dev
```

---

## License

Proprietary — All rights reserved.
