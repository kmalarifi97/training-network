# GPU Network - Training Network Project

## What It Is
Platform connecting idle gaming cafe GPUs (Saudi Arabia) to customers needing AI inference/fine-tuning. Windows agent runs on cafe PCs, reports to central server.

## Current Status: Phase 4 Complete
- Phase 1: Agent foundation (GPU monitor, idle detector, state machine) - DONE
- Phase 2: Server connection (WebSocket, heartbeats) - DONE
- Phase 3: Job execution (submit, assign, run, result) - DONE
- Phase 4: Real inference with llama.cpp (TinyLlama 1.1B) - DONE
- Phase 5: Fine-tuning pipeline - NEXT
- Phase 6: Web dashboard - TODO
- Phase 7: Billing - TODO
- Phase 8: Packaging (.exe installer) - TODO

## Key Files
- `agent/main.py` - Agent loop, state machine (OFFLINE→AVAILABLE→WORKING→BUSY)
- `agent/gpu_monitor.py` - NVIDIA GPU stats via pynvml
- `agent/idle_detector.py` - Win32 API mouse/keyboard idle detection
- `agent/connection.py` - WebSocket client to server
- `agent/job_runner.py` - Real inference via llama-cpp-python
- `agent/model_cache.py` - LRU model cache
- `agent/config.json` - Agent config (idle threshold, heartbeat interval)
- `server/main.py` - FastAPI server, in-memory storage (no DB - POC)
- `README-TRAINING-NETWORK.md` - Full build plan with all 8 phases

## Technical Details
- Test GPU: GeForce GTX 1060 6GB, CUDA 10.1, driver 431.90
- Python: 3.10+, venv at project root `venv/`
- Model: TinyLlama 1.1B GGUF at `C:\GPUNetwork\models\`
- llama-cpp-python: CPU pre-built wheel (no C++ compiler needed)
- Server: FastAPI + in-memory Python dicts (no PostgreSQL - fast POC)
- Testing config: idle threshold 0.17 min (~10s), heartbeat 5s

## How to Run
Terminal 1 (server): `venv\Scripts\python.exe server\main.py`
Terminal 2 (agent): `venv\Scripts\python.exe agent\main.py`
Submit job: `curl -X POST http://localhost:8000/jobs/submit -H "Content-Type: application/json" -d "{\"model_name\": \"tinyllama-1.1b\", \"prompt\": \"your prompt here\", \"job_type\": \"inference\"}"`
Check result: `curl http://localhost:8000/jobs/JOB_ID`

## Setup on New Machine
1. Clone/copy the project
2. `python -m venv venv`
3. `pip install -r requirements.txt`
4. `pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`
5. `pip install huggingface-hub`
6. Download model: `python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', local_dir='C:/GPUNetwork/models')"`

## User Preferences
- Thorough explanations before building
- Fast POC approach (no DB, in-memory)
- Commits at phase boundaries
- Git: local only, user "dev"
