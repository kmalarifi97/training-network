# GPU Network - Training Network Project

## What It Is
Platform connecting idle gaming cafe GPUs (Saudi Arabia) to customers needing AI inference/fine-tuning. Windows agent runs on cafe PCs, reports to central server.

## Current Status: Production Groundwork Complete
- Phase 1: Agent foundation (GPU monitor, idle detector, state machine) - DONE
- Phase 2: Server connection (WebSocket, heartbeats) - DONE
- Phase 3: Job execution (submit, assign, run, result) - DONE
- Phase 4: Real inference with llama.cpp (TinyLlama 1.1B) - DONE
- Phase 4.5: Production groundwork - DONE
  - SQLite persistence (server/database.py)
  - Cafe registration + API key auth
  - WSS/TLS support in agent
  - Model auto-download from HuggingFace
  - Resource limiter (GPU/RAM/disk checks)
  - Windows Service wrapper
  - Inno Setup installer script + PyInstaller build script
  - Config: cross-platform paths, env var overrides
- Phase 5: Fine-tuning pipeline - NEXT
- Phase 6: Web dashboard - TODO
- Phase 7: Billing - TODO
- Phase 8: Packaging (.exe installer) - MOSTLY DONE (build scripts ready)

## Key Files
- `agent/main.py` - Agent loop, state machine (OFFLINE→AVAILABLE→WORKING→BUSY)
- `agent/gpu_monitor.py` - NVIDIA GPU stats via pynvml
- `agent/idle_detector.py` - Win32 API mouse/keyboard idle detection
- `agent/connection.py` - WebSocket client to server (WSS/TLS support)
- `agent/job_runner.py` - Real inference via llama-cpp-python + auto-download models
- `agent/model_cache.py` - LRU model cache
- `agent/resource_limiter.py` - Pre-job and during-job resource checks
- `agent/service_wrapper.py` - Windows Service install/start/stop
- `agent/config.py` - Config with env var overrides, cross-platform paths
- `agent/config.json` - Agent config (idle threshold, heartbeat interval)
- `server/main.py` - FastAPI server with SQLite persistence + cafe auth
- `server/database.py` - SQLite database layer (cafes, agents, jobs)
- `build/build_exe.py` - PyInstaller build script
- `build/installer.iss` - Inno Setup installer script
- `README-TRAINING-NETWORK.md` - Full build plan with all 8 phases

## Technical Details
- Test GPU: GeForce GTX 1060 6GB, CUDA 10.1, driver 431.90
- Python: 3.10+, venv at project root `venv/`
- Models: auto-downloaded from HuggingFace (tinyllama-1.1b, mistral-7b, llama3-8b)
- llama-cpp-python: CPU pre-built wheel (no C++ compiler needed)
- Server: FastAPI + SQLite (WAL mode) via server/database.py
- Testing config: idle threshold 0.17 min (~10s), heartbeat 5s

## How to Run
Terminal 1 (server): `venv\Scripts\python.exe server\main.py`
Terminal 2 (agent): `venv\Scripts\python.exe agent\main.py`

Register a cafe: `curl -X POST http://localhost:8000/cafes/register -H "Content-Type: application/json" -d "{\"name\": \"My Cafe\", \"location\": \"Riyadh\"}"`
Submit job: `curl -X POST http://localhost:8000/jobs/submit -H "Content-Type: application/json" -d "{\"model_name\": \"tinyllama-1.1b\", \"prompt\": \"your prompt here\", \"job_type\": \"inference\"}"`
Check result: `curl http://localhost:8000/jobs/JOB_ID`

## Config: Environment Variable Overrides
- `GPUNET_SERVER_URL` - WebSocket server URL
- `GPUNET_CAFE_ID` - Cafe identifier
- `GPUNET_API_KEY` - Authentication key
- `GPUNET_MODEL_DIR` - Model cache directory
- `GPUNET_LOG_DIR` - Log directory
- `GPUNET_IDLE_THRESHOLD` - Idle threshold in minutes
- `GPUNET_HEARTBEAT_INTERVAL` - Heartbeat interval in seconds

## Deploying to Client Machines
1. Register cafe via API → get cafe_id + api_key
2. Build exe: `python build/build_exe.py`
3. Compile installer: open `build/installer.iss` in Inno Setup
4. Distribute installer to cafe
5. Silent install: `GPUNetworkAgent-Setup.exe /SILENT /CAFE_ID=xxx /API_KEY=yyy /SERVER_URL=wss://...`

## User Preferences
- Thorough explanations before building
- Fast POC approach (no DB, in-memory)
- Commits at phase boundaries
- Git: local only, user "dev"
