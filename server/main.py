import asyncio
import hashlib
import json
import os
import secrets
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi import UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import init_db, create_cafe, get_cafe, get_cafe_by_api_key, list_cafes
from database import upsert_agent, update_agent_heartbeat, set_agent_offline, list_agents, requeue_agent_jobs
from database import create_job, assign_job, update_job_status, get_job, list_jobs, get_pending_jobs
from database import create_dataset, update_dataset, get_dataset, list_datasets, delete_dataset

# Add project root to path so dataprep package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataprep import DatasetGenerator, extract_text, chunk_text

app = FastAPI(title="GPU Network Server")

# CORS — allow all origins for POC
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (web UI)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Live WebSocket connections (in-memory — only for active sockets)
live_connections: dict[str, WebSocket] = {}


# --- Startup ---
@app.on_event("startup")
async def startup():
    init_db()
    print("[*] Database initialized")


# --- Auth helpers ---
def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    return f"gpunet_{secrets.token_urlsafe(32)}"


async def verify_api_key(x_api_key: str = Header(None)) -> dict:
    """Dependency for REST endpoints that require cafe auth."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    cafe = get_cafe_by_api_key(x_api_key)
    if not cafe:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return cafe


# --- Request models ---
class JobSubmission(BaseModel):
    model_name: str = "tinyllama-1.1b"
    prompt: str = ""
    job_type: str = "inference"


class CafeRegistration(BaseModel):
    name: str
    owner_name: str = ""
    location: str = ""


# --- Helper: find an available agent from live connections ---
def find_available_agent() -> str | None:
    agents = list_agents(online_only=True)
    for agent in agents:
        if agent["state"] == "AVAILABLE" and agent["agent_id"] in live_connections:
            return agent["agent_id"]
    return None


# --- Helper: assign pending jobs to available agents ---
async def try_assign_jobs():
    pending = get_pending_jobs()
    for job in pending:
        agent_id = find_available_agent()
        if not agent_id:
            break

        assign_job(job["job_id"], agent_id)

        ws = live_connections.get(agent_id)
        if ws:
            await ws.send_text(json.dumps({
                "type": "job_assign",
                "job_id": job["job_id"],
                "job_type": job["job_type"],
                "model_name": job["model_name"],
                "prompt": job["prompt"],
            }))
            print(f"[>] Job {job['job_id'][:8]} assigned to {agent_id}")


# --- WebSocket endpoint for agents ---
@app.websocket("/agents/connect")
async def agent_connect(websocket: WebSocket):
    await websocket.accept()
    agent_id = None

    try:
        # First message should be registration
        raw = await websocket.receive_text()
        data = json.loads(raw)

        if data.get("type") != "register":
            await websocket.send_text(json.dumps({"type": "error", "message": "First message must be register"}))
            await websocket.close()
            return

        agent_id = data.get("agent_id")
        cafe_id = data.get("cafe_id")
        api_key = data.get("api_key", "")

        # Validate API key if provided
        if api_key:
            cafe = get_cafe_by_api_key(api_key)
            if not cafe:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid API key"}))
                await websocket.close()
                return
            # Override cafe_id from the key's cafe
            cafe_id = cafe["cafe_id"]

        # Register agent in DB and track live connection
        upsert_agent(agent_id, cafe_id, state="OFFLINE")
        live_connections[agent_id] = websocket

        print(f"[+] Agent connected: {agent_id} (Cafe: {cafe_id})")

        await websocket.send_text(json.dumps({
            "type": "registered",
            "message": f"Welcome agent {agent_id}",
        }))

        # Listen for messages
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data["type"] == "heartbeat":
                state = data.get("state", "OFFLINE")
                gpu_info = data.get("gpu_info")
                idle_status = data.get("idle_status")

                update_agent_heartbeat(agent_id, state, gpu_info, idle_status)

                print(
                    f"[~] Heartbeat from {agent_id}: "
                    f"State={state} | "
                    f"GPU={gpu_info.get('name', 'N/A') if gpu_info else 'N/A'} | "
                    f"VRAM={gpu_info.get('vram_used_mb', '?')}/{gpu_info.get('vram_total_mb', '?')}MB"
                    if gpu_info else f"[~] Heartbeat from {agent_id}: State={state} | GPU=unavailable"
                )

                await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

                if state == "AVAILABLE":
                    await try_assign_jobs()

            elif data["type"] == "job_started":
                job_id = data.get("job_id")
                update_job_status(job_id, "RUNNING")
                print(f"[*] Job {job_id[:8]} started on {agent_id}")

            elif data["type"] == "job_progress":
                job_id = data.get("job_id")
                update_job_status(job_id, "RUNNING", progress=data.get("progress", ""))
                print(f"[~] Job {job_id[:8]} progress: {data.get('progress', '')}")

            elif data["type"] == "job_completed":
                job_id = data.get("job_id")
                update_job_status(job_id, "COMPLETED", result=data.get("result", ""))
                print(f"[✓] Job {job_id[:8]} completed by {agent_id}")

            elif data["type"] == "job_failed":
                job_id = data.get("job_id")
                update_job_status(job_id, "FAILED", error=data.get("error", ""))
                print(f"[✗] Job {job_id[:8]} failed on {agent_id}: {data.get('error')}")

            elif data["type"] == "job_cancelled":
                job_id = data.get("job_id")
                reason = data.get("reason", "")
                if reason == "Agent not available":
                    update_job_status(job_id, "PENDING", assigned_to=None)
                    print(f"[!] Job {job_id[:8]} rejected by {agent_id}, back to queue")
                else:
                    update_job_status(job_id, "CANCELLED", cancel_reason=reason)
                    print(f"[!] Job {job_id[:8]} cancelled on {agent_id}: {reason}")

    except WebSocketDisconnect:
        print(f"[-] Agent disconnected: {agent_id}")
    except Exception as e:
        print(f"[!] Error with agent {agent_id}: {e}")
    finally:
        if agent_id:
            live_connections.pop(agent_id, None)
            set_agent_offline(agent_id)
            requeue_agent_jobs(agent_id)


# --- REST endpoints ---
@app.get("/")
async def root():
    # Serve web UI if it exists, otherwise return JSON status
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "GPU Network Server",
        "agents_online": len(live_connections),
        "total_jobs": len(list_jobs(limit=10000)),
    }


# --- Cafe management ---
@app.post("/cafes/register")
async def register_cafe(reg: CafeRegistration):
    cafe_id = f"cafe_{uuid.uuid4().hex[:12]}"
    api_key = generate_api_key()
    cafe = create_cafe(cafe_id, reg.name, api_key, reg.owner_name, reg.location)
    return {
        "cafe_id": cafe["cafe_id"],
        "api_key": cafe["api_key"],
        "message": "Save your API key — it won't be shown again.",
    }


@app.get("/cafes")
async def get_cafes():
    cafes = list_cafes()
    # Strip api_key from list view
    for c in cafes:
        c.pop("api_key", None)
    return {"cafes": cafes}


# --- Agent status ---
@app.get("/agents/status")
async def agents_status():
    agents = list_agents(online_only=False)
    for a in agents:
        a["online"] = a["agent_id"] in live_connections
    return {"agents_online": len(live_connections), "agents": agents}


# --- Job endpoints ---
@app.post("/jobs/submit")
async def submit_job(submission: JobSubmission):
    job_id = uuid.uuid4().hex
    create_job(job_id, submission.job_type, submission.model_name, submission.prompt)
    print(f"[+] Job {job_id[:8]} submitted: {submission.job_type} / {submission.model_name}")

    await try_assign_jobs()

    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.pop("assigned_to", None)
    return job


@app.get("/jobs")
async def list_jobs_endpoint(status: str | None = None, limit: int = 100):
    jobs = list_jobs(status=status, limit=limit)
    for j in jobs:
        j.pop("assigned_to", None)
    return {"total": len(jobs), "jobs": jobs}


# =============================================================================
# Dataset generation (dataprep integration)
# =============================================================================

DATAPREP_MODEL_DIR = os.environ.get(
    "DATAPREP_MODEL_DIR",
    str(Path(__file__).parent.parent / "models" / "dataprep"),
)
DATAPREP_UPLOAD_DIR = os.environ.get(
    "DATAPREP_UPLOAD_DIR",
    str(Path(__file__).parent / "uploads"),
)
DATAPREP_OUTPUT_DIR = os.environ.get(
    "DATAPREP_OUTPUT_DIR",
    str(Path(__file__).parent / "outputs"),
)
DATAPREP_MODEL = os.environ.get("DATAPREP_MODEL", "mistral-7b")
DATAPREP_CHUNK_SIZE = int(os.environ.get("DATAPREP_CHUNK_SIZE", "500"))
DATAPREP_CHUNK_OVERLAP = int(os.environ.get("DATAPREP_CHUNK_OVERLAP", "50"))
DATAPREP_PAIRS_PER_CHUNK = int(os.environ.get("DATAPREP_PAIRS_PER_CHUNK", "3"))
DATAPREP_MAX_UPLOAD_MB = int(os.environ.get("DATAPREP_MAX_UPLOAD_MB", "50"))

Path(DATAPREP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(DATAPREP_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

_generator: DatasetGenerator | None = None
_gen_lock = threading.Lock()


def _get_generator() -> DatasetGenerator:
    global _generator
    if _generator is None:
        _generator = DatasetGenerator(
            model_dir=DATAPREP_MODEL_DIR,
            model_name=DATAPREP_MODEL,
        )
    return _generator


class TextInput(BaseModel):
    text: str
    model_name: str | None = None
    strategy: str = "self-instruct"
    output_format: str = "alpaca"
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    pairs_per_chunk: int | None = None


def _process_dataset(dataset_id: str, source_path: str, model_name: str,
                     strategy: str, output_format: str,
                     chunk_size: int, chunk_overlap: int, pairs_per_chunk: int):
    """Background worker: text -> chunks -> instruction pairs -> JSONL."""
    with _gen_lock:
        try:
            update_dataset(dataset_id, status="PROCESSING",
                           started_at=datetime.now().isoformat())

            text = extract_text(source_path)
            if not text.strip():
                update_dataset(dataset_id, status="FAILED",
                               error="No text extracted from file")
                return

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                update_dataset(dataset_id, status="FAILED",
                               error="Text too short to generate chunks")
                return

            update_dataset(dataset_id, total_chunks=len(chunks))
            print(f"[dataprep] {dataset_id[:8]}: {len(chunks)} chunks from {len(text)} chars")

            gen = _get_generator()
            if model_name != gen.model_name:
                gen.unload()
                gen.model_name = model_name

            all_pairs = []
            for i, chunk in enumerate(chunks):
                pairs = gen.generate_pairs(
                    chunk, strategy=strategy, pairs_per_chunk=pairs_per_chunk,
                )
                all_pairs.extend(pairs)
                update_dataset(dataset_id, processed_chunks=i + 1,
                               total_pairs=len(all_pairs))
                print(f"[dataprep] {dataset_id[:8]}: chunk {i+1}/{len(chunks)} "
                      f"+{len(pairs)} pairs")

            if not all_pairs:
                update_dataset(dataset_id, status="FAILED",
                               error="Model produced no usable pairs")
                return

            output_path = Path(DATAPREP_OUTPUT_DIR) / f"{dataset_id}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for pair in all_pairs:
                    if output_format == "chat":
                        record = {
                            "messages": [
                                {"role": "user", "content": pair["instruction"]},
                                {"role": "assistant", "content": pair["output"]},
                            ]
                        }
                    else:
                        record = {
                            "instruction": pair["instruction"],
                            "input": pair.get("input", ""),
                            "output": pair["output"],
                        }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            update_dataset(
                dataset_id, status="COMPLETED",
                output_file=str(output_path),
                total_pairs=len(all_pairs),
                completed_at=datetime.now().isoformat(),
            )
            print(f"[dataprep] {dataset_id[:8]}: done — {len(all_pairs)} pairs")

        except Exception as e:
            print(f"[dataprep] {dataset_id[:8]}: FAILED — {e}")
            update_dataset(dataset_id, status="FAILED", error=str(e))


@app.post("/datasets/generate/file")
async def generate_from_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Form(None),
    strategy: str = Form("self-instruct"),
    output_format: str = Form("alpaca"),
    chunk_size: int = Form(None),
    chunk_overlap: int = Form(None),
    pairs_per_chunk: int = Form(None),
):
    """Upload a file (txt, pdf, docx) and generate a fine-tuning dataset."""
    max_bytes = DATAPREP_MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(400, f"File too large. Max {DATAPREP_MAX_UPLOAD_MB}MB")

    dataset_id = uuid.uuid4().hex
    upload_path = Path(DATAPREP_UPLOAD_DIR) / f"{dataset_id}_{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(content)

    source_type = Path(file.filename).suffix.lower().lstrip(".")

    create_dataset(
        dataset_id=dataset_id,
        source_filename=file.filename,
        source_type=source_type,
        model_name=model_name or DATAPREP_MODEL,
        strategy=strategy,
        output_format=output_format,
        chunk_size=chunk_size or DATAPREP_CHUNK_SIZE,
        pairs_per_chunk=pairs_per_chunk or DATAPREP_PAIRS_PER_CHUNK,
    )

    background_tasks.add_task(
        _process_dataset,
        dataset_id=dataset_id,
        source_path=str(upload_path),
        model_name=model_name or DATAPREP_MODEL,
        strategy=strategy,
        output_format=output_format,
        chunk_size=chunk_size or DATAPREP_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or DATAPREP_CHUNK_OVERLAP,
        pairs_per_chunk=pairs_per_chunk or DATAPREP_PAIRS_PER_CHUNK,
    )

    return {"dataset_id": dataset_id, "status": "PENDING"}


@app.post("/datasets/generate/text")
async def generate_from_text(req: TextInput, background_tasks: BackgroundTasks):
    """Submit raw text and generate a fine-tuning dataset."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    dataset_id = uuid.uuid4().hex
    upload_path = Path(DATAPREP_UPLOAD_DIR) / f"{dataset_id}_raw.txt"
    with open(upload_path, "w", encoding="utf-8") as f:
        f.write(req.text)

    create_dataset(
        dataset_id=dataset_id,
        source_filename="raw_text",
        source_type="txt",
        model_name=req.model_name or DATAPREP_MODEL,
        strategy=req.strategy,
        output_format=req.output_format,
        chunk_size=req.chunk_size or DATAPREP_CHUNK_SIZE,
        pairs_per_chunk=req.pairs_per_chunk or DATAPREP_PAIRS_PER_CHUNK,
    )

    background_tasks.add_task(
        _process_dataset,
        dataset_id=dataset_id,
        source_path=str(upload_path),
        model_name=req.model_name or DATAPREP_MODEL,
        strategy=req.strategy,
        output_format=req.output_format,
        chunk_size=req.chunk_size or DATAPREP_CHUNK_SIZE,
        chunk_overlap=req.chunk_overlap or DATAPREP_CHUNK_OVERLAP,
        pairs_per_chunk=req.pairs_per_chunk or DATAPREP_PAIRS_PER_CHUNK,
    )

    return {"dataset_id": dataset_id, "status": "PENDING"}


@app.get("/datasets")
async def get_datasets(status: str | None = None, limit: int = 100):
    datasets = list_datasets(status=status, limit=limit)
    return {"total": len(datasets), "datasets": datasets}


@app.get("/datasets/{dataset_id}")
async def get_dataset_status(dataset_id: str):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@app.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if ds["status"] != "COMPLETED":
        raise HTTPException(400, f"Dataset not ready. Status: {ds['status']}")
    if not ds["output_file"] or not Path(ds["output_file"]).exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(
        ds["output_file"],
        media_type="application/jsonl",
        filename=f"dataset_{dataset_id}.jsonl",
    )


@app.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, lines: int = 5):
    """Preview the first N lines of a completed dataset."""
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if ds["status"] != "COMPLETED":
        raise HTTPException(400, f"Dataset not ready. Status: {ds['status']}")
    if not ds["output_file"] or not Path(ds["output_file"]).exists():
        raise HTTPException(404, "Output file not found")
    samples = []
    with open(ds["output_file"], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= lines:
                break
            samples.append(json.loads(line))
    return {"dataset_id": dataset_id, "total_pairs": ds["total_pairs"], "samples": samples}


@app.delete("/datasets/{dataset_id}")
async def remove_dataset(dataset_id: str):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if ds["output_file"]:
        Path(ds["output_file"]).unlink(missing_ok=True)
    for f in Path(DATAPREP_UPLOAD_DIR).glob(f"{dataset_id}_*"):
        f.unlink(missing_ok=True)
    delete_dataset(dataset_id)
    return {"message": "Dataset deleted"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
