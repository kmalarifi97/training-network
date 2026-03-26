import asyncio
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import init_db, create_cafe, get_cafe, get_cafe_by_api_key, list_cafes
from database import upsert_agent, update_agent_heartbeat, set_agent_offline, list_agents, requeue_agent_jobs
from database import create_job, assign_job, update_job_status, get_job, list_jobs, get_pending_jobs, delete_job, delete_all_jobs
from database import create_dataset, update_dataset, get_dataset, list_datasets, delete_dataset

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

                job = get_job(job_id)
                if job and job["job_type"] == "dataprep":
                    try:
                        prompt_data = json.loads(job["prompt"])
                        update_dataset(prompt_data["dataset_id"],
                                       status="PROCESSING",
                                       started_at=datetime.now().isoformat())
                    except Exception:
                        pass

            elif data["type"] == "job_progress":
                job_id = data.get("job_id")
                progress = data.get("progress", "")
                update_job_status(job_id, "RUNNING", progress=progress)
                print(f"[~] Job {job_id[:8]} progress: {progress}")

                # Update linked dataset progress if dataprep
                job = get_job(job_id)
                if job and job["job_type"] == "dataprep":
                    try:
                        prompt_data = json.loads(job["prompt"])
                        dataset_id = prompt_data["dataset_id"]
                        extra = data.get("dataprep_progress", {})
                        if extra:
                            update_dataset(dataset_id,
                                           status="PROCESSING",
                                           total_chunks=extra.get("total_chunks", 0),
                                           processed_chunks=extra.get("processed_chunks", 0),
                                           total_pairs=extra.get("total_pairs", 0))
                    except Exception:
                        pass

            elif data["type"] == "job_completed":
                job_id = data.get("job_id")
                result_data = data.get("result", "")
                update_job_status(job_id, "COMPLETED", result=result_data)
                print(f"[✓] Job {job_id[:8]} completed by {agent_id}")

                # If this was a dataprep job, save the JSONL and update the dataset
                job = get_job(job_id)
                if job and job["job_type"] == "dataprep":
                    try:
                        prompt_data = json.loads(job["prompt"])
                        dataset_id = prompt_data["dataset_id"]
                        result_obj = json.loads(result_data) if isinstance(result_data, str) else result_data
                        pairs = result_obj.get("pairs", [])
                        output_format = prompt_data.get("output_format", "alpaca")

                        output_path = Path(DATAPREP_OUTPUT_DIR) / f"{dataset_id}.jsonl"
                        with open(output_path, "w", encoding="utf-8") as f:
                            for pair in pairs:
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
                            dataset_id,
                            status="COMPLETED",
                            output_file=str(output_path),
                            total_pairs=len(pairs),
                            completed_at=datetime.now().isoformat(),
                        )
                        print(f"[dataprep] {dataset_id[:8]}: saved {len(pairs)} pairs")
                    except Exception as e:
                        print(f"[dataprep] Failed to save dataset: {e}")
                        try:
                            update_dataset(dataset_id, status="FAILED", error=str(e))
                        except Exception:
                            pass

            elif data["type"] == "job_failed":
                job_id = data.get("job_id")
                error_msg = data.get("error", "")
                update_job_status(job_id, "FAILED", error=error_msg)
                print(f"[✗] Job {job_id[:8]} failed on {agent_id}: {error_msg}")

                # Update linked dataset if this was a dataprep job
                job = get_job(job_id)
                if job and job["job_type"] == "dataprep":
                    try:
                        prompt_data = json.loads(job["prompt"])
                        update_dataset(prompt_data["dataset_id"],
                                       status="FAILED", error=error_msg)
                    except Exception:
                        pass

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


@app.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    delete_job(job_id)
    return {"message": "Job deleted"}


@app.delete("/jobs")
async def delete_all_jobs_endpoint():
    delete_all_jobs()
    return {"message": "All jobs deleted"}


# =============================================================================
# Dataset generation — all processing happens on GPU agents
# =============================================================================

DATAPREP_OUTPUT_DIR = os.environ.get(
    "DATAPREP_OUTPUT_DIR",
    str(Path(__file__).parent / "outputs"),
)
DATAPREP_MAX_UPLOAD_MB = int(os.environ.get("DATAPREP_MAX_UPLOAD_MB", "50"))

Path(DATAPREP_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


class TextInput(BaseModel):
    text: str
    model_name: str | None = None
    strategy: str = "self-instruct"
    output_format: str = "alpaca"
    pairs_per_chunk: int | None = None


@app.post("/datasets/generate/file")
async def generate_from_file(
    file: UploadFile = File(...),
    model_name: str = Form("mistral-7b"),
    strategy: str = Form("self-instruct"),
    output_format: str = Form("alpaca"),
    pairs_per_chunk: int = Form(3),
):
    """Upload a file and send it to a GPU agent for dataset generation."""
    max_bytes = DATAPREP_MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(400, f"File too large. Max {DATAPREP_MAX_UPLOAD_MB}MB")

    # Read text from file on server side (lightweight, no model needed)
    import tempfile
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text — only import needed: PyPDF2/python-docx for PDF/DOCX
        if suffix.lower() == ".pdf":
            try:
                import PyPDF2
                parts = []
                with open(tmp_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            parts.append(t)
                text = "\n\n".join(parts)
            except ImportError:
                raise HTTPException(500, "PyPDF2 not installed on server")
        elif suffix.lower() in (".docx", ".doc"):
            try:
                import docx
                doc = docx.Document(tmp_path)
                text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                raise HTTPException(500, "python-docx not installed on server")
        else:
            text = content.decode("utf-8", errors="replace")
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(400, "No text could be extracted from file")

    return await _create_dataprep_job(
        text=text,
        source_filename=file.filename,
        model_name=model_name,
        strategy=strategy,
        output_format=output_format,
        pairs_per_chunk=pairs_per_chunk,
    )


@app.post("/datasets/generate/text")
async def generate_from_text(req: TextInput):
    """Submit raw text — sent to a GPU agent for dataset generation."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    return await _create_dataprep_job(
        text=req.text,
        source_filename="raw_text",
        model_name=req.model_name or "mistral-7b",
        strategy=req.strategy,
        output_format=req.output_format,
        pairs_per_chunk=req.pairs_per_chunk or 3,
    )


async def _create_dataprep_job(text: str, source_filename: str, model_name: str,
                                strategy: str, output_format: str,
                                pairs_per_chunk: int) -> dict:
    """Create a dataset record + a dataprep job, assign to GPU agent."""
    dataset_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex

    create_dataset(
        dataset_id=dataset_id,
        source_filename=source_filename,
        source_type=Path(source_filename).suffix.lower().lstrip(".") or "txt",
        model_name=model_name,
        strategy=strategy,
        output_format=output_format,
        chunk_size=0,
        pairs_per_chunk=pairs_per_chunk,
    )

    # The prompt field carries all the data the agent needs
    job_prompt = json.dumps({
        "text": text,
        "dataset_id": dataset_id,
        "strategy": strategy,
        "output_format": output_format,
        "pairs_per_chunk": pairs_per_chunk,
    })

    create_job(job_id, "dataprep", model_name, job_prompt)
    update_dataset(dataset_id, status="PENDING")

    print(f"[dataprep] {dataset_id[:8]}: job {job_id[:8]} queued ({len(text)} chars)")

    await try_assign_jobs()

    return {"dataset_id": dataset_id, "job_id": job_id, "status": "PENDING"}


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
    delete_dataset(dataset_id)
    return {"message": "Dataset deleted"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
