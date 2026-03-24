import asyncio
import hashlib
import json
import secrets
import uuid
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from pydantic import BaseModel

from database import init_db, create_cafe, get_cafe, get_cafe_by_api_key, list_cafes
from database import upsert_agent, update_agent_heartbeat, set_agent_offline, list_agents, requeue_agent_jobs
from database import create_job, assign_job, update_job_status, get_job, list_jobs, get_pending_jobs

app = FastAPI(title="GPU Network Server")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
