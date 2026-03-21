import asyncio
import json
import uuid
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

app = FastAPI(title="GPU Network Server")

# --- In-memory storage ---
connected_agents: dict[str, dict] = {}
jobs: dict[str, dict] = {}
# Job statuses: PENDING → ASSIGNED → RUNNING → COMPLETED / FAILED / CANCELLED


# --- Request models ---
class JobSubmission(BaseModel):
    model_name: str = "mock-model"
    prompt: str = ""
    job_type: str = "inference"  # "inference" or "fine-tune"


# --- Helper: find an available agent ---
def find_available_agent() -> str | None:
    for agent_id, info in connected_agents.items():
        if info["state"] == "AVAILABLE":
            return agent_id
    return None


# --- Helper: assign a pending job to an available agent ---
async def try_assign_jobs():
    for job_id, job in jobs.items():
        if job["status"] != "PENDING":
            continue

        agent_id = find_available_agent()
        if not agent_id:
            break

        agent = connected_agents[agent_id]
        job["status"] = "ASSIGNED"
        job["assigned_to"] = agent_id
        job["assigned_at"] = datetime.now().isoformat()

        # Push job to agent via WebSocket
        await agent["websocket"].send_text(json.dumps({
            "type": "job_assign",
            "job_id": job_id,
            "job_type": job["job_type"],
            "model_name": job["model_name"],
            "prompt": job["prompt"],
        }))

        print(f"[>] Job {job_id[:8]} assigned to {agent_id}")


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

        connected_agents[agent_id] = {
            "websocket": websocket,
            "cafe_id": cafe_id,
            "gpu_info": None,
            "idle_status": None,
            "state": "OFFLINE",
            "connected_at": datetime.now().isoformat(),
            "last_heartbeat": None,
        }

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
                connected_agents[agent_id].update({
                    "gpu_info": data.get("gpu_info"),
                    "idle_status": data.get("idle_status"),
                    "state": data.get("state"),
                    "last_heartbeat": datetime.now().isoformat(),
                })
                print(
                    f"[~] Heartbeat from {agent_id}: "
                    f"State={data.get('state')} | "
                    f"GPU={data.get('gpu_info', {}).get('name', 'N/A')} | "
                    f"VRAM={data.get('gpu_info', {}).get('vram_used_mb', '?')}/{data.get('gpu_info', {}).get('vram_total_mb', '?')}MB"
                )

                await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

                # After heartbeat, try to assign pending jobs if agent is available
                if data.get("state") == "AVAILABLE":
                    await try_assign_jobs()

            elif data["type"] == "job_started":
                job_id = data.get("job_id")
                if job_id in jobs:
                    jobs[job_id]["status"] = "RUNNING"
                    jobs[job_id]["started_at"] = datetime.now().isoformat()
                print(f"[*] Job {job_id[:8]} started on {agent_id}")

            elif data["type"] == "job_progress":
                job_id = data.get("job_id")
                if job_id in jobs:
                    jobs[job_id]["progress"] = data.get("progress", "")
                print(f"[~] Job {job_id[:8]} progress: {data.get('progress', '')}")

            elif data["type"] == "job_completed":
                job_id = data.get("job_id")
                if job_id in jobs:
                    jobs[job_id]["status"] = "COMPLETED"
                    jobs[job_id]["result"] = data.get("result", "")
                    jobs[job_id]["completed_at"] = datetime.now().isoformat()
                print(f"[✓] Job {job_id[:8]} completed by {agent_id}")

            elif data["type"] == "job_failed":
                job_id = data.get("job_id")
                if job_id in jobs:
                    jobs[job_id]["status"] = "FAILED"
                    jobs[job_id]["error"] = data.get("error", "")
                print(f"[✗] Job {job_id[:8]} failed on {agent_id}: {data.get('error')}")

            elif data["type"] == "job_cancelled":
                job_id = data.get("job_id")
                reason = data.get("reason", "")
                if job_id in jobs:
                    if reason == "Agent not available":
                        # Agent rejected — re-queue the job
                        jobs[job_id]["status"] = "PENDING"
                        jobs[job_id]["assigned_to"] = None
                        print(f"[!] Job {job_id[:8]} rejected by {agent_id}, back to queue")
                    else:
                        jobs[job_id]["status"] = "CANCELLED"
                        jobs[job_id]["cancelled_at"] = datetime.now().isoformat()
                        jobs[job_id]["cancel_reason"] = reason
                        print(f"[!] Job {job_id[:8]} cancelled on {agent_id}: {reason}")

    except WebSocketDisconnect:
        print(f"[-] Agent disconnected: {agent_id}")
    except Exception as e:
        print(f"[!] Error with agent {agent_id}: {e}")
    finally:
        if agent_id and agent_id in connected_agents:
            # Cancel any running jobs on this agent
            for job_id, job in jobs.items():
                if job.get("assigned_to") == agent_id and job["status"] in ("ASSIGNED", "RUNNING"):
                    job["status"] = "PENDING"
                    job["assigned_to"] = None
                    print(f"[!] Job {job_id[:8]} returned to queue (agent disconnected)")
            del connected_agents[agent_id]


# --- REST endpoints ---
@app.get("/")
async def root():
    return {"service": "GPU Network Server", "agents_online": len(connected_agents), "total_jobs": len(jobs)}


@app.get("/agents/status")
async def agents_status():
    agents = []
    for agent_id, info in connected_agents.items():
        agents.append({
            "agent_id": agent_id,
            "cafe_id": info["cafe_id"],
            "state": info["state"],
            "gpu_info": info["gpu_info"],
            "idle_status": info["idle_status"],
            "connected_at": info["connected_at"],
            "last_heartbeat": info["last_heartbeat"],
        })
    return {"agents_online": len(agents), "agents": agents}


@app.post("/jobs/submit")
async def submit_job(submission: JobSubmission):
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "job_id": job_id,
        "job_type": submission.job_type,
        "model_name": submission.model_name,
        "prompt": submission.prompt,
        "status": "PENDING",
        "assigned_to": None,
        "result": None,
        "error": None,
        "submitted_at": datetime.now().isoformat(),
    }
    print(f"[+] Job {job_id[:8]} submitted: {submission.job_type} / {submission.model_name}")

    # Try to assign immediately
    await try_assign_jobs()

    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    job = jobs[job_id].copy()
    job.pop("assigned_to", None)  # don't expose internal agent ID
    return job


@app.get("/jobs")
async def list_jobs():
    return {
        "total": len(jobs),
        "jobs": [
            {k: v for k, v in job.items() if k != "assigned_to"}
            for job in jobs.values()
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
