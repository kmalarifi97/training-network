import asyncio
import json
import time
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI(title="GPU Network Server")

# --- In-memory storage ---
connected_agents: dict[str, dict] = {}
# Key: agent_id (cafe_id + machine hash)
# Value: {websocket, cafe_id, gpu_info, idle_status, state, connected_at, last_heartbeat}


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
                    f"VRAM={data.get('gpu_info', {}).get('vram_used_mb', '?')}/{data.get('gpu_info', {}).get('vram_total_mb', '?')}MB | "
                    f"Idle={data.get('idle_status', {}).get('idle_seconds', '?')}s"
                )

                # Acknowledge heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

    except WebSocketDisconnect:
        print(f"[-] Agent disconnected: {agent_id}")
    except Exception as e:
        print(f"[!] Error with agent {agent_id}: {e}")
    finally:
        if agent_id and agent_id in connected_agents:
            del connected_agents[agent_id]


# --- REST endpoints (for dashboard / monitoring) ---
@app.get("/")
async def root():
    return {"service": "GPU Network Server", "agents_online": len(connected_agents)}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
