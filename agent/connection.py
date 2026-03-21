import asyncio
import json
import logging
import uuid

import websockets

logger = logging.getLogger("agent.connection")


class ServerConnection:
    def __init__(self, server_url: str, cafe_id: str, api_key: str = ""):
        self.server_url = server_url
        self.cafe_id = cafe_id
        self.api_key = api_key
        self.agent_id = f"{cafe_id}_{uuid.uuid4().hex[:8]}"
        self._ws = None
        self._connected = False
        self._reconnect_delay = 5
        self.on_job_received = None  # callback: async def(job_dict)

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self):
        while True:
            try:
                logger.info(f"Connecting to server: {self.server_url}")
                self._ws = await websockets.connect(self.server_url)

                await self._ws.send(json.dumps({
                    "type": "register",
                    "agent_id": self.agent_id,
                    "cafe_id": self.cafe_id,
                    "api_key": self.api_key,
                }))

                raw = await self._ws.recv()
                data = json.loads(raw)

                if data.get("type") == "registered":
                    self._connected = True
                    logger.info(f"Connected to server. Agent ID: {self.agent_id}")
                    return
                else:
                    logger.error(f"Registration failed: {data}")
                    await self._ws.close()
                    self._ws = None

            except Exception as e:
                logger.warning(f"Connection failed: {e}. Retrying in {self._reconnect_delay}s...")
                self._ws = None
                self._connected = False
                await asyncio.sleep(self._reconnect_delay)

    async def send_heartbeat(self, state: str, gpu_info: dict | None, idle_status: dict):
        if not self.connected:
            return False
        try:
            msg = {
                "type": "heartbeat",
                "agent_id": self.agent_id,
                "state": state,
                "gpu_info": gpu_info,
                "idle_status": idle_status,
            }
            await self._ws.send(json.dumps(msg))

            # After sending heartbeat, check for response
            # Server may send heartbeat_ack OR a job assignment
            raw = await self._ws.recv()
            data = json.loads(raw)

            if data.get("type") == "job_assign":
                logger.info(f"Received job assignment: {data.get('job_id', '')[:8]}")
                if self.on_job_received:
                    await self.on_job_received(data)
            elif data.get("type") == "heartbeat_ack":
                pass

            return True

        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            self._connected = False
            self._ws = None
            return False

    async def send_job_started(self, job_id: str):
        await self._send({"type": "job_started", "job_id": job_id})

    async def send_job_progress(self, job_id: str, progress: str):
        await self._send({"type": "job_progress", "job_id": job_id, "progress": progress})

    async def send_job_completed(self, job_id: str, result: str):
        await self._send({"type": "job_completed", "job_id": job_id, "result": result})

    async def send_job_failed(self, job_id: str, error: str):
        await self._send({"type": "job_failed", "job_id": job_id, "error": error})

    async def send_job_cancelled(self, job_id: str, reason: str):
        await self._send({"type": "job_cancelled", "job_id": job_id, "reason": reason})

    async def _send(self, data: dict):
        if not self.connected:
            return
        try:
            await self._ws.send(json.dumps(data))
        except Exception as e:
            logger.warning(f"Send failed: {e}")
            self._connected = False
            self._ws = None

    async def disconnect(self):
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False
            logger.info("Disconnected from server")
