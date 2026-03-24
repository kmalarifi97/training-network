import asyncio
import enum
import logging
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from config import load_config
from connection import ServerConnection
from gpu_monitor import GPUMonitor
from idle_detector import IdleDetector
from job_runner import JobRunner
from resource_limiter import ResourceLimiter


# --- Logging setup ---
def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "agent.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


logger = logging.getLogger("agent.main")


# --- Agent States ---
class AgentState(enum.Enum):
    OFFLINE = "OFFLINE"
    AVAILABLE = "AVAILABLE"
    WORKING = "WORKING"
    BUSY = "BUSY"


class Agent:
    def __init__(self):
        self.config = load_config()
        setup_logging(self.config["log_dir"])

        self.state = AgentState.OFFLINE
        self.gpu_monitor = GPUMonitor()
        self.idle_detector = IdleDetector(
            idle_threshold_minutes=self.config["idle_threshold_minutes"]
        )
        self.connection = ServerConnection(
            server_url=self.config["server_url"],
            cafe_id=self.config["cafe_id"],
            api_key=self.config["api_key"],
        )
        self.job_runner = JobRunner(model_cache_dir=self.config["model_cache_dir"])
        self.resource_limiter = ResourceLimiter(self.config)
        self.heartbeat_interval = self.config["heartbeat_interval_seconds"]
        self._running = False
        self._job_task = None  # asyncio task for running job

        # Wire up job callback
        self.connection.on_job_received = self._on_job_received

    def _set_state(self, new_state: AgentState):
        if new_state != self.state:
            old = self.state.value
            self.state = new_state
            logger.info(f"State: {old} -> {new_state.value}")

    def _evaluate_state(self):
        user_active = self.idle_detector.is_user_active()

        if self.state == AgentState.WORKING:
            if user_active:
                # Gamer returned — cancel job immediately
                logger.info("Gamer returned! Cancelling current job...")
                self.job_runner.cancel()
                self._set_state(AgentState.BUSY)
        elif self.state == AgentState.BUSY:
            if not user_active:
                self._set_state(AgentState.AVAILABLE)
        elif self.state == AgentState.AVAILABLE:
            if user_active:
                self._set_state(AgentState.BUSY)
        elif self.state == AgentState.OFFLINE:
            if not user_active:
                self._set_state(AgentState.AVAILABLE)
            else:
                self._set_state(AgentState.BUSY)

    async def _on_job_received(self, job_data: dict):
        if self.state != AgentState.AVAILABLE:
            logger.warning(f"Received job but state is {self.state.value}, rejecting")
            await self.connection.send_job_cancelled(job_data["job_id"], "Agent not available")
            return

        # Pre-job resource check
        ok, reason = self.resource_limiter.check_pre_job()
        if not ok:
            logger.warning(f"Resource check failed: {reason}")
            await self.connection.send_job_cancelled(job_data["job_id"], f"Resources insufficient: {reason}")
            return

        self._set_state(AgentState.WORKING)

        # Notify server we started
        await self.connection.send_job_started(job_data["job_id"])

        # Run job in background
        self._job_task = asyncio.create_task(self._execute_job(job_data))

    async def _execute_job(self, job_data: dict):
        job_id = job_data["job_id"]

        async def progress_callback(jid: str, progress: str):
            await self.connection.send_job_progress(jid, progress)

        result = await self.job_runner.run_job(job_data, progress_callback)

        # Send result to server
        if result["status"] == "completed":
            await self.connection.send_job_completed(job_id, result.get("result", ""))
        elif result["status"] == "cancelled":
            await self.connection.send_job_cancelled(job_id, result.get("reason", ""))
        elif result["status"] == "failed":
            await self.connection.send_job_failed(job_id, result.get("error", ""))

        # Return to appropriate state
        if self.state == AgentState.WORKING:
            self._set_state(AgentState.AVAILABLE)
        self._job_task = None

    async def _heartbeat(self):
        gpu = self.gpu_monitor.get_info()
        idle_status = self.idle_detector.get_status()
        gpu_dict = asdict(gpu) if gpu else None

        if gpu:
            logger.info(
                f"[HEARTBEAT] State={self.state.value} | "
                f"GPU={gpu.name} | "
                f"VRAM={gpu.vram_used_mb}/{gpu.vram_total_mb}MB | "
                f"Temp={gpu.temperature_c}C | "
                f"GPU_Util={gpu.utilization_percent}% | "
                f"Idle={idle_status['idle_seconds']}s | "
                f"UserActive={idle_status['user_active']}"
            )
        else:
            logger.info(
                f"[HEARTBEAT] State={self.state.value} | GPU=unavailable | "
                f"Idle={idle_status['idle_seconds']}s"
            )

        if self.connection.connected:
            await self.connection.send_heartbeat(
                state=self.state.value,
                gpu_info=gpu_dict,
                idle_status=idle_status,
            )

    async def run(self):
        logger.info("=" * 60)
        logger.info("GPU Network Agent starting...")
        logger.info(f"Cafe ID: {self.config['cafe_id']}")
        logger.info(f"Heartbeat interval: {self.heartbeat_interval}s")
        logger.info(f"Idle threshold: {self.config['idle_threshold_minutes']} min")
        logger.info("=" * 60)

        self.gpu_monitor.initialize()
        self._running = True

        await self.connection.connect()

        self._evaluate_state()
        await self._heartbeat()

        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                self._evaluate_state()
                await self._heartbeat()

                if not self.connection.connected:
                    logger.warning("Lost connection to server. Reconnecting...")
                    await self.connection.connect()
        except asyncio.CancelledError:
            pass
        finally:
            if self._job_task:
                self.job_runner.cancel()
                self._job_task.cancel()
            await self.connection.disconnect()
            self.gpu_monitor.shutdown()
            logger.info("Agent stopped.")

    def stop(self):
        self._running = False
        self._set_state(AgentState.OFFLINE)


def main():
    agent = Agent()

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal...")
        agent.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    main()
