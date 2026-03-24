"""Resource limiter — checks GPU/RAM/disk before and during jobs.

Usage:
    limiter = ResourceLimiter(config)
    ok, reason = limiter.check_pre_job(required_vram_mb=4000)
    if not ok:
        reject_job(reason)
"""

import logging
import shutil
from pathlib import Path

import psutil

logger = logging.getLogger("agent.resource_limiter")


class ResourceLimiter:
    def __init__(self, config: dict):
        self.max_gpu_percent = config.get("max_gpu_usage_percent", 90)
        self.max_ram_percent = config.get("max_ram_usage_percent", 80)
        self.max_disk_gb = config.get("max_disk_usage_gb", 50)
        self.model_cache_dir = Path(config.get("model_cache_dir", "."))

    def check_pre_job(self, required_vram_mb: int = 0) -> tuple[bool, str]:
        """Check if system has enough resources to start a job.
        Returns (ok, reason).
        """
        # Check RAM
        ram = psutil.virtual_memory()
        if ram.percent > self.max_ram_percent:
            msg = f"RAM usage too high: {ram.percent}% (limit: {self.max_ram_percent}%)"
            logger.warning(msg)
            return False, msg

        # Check disk space for model cache
        try:
            disk = shutil.disk_usage(str(self.model_cache_dir))
            free_gb = disk.free / (1024 ** 3)
            if free_gb < 5:  # need at least 5GB free
                msg = f"Disk space too low: {free_gb:.1f}GB free (need 5GB+)"
                logger.warning(msg)
                return False, msg
        except OSError:
            pass  # directory might not exist yet

        # Check GPU VRAM if pynvml available
        if required_vram_mb > 0:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mb = mem.free // (1024 * 1024)
                if free_mb < required_vram_mb:
                    msg = f"Not enough VRAM: {free_mb}MB free, need {required_vram_mb}MB"
                    logger.warning(msg)
                    return False, msg
            except Exception:
                pass  # no GPU or pynvml not available

        logger.info("Resource check passed")
        return True, ""

    def check_during_job(self) -> tuple[bool, str]:
        """Lightweight check during job execution.
        Returns (ok, reason) — if not ok, job should be paused or cancelled.
        """
        ram = psutil.virtual_memory()
        if ram.percent > 95:
            return False, f"RAM critical: {ram.percent}%"

        try:
            disk = shutil.disk_usage(str(self.model_cache_dir))
            free_gb = disk.free / (1024 ** 3)
            if free_gb < 1:
                return False, f"Disk critically low: {free_gb:.1f}GB"
        except OSError:
            pass

        return True, ""

    def get_status(self) -> dict:
        """Return current resource usage snapshot."""
        ram = psutil.virtual_memory()
        try:
            disk = shutil.disk_usage(str(self.model_cache_dir))
            disk_info = {
                "total_gb": round(disk.total / (1024 ** 3), 1),
                "used_gb": round(disk.used / (1024 ** 3), 1),
                "free_gb": round(disk.free / (1024 ** 3), 1),
            }
        except OSError:
            disk_info = None

        return {
            "ram_percent": ram.percent,
            "ram_available_mb": ram.available // (1024 * 1024),
            "disk": disk_info,
            "limits": {
                "max_gpu_percent": self.max_gpu_percent,
                "max_ram_percent": self.max_ram_percent,
                "max_disk_gb": self.max_disk_gb,
            },
        }
