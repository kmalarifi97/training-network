import logging
from dataclasses import dataclass

import pynvml

logger = logging.getLogger("agent.gpu_monitor")


@dataclass
class GPUInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    temperature_c: int
    utilization_percent: int


class GPUMonitor:
    def __init__(self):
        self._initialized = False
        self._handle = None

    def initialize(self):
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._initialized = True
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            logger.info(f"GPU Monitor initialized: {name}")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self._initialized = False

    def get_info(self) -> GPUInfo | None:
        if not self._initialized:
            return None
        try:
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)

            return GPUInfo(
                name=name,
                vram_total_mb=mem.total // (1024 * 1024),
                vram_used_mb=mem.used // (1024 * 1024),
                vram_free_mb=mem.free // (1024 * 1024),
                temperature_c=temp,
                utilization_percent=util.gpu,
            )
        except pynvml.NVMLError as e:
            logger.error(f"Error reading GPU info: {e}")
            return None

    def shutdown(self):
        if self._initialized:
            pynvml.nvmlShutdown()
            self._initialized = False
            logger.info("GPU Monitor shut down")
