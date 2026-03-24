import json
import os
import sys
from pathlib import Path


def _default_data_dir() -> str:
    """Platform-appropriate data directory."""
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA", "C:\\GPUNetwork")
        return str(Path(base) / "GPUNetwork")
    else:
        return str(Path.home() / ".gpunetwork")


def _default_config() -> dict:
    data_dir = _default_data_dir()
    return {
        "server_url": "ws://localhost:8000/agents/connect",
        "cafe_id": "test-cafe-001",
        "api_key": "",
        "idle_threshold_minutes": 5,
        "max_gpu_usage_percent": 90,
        "max_ram_usage_percent": 80,
        "max_disk_usage_gb": 50,
        "model_cache_dir": str(Path(data_dir) / "models"),
        "log_dir": str(Path(data_dir) / "logs"),
        "heartbeat_interval_seconds": 30,
    }


# Search order for config.json:
# 1. GPUNET_CONFIG env var
# 2. Next to the executable (for PyInstaller builds)
# 3. Next to this script (agent/config.json for dev)
def _find_config_path() -> Path | None:
    env_path = os.environ.get("GPUNET_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # PyInstaller frozen exe directory
    if getattr(sys, "frozen", False):
        p = Path(sys.executable).parent / "config.json"
        if p.exists():
            return p

    # Dev: agent/config.json
    p = Path(__file__).parent / "config.json"
    if p.exists():
        return p

    return None


def load_config() -> dict:
    config = _default_config()

    # Load from file
    config_path = _find_config_path()
    if config_path:
        with open(config_path, "r") as f:
            user_config = json.load(f)
        config.update(user_config)

    # Environment variable overrides (GPUNET_SERVER_URL, GPUNET_CAFE_ID, etc.)
    env_map = {
        "GPUNET_SERVER_URL": "server_url",
        "GPUNET_CAFE_ID": "cafe_id",
        "GPUNET_API_KEY": "api_key",
        "GPUNET_IDLE_THRESHOLD": ("idle_threshold_minutes", float),
        "GPUNET_HEARTBEAT_INTERVAL": ("heartbeat_interval_seconds", int),
        "GPUNET_MODEL_DIR": "model_cache_dir",
        "GPUNET_LOG_DIR": "log_dir",
        "GPUNET_MAX_GPU_PERCENT": ("max_gpu_usage_percent", int),
        "GPUNET_MAX_RAM_PERCENT": ("max_ram_usage_percent", int),
        "GPUNET_MAX_DISK_GB": ("max_disk_usage_gb", int),
    }

    for env_key, target in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if isinstance(target, tuple):
                config_key, cast = target
                config[config_key] = cast(val)
            else:
                config[target] = val

    return config


def save_default_config():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(_default_config(), f, indent=2)
        print(f"Default config written to {config_path}")
