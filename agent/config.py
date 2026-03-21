import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "server_url": "ws://localhost:8000/agents/connect",
    "cafe_id": "test-cafe-001",
    "api_key": "",
    "idle_threshold_minutes": 5,
    "max_gpu_usage_percent": 90,
    "max_ram_usage_percent": 80,
    "max_disk_usage_gb": 50,
    "model_cache_dir": "C:\\GPUNetwork\\models",
    "log_dir": "C:\\GPUNetwork\\logs",
    "heartbeat_interval_seconds": 30,
}

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def save_default_config():
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Default config written to {CONFIG_PATH}")
