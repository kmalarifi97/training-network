import hashlib
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("agent.model_cache")


class ModelCache:
    def __init__(self, cache_dir: str, max_disk_gb: int = 50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = max_disk_gb * 1024 * 1024 * 1024

    def get_model_path(self, model_name: str) -> Path | None:
        path = self.cache_dir / model_name
        if path.exists():
            logger.info(f"Model '{model_name}' found in cache")
            return path
        return None

    def save_model(self, model_name: str, data: bytes) -> Path:
        self._evict_if_needed(len(data))
        path = self.cache_dir / model_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.info(f"Model '{model_name}' saved to cache ({len(data)} bytes)")
        return path

    def get_cache_size(self) -> int:
        total = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def _evict_if_needed(self, incoming_bytes: int):
        current = self.get_cache_size()
        if current + incoming_bytes <= self.max_disk_bytes:
            return

        # LRU eviction — delete oldest accessed files first
        files = []
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                files.append((f, f.stat().st_atime, f.stat().st_size))
        files.sort(key=lambda x: x[1])  # oldest first

        freed = 0
        needed = (current + incoming_bytes) - self.max_disk_bytes
        for f, _, size in files:
            f.unlink()
            freed += size
            logger.info(f"Evicted '{f.name}' from cache ({size} bytes)")
            if freed >= needed:
                break

    def list_models(self) -> list[str]:
        return [f.name for f in self.cache_dir.iterdir() if f.is_file()]
