"""SQLite persistence layer for GPU Network Server.

Replaces in-memory dicts with a real database so agents and jobs
survive server restarts.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

DB_PATH = os.environ.get("GPUNET_DB_PATH", str(Path(__file__).parent / "gpunetwork.db"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables on startup. Safe to call multiple times."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cafes (
                cafe_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                api_key TEXT NOT NULL UNIQUE,
                owner_name TEXT DEFAULT '',
                location TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                cafe_id TEXT NOT NULL,
                gpu_info TEXT,
                idle_status TEXT,
                state TEXT DEFAULT 'OFFLINE',
                connected_at TEXT,
                last_heartbeat TEXT,
                FOREIGN KEY (cafe_id) REFERENCES cafes(cafe_id)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt TEXT DEFAULT '',
                status TEXT DEFAULT 'PENDING',
                assigned_to TEXT,
                result TEXT,
                error TEXT,
                progress TEXT,
                cancel_reason TEXT,
                submitted_at TEXT NOT NULL,
                assigned_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                cancelled_at TEXT,
                submitted_by TEXT DEFAULT 'api'
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_assigned ON jobs(assigned_to);
            CREATE INDEX IF NOT EXISTS idx_agents_cafe ON agents(cafe_id);

            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'PENDING',
                source_filename TEXT,
                source_type TEXT,
                model_name TEXT,
                strategy TEXT DEFAULT 'self-instruct',
                output_format TEXT DEFAULT 'alpaca',
                chunk_size INTEGER,
                pairs_per_chunk INTEGER,
                total_chunks INTEGER DEFAULT 0,
                processed_chunks INTEGER DEFAULT 0,
                total_pairs INTEGER DEFAULT 0,
                output_file TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);

            CREATE TABLE IF NOT EXISTS finetune_runs (
                run_id TEXT PRIMARY KEY,
                base_model TEXT NOT NULL,
                dataset_id TEXT,
                dataset_file TEXT,
                status TEXT DEFAULT 'PENDING',
                job_id TEXT,
                -- Training config
                epochs INTEGER DEFAULT 3,
                batch_size INTEGER DEFAULT 4,
                learning_rate REAL DEFAULT 0.0002,
                lora_r INTEGER DEFAULT 16,
                lora_alpha INTEGER DEFAULT 32,
                max_steps INTEGER DEFAULT -1,
                -- Progress
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER DEFAULT 0,
                current_epoch REAL DEFAULT 0,
                current_loss REAL,
                -- Output
                adapter_path TEXT,
                training_log TEXT,
                error TEXT,
                -- Timestamps
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_finetune_status ON finetune_runs(status);
        """)


# --- Cafe operations ---

def create_cafe(cafe_id: str, name: str, api_key: str, owner_name: str = "", location: str = "") -> dict:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO cafes (cafe_id, name, api_key, owner_name, location, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cafe_id, name, api_key, owner_name, location, datetime.now().isoformat()),
        )
    return {"cafe_id": cafe_id, "name": name, "api_key": api_key}


def get_cafe_by_api_key(api_key: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM cafes WHERE api_key = ? AND active = 1", (api_key,)).fetchone()
    return dict(row) if row else None


def get_cafe(cafe_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM cafes WHERE cafe_id = ?", (cafe_id,)).fetchone()
    return dict(row) if row else None


def list_cafes() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM cafes WHERE active = 1").fetchall()
    return [dict(r) for r in rows]


# --- Agent operations ---

def upsert_agent(agent_id: str, cafe_id: str, state: str = "OFFLINE") -> dict:
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO agents (agent_id, cafe_id, state, connected_at, last_heartbeat)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                cafe_id = excluded.cafe_id,
                state = excluded.state,
                connected_at = excluded.connected_at,
                last_heartbeat = excluded.last_heartbeat
        """, (agent_id, cafe_id, state, now, now))
    return {"agent_id": agent_id, "cafe_id": cafe_id, "state": state}


def update_agent_heartbeat(agent_id: str, state: str, gpu_info: dict | None, idle_status: dict | None):
    with get_db() as conn:
        conn.execute("""
            UPDATE agents SET state = ?, gpu_info = ?, idle_status = ?, last_heartbeat = ?
            WHERE agent_id = ?
        """, (
            state,
            json.dumps(gpu_info) if gpu_info else None,
            json.dumps(idle_status) if idle_status else None,
            datetime.now().isoformat(),
            agent_id,
        ))


def set_agent_offline(agent_id: str):
    with get_db() as conn:
        conn.execute("UPDATE agents SET state = 'OFFLINE' WHERE agent_id = ?", (agent_id,))


def get_agent(agent_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["gpu_info"] = json.loads(d["gpu_info"]) if d["gpu_info"] else None
    d["idle_status"] = json.loads(d["idle_status"]) if d["idle_status"] else None
    return d


def list_agents(online_only: bool = False) -> list[dict]:
    with get_db() as conn:
        if online_only:
            rows = conn.execute("SELECT * FROM agents WHERE state != 'OFFLINE'").fetchall()
        else:
            rows = conn.execute("SELECT * FROM agents").fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["gpu_info"] = json.loads(d["gpu_info"]) if d["gpu_info"] else None
        d["idle_status"] = json.loads(d["idle_status"]) if d["idle_status"] else None
        results.append(d)
    return results


def delete_agent(agent_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))


# --- Job operations ---

def create_job(job_id: str, job_type: str, model_name: str, prompt: str, submitted_by: str = "api") -> dict:
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO jobs (job_id, job_type, model_name, prompt, status, submitted_at, submitted_by)
            VALUES (?, ?, ?, ?, 'PENDING', ?, ?)
        """, (job_id, job_type, model_name, prompt, now, submitted_by))
    return {
        "job_id": job_id, "job_type": job_type, "model_name": model_name,
        "prompt": prompt, "status": "PENDING", "submitted_at": now,
    }


def assign_job(job_id: str, agent_id: str):
    with get_db() as conn:
        conn.execute("""
            UPDATE jobs SET status = 'ASSIGNED', assigned_to = ?, assigned_at = ?
            WHERE job_id = ?
        """, (agent_id, datetime.now().isoformat(), job_id))


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status with optional extra fields (result, error, progress, cancel_reason)."""
    now = datetime.now().isoformat()
    sets = ["status = ?"]
    vals = [status]

    if status == "RUNNING":
        sets.append("started_at = ?")
        vals.append(now)
    elif status == "COMPLETED":
        sets.append("completed_at = ?")
        vals.append(now)
    elif status == "CANCELLED":
        sets.append("cancelled_at = ?")
        vals.append(now)

    for key in ("result", "error", "progress", "cancel_reason", "assigned_to"):
        if key in kwargs:
            sets.append(f"{key} = ?")
            vals.append(kwargs[key])

    vals.append(job_id)
    with get_db() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals)


def get_job(job_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(status: str | None = None, limit: int = 100) -> list[dict]:
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY submitted_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_pending_jobs() -> list[dict]:
    return list_jobs(status="PENDING")


def delete_job(job_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))


def delete_all_jobs():
    with get_db() as conn:
        conn.execute("DELETE FROM jobs")


def requeue_agent_jobs(agent_id: str):
    """Return all in-flight jobs for an agent back to PENDING."""
    with get_db() as conn:
        conn.execute("""
            UPDATE jobs SET status = 'PENDING', assigned_to = NULL
            WHERE assigned_to = ? AND status IN ('ASSIGNED', 'RUNNING')
        """, (agent_id,))


# --- Dataset operations ---

def create_dataset(dataset_id: str, source_filename: str, source_type: str,
                   model_name: str, strategy: str, output_format: str,
                   chunk_size: int, pairs_per_chunk: int) -> dict:
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO datasets (dataset_id, source_filename, source_type, model_name,
                                  strategy, output_format, chunk_size, pairs_per_chunk, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset_id, source_filename, source_type, model_name,
              strategy, output_format, chunk_size, pairs_per_chunk, now))
    return {"dataset_id": dataset_id, "status": "PENDING", "created_at": now}


def update_dataset(dataset_id: str, **kwargs):
    sets = []
    vals = []
    for key, val in kwargs.items():
        sets.append(f"{key} = ?")
        vals.append(val)
    vals.append(dataset_id)
    with get_db() as conn:
        conn.execute(f"UPDATE datasets SET {', '.join(sets)} WHERE dataset_id = ?", vals)


def get_dataset(dataset_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    return dict(row) if row else None


def list_datasets(status: str | None = None, limit: int = 100) -> list[dict]:
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def delete_dataset(dataset_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))


# --- Fine-tune run operations ---

def create_finetune_run(run_id: str, base_model: str, dataset_file: str,
                        dataset_id: str | None = None,
                        epochs: int = 3, batch_size: int = 4,
                        learning_rate: float = 2e-4,
                        lora_r: int = 16, lora_alpha: int = 32,
                        max_steps: int = -1) -> dict:
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO finetune_runs (run_id, base_model, dataset_id, dataset_file,
                                       epochs, batch_size, learning_rate,
                                       lora_r, lora_alpha, max_steps, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, base_model, dataset_id, dataset_file,
              epochs, batch_size, learning_rate,
              lora_r, lora_alpha, max_steps, now))
    return {"run_id": run_id, "status": "PENDING", "created_at": now}


def update_finetune_run(run_id: str, **kwargs):
    sets = []
    vals = []
    for key, val in kwargs.items():
        sets.append(f"{key} = ?")
        vals.append(val)
    vals.append(run_id)
    with get_db() as conn:
        conn.execute(f"UPDATE finetune_runs SET {', '.join(sets)} WHERE run_id = ?", vals)


def get_finetune_run(run_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM finetune_runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_finetune_runs(status: str | None = None, limit: int = 100) -> list[dict]:
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM finetune_runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM finetune_runs ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def delete_finetune_run(run_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM finetune_runs WHERE run_id = ?", (run_id,))
