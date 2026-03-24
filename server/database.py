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


def requeue_agent_jobs(agent_id: str):
    """Return all in-flight jobs for an agent back to PENDING."""
    with get_db() as conn:
        conn.execute("""
            UPDATE jobs SET status = 'PENDING', assigned_to = NULL
            WHERE assigned_to = ? AND status IN ('ASSIGNED', 'RUNNING')
        """, (agent_id,))
