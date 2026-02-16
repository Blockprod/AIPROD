"""
JobStore Souverain — stockage persistant des jobs vidéo via SQLite.

Architecture :
    API Gateway → job_store.enqueue(job) → SQLite → worker.dequeue() → GPU

Zéro dépendance cloud. Le fichier SQLite est local et portable.
Le store expose une interface thread-safe avec des opérations atomiques.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'queued',
    priority    INTEGER NOT NULL DEFAULT 0,
    prompt      TEXT NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',
    result      TEXT DEFAULT NULL,
    error       TEXT DEFAULT NULL,
    created_at  REAL NOT NULL,
    started_at  REAL DEFAULT NULL,
    finished_at REAL DEFAULT NULL,
    user_id     TEXT DEFAULT NULL,
    tier        TEXT DEFAULT 'free'
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
"""


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------


class JobStore:
    """
    Store persistant pour les jobs de génération vidéo.
    
    Utilise SQLite pour la persistance locale — survit aux redémarrages.
    Thread-safe grâce à un lock Python + SQLite WAL mode.
    """

    def __init__(self, db_path: str = "data/jobs.db"):
        self._db_path = db_path
        self._lock = threading.Lock()

        # Créer le dossier parent si nécessaire
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialiser la DB
        self._init_db()
        logger.info("JobStore initialized at %s", db_path)

    def _init_db(self) -> None:
        """Crée les tables si elles n'existent pas."""
        with self._get_connection() as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager pour une connexion SQLite thread-safe."""
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- CRUD Operations ---------------------------------------------------

    def enqueue(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        user_id: Optional[str] = None,
        tier: str = "free",
        job_id: Optional[str] = None,
    ) -> str:
        """
        Ajoute un job à la queue.
        
        Returns:
            job_id — identifiant unique du job
        """
        job_id = job_id or str(uuid.uuid4())
        params = params or {}

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (job_id, status, priority, prompt, params, created_at, user_id, tier)
                    VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        priority,
                        prompt,
                        json.dumps(params),
                        time.time(),
                        user_id,
                        tier,
                    ),
                )

        logger.info("Job %s enqueued (priority=%d, tier=%s)", job_id, priority, tier)
        return job_id

    def dequeue(self) -> Optional[Dict[str, Any]]:
        """
        Prend le prochain job en attente (FIFO avec priorité).
        
        Le job passe en status 'processing'.
        Returns None si la queue est vide.
        """
        with self._lock:
            with self._get_connection() as conn:
                # Récupérer le job de plus haute priorité, puis le plus ancien
                row = conn.execute(
                    """
                    SELECT * FROM jobs 
                    WHERE status = 'queued'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                    """
                ).fetchone()

                if row is None:
                    return None

                job_id = row["job_id"]
                conn.execute(
                    """
                    UPDATE jobs SET status = 'processing', started_at = ?
                    WHERE job_id = ?
                    """,
                    (time.time(), job_id),
                )

        job = self._row_to_dict(row)
        job["status"] = "processing"  # Refléter la mise à jour
        logger.info("Job %s dequeued for processing", job_id)
        return job

    def update_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Met à jour le statut et le résultat d'un job."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE jobs SET 
                        status = ?,
                        result = ?,
                        error = ?,
                        finished_at = CASE WHEN ? IN ('completed', 'failed', 'cancelled') 
                                      THEN ? ELSE finished_at END
                    WHERE job_id = ?
                    """,
                    (
                        status,
                        json.dumps(result) if result else None,
                        error,
                        status,
                        time.time(),
                        job_id,
                    ),
                )
        logger.info("Job %s → %s", job_id, status)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un job par son ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def list_pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Liste les jobs en attente."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_all(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste tous les jobs avec filtre optionnel par statut."""
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_by_status(self) -> Dict[str, int]:
        """Compte les jobs par statut."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
            ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def cancel_job(self, job_id: str) -> bool:
        """Annule un job s'il est encore en attente."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE jobs SET status = 'cancelled', finished_at = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (time.time(), job_id),
                )
                return cursor.rowcount > 0

    def cleanup_stale(self, timeout_sec: float = 3600) -> int:
        """
        Remet en queue les jobs 'processing' qui sont bloqués depuis plus de timeout_sec.
        Utile en cas de crash du worker.
        """
        cutoff = time.time() - timeout_sec
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE jobs SET status = 'queued', started_at = NULL
                    WHERE status = 'processing' AND started_at < ?
                    """,
                    (cutoff,),
                )
                count = cursor.rowcount
        if count > 0:
            logger.warning("Reset %d stale jobs back to queued", count)
        return count

    def purge_completed(self, older_than_sec: float = 86400 * 7) -> int:
        """Supprime les jobs terminés plus vieux que older_than_sec."""
        cutoff = time.time() - older_than_sec
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM jobs 
                    WHERE status IN ('completed', 'failed', 'cancelled') 
                    AND finished_at < ?
                    """,
                    (cutoff,),
                )
                count = cursor.rowcount
        if count > 0:
            logger.info("Purged %d old jobs", count)
        return count

    # ---- Utility -----------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convertit un Row SQLite en dict, avec parsing JSON des champs."""
        d = dict(row)
        # Parser les champs JSON
        if d.get("params"):
            try:
                d["params"] = json.loads(d["params"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def stats(self) -> Dict[str, Any]:
        """Statistiques du store."""
        counts = self.count_by_status()
        return {
            "total_jobs": sum(counts.values()),
            "by_status": counts,
            "db_path": self._db_path,
        }
