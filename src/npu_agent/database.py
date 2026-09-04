from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import Candidate, CompileResult, Evaluation, KernelManifest, TargetProfile, TranslationRequest


_VOLATILE_ERROR = re.compile(r"(?:0x[0-9a-f]+|/[^\s:]+|\b\d{2,}\b)", re.IGNORECASE)


def normalize_error(value: str) -> str:
    normalized = _VOLATILE_ERROR.sub("<value>", value.lower())
    normalized = " ".join(normalized.split())[:2000]
    return hashlib.sha256(normalized.encode()).hexdigest()


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self.seed_builtin_knowledge()

    def _create_schema(self) -> None:
        with self.lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_runs (
                    id TEXT PRIMARY KEY,
                    kernel_name TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    experiment_id TEXT,
                    method TEXT NOT NULL DEFAULT 'agentic',
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES app_runs(id),
                    target_id TEXT NOT NULL,
                    parent_id TEXT,
                    depth INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    bundle_json TEXT NOT NULL,
                    debug_attempt INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    candidate_id TEXT PRIMARY KEY REFERENCES candidates(id),
                    evaluation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS compile_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidates(id),
                    attempt INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    exit_code INTEGER NOT NULL,
                    compiler_fingerprint TEXT NOT NULL,
                    error_fingerprint TEXT,
                    stdout TEXT NOT NULL,
                    stderr TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    evaluation_duration_seconds REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidates(id),
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS agent_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES app_runs(id),
                    role TEXT NOT NULL,
                    target_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT,
                    executable TEXT,
                    provider_version TEXT,
                    prompt_sha256 TEXT NOT NULL,
                    schema_sha256 TEXT NOT NULL,
                    exit_code INTEGER NOT NULL,
                    schema_valid INTEGER NOT NULL DEFAULT 1,
                    usage_json TEXT NOT NULL,
                    telemetry_json TEXT NOT NULL DEFAULT '[]',
                    response_json TEXT NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_role TEXT NOT NULL,
                    dialect TEXT,
                    target_id TEXT,
                    vendor TEXT,
                    backend TEXT,
                    hardware TEXT,
                    compiler_fingerprint TEXT,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    source_url TEXT,
                    source_revision TEXT,
                    content_sha256 TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    item_id UNINDEXED, title, body, tags
                );
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    hardware TEXT NOT NULL,
                    compiler_fingerprint TEXT NOT NULL,
                    operation_tags TEXT NOT NULL,
                    error_fingerprint TEXT NOT NULL,
                    diagnosis TEXT NOT NULL,
                    fix_summary TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
                    lesson_id UNINDEXED, diagnosis, fix_summary, operation_tags
                );
                PRAGMA user_version=6;
                """
            )
            migrations = {
                "app_runs": {
                    "experiment_id": "TEXT",
                    "method": "TEXT NOT NULL DEFAULT 'agentic'",
                    "duration_seconds": "REAL",
                },
                "compile_attempts": {
                    "duration_seconds": "REAL NOT NULL DEFAULT 0",
                    "evaluation_duration_seconds": "REAL NOT NULL DEFAULT 0",
                    "result_json": "TEXT NOT NULL DEFAULT '{}'",
                },
                "agent_calls": {
                    "target_id": "TEXT",
                    "telemetry_json": "TEXT NOT NULL DEFAULT '[]'",
                    "duration_seconds": "REAL NOT NULL DEFAULT 0",
                    "schema_valid": "INTEGER NOT NULL DEFAULT 1",
                },
                "knowledge_items": {
                    "target_id": "TEXT",
                    "vendor": "TEXT",
                    "compiler_fingerprint": "TEXT",
                    "source_url": "TEXT",
                    "source_revision": "TEXT",
                    "content_sha256": "TEXT",
                },
            }
            for table, columns in migrations.items():
                existing = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
                for column, definition in columns.items():
                    if column not in existing:
                        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            lesson_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(lessons)")}
            if "vendor" not in lesson_columns:
                self.connection.execute("ALTER TABLE lessons ADD COLUMN vendor TEXT NOT NULL DEFAULT ''")
                self.connection.execute(
                    "UPDATE lessons SET vendor=CASE WHEN backend='amd_xdna2' THEN 'amd' ELSE 'intel' END"
                )

    def start_run(
        self,
        run_id: str,
        manifest: KernelManifest,
        request: TranslationRequest,
        method: str = "agentic",
        experiment_id: str | None = None,
    ) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO app_runs(id, kernel_name, request_json, experiment_id, method, status)
                   VALUES (?, ?, ?, ?, ?, 'running')""",
                (run_id, manifest.name, request.model_dump_json(), experiment_id, method),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """UPDATE app_runs SET status=?, result_json=?, error=?, duration_seconds=?,
                   updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (status, json.dumps(result) if result is not None else None, error, duration_seconds, run_id),
            )

    def add_candidate(self, candidate: Candidate) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO candidates
                   (id, run_id, target_id, parent_id, depth, label, rationale, bundle_json, debug_attempt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.id,
                    candidate.run_id,
                    candidate.target_id,
                    candidate.parent_id,
                    candidate.depth,
                    candidate.label,
                    candidate.rationale,
                    candidate.bundle.model_dump_json(),
                    candidate.debug_attempt,
                ),
            )

    def add_evaluation(self, evaluation: Evaluation) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO evaluations(candidate_id, evaluation_json) VALUES (?, ?)",
                (evaluation.candidate_id, evaluation.model_dump_json()),
            )

    def add_compile_attempt(self, candidate_id: str, attempt: int, result: CompileResult) -> None:
        error_fingerprint = normalize_error(result.stderr) if result.stderr else None
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO compile_attempts
                   (candidate_id, attempt, success, exit_code, compiler_fingerprint, error_fingerprint,
                    stdout, stderr, artifacts_json, result_json, duration_seconds, evaluation_duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_id,
                    attempt,
                    result.success,
                    result.exit_code,
                    result.compiler_fingerprint,
                    error_fingerprint,
                    result.stdout,
                    result.stderr,
                    json.dumps(result.artifacts, sort_keys=True),
                    result.model_dump_json(),
                    result.duration_seconds,
                    result.evaluation_duration_seconds,
                ),
            )

    def add_artifact(
        self,
        candidate_id: str,
        kind: str,
        path: str,
        sha256: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO artifacts(candidate_id, kind, path, sha256, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (candidate_id, kind, path, sha256, json.dumps(metadata or {}, sort_keys=True)),
            )

    def add_agent_call(
        self,
        run_id: str,
        role: str,
        metadata: dict[str, Any],
        response: dict[str, Any],
        target_id: str | None = None,
    ) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO agent_calls
                   (run_id, role, target_id, provider, model, executable, provider_version, prompt_sha256,
                    schema_sha256, exit_code, schema_valid, usage_json, telemetry_json, response_json,
                    duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    role,
                    target_id,
                    metadata["provider"],
                    metadata.get("model"),
                    metadata.get("executable"),
                    metadata.get("provider_version"),
                    metadata["prompt_sha256"],
                    metadata["schema_sha256"],
                    metadata.get("exit_code", 0),
                    metadata.get("schema_valid", True),
                    json.dumps(metadata.get("usage", {}), sort_keys=True),
                    json.dumps(metadata.get("telemetry", []), sort_keys=True),
                    json.dumps(response, sort_keys=True),
                    metadata.get("duration_seconds", 0.0),
                ),
            )

    def import_knowledge(
        self,
        role: str,
        title: str,
        body: str,
        tags: str = "",
        dialect: str | None = None,
        backend: str | None = None,
        hardware: str | None = None,
        *,
        target_id: str | None = None,
        vendor: str | None = None,
        compiler_fingerprint: str | None = None,
        source_url: str | None = None,
        source_revision: str | None = None,
        content_sha256: str | None = None,
    ) -> int:
        digest = content_sha256 or hashlib.sha256(body.encode()).hexdigest()
        with self.lock, self.connection:
            cursor = self.connection.execute(
                """INSERT INTO knowledge_items
                   (agent_role, dialect, target_id, vendor, backend, hardware, compiler_fingerprint,
                    title, body, tags, source_url, source_revision, content_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    role, dialect, target_id, vendor, backend, hardware, compiler_fingerprint,
                    title, body, tags, source_url, source_revision, digest,
                ),
            )
            item_id = int(cursor.lastrowid)
            self.connection.execute(
                "INSERT INTO knowledge_fts(item_id, title, body, tags) VALUES (?, ?, ?, ?)",
                (item_id, title, body, tags),
            )
        return item_id

    def retrieve_knowledge(
        self,
        role: str,
        query: str,
        dialect: str | None = None,
        backend: str | None = None,
        hardware: str | None = None,
        target_id: str | None = None,
        vendor: str | None = None,
        compiler_fingerprint: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)[:12] or ["kernel"]
        terms = " OR ".join(f'"{token}"' for token in tokens)
        sql = """
            SELECT k.*, bm25(knowledge_fts) AS score
            FROM knowledge_fts JOIN knowledge_items k ON k.id=knowledge_fts.item_id
            WHERE knowledge_fts MATCH ? AND k.enabled=1 AND k.agent_role=?
              AND (k.dialect IS NULL OR k.dialect=?)
              AND (k.target_id IS NULL OR k.target_id=?)
              AND (k.vendor IS NULL OR k.vendor=?)
              AND (k.backend IS NULL OR k.backend=?)
              AND (k.hardware IS NULL OR k.hardware=?)
              AND (k.compiler_fingerprint IS NULL OR k.compiler_fingerprint=?)
            ORDER BY score LIMIT ?
        """
        with self.lock:
            rows = self.connection.execute(
                sql,
                (terms, role, dialect, target_id, vendor, backend, hardware, compiler_fingerprint, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def seed_builtin_knowledge(self) -> dict[str, int]:
        catalog = json.loads(
            importlib.resources.files("npu_agent").joinpath("data/knowledge.json").read_text(encoding="utf-8")
        )
        inserted = 0
        skipped = 0
        for item in catalog["entries"]:
            digest = hashlib.sha256(item["body"].encode()).hexdigest()
            if item.get("content_sha256") != digest:
                raise ValueError(f"knowledge content hash mismatch: {item['title']}")
            for role in item["roles"]:
                with self.lock:
                    exists = self.connection.execute(
                        """SELECT 1 FROM knowledge_items
                           WHERE agent_role=? AND target_id IS ? AND content_sha256=?""",
                        (role, item.get("target_id"), digest),
                    ).fetchone()
                if exists:
                    skipped += 1
                    continue
                self.import_knowledge(
                    role,
                    item["title"],
                    item["body"],
                    item.get("tags", ""),
                    item.get("dialect"),
                    item.get("backend"),
                    item.get("hardware"),
                    target_id=item.get("target_id"),
                    vendor=item.get("vendor"),
                    compiler_fingerprint=item.get("compiler_fingerprint"),
                    source_url=item.get("source_url"),
                    source_revision=item.get("source_revision"),
                    content_sha256=digest,
                )
                inserted += 1
        return {"inserted": inserted, "skipped": skipped}

    def add_lesson(
        self,
        target: TargetProfile,
        compiler_fingerprint: str,
        operation_tags: list[str],
        error_fingerprint: str,
        diagnosis: str,
        fix_summary: str,
        verified: bool,
    ) -> int:
        tags = " ".join(operation_tags)
        with self.lock, self.connection:
            cursor = self.connection.execute(
                """INSERT INTO lessons
                   (vendor, backend, hardware, compiler_fingerprint, operation_tags, error_fingerprint,
                    diagnosis, fix_summary, verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    target.vendor,
                    target.backend.value,
                    target.hardware,
                    compiler_fingerprint,
                    tags,
                    error_fingerprint,
                    diagnosis,
                    fix_summary,
                    verified,
                ),
            )
            lesson_id = int(cursor.lastrowid)
            self.connection.execute(
                "INSERT INTO lessons_fts(lesson_id, diagnosis, fix_summary, operation_tags) VALUES (?, ?, ?, ?)",
                (lesson_id, diagnosis, fix_summary, tags),
            )
        return lesson_id

    def retrieve_lessons(self, target: TargetProfile, query: str, limit: int = 8) -> list[dict[str, Any]]:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)[:12] or ["compile"]
        terms = " OR ".join(f'"{token}"' for token in tokens)
        sql = """
            SELECT l.*, bm25(lessons_fts) AS score
            FROM lessons_fts JOIN lessons l ON l.id=lessons_fts.lesson_id
            WHERE lessons_fts MATCH ? AND l.verified=1 AND l.vendor=? AND l.backend=? AND l.hardware=?
              AND l.compiler_fingerprint=?
            ORDER BY score LIMIT ?
        """
        compiler_fingerprint = f"{target.id}:{target.compiler_version}"
        with self.lock:
            rows = self.connection.execute(
                sql,
                (terms, target.vendor, target.backend.value, target.hardware, compiler_fingerprint, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_memory(
        self, hardware: str | None = None, target_id: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        with self.lock:
            if target_id:
                knowledge_rows = self.connection.execute(
                    "SELECT * FROM knowledge_items WHERE target_id IS NULL OR target_id=? ORDER BY id",
                    (target_id,),
                )
            else:
                knowledge_rows = self.connection.execute("SELECT * FROM knowledge_items ORDER BY id")
            knowledge = [dict(row) for row in knowledge_rows]
            if target_id:
                lessons_rows = self.connection.execute(
                    "SELECT * FROM lessons WHERE compiler_fingerprint LIKE ? ORDER BY id",
                    (f"{target_id}:%",),
                )
            elif hardware:
                lessons_rows = self.connection.execute("SELECT * FROM lessons WHERE hardware=? ORDER BY id", (hardware,))
            else:
                lessons_rows = self.connection.execute("SELECT * FROM lessons ORDER BY id")
            lessons = [dict(row) for row in lessons_rows]
        return {"knowledge": knowledge, "lessons": lessons}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM app_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.connection.close()
