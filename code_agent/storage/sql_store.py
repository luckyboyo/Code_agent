"""PostgreSQL 持久化：会话、消息、任务状态和工具审计日志。"""
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from code_agent.config import get_env, get_setting


logger = logging.getLogger(__name__)


class SQLStore:
    """PostgreSQL 业务数据存储；Redis 仅负责 checkpoint 和缓存。"""

    _instance: Optional["SQLStore"] = None

    def __init__(self):
        self.enabled = bool(get_setting("postgres", "enabled"))
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "SQLStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _parse_datetime(value):
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _iso(value):
        return value.isoformat() if isinstance(value, datetime) else value

    def _connect(self):
        return psycopg2.connect(
            host=get_setting("postgres", "host"),
            port=get_setting("postgres", "port"),
            database=get_setting("postgres", "database"),
            user=get_setting("postgres", "user"),
            password=get_env(get_setting("postgres", "password_env")),
        )

    @contextmanager
    def _get_conn(self, required: bool = False):
        """Open a transaction. Connection failures may be ignored only by optional logging."""
        if not self.enabled:
            if required:
                raise RuntimeError(
                    "PostgreSQL 未启用，请在 config/settings.yml 中设置 postgres.enabled: true"
                )
            yield None
            return

        try:
            conn = self._connect()
        except Exception:
            if required:
                raise
            yield None
            return

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_tables(self, conn):
        """Create the small set of tables used by the CLI, if absent."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                workspace_dir TEXT NOT NULL DEFAULT '',
                turn INTEGER NOT NULL DEFAULT 0,
                history JSONB NOT NULL DEFAULT '[]'::jsonb,
                completed_task_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                last_task_id VARCHAR(64),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_tasks (
                task_id VARCHAR(64) PRIMARY KEY,
                thread_id VARCHAR(64) NOT NULL,
                session_id VARCHAR(64) REFERENCES agent_sessions(session_id) ON DELETE SET NULL,
                session_turn INTEGER NOT NULL DEFAULT 0,
                status VARCHAR(32) NOT NULL DEFAULT 'queued',
                user_request TEXT NOT NULL DEFAULT '',
                workspace_dir TEXT NOT NULL DEFAULT '',
                history JSONB NOT NULL DEFAULT '[]'::jsonb,
                pause_reason TEXT,
                final_response TEXT,
                last_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
                task_id VARCHAR(64) NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                turn_no INTEGER NOT NULL DEFAULT 0,
                role VARCHAR(16) NOT NULL CHECK (role IN ('human', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (task_id, role)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_audit_log (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                agent_name VARCHAR(32) NOT NULL,
                action_type VARCHAR(32) NOT NULL,
                action_detail JSONB NOT NULL DEFAULT '{}'::jsonb,
                file_path VARCHAR(512),
                diff_content TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_storage_migrations (
                migration_name VARCHAR(128) PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated ON agent_sessions(updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_session_turn ON agent_tasks(session_id, session_turn)",
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_status_updated ON agent_tasks(status, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_session_turn ON agent_messages(session_id, turn_no, id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_session ON agent_audit_log(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_agent ON agent_audit_log(agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_audit_created ON agent_audit_log(created_at)",
        ]
        with conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

    def initialize(self) -> bool:
        """Create the schema and verify PostgreSQL connectivity."""
        if not self.enabled:
            return False
        with self._get_conn(required=True) as conn:
            self._ensure_tables(conn)
        self._initialized = True
        return True

    def _ensure_ready(self):
        if not self.enabled:
            raise RuntimeError("PostgreSQL is disabled")
        if not self._initialized:
            self.initialize()

    def _write_session(self, cursor, session_id: str, meta: dict, overwrite: bool = True):
        values = (
            session_id,
            meta.get("workspace_dir", ""),
            int(meta.get("turn", 0) or 0),
            Json(meta.get("history", [])),
            Json(meta.get("completed_task_ids", [])),
            meta.get("last_task_id"),
            self._parse_datetime(meta.get("created_at")),
        )
        if overwrite:
            conflict = """
            ON CONFLICT (session_id) DO UPDATE SET
                workspace_dir = EXCLUDED.workspace_dir,
                turn = EXCLUDED.turn,
                history = EXCLUDED.history,
                completed_task_ids = EXCLUDED.completed_task_ids,
                last_task_id = COALESCE(EXCLUDED.last_task_id, agent_sessions.last_task_id),
                updated_at = NOW()
            """
        else:
            conflict = "ON CONFLICT (session_id) DO NOTHING"
        cursor.execute(
            f"""
            INSERT INTO agent_sessions
                (session_id, workspace_dir, turn, history, completed_task_ids,
                 last_task_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
            {conflict}
            """,
            values,
        )

    def save_session(self, session_id: str, meta: dict):
        self._ensure_ready()
        saved = {**meta, "session_id": session_id}
        with self._get_conn(required=True) as conn:
            with conn.cursor() as cursor:
                self._write_session(cursor, session_id, saved)

    @staticmethod
    def _session_from_row(row) -> dict:
        return {
            "session_id": row["session_id"],
            "workspace_dir": row["workspace_dir"],
            "turn": row["turn"],
            "history": row["history"] or [],
            "completed_task_ids": row["completed_task_ids"] or [],
            "last_task_id": row["last_task_id"],
            "created_at": SQLStore._iso(row["created_at"]),
            "updated_at": SQLStore._iso(row["updated_at"]),
        }

    def get_session(self, session_id: str) -> Optional[dict]:
        if not self.enabled:
            return None
        self._ensure_ready()
        with self._get_conn(required=True) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM agent_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        if not self.enabled or limit <= 0:
            return []
        self._ensure_ready()
        with self._get_conn(required=True) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT s.* FROM agent_sessions AS s
                    WHERE s.turn > 0 OR EXISTS (
                        SELECT 1 FROM agent_tasks AS t
                        WHERE t.session_id = s.session_id
                    )
                    ORDER BY s.updated_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [self._session_from_row(row) for row in rows]

    def _write_task(self, cursor, task_id: str, meta: dict, overwrite: bool = True):
        saved = {**meta, "task_id": task_id}
        session_id = saved.get("session_id")
        if session_id:
            self._write_session(
                cursor,
                session_id,
                {
                    "workspace_dir": saved.get("workspace_dir", ""),
                    "turn": saved.get("session_turn", 0),
                    "history": [],
                    "completed_task_ids": [],
                },
                overwrite=False,
            )

        if overwrite:
            conflict = """
            ON CONFLICT (task_id) DO UPDATE SET
                thread_id = EXCLUDED.thread_id,
                session_id = EXCLUDED.session_id,
                session_turn = EXCLUDED.session_turn,
                status = EXCLUDED.status,
                user_request = EXCLUDED.user_request,
                workspace_dir = EXCLUDED.workspace_dir,
                history = EXCLUDED.history,
                pause_reason = EXCLUDED.pause_reason,
                final_response = EXCLUDED.final_response,
                last_error = EXCLUDED.last_error,
                updated_at = NOW(),
                completed_at = EXCLUDED.completed_at,
                metadata = EXCLUDED.metadata
            """
            message_conflict = """
            ON CONFLICT (task_id, role) DO UPDATE SET
                content = EXCLUDED.content,
                updated_at = NOW()
            """
        else:
            conflict = "ON CONFLICT (task_id) DO NOTHING"
            message_conflict = "ON CONFLICT (task_id, role) DO NOTHING"

        cursor.execute(
            f"""
            INSERT INTO agent_tasks
                (task_id, thread_id, session_id, session_turn, status, user_request,
                 workspace_dir, history, pause_reason, final_response, last_error,
                 created_at, completed_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    COALESCE(%s, NOW()), %s, %s)
            {conflict}
            """,
            (
                task_id,
                saved.get("thread_id", task_id),
                session_id,
                int(saved.get("session_turn", 0) or 0),
                saved.get("status", "queued"),
                saved.get("user_request", ""),
                saved.get("workspace_dir", ""),
                Json(saved.get("history", [])),
                saved.get("pause_reason"),
                saved.get("final_response"),
                saved.get("last_error"),
                self._parse_datetime(saved.get("created_at")),
                self._parse_datetime(saved.get("completed_at")),
                Json(saved),
            ),
        )

        if session_id and saved.get("user_request") is not None:
            cursor.execute(
                f"""
                INSERT INTO agent_messages (session_id, task_id, turn_no, role, content)
                VALUES (%s, %s, %s, 'human', %s)
                {message_conflict}
                """,
                (session_id, task_id, int(saved.get("session_turn", 0) or 0), saved["user_request"]),
            )
            if "final_response" in saved and saved.get("final_response") is not None:
                cursor.execute(
                    f"""
                    INSERT INTO agent_messages (session_id, task_id, turn_no, role, content)
                    VALUES (%s, %s, %s, 'assistant', %s)
                    {message_conflict}
                    """,
                    (session_id, task_id, int(saved.get("session_turn", 0) or 0), saved["final_response"]),
                )

    def save_task(self, task_id: str, meta: dict):
        self._ensure_ready()
        with self._get_conn(required=True) as conn:
            with conn.cursor() as cursor:
                self._write_task(cursor, task_id, meta)

    @staticmethod
    def _task_from_row(row) -> dict:
        meta = dict(row["metadata"] or {})
        meta.update({
            "task_id": row["task_id"],
            "thread_id": row["thread_id"],
            "session_id": row["session_id"],
            "session_turn": row["session_turn"],
            "status": row["status"],
            "user_request": row["user_request"],
            "workspace_dir": row["workspace_dir"],
            "history": row["history"] or [],
            "pause_reason": row["pause_reason"],
            "final_response": row["final_response"],
            "last_error": row["last_error"],
            "created_at": SQLStore._iso(row["created_at"]),
            "updated_at": SQLStore._iso(row["updated_at"]),
            "completed_at": SQLStore._iso(row["completed_at"]),
        })
        return meta

    def get_task(self, task_id: str) -> Optional[dict]:
        if not self.enabled:
            return None
        self._ensure_ready()
        with self._get_conn(required=True) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM agent_tasks WHERE task_id = %s", (task_id,))
                row = cursor.fetchone()
        return self._task_from_row(row) if row else None

    def update_task(self, task_id: str, **updates) -> Optional[dict]:
        meta = self.get_task(task_id)
        if meta is None:
            return None
        meta.update(updates)
        self.save_task(task_id, meta)
        return meta

    def _query_tasks(self, query: str, params: tuple) -> list[dict]:
        self._ensure_ready()
        with self._get_conn(required=True) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        return [self._task_from_row(row) for row in rows]

    def list_tasks(self, limit: int = 20) -> list[dict]:
        if not self.enabled or limit <= 0:
            return []
        return self._query_tasks(
            "SELECT * FROM agent_tasks ORDER BY updated_at DESC LIMIT %s",
            (limit,),
        )

    def list_session_tasks(self, session_id: str, limit: int | None = None) -> list[dict]:
        if not self.enabled or (limit is not None and limit <= 0):
            return []
        if limit is None:
            return self._query_tasks(
                "SELECT * FROM agent_tasks WHERE session_id = %s ORDER BY session_turn, created_at",
                (session_id,),
            )
        return self._query_tasks(
            "SELECT * FROM agent_tasks WHERE session_id = %s ORDER BY session_turn, created_at LIMIT %s",
            (session_id, limit),
        )

    def migrate_legacy_redis_data(self, redis_store) -> tuple[int, int]:
        """Import old Redis session/task records once; new records are written only to PostgreSQL."""
        if not self.enabled:
            return 0, 0
        self._ensure_ready()
        sessions = redis_store.list_legacy_sessions(limit=100000)
        tasks = redis_store.list_legacy_tasks(limit=100000)
        migration_name = "redis_session_task_metadata_v1"

        with self._get_conn(required=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM agent_storage_migrations WHERE migration_name = %s",
                    (migration_name,),
                )
                if cursor.fetchone():
                    return 0, 0

                for session in sessions:
                    session_id = session.get("session_id")
                    if session_id:
                        self._write_session(cursor, session_id, session, overwrite=False)

                for task in tasks:
                    task_id = task.get("task_id") or task.get("thread_id")
                    if not task_id:
                        continue
                    session_id = task.get("session_id")
                    if session_id:
                        self._write_session(
                            cursor,
                            session_id,
                            {
                                "workspace_dir": task.get("workspace_dir", ""),
                                "turn": task.get("session_turn", 0),
                            },
                            overwrite=False,
                        )
                    self._write_task(cursor, task_id, task, overwrite=False)

                cursor.execute(
                    "INSERT INTO agent_storage_migrations (migration_name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (migration_name,),
                )
        return len(sessions), len(tasks)

    def log(
        self,
        session_id: str,
        agent_name: str,
        action_type: str,
        detail: dict,
        file_path: Optional[str] = None,
        diff_content: Optional[str] = None,
    ):
        """Write an audit event; audit failures must not fail the Agent tool call."""
        if not self.enabled:
            return
        try:
            self._ensure_ready()
            with self._get_conn(required=False) as conn:
                if conn is None:
                    return
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO agent_audit_log
                            (session_id, agent_name, action_type, action_detail, file_path, diff_content)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (session_id, agent_name, action_type, Json(detail), file_path, diff_content),
                    )
        except Exception:
            logger.exception("PostgreSQL audit log write failed")
