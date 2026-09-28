"""PostgreSQL — Agent 操作审计日志"""
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional
import psycopg2
from psycopg2.extras import Json
from code_agent.config import get_setting, get_env


class SQLStore:
    """PostgreSQL 审计日志存储（可选，PG 不可用时优雅降级）"""

    _instance: Optional["SQLStore"] = None

    def __init__(self):
        self.enabled = get_setting("postgres", "enabled")
        self._conn = None

    @classmethod
    def get_instance(cls) -> "SQLStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_audit_log (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                agent_name VARCHAR(32) NOT NULL,
                action_type VARCHAR(32) NOT NULL,
                action_detail JSONB NOT NULL DEFAULT '{}',
                file_path VARCHAR(512),
                diff_content TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_audit_session ON agent_audit_log(session_id);
            CREATE INDEX IF NOT EXISTS idx_audit_agent ON agent_audit_log(agent_name);
            CREATE INDEX IF NOT EXISTS idx_audit_created ON agent_audit_log(created_at);
        """)

    @contextmanager
    def _get_conn(self):
        if not self.enabled:
            yield None
            return
        try:
            c = psycopg2.connect(
                host=get_setting("postgres", "host"),
                port=get_setting("postgres", "port"),
                database=get_setting("postgres", "database"),
                user=get_setting("postgres", "user"),
                password=get_env(get_setting("postgres", "password_env")),
            )
            yield c
            c.commit()
        except Exception:
            yield None
        finally:
            try:
                c.close()
            except Exception:
                pass

    def log(self, session_id: str, agent_name: str, action_type: str,
            detail: dict, file_path: Optional[str] = None,
            diff_content: Optional[str] = None):
        with self._get_conn() as conn:
            if conn is None:
                return
            self._ensure_table(conn)
            conn.cursor().execute(
                """INSERT INTO agent_audit_log
                   (session_id, agent_name, action_type, action_detail, file_path, diff_content)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (session_id, agent_name, action_type, Json(detail), file_path, diff_content)
            )
