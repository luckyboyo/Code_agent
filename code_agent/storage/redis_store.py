"""Redis — Checkpoint 持久化 + 工具结果缓存"""
import hashlib
import json
from typing import Optional
import redis
from langgraph.checkpoint.redis import RedisSaver
from code_agent.config import get_setting


class RedisStore:
    """Redis 管理 LangGraph checkpoint 和可过期缓存。"""

    _instance: Optional["RedisStore"] = None

    def __init__(self):
        connection = {
            "host": get_setting("redis", "host"),
            "port": get_setting("redis", "port"),
            "db": get_setting("redis", "db"),
            "password": get_setting("redis", "password") or None,
        }
        # 业务缓存使用文本响应；RedisSaver 按其适配器要求使用 bytes 响应。
        self.client = redis.Redis(**connection, decode_responses=True)
        self.checkpoint_client = redis.Redis(**connection, decode_responses=False)
        # 新版构造函数的第一个位置参数是 redis_url；客户端必须使用关键字传入。
        self.saver = RedisSaver(redis_client=self.checkpoint_client)
        self._checkpointer_ready = False

    @classmethod
    def get_instance(cls) -> "RedisStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Checkpoint（断点续聊） ──
    def get_checkpointer(self) -> RedisSaver:
        if not self._checkpointer_ready:
            self.checkpoint_client.ping()
            self.saver.setup()
            self._checkpointer_ready = True
        return self.saver

    # ── 旧版本 Redis 业务数据，只用于首次迁移至 PostgreSQL ──
    def list_legacy_tasks(self, limit: int = 100000) -> list[dict]:
        if limit <= 0:
            return []
        task_ids = self.client.zrevrange("sh-agent:tasks", 0, limit - 1)
        tasks = []
        for task_id in task_ids:
            value = self.client.get(f"sh-agent:task:{task_id}")
            if value:
                tasks.append(json.loads(value))
        return tasks

    def list_legacy_sessions(self, limit: int = 100000) -> list[dict]:
        if limit <= 0:
            return []
        session_ids = self.client.zrevrange("sh-agent:sessions", 0, limit - 1)
        sessions = []
        for session_id in session_ids:
            value = self.client.get(f"sh-agent:session:{session_id}")
            if value:
                sessions.append(json.loads(value))
        return sessions

    # ── 文件内容缓存 ──
    @staticmethod
    def _cache_key(kind: str, identity: str) -> str:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"sh-agent:cache:{kind}:{digest}"

    def cache_file(self, file_path: str, content: str, version: str = ""):
        ttl = int(get_setting("redis", "ttl_file_cache") or 0)
        if ttl <= 0:
            return
        key = self._cache_key("file", file_path)
        value = json.dumps(
            {"version": version, "content": content}, ensure_ascii=False
        )
        self.client.setex(key, ttl, value)

    def get_cached_file(self, file_path: str, version: str = "") -> Optional[str]:
        key = self._cache_key("file", file_path)
        value = self.client.get(key)
        if value is None:
            return None
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != version:
            return None
        return payload.get("content")

    def invalidate_file(self, file_path: str):
        self.client.delete(self._cache_key("file", file_path))

    # ── Grep 结果缓存 ──
    def cache_grep(self, identity: str, result: str):
        ttl = int(get_setting("redis", "ttl_grep_cache") or 0)
        if ttl <= 0:
            return
        key = self._cache_key("grep", identity)
        self.client.setex(key, ttl, result)

    def get_cached_grep(self, identity: str) -> Optional[str]:
        key = self._cache_key("grep", identity)
        return self.client.get(key)

    def clear_grep_cache(self):
        batch = []
        for key in self.client.scan_iter(match="sh-agent:cache:grep:*", count=200):
            batch.append(key)
            if len(batch) >= 200:
                self.client.delete(*batch)
                batch.clear()
        if batch:
            self.client.delete(*batch)
