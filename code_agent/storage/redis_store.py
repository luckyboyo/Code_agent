"""Redis — Checkpoint 持久化 + 工具结果缓存"""
import hashlib
import json
from typing import Optional
import redis
from langgraph.checkpoint.redis import RedisSaver
from code_agent.config import get_setting


class RedisStore:
    """Redis 统一管理：Checkpoint + 业务缓存"""

    _instance: Optional["RedisStore"] = None

    def __init__(self):
        self.client = redis.Redis(
            host=get_setting("redis", "host"),
            port=get_setting("redis", "port"),
            db=get_setting("redis", "db"),
            password=get_setting("redis", "password") or None,
            decode_responses=True,
        )
        self.saver = RedisSaver(self.client)

    @classmethod
    def get_instance(cls) -> "RedisStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Checkpoint（断点续聊） ──
    def get_checkpointer(self) -> RedisSaver:
        return self.saver

    # ── 文件内容缓存 ──
    def cache_file(self, file_path: str, content: str):
        key = f"file:{hashlib.md5(file_path.encode()).hexdigest()}"
        ttl = get_setting("redis", "ttl_file_cache")
        self.client.setex(key, ttl, content)

    def get_cached_file(self, file_path: str) -> Optional[str]:
        key = f"file:{hashlib.md5(file_path.encode()).hexdigest()}"
        return self.client.get(key)

    # ── Grep 结果缓存 ──
    def cache_grep(self, pattern: str, result: str):
        key = f"grep:{hashlib.md5(pattern.encode()).hexdigest()}"
        ttl = get_setting("redis", "ttl_grep_cache")
        self.client.setex(key, ttl, result)

    def get_cached_grep(self, pattern: str) -> Optional[str]:
        key = f"grep:{hashlib.md5(pattern.encode()).hexdigest()}"
        return self.client.get(key)

    # ── 会话元数据 ──
    def save_session_meta(self, session_id: str, meta: dict):
        self.client.hset(f"session:{session_id}", mapping=meta)
        self.client.expire(f"session:{session_id}", 3600 * 24)

    def get_session_meta(self, session_id: str) -> dict:
        return self.client.hgetall(f"session:{session_id}")
