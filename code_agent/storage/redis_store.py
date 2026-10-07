"""Redis — Checkpoint 持久化 + 工具结果缓存"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional
import redis
from langgraph.checkpoint.redis import RedisSaver
from code_agent.config import get_setting


class RedisStore:
    """Redis 统一管理：Checkpoint + 业务缓存"""

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

    # ── 可恢复任务索引 ──
    def save_task(self, task_id: str, meta: dict):
        """持久化任务元数据，并加入可查询任务索引。"""
        now = datetime.now(timezone.utc).isoformat()
        saved = {**meta, "task_id": task_id, "updated_at": now}
        self.client.set(
            f"sh-agent:task:{task_id}",
            json.dumps(saved, ensure_ascii=False),
        )
        self.client.zadd("sh-agent:tasks", {task_id: datetime.now(timezone.utc).timestamp()})

    def get_task(self, task_id: str) -> Optional[dict]:
        value = self.client.get(f"sh-agent:task:{task_id}")
        return json.loads(value) if value else None

    def update_task(self, task_id: str, **updates) -> Optional[dict]:
        meta = self.get_task(task_id)
        if meta is None:
            return None
        meta.update(updates)
        self.save_task(task_id, meta)
        return meta

    def list_tasks(self, limit: int = 20) -> list[dict]:
        task_ids = self.client.zrevrange("sh-agent:tasks", 0, max(0, limit - 1))
        tasks = []
        for task_id in task_ids:
            meta = self.get_task(task_id)
            if meta is not None:
                tasks.append(meta)
        return tasks

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
