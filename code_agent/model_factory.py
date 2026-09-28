"""模型工厂 — 从配置创建 ChatModel（惰性初始化）"""
from threading import Lock
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from code_agent.config import get_setting, get_env

load_dotenv()

_chat_model = None
_lock = Lock()


def get_chat_model() -> ChatOpenAI:
    global _chat_model
    if _chat_model is None:
        with _lock:
            if _chat_model is None:
                _chat_model = ChatOpenAI(
                    api_key=get_env(get_setting("model", "api_key_env")),
                    base_url=get_env(get_setting("model", "base_url_env")),
                    model=get_setting("model", "chat_model"),
                    temperature=get_setting("agent", "model_temperature"),
                    streaming=True,
                )
    return _chat_model
