"""Token 级流式输出 — 逐 token 渲染 LLM 输出"""
from rich.text import Text
from langchain_core.callbacks import BaseCallbackHandler
from code_agent.ui.terminal import console


class TokenStreamHandler(BaseCallbackHandler):
    """逐 token 输出到 Rich 控制台，实现打字机效果"""

    def __init__(self):
        self._buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._buffer += token
        console.print(token, end="", highlight=False)

    def on_llm_end(self, *args, **kwargs) -> None:
        if self._buffer:
            self._buffer = ""
