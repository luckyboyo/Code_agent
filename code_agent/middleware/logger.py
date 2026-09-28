"""日志中间件 — 工具调用监控 + 模型前日志"""
import logging
from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model
from langgraph.runtime import Runtime
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("sh-agent")


@wrap_tool_call
def tool_monitor(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
):
    tool_name = request.tool_call["name"]
    args_preview = str(request.tool_call.get("args", {}))[:100]
    logger.info(f"🔧 {tool_name} | {args_preview}")

    result = handler(request)

    content = result.content if hasattr(result, "content") else str(result)
    preview = content[:80].replace("\n", " ")
    logger.info(f"   ✅ {tool_name} → {preview}...")
    return result


@before_model
def log_model_call(state: AgentState, runtime: Runtime):
    msg_count = len(state.get("messages", []))
    logger.info(f"🤖 调用模型 | 共 {msg_count} 条消息")
    return None
