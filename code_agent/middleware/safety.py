"""安全中间件 — 工具调用前置检查 + 操作日志"""
from typing import Callable
from langchain.agents.middleware import wrap_tool_call
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from code_agent.config import get_setting
from code_agent.storage.sql_store import SQLStore
import fnmatch


@wrap_tool_call
def safety_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
):
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    # ── 文件路径安全检查 ──
    file_path = tool_args.get("file_path", "")
    if file_path:
        confirm_patterns = get_setting("safety", "confirm_paths")
        for pattern in confirm_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return ToolMessage(
                    content=f"[安全拦截] 文件 {file_path} 匹配受保护模式 {pattern}，操作被阻止",
                    tool_call_id=request.tool_call["id"]
                )

    # ── 执行工具 ──
    try:
        result = handler(request)
    except Exception as e:
        result = ToolMessage(
            content=f"工具执行失败: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

    # ── 审计日志 ──
    session_id = request.runtime.context.get("session_id", "unknown")
    sql = SQLStore.get_instance()
    sql.log(
        session_id=session_id,
        agent_name=request.runtime.context.get("current_agent", "unknown"),
        action_type="tool_call",
        detail={"tool": tool_name, "args": tool_args},
        file_path=file_path,
    )

    return result
