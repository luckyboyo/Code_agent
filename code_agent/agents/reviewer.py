"""Reviewer Agent — 审查代码改动，检查质量"""
from pathlib import Path
from threading import Lock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from code_agent.model_factory import get_chat_model
from code_agent.state import CodingState
from code_agent.project_context import get_project_context, format_project_context
from code_agent.config import get_setting
from code_agent.tools.file_tools import read_file
from code_agent.tools.search_tools import grep, list_dir

REVIEWER_PROMPT = """你是 Code Reviewer，负责把关代码质量。

## 审查清单（逐条检查）
- 逻辑正确性：改动是否准确实现了需求，有无遗漏或曲解
- 安全性：有无注入风险、路径穿越、密钥泄露、权限绕过
- 边界处理：空输入、None、大文件、并发场景是否安全
- 代码风格：缩进、命名、引号、import 顺序是否与项目一致
- 副作用：改动是否影响其他模块、是否破坏公共接口

## 工具使用约束
- 只使用只读工具，不要尝试修改文件或执行命令
- 优先读取用户指定的待审查文件；只有需要追查引用关系时再使用 grep 或 list_dir
- 不要重复相同的工具调用；获得足够证据后立即输出结论
- 最多使用 {max_tool_calls} 次工具调用；达到上限后基于已有信息完成审查

## 输出格式
- 通过：回复以 **审查通过** 开头，可选附小建议
- 不通过：回复以 **审查不通过** 开头，逐条列出问题 + 修复建议

示例：
```
**审查通过**
改动逻辑正确，无安全风险，与现有风格一致。
小建议：第 23 行变量名可更语义化（非阻塞项）。
```
"""


def _make_bounded_tools(workspace: str, max_tool_calls: int) -> list[StructuredTool]:
    """给 Reviewer 创建带调用上限和工作区边界的只读工具。"""
    workspace_root = Path(workspace).resolve()
    counter = {"calls": 0}
    counter_lock = Lock()
    source_tools = [read_file, grep, list_dir]
    bounded_tools = []

    for source_tool in source_tools:
        def invoke_limited(_tool=source_tool, **kwargs):
            with counter_lock:
                if counter["calls"] >= max_tool_calls:
                    return (
                        f"已达到 Reviewer 的 {max_tool_calls} 次工具调用上限。"
                        "请停止调用工具，并根据已有信息输出审查结论。"
                    )
                counter["calls"] += 1

            # 工具调用固定在当前工作区，拒绝越界路径。
            kwargs["workspace_dir"] = str(workspace_root)
            path_key = "file_path" if "file_path" in kwargs else "path"
            if path_key in kwargs:
                candidate = (workspace_root / kwargs[path_key]).resolve()
                try:
                    candidate.relative_to(workspace_root)
                except ValueError:
                    return f"[安全拦截] 路径超出工作区：{kwargs[path_key]}"
                kwargs[path_key] = str(candidate)

            return _tool.invoke(kwargs)

        bounded_tools.append(StructuredTool.from_function(
            func=invoke_limited,
            name=source_tool.name,
            description=source_tool.description,
            args_schema=source_tool.args_schema,
        ))

    return bounded_tools


def reviewer_node(state: CodingState) -> dict:
    task = state.get("user_request", "")
    workspace = state.get("workspace_dir", ".")
    exploration = state.get("exploration_result", "")
    relevant_files = state.get("relevant_files") or []
    max_tool_calls = get_setting("agent", "max_reviewer_tool_calls")

    ctx = get_project_context(workspace)
    proj_info = format_project_context(ctx)

    prompt_parts = [
        f"## 项目信息\n{proj_info}",
        f"\n## 用户原需求\n{task}",
        f"\n## Explorer 分析\n{exploration}",
    ]

    if relevant_files:
        prompt_parts.append("\n## 优先审查文件\n" + "\n".join(f"- {path}" for path in relevant_files[:5]))
    else:
        prompt_parts.append("\n未提供优先审查文件，请根据用户需求和 Explorer 分析谨慎定位；若无法确认目标文件，明确说明审查受限。")

    previous_feedback = state.get("review_feedback")
    if previous_feedback:
        prompt_parts.append(f"\n## 上一轮审查意见\n{previous_feedback}")

    prompt = "\n".join(prompt_parts)
    tools = _make_bounded_tools(workspace, max_tool_calls)
    agent = create_react_agent(
        model=get_chat_model(),
        tools=tools,
        prompt=REVIEWER_PROMPT.replace("{max_tool_calls}", str(max_tool_calls)),
    )

    # 不传入其他 Agent 的完整工具历史，避免旧工具消息干扰 Reviewer。
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": max(12, max_tool_calls * 3 + 3)},
        )
        last_msg = result["messages"][-1].content
        if not isinstance(last_msg, str):
            last_msg = str(last_msg)
    except GraphRecursionError:
        last_msg = (
            "**审查不通过**\nReviewer 达到递归保护上限，本轮审查未能完成。"
            "请缩小审查文件范围后重试。"
        )

    approved = "审查通过" in last_msg

    return {
        "review_feedback": last_msg,
        "review_approved": approved,
        "messages": [AIMessage(content=last_msg)],
    }
