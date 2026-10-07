"""Explorer Agent — 搜索、阅读、理解代码"""
from pathlib import Path
from threading import Lock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from code_agent.model_factory import get_chat_model
from code_agent.tools.registry import AGENT_TOOLS
from code_agent.state import CodingState
from code_agent.project_context import get_project_context, format_project_context
from code_agent.config import get_setting

EXPLORER_PROMPT = """你是 Code Explorer，负责深入理解代码库。

## 工作流程
1. 先用 list_dir 了解目录结构
2. 用 glob_files 按文件名模式找相关文件
3. 用 grep 搜索关键符号（函数名、类名、import）
4. 用 read_file 精读关键代码段

## 输出要求
- 每个发现后总结：文件路径、关键代码段、与你任务的关系
- 不确定的地方标注 "[需确认]"
- 如果项目有规范的模块结构，指出代码的组织方式
- 完成探索后在末尾写 [探索完成]
- 最多调用工具 {max_tool_calls} 次；达到上限后立即根据已有信息总结，不要继续调用工具

## 注意
- 不要读超大文件（>500行）的全部内容，用 grep 定位关键区域
- 优先读入口文件、配置文件、与需求直接相关的模块
- 读代码时关注：对外接口、数据流向、错误处理方式
"""


def _make_bounded_tools(workspace: str, max_tool_calls: int) -> list[StructuredTool]:
    """限制 Explorer 的工具调用次数，并把搜索范围固定在当前工作区。"""
    workspace_root = Path(workspace).resolve()
    counter = {"calls": 0}
    counter_lock = Lock()
    bounded_tools = []

    for source_tool in AGENT_TOOLS["explorer"]:
        def invoke_limited(_tool=source_tool, **kwargs):
            with counter_lock:
                if counter["calls"] >= max_tool_calls:
                    return (
                        f"已达到 Explorer 的 {max_tool_calls} 次工具调用上限。"
                        "请根据已有结果完成探索总结。"
                    )
                counter["calls"] += 1
            kwargs["workspace_dir"] = str(workspace_root)
            return _tool.invoke(kwargs)

        bounded_tools.append(StructuredTool.from_function(
            func=invoke_limited,
            name=source_tool.name,
            description=source_tool.description,
            args_schema=source_tool.args_schema,
        ))

    return bounded_tools


def explorer_node(state: CodingState) -> dict:
    max_tool_calls = int(get_setting("agent", "max_tool_calls_per_turn") or 10)
    agent = create_react_agent(
        model=get_chat_model(),
        tools=_make_bounded_tools(state.get("workspace_dir", "."), max_tool_calls),
        prompt=EXPLORER_PROMPT.replace("{max_tool_calls}", str(max_tool_calls)),
    )

    task = state.get("user_request", "")
    workspace = state.get("workspace_dir", ".")
    plan = state.get("task_plan", "")

    ctx = get_project_context(workspace)
    proj_info = format_project_context(ctx)

    prompt = (
        f"## 项目信息\n{proj_info}\n\n"
        f"## 任务计划\n{plan}\n\n"
        f"## 用户需求\n{task}\n\n"
        f"请按照工作流程探索代码库，先了解结构再深入细节。"
    )

    existing = list(state.get("messages", []))
    try:
        result = agent.invoke(
            {"messages": existing + [HumanMessage(content=prompt)]},
            config={"recursion_limit": max(25, max_tool_calls * 3 + 3)},
        )
        last_msg = result["messages"][-1].content
        messages = result["messages"]
    except GraphRecursionError:
        last_msg = (
            "[探索受限] Explorer 达到递归保护上限，未能完成全部检索。"
            "请基于已收集的信息继续，不要重复调用 Explorer。"
        )
        messages = existing + [AIMessage(content=last_msg)]

    relevant_files = _extract_files(last_msg, workspace)

    return {
        "exploration_result": last_msg,
        "relevant_files": relevant_files,
        "messages": messages,
    }


def _extract_files(text: str, workspace: str) -> list[str]:
    """从探索结果中提取文件路径"""
    import re
    import os
    files = set()
    patterns = [
        r'[\w/\-]+\.\w{1,6}',  # 通用文件路径
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            abs_path = os.path.join(workspace, match)
            if os.path.isfile(abs_path):
                files.add(match)
    return list(files)[:10]
