"""Explorer Agent — 搜索、阅读、理解代码"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from code_agent.model_factory import get_chat_model
from code_agent.tools.registry import AGENT_TOOLS
from code_agent.state import CodingState
from code_agent.project_context import get_project_context, format_project_context

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

## 注意
- 不要读超大文件（>500行）的全部内容，用 grep 定位关键区域
- 优先读入口文件、配置文件、与需求直接相关的模块
- 读代码时关注：对外接口、数据流向、错误处理方式
"""


def explorer_node(state: CodingState) -> dict:
    agent = create_react_agent(
        model=get_chat_model(),
        tools=AGENT_TOOLS["explorer"],
        prompt=EXPLORER_PROMPT,
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
    result = agent.invoke({"messages": existing + [HumanMessage(content=prompt)]})
    last_msg = result["messages"][-1].content

    relevant_files = _extract_files(last_msg, workspace)

    return {
        "exploration_result": last_msg,
        "relevant_files": relevant_files,
        "messages": result["messages"],
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
