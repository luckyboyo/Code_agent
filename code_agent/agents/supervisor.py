"""Supervisor Agent — 任务拆解 + 动态调度"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from code_agent.model_factory import get_chat_model
from code_agent.state import CodingState
from code_agent.project_context import get_project_context, format_project_context

SUPERVISOR_PROMPT = """你是 Supervisor，一个 Multi-Agent 编码系统的调度核心。

## 核心职责
你本身不持有任何工具，只做决策：
1. 理解用户意图 — 是只读查询、代码修改、还是运行验证
2. 规划执行路径 — 选择最少 Agent 步骤完成目标
3. 判断终止时机 — 目标达成时果断 finish，不要过度调度

## Agent 能力矩阵
| Agent     | 工具                                     | 适用场景                  |
|-----------|------------------------------------------|---------------------------|
| Explorer  | read_file, grep, glob_files, list_dir   | 搜索代码、理解结构、定位文件 |
| Coder     | read_file, write_file, edit_file, grep  | 写新代码、修改现有代码     |
| Reviewer  | read_file, grep, list_dir, bash         | 审查改动、检查安全与质量   |
| Executor  | bash, read_file                         | 运行测试、执行命令验证     |

## 路由策略
- 只读任务（分析、解释、查找）：Explorer → finish
- 代码修改：Explorer → Coder → Reviewer → Executor → finish
- 运行命令：Executor → finish
- 代码审查（不改）：Explorer → Reviewer → finish
- Reviewer 不通过时 Coder→Reviewer 自动循环，最多 {max_retries} 轮，你无需干预

## 输出格式
```
任务分析: <一句话>
执行计划: <步骤>
决策: <explore/code/review/execute/finish>
```"""


def supervisor_node(state: CodingState) -> dict:
    workspace = state.get("workspace_dir", ".")
    max_retries = state.get("max_retries", 3)
    prompt_text = SUPERVISOR_PROMPT.replace("{max_retries}", str(max_retries))

    agent = create_react_agent(
        model=get_chat_model(),
        tools=[],
        prompt=prompt_text,
    )

    # 项目上下文
    ctx = get_project_context(workspace)
    proj_info = format_project_context(ctx)

    task = state.get("user_request", "")
    exploration = state.get("exploration_result", "")
    review_feedback = state.get("review_feedback", "")
    review_approved = state.get("review_approved", False)
    test_result = state.get("test_result", "")
    retry_count = state.get("retry_count", 0)

    prompt_parts = [
        f"## 项目信息\n{proj_info}",
        f"\n## 用户需求\n{task}",
    ]

    # 反馈当前状态
    status = []
    if exploration:
        status.append("✅ Explorer 已完成代码分析")
    else:
        status.append("⬜ 尚未探索代码")

    if review_feedback:
        if review_approved:
            status.append("✅ Reviewer 审查通过")
        else:
            status.append(f"❌ Reviewer 要求修改 (第{retry_count}/{max_retries}轮)")
    else:
        status.append("⬜ 尚未审查")

    if test_result:
        status.append("✅ Executor 已执行测试")

    prompt_parts.append(f"\n## 当前状态\n" + "\n".join(f"- {s}" for s in status))
    prompt_parts.append(f"\n审修轮次: {retry_count}/{max_retries}")
    prompt_parts.append("\n请输出你的任务分析和决策。")

    prompt = "\n".join(prompt_parts)

    # 传入完整历史消息，保留多轮对话上下文
    existing = list(state.get("messages", []))
    result = agent.invoke({"messages": existing + [HumanMessage(content=prompt)]})
    last_msg = result["messages"][-1].content

    decision = _parse_decision(last_msg, state)
    task_plan = _extract_plan(last_msg)

    return {
        "task_plan": task_plan,
        "current_agent": decision,
        "messages": result["messages"],
    }


def parse_decision(text: str, exploration_result: str | None = None,
                   review_feedback: str | None = None, review_approved: bool = False,
                   test_result: str | None = None, retry_count: int = 0,
                   max_retries: int = 3) -> str:
    """从 Supervisor 输出中解析路由决策（纯函数，可测试）"""
    text_lower = text.lower()

    if "finish" in text_lower:
        return "finish"

    if not exploration_result:
        return "explore"

    if not review_feedback:
        if "code" in text_lower:
            return "code"
        if "review" in text_lower:
            return "review"
        if "execute" in text_lower:
            return "execute"
        return "finish"

    if not review_approved:
        if retry_count < max_retries:
            return "code"
        return "finish"

    if not test_result:
        return "execute"

    return "finish"


def _parse_decision(text: str, state: CodingState) -> str:
    """从 Supervisor 输出中解析路由决策（兼容旧接口）"""
    return parse_decision(
        text=text,
        exploration_result=state.get("exploration_result"),
        review_feedback=state.get("review_feedback"),
        review_approved=state.get("review_approved", False),
        test_result=state.get("test_result"),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", 3),
    )


def _extract_plan(text: str) -> str:
    """提取任务计划文本"""
    for line in text.split("\n"):
        if "任务分析" in line or "执行计划" in line:
            continue
    return text[:300]
