"""Reviewer Agent — 审查代码改动，检查质量"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from code_agent.model_factory import get_chat_model
from code_agent.tools.registry import AGENT_TOOLS
from code_agent.state import CodingState
from code_agent.project_context import get_project_context, format_project_context

REVIEWER_PROMPT = """你是 Code Reviewer，负责把关代码质量。

## 审查清单（逐条检查）
- 逻辑正确性：改动是否准确实现了需求，有无遗漏或曲解
- 安全性：有无注入风险、路径穿越、密钥泄露、权限绕过
- 边界处理：空输入、None、大文件、并发场景是否安全
- 代码风格：缩进、命名、引号、import 顺序是否与项目一致
- 副作用：改动是否影响其他模块、是否破坏公共接口

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


def reviewer_node(state: CodingState) -> dict:
    agent = create_react_agent(
        model=get_chat_model(),
        tools=AGENT_TOOLS["reviewer"],
        prompt=REVIEWER_PROMPT,
    )

    task = state.get("user_request", "")
    workspace = state.get("workspace_dir", ".")
    exploration = state.get("exploration_result", "")
    relevant_files = state.get("relevant_files", [])

    ctx = get_project_context(workspace)
    proj_info = format_project_context(ctx)

    prompt_parts = [
        f"## 项目信息\n{proj_info}",
        f"\n## 用户原需求\n{task}",
        f"\n## Explorer 分析\n{exploration}",
    ]

    if relevant_files:
        prompt_parts.append(f"\n## 待审查文件\n" + "\n".join(f"- {f}" for f in relevant_files))
        prompt_parts.append("\n请先 read_file 读取每个文件，然后逐条对照审查清单检查。")

    prompt = "\n".join(prompt_parts)

    existing = list(state.get("messages", []))
    result = agent.invoke({"messages": existing + [HumanMessage(content=prompt)]})
    last_msg = result["messages"][-1].content

    approved = "审查通过" in last_msg

    return {
        "review_feedback": last_msg,
        "review_approved": approved,
        "messages": result["messages"],
    }
