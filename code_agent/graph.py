"""LangGraph StateGraph — Multi-Agent 编排"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from code_agent.state import CodingState
from code_agent.agents.supervisor import supervisor_node
from code_agent.agents.explorer import explorer_node
from code_agent.agents.coder import coder_node
from code_agent.agents.reviewer import reviewer_node
from code_agent.agents.executor import executor_node
from code_agent.storage.redis_store import RedisStore


def route_after_supervisor(state: CodingState) -> str:
    agent = state.get("current_agent", "finish")
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # 审修轮次超限 → 强制结束
    if (state.get("review_feedback")
        and not state.get("review_approved")
        and retry >= max_retries):
        return "finish"

    return agent


def route_after_coder(state: CodingState) -> str:
    """Coder 完成后：直接去 Reviewer 审查，不经过 Supervisor"""
    return "reviewer"


def route_after_reviewer_update(state: CodingState) -> str:
    """审查通过后，代码变更先等待人工确认；未通过则回到 Coder 修复。"""
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if state.get("review_approved"):
        return "approval" if state.get("approval_required") else "execute"
    if retry < max_retries:
        return "coder"
    return "supervisor"


def approval_node(state: CodingState) -> dict:
    """在执行验证前暂停，等待用户批准或拒绝。"""
    decision = interrupt({
        "type": "code_change_approval",
        "message": "代码已通过 Reviewer，是否继续运行 Executor 验证？",
        "user_request": state.get("user_request", ""),
        "review_feedback": state.get("review_feedback", ""),
    })

    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
    else:
        approved = decision is True

    return {"approval_decision": "approved" if approved else "rejected"}


def route_after_approval(state: CodingState) -> str:
    """只有人工批准后才允许运行 Executor。"""
    if state.get("approval_decision") == "approved":
        return "execute"
    return "finish"


def after_reviewer_update(state: CodingState) -> dict:
    """Reviewer 返回不通过时，增加重试计数"""
    if not state.get("review_approved"):
        return {"retry_count": (state.get("retry_count") or 0) + 1}
    return {}


def finalize(state: CodingState) -> dict:
    """汇总最终结果"""
    parts = []

    exploration = state.get("exploration_result", "")
    if exploration:
        parts.append(f"## 代码分析\n\n{exploration}")

    review = state.get("review_feedback", "")
    if review:
        parts.append(f"## 审查结果\n\n{review}")

    test = state.get("test_result", "")
    if test:
        parts.append(f"## 测试结果\n\n{test}")

    if state.get("approval_decision") == "rejected":
        parts.append(
            "## 执行状态\n\n人工拒绝了 Executor 验证请求；任务已停止，未运行测试或命令。"
        )

    final = "\n\n".join(parts) if parts else "任务完成。"
    return {
        "final_response": final,
        "task_complete": True,
    }


def build_graph() -> StateGraph:
    builder = StateGraph(CodingState)

    # 注册节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("explorer", explorer_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("reviewer_update", after_reviewer_update)
    builder.add_node("approval", approval_node)
    builder.add_node("executor", executor_node)
    builder.add_node("finalizer", finalize)

    # 入口
    builder.set_entry_point("supervisor")

    # Supervisor → 条件路由
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "explore": "explorer",
            "code": "coder",
            "review": "reviewer",
            "execute": "executor",
            "finish": "finalizer",
        }
    )

    # Explorer / Executor 完成后回 Supervisor
    builder.add_edge("explorer", "supervisor")
    builder.add_edge("executor", "supervisor")

    # Coder → Reviewer（自包含审修闭环）
    builder.add_conditional_edges(
        "coder",
        route_after_coder,
        {"reviewer": "reviewer"}
    )

    # Reviewer → 更新计数 → 条件路由
    builder.add_edge("reviewer", "reviewer_update")
    builder.add_conditional_edges(
        "reviewer_update",
        route_after_reviewer_update,
        {
            "approval": "approval",
            "execute": "executor",
            "coder": "coder",
            "supervisor": "supervisor",
        }
    )

    # 人工确认后才进入 Executor；拒绝则直接汇总并结束。
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"execute": "executor", "finish": "finalizer"},
    )

    # 结束
    builder.add_edge("finalizer", END)

    return builder


def compile_graph(
    with_checkpoint: bool = True,
    in_memory_checkpoint: bool = False,
) -> StateGraph:
    builder = build_graph()

    if with_checkpoint:
        redis_store = RedisStore.get_instance()
        checkpointer = redis_store.get_checkpointer()
        return builder.compile(checkpointer=checkpointer)

    if in_memory_checkpoint:
        # 无 Redis 时仍支持当前进程内的人工暂停/继续；进程退出后状态不会保留。
        return builder.compile(checkpointer=MemorySaver())

    return builder.compile()
