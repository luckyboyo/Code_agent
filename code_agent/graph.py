"""LangGraph StateGraph — Multi-Agent 编排"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
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
    """审查后决定：通过→执行，不通过→Coder 修复（自包含闭环）"""
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if state.get("review_approved"):
        return "execute"
    if retry < max_retries:
        return "coder"
    return "supervisor"


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
        parts.append(f"## 代码分析\n\n{exploration[:500]}")

    review = state.get("review_feedback", "")
    if review:
        parts.append(f"## 审查结果\n\n{review[:500]}")

    test = state.get("test_result", "")
    if test:
        parts.append(f"## 测试结果\n\n{test[:500]}")

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
            "execute": "executor",
            "coder": "coder",
            "supervisor": "supervisor",
        }
    )

    # 结束
    builder.add_edge("finalizer", END)

    return builder


def compile_graph(with_checkpoint: bool = True) -> StateGraph:
    builder = build_graph()

    if with_checkpoint:
        try:
            redis_store = RedisStore.get_instance()
            checkpointer = redis_store.get_checkpointer()
            return builder.compile(checkpointer=checkpointer)
        except Exception:
            pass  # Redis 不可用时降级为无 checkpoint

    return builder.compile()
