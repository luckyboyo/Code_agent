"""测试路由决策逻辑 — 覆盖全部状态分支"""
import pytest
from code_agent.agents.supervisor import parse_decision


class TestParseDecision:
    """parse_decision 纯函数测试 — 不依赖 LLM 调用"""

    # ── 无探索结果 → 必须先去 explore ──
    def test_no_exploration_forces_explore(self):
        result = parse_decision("some text", exploration_result=None)
        assert result == "explore"

    def test_no_exploration_even_if_text_says_finish(self):
        result = parse_decision("finish now", exploration_result=None)
        assert result == "finish"

    # ── 有探索结果，无审查反馈 ──
    def test_explored_no_review_llm_says_code(self):
        result = parse_decision("code", exploration_result="found files")
        assert result == "code"

    def test_explored_no_review_llm_says_review(self):
        result = parse_decision("review", exploration_result="found files")
        assert result == "review"

    def test_explored_no_review_llm_says_execute(self):
        result = parse_decision("execute", exploration_result="found files")
        assert result == "execute"

    def test_explored_no_review_default_finish(self):
        result = parse_decision("no keyword here", exploration_result="found files")
        assert result == "finish"

    # ── 审查不通过，未超轮次 → 回 Coder ──
    def test_review_failed_within_retries(self):
        result = parse_decision(
            "fix",
            exploration_result="found",
            review_feedback="needs changes",
            review_approved=False,
            retry_count=1,
            max_retries=3,
        )
        assert result == "code"

    # ── 审查不通过，超轮次 → 强制结束 ──
    def test_review_failed_max_retries(self):
        result = parse_decision(
            "fix",
            exploration_result="found",
            review_feedback="needs changes",
            review_approved=False,
            retry_count=3,
            max_retries=3,
        )
        assert result == "finish"

    def test_review_failed_exceeded_retries(self):
        result = parse_decision(
            "fix",
            exploration_result="found",
            review_feedback="needs changes",
            review_approved=False,
            retry_count=5,
            max_retries=3,
        )
        assert result == "finish"

    # ── 审查通过，未测试 → execute ──
    def test_review_passed_no_test(self):
        result = parse_decision(
            "done",
            exploration_result="found",
            review_feedback="looks good",
            review_approved=True,
            test_result=None,
        )
        assert result == "execute"

    # ── 审查通过，已测试 → finish ──
    def test_review_passed_with_test(self):
        result = parse_decision(
            "done",
            exploration_result="found",
            review_feedback="looks good",
            review_approved=True,
            test_result="10 passed",
        )
        assert result == "finish"

    # ── LLM 明确说 finish ──
    def test_finish_explicit(self):
        result = parse_decision(
            "task complete, finish",
            exploration_result="found",
            review_feedback="ok",
            review_approved=True,
            test_result="done",
        )
        assert result == "finish"

    # ── 空文本 ──
    def test_empty_text_no_exploration(self):
        result = parse_decision("", exploration_result=None)
        assert result == "explore"

    def test_empty_text_with_exploration(self):
        result = parse_decision("", exploration_result="found")
        assert result == "finish"


class TestRouteAfterSupervisor:
    """graph.py 中的 route_after_supervisor 测试"""

    def test_route_to_explore(self):
        from code_agent.graph import route_after_supervisor
        state = {"current_agent": "explore", "retry_count": 0, "max_retries": 3,
                 "review_feedback": None, "review_approved": False}
        assert route_after_supervisor(state) == "explore"

    def test_route_to_finish(self):
        from code_agent.graph import route_after_supervisor
        state = {"current_agent": "finish", "retry_count": 0, "max_retries": 3,
                 "review_feedback": None, "review_approved": False}
        assert route_after_supervisor(state) == "finish"

    def test_retry_exceeded_forces_finish(self):
        from code_agent.graph import route_after_supervisor
        state = {"current_agent": "code", "retry_count": 3, "max_retries": 3,
                 "review_feedback": "fix it", "review_approved": False}
        assert route_after_supervisor(state) == "finish"


class TestRouteAfterReviewerUpdate:
    """审修闭环路由测试"""

    def test_approved_goes_to_execute(self):
        from code_agent.graph import route_after_reviewer_update
        state = {"review_approved": True, "retry_count": 0, "max_retries": 3}
        assert route_after_reviewer_update(state) == "execute"

    def test_not_approved_under_limit_goes_to_coder(self):
        from code_agent.graph import route_after_reviewer_update
        state = {"review_approved": False, "retry_count": 1, "max_retries": 3}
        assert route_after_reviewer_update(state) == "coder"

    def test_not_approved_at_limit_goes_to_supervisor(self):
        from code_agent.graph import route_after_reviewer_update
        state = {"review_approved": False, "retry_count": 3, "max_retries": 3}
        assert route_after_reviewer_update(state) == "supervisor"
