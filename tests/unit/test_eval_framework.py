"""测试评测框架本身 — 验证 Judge、Report、Runner 正常工作"""
import pytest
from tests.eval.judge import Judge, JudgeResult
from tests.eval.report import EvalReport, CaseResult


class TestJudge:
    """评判引擎测试"""

    def test_must_contain_pass(self):
        judge = Judge("test_case", {"must_contain": ["hello", "world"]})
        result = judge.evaluate("hello world foo bar")
        assert result.passed
        assert result.score == 1.0

    def test_must_contain_fail(self):
        judge = Judge("test_case", {"must_contain": ["hello", "missing"]})
        result = judge.evaluate("hello world")
        assert not result.passed
        assert result.score == 0.5  # 1/2 checks passed

    def test_must_not_contain_pass(self):
        judge = Judge("test_case", {"must_not_contain": ["evil", "hack"]})
        result = judge.evaluate("clean code")
        assert result.passed

    def test_must_not_contain_fail(self):
        judge = Judge("test_case", {"must_not_contain": ["evil"]})
        result = judge.evaluate("this is evil code")
        assert not result.passed

    def test_must_match_regex(self):
        judge = Judge("test_case", {"must_match": [r"\d+ passed"]})
        result = judge.evaluate("89 passed in 0.5s")
        assert result.passed

    def test_must_match_regex_fail(self):
        judge = Judge("test_case", {"must_match": [r"\d+ passed"]})
        result = judge.evaluate("all tests failed")
        assert not result.passed

    def test_route_decision_match(self):
        judge = Judge("test_case", {"route_decision": "explore"})
        result = judge.evaluate("explore the code", {"decision": "explore"})
        assert result.passed

    def test_route_decision_mismatch(self):
        judge = Judge("test_case", {"route_decision": "explore"})
        result = judge.evaluate("", {"decision": "code"})
        assert not result.passed

    def test_empty_expected(self):
        judge = Judge("test_case", {})
        result = judge.evaluate("anything")
        assert result.score == 0.5  # 中性分
        assert result.passed  # 无检查则通过

    def test_composite_rules(self):
        judge = Judge("test_case", {
            "must_contain": ["审查通过"],
            "must_not_contain": ["审查不通过"],
        })
        output = "**审查通过** 改动正确，小建议：第3行可优化"
        result = judge.evaluate(output)
        assert result.passed
        assert result.score == 1.0

    def test_case_insensitive(self):
        judge = Judge("test_case", {"must_contain": ["HELLO"]})
        result = judge.evaluate("hello world")
        assert result.passed


class TestEvalReport:
    """评测报告测试"""

    def test_empty_report(self):
        report = EvalReport(agent="supervisor")
        assert report.total == 0
        assert report.passed == 0
        assert report.pass_rate == 0.0
        assert report.avg_score == 0.0

    def test_single_pass(self):
        report = EvalReport(agent="supervisor")
        result = JudgeResult(passed=True, score=1.0, checks=[{"check": "test", "pass": True}])
        report.add("case1", result)
        assert report.total == 1
        assert report.passed == 1
        assert report.pass_rate == 1.0
        assert report.avg_score == 1.0

    def test_mixed_results(self):
        report = EvalReport(agent="explorer")
        report.add("case1", JudgeResult(passed=True, score=1.0))
        report.add("case2", JudgeResult(passed=False, score=0.3))
        report.add("case3", JudgeResult(passed=True, score=0.8))
        assert report.total == 3
        assert report.passed == 2
        assert report.pass_rate == pytest.approx(2/3)
        assert report.avg_score == pytest.approx(2.1 / 3)
