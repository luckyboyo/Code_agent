"""测试审查结果解析 — 通过/不通过的判断逻辑"""
import pytest
from code_agent.agents.reviewer import reviewer_node


class TestReviewParsing:
    """审查批准判断测试"""

    def test_approved_detected(self):
        text = "**审查通过**\n改动逻辑正确，无安全风险。"
        assert "审查通过" in text

    def test_rejected_detected(self):
        text = "**审查不通过**\n第23行存在安全问题。"
        assert "审查不通过" in text

    def test_approved_in_code_block_not_confused(self):
        """审查通过 出现在代码块里时不应混淆"""
        text = '''**审查不通过**
发现问题如下：
```
# 这段代码审查通过不了
```
请修复注入问题。'''
        assert "审查不通过" in text

    def test_english_approved(self):
        text = "Approved. The changes look good."
        assert "审查通过" not in text

    # ── 边界情况 ──
    @pytest.mark.parametrize("text,expected_approved", [
        ("**审查通过**", True),
        ("**审查不通过**", False),
        ("审查通过✅", True),
        ("审查不通过❌", False),
        ("前置文字 **审查通过** 后置文字", True),
        ("前置文字 **审查不通过** 后置文字", False),
    ])
    def test_approval_patterns(self, text, expected_approved):
        approved = "审查通过" in text
        rejected = "审查不通过" in text
        if expected_approved:
            assert approved and not rejected
        else:
            assert rejected


class TestTestPassedDetection:
    """执行器中 test_passed 判断测试"""

    def test_all_passed(self):
        output = "10 passed in 0.5s"
        assert "failed" not in output.lower() and "error" not in output.lower()

    def test_some_failed(self):
        output = "8 passed, 2 failed in 1.2s"
        assert "failed" in output.lower()

    def test_error_occurred(self):
        output = "ERROR: module not found"
        assert "error" in output.lower()

    def test_import_error_is_failure(self):
        output = "ImportError: No module named 'foo'"
        assert "error" in output.lower()
