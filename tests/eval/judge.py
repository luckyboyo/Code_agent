"""评判引擎 — 结构断言 + 语义评分"""
from dataclasses import dataclass, field
import re


@dataclass
class JudgeResult:
    passed: bool
    score: float  # 0.0 ~ 1.0
    checks: list[dict] = field(default_factory=list)
    reason: str = ""


class Judge:
    """对 Agent 输出执行多维度评判"""

    def __init__(self, case_name: str, expected: dict):
        self.case_name = case_name
        self.expected = expected

    def evaluate(self, output: str, metadata: dict | None = None) -> JudgeResult:
        checks = []
        all_pass = True

        # ── 结构断言 ──
        if "must_contain" in self.expected:
            for keyword in self.expected["must_contain"]:
                ok = keyword.lower() in output.lower()
                checks.append({"check": f"must_contain: {keyword}", "pass": ok})
                if not ok:
                    all_pass = False

        if "must_not_contain" in self.expected:
            for keyword in self.expected["must_not_contain"]:
                ok = keyword.lower() not in output.lower()
                checks.append({"check": f"must_not_contain: {keyword}", "pass": ok})
                if not ok:
                    all_pass = False

        if "must_match" in self.expected:
            for pattern in self.expected["must_match"]:
                ok = bool(re.search(pattern, output, re.IGNORECASE))
                checks.append({"check": f"must_match: {pattern}", "pass": ok})
                if not ok:
                    all_pass = False

        # ── 路由决策断言 ──
        if "route_decision" in self.expected and metadata:
            actual = metadata.get("decision", "")
            expected_dec = self.expected["route_decision"]
            ok = actual == expected_dec
            checks.append({"check": f"route_decision: {expected_dec}", "pass": ok})
            if not ok:
                all_pass = False

        # ── 计分 ──
        if checks:
            score = sum(1 for c in checks if c["pass"]) / len(checks)
        else:
            score = 0.5  # 无检查项时中性分

        reason = "全部通过" if all_pass else f"{sum(1 for c in checks if not c['pass'])}/{len(checks)} 项未通过"

        return JudgeResult(
            passed=all_pass,
            score=score,
            checks=checks,
            reason=reason,
        )
