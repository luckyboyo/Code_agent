"""评测报告渲染 — Rich 表格输出"""
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tests.eval.judge import JudgeResult


@dataclass
class CaseResult:
    agent: str
    case: str
    result: JudgeResult


@dataclass
class EvalReport:
    agent: str
    results: list[CaseResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0

    def add(self, case: str, result: JudgeResult):
        self.results.append(CaseResult(self.agent, case, result))
        self.total += 1
        if result.passed:
            self.passed += 1

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.result.score for r in self.results) / len(self.results)


class ReportRenderer:
    """Rich 渲染评测报告"""

    def __init__(self):
        self.console = Console()
        self.reports: dict[str, EvalReport] = {}

    def add_report(self, report: EvalReport):
        self.reports[report.agent] = report

    def render_summary(self) -> str:
        table = Table(title="Agent 评测结果", header_style="bold cyan")
        table.add_column("Agent", style="bold")
        table.add_column("通过率", justify="right")
        table.add_column("均分", justify="right")
        table.add_column("通过/总用例", justify="right")
        table.add_column("典型失败", style="dim")

        for agent, report in self.reports.items():
            rate = f"{report.pass_rate:.0%}"
            score = f"{report.avg_score:.2f}"
            count = f"{report.passed}/{report.total}"

            failures = [r for r in report.results if not r.result.passed]
            typical = ""
            if failures:
                typical = failures[0].case[:30] + ("..." if len(failures[0].case) > 30 else "")
                if len(failures) > 1:
                    typical += f" (+{len(failures)-1})"

            style = "green" if report.pass_rate >= 0.8 else ("yellow" if report.pass_rate >= 0.6 else "red")
            table.add_row(agent, f"[{style}]{rate}[/{style}]", score, count, typical)

        self.console.print(table)
        return ""

    def render_detail(self, agent: str):
        report = self.reports.get(agent)
        if not report:
            self.console.print(f"[yellow]无 {agent} 的评测数据[/yellow]")
            return

        self.console.print(f"\n[bold]{agent} 详细结果[/bold]")
        for cr in report.results:
            icon = "[green]✓[/green]" if cr.result.passed else "[red]✗[/red]"
            self.console.print(f"  {icon} {cr.case} [dim]({cr.result.reason})[/dim]")
            for check in cr.result.checks:
                c_icon = "  ✓" if check["pass"] else "  ✗"
                self.console.print(f"    {c_icon} {check['check']}")
