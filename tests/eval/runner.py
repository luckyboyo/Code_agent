"""评测运行器 — 加载 Golden Dataset，驱动 Agent 执行，收集结果"""
import yaml
import os
import shutil
import tempfile
import time
from pathlib import Path

from rich.console import Console
from tests.eval.judge import Judge, JudgeResult
from tests.eval.report import EvalReport, ReportRenderer


DATASET_DIR = Path(__file__).parent / "datasets"


class EvalRunner:
    """加载数据集并驱动评测"""

    def __init__(self, mock: bool = True):
        """
        mock=True: 不调用 LLM，只测框架逻辑 (离线模式)
        mock=False: 真实调用 Agent (需要 API Key)
        """
        self.mock = mock
        self.renderer = ReportRenderer()
        self.console = Console()

    def run_all(self, agents: list[str] | None = None) -> ReportRenderer:
        if agents is None:
            agents = ["supervisor", "explorer", "coder", "reviewer", "executor"]

        for agent_name in agents:
            dataset_path = DATASET_DIR / f"{agent_name}.yaml"
            if not dataset_path.exists():
                continue
            report = self.run_agent(agent_name, dataset_path)
            self.renderer.add_report(report)

        return self.renderer

    def run_agent(self, agent_name: str, dataset_path: Path) -> EvalReport:
        dataset = self._load_dataset(dataset_path)
        report = EvalReport(agent=agent_name)

        total_cases = len(dataset)
        for case_index, case in enumerate(dataset, start=1):
            case_name = case["name"]
            expected = case.get("expected", {})
            started_at = time.perf_counter()
            output = ""

            if not self.mock:
                self.console.print(
                    f"[cyan][LIVE] {agent_name} {case_index}/{total_cases} 开始：{case_name}[/cyan]"
                )

            try:
                if self.mock:
                    # 离线模式：用模拟输出进行结构评测
                    output = case.get("mock_output", "")
                    metadata = case.get("mock_metadata", {})
                else:
                    # 真实模式：在临时项目副本中调用 Agent，避免评测改动当前工作区
                    output, metadata = self._invoke_agent_in_temp_workspace(agent_name, case)

                judge = Judge(case_name, expected)
                result = judge.evaluate(output, metadata)
            except Exception as exc:
                result = JudgeResult(
                    passed=False,
                    score=0.0,
                    checks=[{"check": "agent_execution", "pass": False}],
                    reason=f"执行异常: {type(exc).__name__}: {exc}",
                )

            report.add(case_name, result)
            if not self.mock:
                elapsed = time.perf_counter() - started_at
                style = "green" if result.passed else "red"
                self.console.print(
                    f"[{style}][LIVE] {agent_name} {case_index}/{total_cases} "
                    f"{'通过' if result.passed else '失败'}，得分 {result.score:.2f}，耗时 {elapsed:.1f}s[/{style}]"
                )
                if not result.passed:
                    for check in result.checks:
                        if not check.get("pass", False):
                            self.console.print(f"[red]  未通过断言：{check.get('check', '未知检查')}[/red]")
                    if result.reason.startswith("执行异常:"):
                        self.console.print(f"[red]  {result.reason}[/red]")
                    elif output:
                        preview = str(output).strip()
                        if len(preview) > 1200:
                            preview = preview[:1200] + "…（已截断）"
                        self.console.print(f"[yellow]  Agent 实际输出：\n{preview}[/yellow]")

        return report

    def _load_dataset(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    def _invoke_agent_in_temp_workspace(self, agent_name: str, case: dict) -> tuple[str, dict]:
        """在不含 .env/.git 的临时项目副本中调用真实 Agent。"""
        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="sh-agent-eval-") as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            shutil.copytree(
                project_root,
                workspace,
                ignore=shutil.ignore_patterns(
                    ".git", ".env", ".pytest_cache", "__pycache__", "*.pyc",
                    "*.egg-info", ".venv", "venv", "build", "dist",
                ),
            )

            previous_cwd = os.getcwd()
            try:
                # 工具默认以当前目录为工作区；切到临时副本，让文件和测试操作落在副本中。
                os.chdir(workspace)
                return self._invoke_agent(agent_name, case, workspace)
            finally:
                os.chdir(previous_cwd)

    def _invoke_agent(self, agent_name: str, case: dict, workspace: Path) -> tuple[str, dict]:
        """调用指定 Agent 节点并收集输出及元数据。"""
        from code_agent.state import CodingState
        from langchain_core.messages import HumanMessage

        context = case.get("context", {})
        state: CodingState = {
            "messages": [HumanMessage(content=case.get("input", ""))],
            "user_request": case.get("input", ""),
            "workspace_dir": str(workspace),
            "task_plan": "",
            "current_agent": agent_name,
            "exploration_result": context.get("exploration_result"),
            "relevant_files": context.get("relevant_files"),
            "code_changes": context.get("code_changes"),
            "review_feedback": context.get("review_feedback"),
            "review_approved": context.get("review_approved", False),
            "test_result": context.get("test_result"),
            "test_passed": context.get("test_passed", False),
            "retry_count": context.get("retry_count", 0),
            "max_retries": context.get("max_retries", 3),
            "final_response": None,
            "task_complete": False,
        }

        metadata = {}

        if agent_name == "supervisor":
            from code_agent.agents.supervisor import supervisor_node, parse_decision
            result = supervisor_node(state)
            output = result.get("messages", [None])[-1].content if result.get("messages") else ""
            metadata["decision"] = parse_decision(
                output,
                exploration_result=state.get("exploration_result"),
                review_feedback=state.get("review_feedback"),
                review_approved=state.get("review_approved", False),
                test_result=state.get("test_result"),
                retry_count=state.get("retry_count", 0),
                max_retries=state.get("max_retries", 3),
            )

        elif agent_name == "explorer":
            from code_agent.agents.explorer import explorer_node
            result = explorer_node(state)
            output = result.get("exploration_result", "")
            metadata["files"] = result.get("relevant_files", [])

        elif agent_name == "coder":
            from code_agent.agents.coder import coder_node
            result = coder_node(state)
            messages = result.get("messages", [])
            output = next(
                (message.content for message in reversed(messages)
                 if getattr(message, "type", None) == "ai" and getattr(message, "content", "")),
                "",
            )

        elif agent_name == "reviewer":
            from code_agent.agents.reviewer import reviewer_node
            result = reviewer_node(state)
            output = result.get("review_feedback", "")
            metadata["approved"] = result.get("review_approved", False)

        elif agent_name == "executor":
            from code_agent.agents.executor import executor_node
            result = executor_node(state)
            output = result.get("test_result", "")
            metadata["passed"] = result.get("test_passed", False)

        else:
            output = ""
            metadata = {}

        return output, metadata


def run_eval(mock: bool = True, agents: list[str] | None = None) -> ReportRenderer:
    runner = EvalRunner(mock=mock)
    return runner.run_all(agents)
