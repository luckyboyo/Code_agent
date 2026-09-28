"""评测运行器 — 加载 Golden Dataset，驱动 Agent 执行，收集结果"""
import yaml
from pathlib import Path
from typing import Optional

from tests.eval.judge import Judge
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

        for case in dataset:
            case_name = case["name"]
            expected = case.get("expected", {})

            if self.mock:
                # 离线模式：用模拟输出进行结构评测
                mock_output = case.get("mock_output", "")
                mock_metadata = case.get("mock_metadata", {})
                judge = Judge(case_name, expected)
                result = judge.evaluate(mock_output, mock_metadata)
            else:
                # 真实模式：调用 Agent 节点
                output, metadata = self._invoke_agent(agent_name, case)
                judge = Judge(case_name, expected)
                result = judge.evaluate(output, metadata)

            report.add(case_name, result)

        return report

    def _load_dataset(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    def _invoke_agent(self, agent_name: str, case: dict) -> tuple[str, dict]:
        """真实调用 Agent 节点（需要 LLM）"""
        from code_agent.state import CodingState
        from langchain_core.messages import HumanMessage

        state: CodingState = {
            "messages": [HumanMessage(content=case.get("input", ""))],
            "user_request": case.get("input", ""),
            "workspace_dir": ".",
            "task_plan": "",
            "current_agent": agent_name,
            "exploration_result": case.get("context", {}).get("exploration_result"),
            "relevant_files": None,
            "code_changes": None,
            "review_feedback": case.get("context", {}).get("review_feedback"),
            "review_approved": case.get("context", {}).get("review_approved", False),
            "test_result": case.get("context", {}).get("test_result"),
            "test_passed": False,
            "retry_count": case.get("context", {}).get("retry_count", 0),
            "max_retries": 3,
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
