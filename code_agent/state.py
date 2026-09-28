"""CodingState — Multi-Agent 共享状态定义"""
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class FileChange(TypedDict):
    file_path: str
    original: str
    replacement: str
    reason: str


class CodingState(TypedDict):
    # ── 全局消息流（所有 Agent 可见） ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 用户原始需求 ──
    user_request: str

    # ── 工作目录 ──
    workspace_dir: str

    # ── Supervisor 规划 ──
    task_plan: str                          # 任务拆解计划
    current_agent: str                      # 当前激活的 Agent 名

    # ── Explorer 产出 ──
    exploration_result: Optional[str]       # 代码分析结果
    relevant_files: Optional[list[str]]     # 相关文件列表

    # ── Coder 产出 ──
    code_changes: Optional[list[FileChange]]  # 改动记录

    # ── Reviewer 产出 ──
    review_feedback: Optional[str]          # 审查意见
    review_approved: bool                   # 是否通过审查

    # ── Executor 产出 ──
    test_result: Optional[str]              # 测试/运行结果
    test_passed: bool                       # 测试是否通过

    # ── 循环控制 ──
    retry_count: int                        # 当前审修轮次
    max_retries: int                        # 最大审修轮次

    # ── 最终产出 ──
    final_response: Optional[str]           # 给用户的最终回复
    task_complete: bool                     # 任务是否完成
