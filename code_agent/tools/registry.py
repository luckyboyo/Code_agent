"""工具注册 — 汇总所有工具并按 Agent 分配"""
from code_agent.tools.file_tools import read_file, write_file, edit_file
from code_agent.tools.search_tools import grep, glob_files, list_dir
from code_agent.tools.shell_tools import bash

# ── 全部工具 ──
ALL_TOOLS = [read_file, write_file, edit_file, grep, glob_files, list_dir, bash]

# ── 各 Agent 的工具权限分配 ──
AGENT_TOOLS = {
    "explorer":    [read_file, grep, glob_files, list_dir],
    "coder":       [read_file, write_file, edit_file, grep, glob_files, list_dir],
    "reviewer":    [read_file, grep, list_dir, bash],
    "executor":    [bash, read_file],
    "supervisor":  [],   # Supervisor 纯推理，不持有工具
}
