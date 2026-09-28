"""文件操作工具 — Read / Write / Edit"""
import os
import difflib
from pathlib import Path
from langchain_core.tools import tool


def _safe_path(base_dir: str, file_path: str) -> str:
    """防止路径穿越，确保在 workspace 内"""
    base_dir = os.path.abspath(base_dir)
    full = os.path.normpath(os.path.join(base_dir, file_path))
    if not full.startswith(base_dir):
        raise ValueError(f"路径越权: {file_path}")
    return full


def _make_diff(file_path: str, original: str, replacement: str) -> str:
    """生成 unified diff 字符串"""
    orig_lines = original.splitlines(keepends=True) or ["\n"]
    repl_lines = replacement.splitlines(keepends=True) or ["\n"]
    diff = difflib.unified_diff(
        orig_lines, repl_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


@tool(description="读取指定文件的全部内容。入参: file_path(相对于工作目录的文件路径)")
def read_file(file_path: str, workspace_dir: str = ".") -> str:
    abs_path = _safe_path(workspace_dir, file_path)
    if not os.path.exists(abs_path):
        return f"[错误] 文件不存在: {file_path}"
    if not os.path.isfile(abs_path):
        return f"[错误] 不是文件: {file_path}"
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    # 加上行号
    lines = content.split("\n")
    numbered = "\n".join(f"{i+1:4d}| {line}" for i, line in enumerate(lines))
    return f"── {file_path} ({len(lines)} 行) ──\n{numbered}"


@tool(description="创建或覆盖文件。入参: file_path(相对路径), content(文件内容)")
def write_file(file_path: str, content: str, workspace_dir: str = ".") -> str:
    abs_path = _safe_path(workspace_dir, file_path)
    existed = os.path.exists(abs_path)
    old_content = None
    if existed:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            old_content = f.read()

    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    lines = content.count("\n") + 1

    tag = "[DIFF:write]"
    diff_text = ""
    if existed and old_content is not None:
        diff_text = _make_diff(file_path, old_content, content)
    else:
        diff_text = f"@@ 新文件: {file_path} ({lines} 行, {len(content)} 字符)"

    return f"{tag}\n{diff_text}\n[已写入] {file_path} ({lines} 行, {len(content)} 字符)"


@tool(description="精确替换文件中的字符串。入参: file_path(相对路径), old_string(要被替换的内容), new_string(替换后的内容)")
def edit_file(file_path: str, old_string: str, new_string: str, workspace_dir: str = ".") -> str:
    abs_path = _safe_path(workspace_dir, file_path)
    if not os.path.exists(abs_path):
        return f"[错误] 文件不存在: {file_path}"

    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_string)
    if count == 0:
        return f"[错误] 未找到要替换的内容，请确认 old_string 与文件内容精确匹配"
    if count > 1:
        return (f"[错误] old_string 匹配到 {count} 处，请提供更多上下文使匹配唯一。"
                f"\n匹配位置: {_find_positions(content, old_string)}")

    new_content = content.replace(old_string, new_string, 1)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    diff_text = _make_diff(file_path, content, new_content)
    return f"[DIFF:edit]\n{diff_text}\n[已修改] {file_path} (1 处替换)"


def _find_positions(content: str, target: str, max_shown: int = 5) -> str:
    positions = []
    start = 0
    while True:
        idx = content.find(target, start)
        if idx == -1:
            break
        line_num = content[:idx].count("\n") + 1
        positions.append(f"第{line_num}行")
        start = idx + 1
        if len(positions) >= max_shown:
            remaining = content[start:].count(target)
            if remaining > 0:
                positions.append(f"...及其他{remaining}处")
            break
    return "、".join(positions)
