"""搜索工具 — Grep / Glob / ListDir"""
import os
import glob as glob_mod
import subprocess
from langchain_core.tools import tool


def _safe_dir(workspace_dir: str, sub_path: str = "") -> str:
    workspace_dir = os.path.abspath(workspace_dir)
    full = os.path.normpath(os.path.join(workspace_dir, sub_path))
    if not full.startswith(workspace_dir):
        raise ValueError(f"路径越权: {sub_path}")
    return full


@tool(description="在文件中搜索文本模式（正则）。入参: pattern(搜索模式), path(搜索目录，默认'.'), glob(文件过滤，如'*.py')")
def grep(pattern: str, path: str = ".", glob: str = "*", workspace_dir: str = ".") -> str:
    search_dir = _safe_dir(workspace_dir, path)
    try:
        result = subprocess.run(
            ["rg", "--line-number", "--max-count=30", "--glob", glob, pattern, search_dir],
            capture_output=True, text=True, timeout=15, encoding="utf-8"
        )
        output = result.stdout.strip()
        return output if output else f"[无匹配] {pattern}"
    except FileNotFoundError:
        # 如果没有 ripgrep，回退到 Python
        import re
        matches = []
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if not glob_mod.fnmatch.fnmatch(fname, glob):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if re.search(pattern, line) and len(matches) < 30:
                                rel = os.path.relpath(fpath, workspace_dir)
                                matches.append(f"{rel}:{i}: {line.strip()[:120]}")
                except Exception:
                    continue
        return "\n".join(matches) if matches else f"[无匹配] {pattern}"


@tool(description="按文件名模式查找文件。入参: pattern(glob模式，如'**/*.py')")
def glob_files(pattern: str, workspace_dir: str = ".") -> str:
    search_dir = _safe_dir(workspace_dir)
    files = glob_mod.glob(pattern, root_dir=search_dir, recursive=True)
    if not files:
        return f"[无匹配] {pattern}"
    return "\n".join(sorted(files)[:50])


@tool(description="列出目录内容。入参: path(相对路径，默认'.')")
def list_dir(path: str = ".", workspace_dir: str = ".") -> str:
    target = _safe_dir(workspace_dir, path)
    if not os.path.isdir(target):
        return f"[错误] 不是目录: {path}"
    items = os.listdir(target)
    if not items:
        return f"[空目录] {path}"

    lines = []
    for name in sorted(items):
        full = os.path.join(target, name)
        tag = "/" if os.path.isdir(full) else ""
        lines.append(f"  {name}{tag}")
    return f"── {path or '.'}/ ──\n" + "\n".join(lines)
