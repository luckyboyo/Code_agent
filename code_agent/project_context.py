"""项目上下文 — 自动检测项目类型、结构，注入 Agent prompt"""
import os


def get_project_context(workspace_dir: str) -> dict:
    """扫描工作目录，返回项目上下文"""
    ctx = {
        "type": "unknown",
        "name": os.path.basename(os.path.abspath(workspace_dir)),
        "top_files": [],
        "top_dirs": [],
        "build_system": None,
        "test_dir": None,
        "main_source": None,
    }

    try:
        entries = sorted(os.listdir(workspace_dir))
    except PermissionError:
        return ctx

    for entry in entries:
        full = os.path.join(workspace_dir, entry)
        if entry.startswith("."):
            continue
        if os.path.isdir(full):
            ctx["top_dirs"].append(entry)
        else:
            ctx["top_files"].append(entry)

    # 检测项目类型
    if "pyproject.toml" in ctx["top_files"] or "setup.py" in ctx["top_files"]:
        ctx["type"] = "python"
        ctx["build_system"] = "pyproject.toml" if "pyproject.toml" in ctx["top_files"] else "setup.py"
    elif "package.json" in ctx["top_files"]:
        ctx["type"] = "node"
        ctx["build_system"] = "package.json"
    elif "go.mod" in ctx["top_files"]:
        ctx["type"] = "go"
        ctx["build_system"] = "go.mod"
    elif "Cargo.toml" in ctx["top_files"]:
        ctx["type"] = "rust"
        ctx["build_system"] = "Cargo.toml"

    # 检测主源码目录
    for d in ctx["top_dirs"]:
        lower = d.lower()
        if lower in ("src", "lib", "code_agent", "app"):
            ctx["main_source"] = d
            break

    # 检测测试目录
    for d in ctx["top_dirs"]:
        if "test" in d.lower():
            ctx["test_dir"] = d
            break

    return ctx


def format_project_context(ctx: dict) -> str:
    """将项目上下文格式化为 prompt 可用文本"""
    lines = [f"项目名称: {ctx['name']}", f"项目类型: {ctx['type']}"]

    if ctx["build_system"]:
        lines.append(f"构建系统: {ctx['build_system']}")
    if ctx["main_source"]:
        lines.append(f"主源码目录: {ctx['main_source']}/")
    if ctx["test_dir"]:
        lines.append(f"测试目录: {ctx['test_dir']}/")

    if ctx["top_dirs"]:
        dirs = " ".join(f"{d}/" for d in ctx["top_dirs"][:10])
        lines.append(f"顶层目录: {dirs}")
    if ctx["top_files"]:
        files = " ".join(ctx["top_files"][:10])
        lines.append(f"顶层文件: {files}")

    return "\n".join(lines)
