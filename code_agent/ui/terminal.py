"""终端 UI — Rich 渲染 Multi-Agent 输出"""
import sys
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from pathlib import Path


console = Console()

AGENT_COLORS = {
    "supervisor": "cyan",
    "explorer": "green",
    "coder": "yellow",
    "reviewer": "magenta",
    "executor": "blue",
    "finalizer": "white",
}


def render_agent_header(agent_name: str) -> Panel:
    color = AGENT_COLORS.get(agent_name, "white")
    emoji_map = {
        "supervisor": "🧠",
        "explorer": "🔍",
        "coder": "✏️",
        "reviewer": "👀",
        "executor": "🧪",
    }
    emoji = emoji_map.get(agent_name, "📋")
    title = f"{emoji} {agent_name.upper()}"
    return Panel(title, style=color, padding=(0, 2))


def render_banner():
    console.print()
    console.print(
        Panel(
            "[bold cyan]SH Agent v0.1.0[/bold cyan]\n"
            "[dim]Multi-Agent 编码助手 | LangGraph + Redis + PostgreSQL[/dim]\n\n"
            "[bold white]你好！我是 SH Agent 🤖[/bold white]\n"
            "由 5 个 AI Agent 协作完成你的编码任务：\n"
            "  🧠 Supervisor  ·  调度决策\n"
            "  🔍 Explorer   ·  分析代码\n"
            "  ✏️ Coder      ·  编写修改\n"
            "  👀 Reviewer   ·  审查质量\n"
            "  🧪 Executor   ·  运行验证\n\n"
            "[bold]快速开始 — 试试这些:[/bold]\n"
            "  [cyan]1.[/cyan] 列出当前项目的文件结构\n"
            "  [cyan]2.[/cyan] 帮我分析 code_agent/graph.py 的代码逻辑\n"
            "  [cyan]3.[/cyan] 审查 code_agent/agents/supervisor.py 的代码质量\n"
            "  [cyan]4.[/cyan] 在 code_agent/ 下新增一个单元测试模块\n\n"
            "输入 [bold]/help[/bold] 查看命令   输入 [bold]/quit[/bold] 退出",
            title="🚀 欢迎",
            border_style="cyan",
        )
    )
    console.print()


def render_help():
    console.print(
        Panel(
            "[bold]可用命令:[/bold]\n"
            "  [cyan]/help[/cyan]    显示此帮助\n"
            "  [cyan]/new[/cyan]     开始新会话（清空上下文）\n"
            "  [cyan]/quit[/cyan]    退出程序\n"
            "  [cyan]/clear[/cyan]   清屏\n"
            "  [cyan]/status[/cyan]  查看当前会话状态\n"
            "  [cyan]/setup[/cyan]   重新配置 API\n"
            "  [cyan]/eval[/cyan]    运行 Agent 评测（离线模式）\n\n"
            "[bold]直接输入编程问题即可开始:[/bold]\n"
            "  - 帮我修复 xxx.py 中的空指针异常\n"
            "  - 在 src/ 下新增一个用户认证模块\n"
            "  - 这段代码有什么问题？[粘贴代码]\n"
            "  - 运行测试并告诉我哪些失败了",
            title="📖 帮助",
            border_style="cyan",
        )
    )


def create_prompt_session() -> PromptSession:
    history_file = Path.home() / ".sh_agent_history"
    style = Style.from_dict({
        "prompt": "bold cyan",
    })
    return PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        style=style,
    )
