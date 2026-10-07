"""CLI 入口 — 终端 REPL"""
import os
import sys
import uuid
from datetime import datetime, timezone
from rich.markdown import Markdown
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markup import escape
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from code_agent.state import CodingState
from code_agent.graph import compile_graph
from code_agent.config import get_setting
from code_agent.ui.terminal import (
    console, render_banner, render_help, render_agent_header, create_prompt_session,
)
from code_agent.ui.stream_handler import TokenStreamHandler


console = Console()
LOCAL_TASKS: dict[str, dict] = {}
LOCAL_SESSIONS: dict[str, dict] = {}


def main():
    render_banner()

    # 没有 .env 则自动启动配置向导
    if not os.path.exists(".env"):
        console.print("[yellow]未检测到 .env 配置文件，启动配置向导...[/yellow]")
        _setup_wizard()

    # 编译 graph
    console.print("[dim]正在初始化 Multi-Agent 系统...[/dim]")
    try:
        graph = compile_graph(with_checkpoint=True)
        from code_agent.storage.redis_store import RedisStore
        redis_store = RedisStore.get_instance()
        console.print("[green]Redis checkpoint 和缓存已启用[/green]")
    except Exception as exc:
        console.print(f"[yellow]Redis checkpoint 不可用（{exc}）[/yellow]")
        console.print("[yellow]改用进程内 checkpoint；退出程序后无法恢复任务[/yellow]")
        graph = compile_graph(with_checkpoint=False, in_memory_checkpoint=True)
        redis_store = None

    from code_agent.storage.sql_store import SQLStore
    sql_store = SQLStore.get_instance()
    try:
        if sql_store.initialize():
            migrated_sessions, migrated_tasks = (
                sql_store.migrate_legacy_redis_data(redis_store)
                if redis_store else (0, 0)
            )
            console.print("[green]PostgreSQL 会话、消息和任务状态存储已启用[/green]")
            if migrated_sessions or migrated_tasks:
                console.print(
                    f"[dim]已从旧 Redis 数据迁移 {migrated_sessions} 个会话、"
                    f"{migrated_tasks} 个任务[/dim]"
                )
        else:
            console.print("[yellow]PostgreSQL 存储未启用，会话和任务只保存在当前进程[/yellow]")
            sql_store = None
    except Exception as exc:
        console.print(f"[yellow]PostgreSQL 不可用（{exc}）[/yellow]")
        console.print("[yellow]会话和任务暂存于当前进程；配置数据库后重启即可启用持久化[/yellow]")
        sql_store = None

    console.print(
        "\n[dim]💡 输入数字 1-4 快速开始，或直接输入你的编程问题[/dim]"
    )

    # 终端 REPL
    session = create_prompt_session()
    workspace_dir = os.getcwd()

    # session_id 标识一段可恢复会话；每个问题仍使用独立 task_id/thread_id。
    session_id = uuid.uuid4().hex[:12]
    messages_history: list = []
    turn = 0
    console.print(f"[dim]当前会话 ID: {session_id}[/dim]")

    # 快捷问题映射
    shortcuts = {
        "1": "列出当前项目的文件结构",
        "2": "帮我分析 code_agent/graph.py 的代码逻辑",
        "3": "审查 code_agent/agents/supervisor.py 的代码质量",
        "4": "在 code_agent/ 下新增一个单元测试模块",
    }

    while True:
        try:
            prompt_text = f"\n[{turn + 1}] > " if messages_history else "\n> "
            user_input = session.prompt([("class:prompt", prompt_text)]).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue

        # 快捷数字映射
        if user_input in shortcuts:
            user_input = shortcuts[user_input]
            console.print(f"[dim]→ {user_input}[/dim]")

        # 内置命令
        if user_input.startswith("/"):
            if user_input.lower().strip() in ("/new",):
                messages_history.clear()
                turn = 0
                session_id = uuid.uuid4().hex[:12]
                console.clear()
                render_banner()
                console.print(f"[green]✅ 已开始新会话，ID: {session_id}[/green]")
                continue
            command_result = _handle_command(
                user_input,
                console,
                messages_history,
                graph=graph,
                sql_store=sql_store,
                workspace_dir=workspace_dir,
                session_id=session_id,
                turn=turn,
            )
            if command_result and command_result.get("action") == "resume_session":
                session_id = command_result["session_id"]
                messages_history = command_result["messages_history"]
                workspace_dir = command_result["workspace_dir"]
                turn = command_result["turn"]
                console.print(f"[green]✅ 已恢复会话 {session_id}（{turn} 轮）[/green]")
            continue

        turn += 1

        # 每个任务使用独立且持久的 thread_id；恢复时必须复用同一个 ID。
        task_id = uuid.uuid4().hex[:12]
        config = {
            "configurable": {"thread_id": task_id},
            "callbacks": [TokenStreamHandler()],
            "recursion_limit": int(get_setting("agent", "graph_recursion_limit") or 64),
        }

        initial_state = _build_initial_state(user_input, workspace_dir, messages_history)
        meta = {
            "thread_id": task_id,
            "session_id": session_id,
            "session_turn": turn,
            "status": "running",
            "user_request": user_input,
            "workspace_dir": workspace_dir,
            "history": _serialize_history(messages_history),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if sql_store:
            try:
                sql_store.save_task(task_id, meta)
            except Exception as exc:
                console.print(f"[red]无法登记可恢复任务，未启动执行: {exc}[/red]")
                continue
        else:
            LOCAL_TASKS[task_id] = {"task_id": task_id, **meta}

        # 空会话不落库；第一个真实任务登记成功后才创建持久化会话。
        _persist_session(
            session_id, workspace_dir, messages_history, turn, sql_store,
            last_task_id=task_id,
        )

        result = _run_task(graph, initial_state, config, task_id, meta, sql_store)
        final = result.get("final_response")
        if final:
            messages_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=final),
            ])
            if len(messages_history) > 40:
                messages_history = messages_history[-40:]
            _persist_session(
                session_id, workspace_dir, messages_history, turn, sql_store,
                last_task_id=task_id,
                completed_task_id=task_id,
            )

        console.print("[dim]─" * 60 + "[/dim]")


def _render_message(message, agent_name: str):
    """渲染一条消息 — AI 文本已流式输出，只处理工具调用/结果/用户输入/diff"""
    content = message.content if hasattr(message, "content") else str(message)
    if not content:
        return

    # 工具调用消息
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            name = tc.get("name", "?")
            args = str(tc.get("args", {}))[:80]
            console.print(f"  [dim]🔧 {name}({args})[/dim]")
        return

    # 工具返回结果 — 检查是否包含 diff
    if hasattr(message, "name") and message.name:
        _render_tool_result(content)
        return

    # 用户消息
    if hasattr(message, "type") and message.type == "human":
        console.print(f"[bold white]{content}[/bold white]")
        return

    # AI 消息内容已通过 TokenStreamHandler 流式输出，此处跳过


def _render_tool_result(content: str):
    """渲染工具返回结果，diff 内容高亮显示"""
    # 检测 diff 标记
    if content.startswith("[DIFF:edit]") or content.startswith("[DIFF:write]"):
        parts = content.split("\n", 1)
        if len(parts) < 2:
            return
        body = parts[1]

        diff_part = body
        status_line = ""
        tag = "修改"

        if "\n[已修改]" in body:
            diff_part, status_line = body.rsplit("\n[已修改]", 1)
            tag = "已修改"
        elif "\n[已写入]" in body:
            diff_part, status_line = body.rsplit("\n[已写入]", 1)
            tag = "已写入"

        if diff_part.strip():
            if diff_part.startswith("@@ 新文件:"):
                console.print(f"  [dim]  → {diff_part}[/dim]")
            else:
                syntax = Syntax(
                    diff_part, "diff", theme="monokai",
                    line_numbers=False, word_wrap=True
                )
                panel = Panel(
                    syntax,
                    title=f"[bold yellow]📝 {tag}[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
                console.print(panel)

        if status_line:
            console.print(f"  [dim]  → [{tag}] {status_line}[/dim]")
        return

    # 普通工具结果
    preview = content[:200].replace("\n", " ")
    if len(content) > 200:
        preview += "..."
    console.print(f"  [dim]  → {preview}[/dim]")


def _build_initial_state(user_input: str, workspace_dir: str, history: list) -> CodingState:
    """为新任务构建图状态；任务恢复时从 Redis checkpoint 读取该状态。"""
    return {
        "messages": list(history) + [HumanMessage(content=user_input)],
        "user_request": user_input,
        "workspace_dir": workspace_dir,
        "task_plan": "",
        "current_agent": "supervisor",
        "exploration_result": None,
        "relevant_files": None,
        "code_changes": None,
        "review_feedback": None,
        "review_approved": False,
        "approval_required": False,
        "approval_decision": None,
        "test_result": None,
        "test_passed": False,
        "retry_count": 0,
        "max_retries": get_setting("agent", "max_review_retries"),
        "final_response": None,
        "task_complete": False,
    }


def _serialize_history(messages: list) -> list[dict]:
    """将 REPL 的人类/助手文本历史转成可放入 Redis JSON 的结构。"""
    serialized = []
    for message in messages:
        role = "human" if getattr(message, "type", "") == "human" else "assistant"
        content = getattr(message, "content", "")
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
    return serialized


def _restore_history(serialized: list[dict]) -> list:
    messages = []
    for item in serialized:
        content = item.get("content", "")
        if item.get("role") == "human":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_session_meta(sql_store, session_id: str) -> dict | None:
    if sql_store:
        try:
            return sql_store.get_session(session_id) or LOCAL_SESSIONS.get(session_id)
        except Exception as exc:
            console.print(f"[yellow]读取 PostgreSQL 会话失败: {exc}[/yellow]")
    return LOCAL_SESSIONS.get(session_id)


def _persist_session(
    session_id: str,
    workspace_dir: str,
    messages_history: list,
    turn: int,
    sql_store,
    last_task_id: str | None = None,
    completed_task_id: str | None = None,
):
    """保存会话元数据和供模型使用的近期上下文。完整 transcript 从关联任务记录读取。"""
    existing = _get_session_meta(sql_store, session_id) or {}
    completed_ids = list(existing.get("completed_task_ids", []))
    if completed_task_id and completed_task_id not in completed_ids:
        completed_ids.append(completed_task_id)

    meta = {
        "session_id": session_id,
        "workspace_dir": workspace_dir,
        # 这里只保留最近 20 轮左右供下一轮模型调用；完整历史由 task 记录组成。
        "history": _serialize_history(messages_history[-40:]),
        "turn": turn,
        "completed_task_ids": completed_ids,
        "last_task_id": last_task_id or existing.get("last_task_id"),
        "created_at": existing.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }
    if sql_store:
        try:
            sql_store.save_session(session_id, meta)
            return
        except Exception as exc:
            console.print(f"[yellow]保存 PostgreSQL 会话失败，当前进程会暂存: {exc}[/yellow]")
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    LOCAL_SESSIONS[session_id] = meta


def _list_session_tasks(sql_store, session_id: str) -> list[dict]:
    """返回按轮次排序的 PostgreSQL 任务，失败时使用进程内回退数据。"""
    if sql_store:
        try:
            return sql_store.list_session_tasks(session_id)
        except Exception as exc:
            console.print(f"[yellow]读取 PostgreSQL 会话任务失败: {exc}[/yellow]")
    tasks = [
        {"task_id": task_id, **meta}
        for task_id, meta in LOCAL_TASKS.items()
        if meta.get("session_id") == session_id
    ]
    return sorted(
        tasks,
        key=lambda item: (
            _as_int(item.get("session_turn")), str(item.get("created_at", ""))
        ),
    )


def _render_session_history(session_id: str, sql_store, session_meta: dict | None = None):
    """把会话关联的每轮用户输入、回答和任务状态完整显示出来。"""
    session_meta = session_meta or _get_session_meta(sql_store, session_id) or {}
    tasks = _list_session_tasks(sql_store, session_id)
    console.print(Panel(
        f"[bold]会话 ID:[/bold] {escape(session_id)}\n"
        f"[bold]工作目录:[/bold] {escape(str(session_meta.get('workspace_dir', '未知')))}\n"
        f"[bold]任务轮数:[/bold] {session_meta.get('turn', len(tasks))}",
        title="💬 会话记录",
        border_style="cyan",
    ))

    # 老版本只存了 messages_history，没有任务到会话的索引，提供兼容显示。
    if not tasks:
        legacy_history = session_meta.get("history", [])
        if not legacy_history:
            console.print("[dim]这个会话还没有对话记录。[/dim]")
            return
        for index in range(0, len(legacy_history), 2):
            user = legacy_history[index].get("content", "")
            assistant = (
                legacy_history[index + 1].get("content", "")
                if index + 1 < len(legacy_history) else ""
            )
            console.print(Panel(Markdown(user), title=f"你 · 第 {index // 2 + 1} 轮"))
            if assistant:
                console.print(Panel(Markdown(assistant), title="SH Agent"))
        return

    for task in tasks:
        task_id = str(task.get("task_id", task.get("thread_id", "未知")))
        task_turn = task.get("session_turn", "?")
        status = str(task.get("status", "unknown"))
        title = f"你 · 第 {task_turn} 轮 · {status} · task {task_id}"
        request = str(task.get("user_request", "")) or "（没有保存用户输入）"
        console.print(Panel(Markdown(request), title=title, border_style="cyan"))

        response = task.get("final_response")
        if response:
            console.print(Panel(
                Markdown(str(response)), title="SH Agent", border_style="green"
            ))
        elif status in {"paused", "interrupted", "running", "failed"}:
            detail = f"当前状态：{status}。"
            if task.get("last_error"):
                detail += f"\n最近错误：{task['last_error']}"
            detail += f"\n需要继续此任务时，输入：/resume {task_id}"
            console.print(Panel(detail, title="任务尚未完成", border_style="yellow"))
        else:
            console.print(Panel(
                f"任务状态：{status}，但没有保存最终文本回答。",
                title="SH Agent",
                border_style="yellow",
            ))


def _render_session_tasks(session_id: str, sql_store):
    """显示某个会话下所有任务 ID，便于恢复具体一轮的图执行。"""
    tasks = _list_session_tasks(sql_store, session_id)
    if not tasks:
        console.print(f"[dim]会话 {escape(session_id)} 下还没有任务。[/dim]")
        return
    console.print(f"[bold cyan]会话 {escape(session_id)} 的任务[/bold cyan]")
    for task in tasks:
        task_id = str(task.get("task_id", task.get("thread_id", "未知")))
        status = str(task.get("status", "unknown"))
        request = str(task.get("user_request", "")).replace("\n", " ")
        if len(request) > 96:
            request = request[:96] + "..."
        resumable = "  [yellow]可用 /resume 恢复[/yellow]" if status in {
            "paused", "interrupted", "running", "failed"
        } else ""
        console.print(
            f"第 {task.get('session_turn', '?')} 轮  "
            f"[cyan]{escape(task_id)}[/cyan]  "
            f"[dim]{escape(status)}[/dim]  {escape(request)}{resumable}"
        )


def _record_resumed_task_result(
    task_id: str,
    task_meta: dict,
    final_response: str,
    sql_store,
    active_session_id: str,
    active_history: list,
):
    """把恢复完成的任务结果写回它所属的会话，避免重复追加。"""
    owner_session_id = task_meta.get("session_id")
    if not owner_session_id or not final_response:
        return

    session_meta = _get_session_meta(sql_store, owner_session_id) or {
        "workspace_dir": task_meta.get("workspace_dir", os.getcwd()),
        "history": [],
        "turn": task_meta.get("session_turn", 0),
    }
    linked_tasks = _list_session_tasks(sql_store, owner_session_id)
    if not any(
        item.get("task_id", item.get("thread_id")) == task_id
        for item in linked_tasks
    ):
        linked_tasks.append({
            **task_meta,
            "task_id": task_id,
            "final_response": final_response,
        })
    linked_tasks.sort(
        key=lambda item: (
            _as_int(item.get("session_turn")), str(item.get("created_at", ""))
        )
    )
    history = []
    for linked_task in linked_tasks:
        linked_id = linked_task.get("task_id", linked_task.get("thread_id"))
        response = (
            final_response if linked_id == task_id
            else linked_task.get("final_response", "")
        )
        if response:
            history.extend([
                HumanMessage(content=linked_task.get("user_request", "")),
                AIMessage(content=response),
            ])
    if not history:
        history = _restore_history(session_meta.get("history", []))
    history = history[-40:]
    _persist_session(
        owner_session_id,
        session_meta.get("workspace_dir", task_meta.get("workspace_dir", os.getcwd())),
        history,
        session_meta.get("turn", task_meta.get("session_turn", 0)),
        sql_store,
        last_task_id=task_id,
        completed_task_id=task_id,
    )

    if owner_session_id == active_session_id:
        active_history[:] = history[-40:]


def _stream_graph(graph, graph_input, config: dict) -> dict:
    """运行/恢复一段图执行，并保留现有的 Agent 与工具输出渲染。"""
    console.print()
    last_agent = None
    latest_state = {}

    for chunk in graph.stream(graph_input, config, stream_mode="values"):
        latest_state = chunk
        current_agent = chunk.get("current_agent", "")
        messages = chunk.get("messages", [])

        if current_agent and current_agent != last_agent:
            console.print(render_agent_header(current_agent))
            last_agent = current_agent

        if messages:
            latest = messages[-1]
            if hasattr(latest, "content") and latest.content:
                _render_message(latest, current_agent)

    return latest_state


def _pending_interrupt(graph, config: dict):
    """读取 checkpoint 中等待恢复的 interrupt；无中断时返回 None。"""
    snapshot = graph.get_state(config)
    for task in getattr(snapshot, "tasks", ()):
        for pending in getattr(task, "interrupts", ()):
            value = getattr(pending, "value", None)
            if value is not None:
                return value

    # 兼容 checkpoint 结构不暴露 task.interrupts 的 LangGraph 小版本。
    if "approval" in getattr(snapshot, "next", ()):
        values = getattr(snapshot, "values", {}) or {}
        return {
            "message": "代码已通过 Reviewer，是否继续运行 Executor 验证？",
            "user_request": values.get("user_request", ""),
            "review_feedback": values.get("review_feedback", ""),
        }
    return None


def _ask_approval(payload: dict) -> bool | None:
    """True=批准，False=拒绝，None=稍后再决定并保持暂停。"""
    message = escape(str(payload.get("message", "任务等待人工确认")))
    user_request = escape(str(payload.get("user_request", "")))
    review_feedback = str(payload.get("review_feedback", "")).strip()
    review_text = (
        f"\n\n[cyan]Reviewer 意见：[/cyan]{escape(review_feedback)}"
        if review_feedback else ""
    )
    console.print(Panel(
        f"[bold]{message}[/bold]\n\n"
        f"[cyan]用户需求：[/cyan]{user_request}"
        f"{review_text}\n\n"
        f"[dim]确认后将进入 Executor；选择稍后可用 /resume 恢复。[/dim]",
        title="⏸ 等待人工确认",
        border_style="yellow",
    ))
    choice = Prompt.ask(
        "选择 [y=继续 / n=拒绝 / later=稍后]",
        choices=["y", "n", "later"],
        default="later",
    )
    if choice == "later":
        return None
    return choice == "y"


def _update_task(sql_store, task_id: str, **updates):
    if sql_store is None:
        if task_id in LOCAL_TASKS:
            LOCAL_TASKS[task_id].update(updates)
            LOCAL_TASKS[task_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        return
    try:
        sql_store.update_task(task_id, **updates)
    except Exception as exc:
        console.print(f"[yellow]PostgreSQL 任务状态更新失败: {exc}[/yellow]")


def _run_task(graph, graph_input, config: dict, task_id: str, meta: dict, sql_store):
    """执行或继续单个持久化任务，处理人工中断和状态登记。"""
    _update_task(sql_store, task_id, status="running", last_error=None)
    current_input = graph_input

    try:
        while True:
            latest_state = _stream_graph(graph, current_input, config)
            pending = _pending_interrupt(graph, config)
            if pending is not None:
                _update_task(sql_store, task_id, status="paused", pause_reason="approval")
                decision = _ask_approval(pending)
                if decision is None:
                    console.print(
                        f"[yellow]任务已暂停。稍后输入 /resume {task_id} 继续。[/yellow]"
                    )
                    return {"status": "paused"}

                current_input = Command(resume={"approved": decision})
                _update_task(sql_store, task_id, status="running", pause_reason=None)
                continue

            final = latest_state.get("final_response", "")
            if final:
                console.print()
                console.print(Markdown(final))

            approval_decision = latest_state.get("approval_decision")
            status = "rejected" if approval_decision == "rejected" else "completed"
            _update_task(
                sql_store,
                task_id,
                status=status,
                final_response=final,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return {"status": status, "final_response": final}

    except KeyboardInterrupt:
        _update_task(sql_store, task_id, status="interrupted")
        console.print(
            f"\n[yellow]已请求停止。稍后可输入 /resume {task_id} 从 checkpoint 恢复。[/yellow]"
        )
        return {"status": "interrupted"}
    except Exception as exc:
        _update_task(
            sql_store,
            task_id,
            status="failed",
            last_error=str(exc)[:2000],
        )
        console.print(f"\n[red]执行出错: {exc}[/red]")
        console.print(f"[dim]可用 /resume {task_id} 尝试从最近 checkpoint 恢复。[/dim]")
        return {"status": "failed"}


def _sync_terminal_task(task_id: str, meta: dict, values: dict, sql_store) -> tuple[str, str]:
    """把已到达图终态的 checkpoint 同步到任务记录，返回状态和最终回答。"""
    final = values.get("final_response") or meta.get("final_response", "")
    terminal_status = (
        "rejected" if values.get("approval_decision") == "rejected" else "completed"
    )
    updates = {
        "status": terminal_status,
        "completed_at": meta.get("completed_at") or datetime.now(timezone.utc).isoformat(),
        "pause_reason": None,
        "last_error": None,
    }
    if final:
        updates["final_response"] = final
        meta["final_response"] = final
    _update_task(sql_store, task_id, **updates)
    meta["status"] = terminal_status
    meta["completed_at"] = updates["completed_at"]
    return terminal_status, final


def _resume_task(
    task_id: str,
    graph,
    sql_store,
    console: Console,
    messages_history: list,
    active_session_id: str,
):
    meta = sql_store.get_task(task_id) if sql_store else LOCAL_TASKS.get(task_id)
    if meta is None:
        console.print(f"[yellow]没有找到任务 {task_id}。输入 /tasks 查看任务 ID。[/yellow]")
        return

    thread_id = meta.get("thread_id", task_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [TokenStreamHandler()],
        "recursion_limit": int(get_setting("agent", "graph_recursion_limit") or 64),
    }

    try:
        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", {}) or {}
        next_nodes = getattr(snapshot, "next", ())

        if values and not next_nodes:
            terminal_status, final = _sync_terminal_task(
                task_id, meta, values, sql_store
            )

            if final:
                console.print(Markdown(final))
                _record_resumed_task_result(
                    task_id, meta, final, sql_store, active_session_id, messages_history
                )
            console.print(f"[dim]任务 {task_id} 已经结束（状态：{terminal_status}）。[/dim]")
            return

        if values:
            pending = _pending_interrupt(graph, config)
            if pending is not None:
                decision = _ask_approval(pending)
                if decision is None:
                    console.print(f"[yellow]任务保持暂停。稍后输入 /resume {task_id}。[/yellow]")
                    _update_task(sql_store, task_id, status="paused", pause_reason="approval")
                    return
                graph_input = Command(resume={"approved": decision})
            else:
                # 从上一个已保存节点继续；失败中的节点会按 LangGraph 语义重新执行。
                graph_input = None
        else:
            # 任务尚未完成第一个 checkpoint，使用 PostgreSQL 任务记录重构首次输入。
            history = _restore_history(meta.get("history", []))
            graph_input = _build_initial_state(
                meta.get("user_request", ""),
                meta.get("workspace_dir", os.getcwd()),
                history,
            )

        result = _run_task(graph, graph_input, config, task_id, meta, sql_store)
        final = result.get("final_response")
        if final:
            _record_resumed_task_result(
                task_id, meta, final, sql_store, active_session_id, messages_history
            )
    except Exception as exc:
        console.print(f"[red]恢复任务失败: {exc}[/red]")


def _handle_command(
    cmd: str,
    console: Console,
    messages_history: list = None,
    graph=None,
    sql_store=None,
    workspace_dir: str | None = None,
    session_id: str = "",
    turn: int = 0,
):
    raw_cmd = cmd.strip()
    parts = raw_cmd.split()
    command = parts[0].lower() if parts else ""
    if command == "/quit" or command == "/q":
        console.print("[dim]再见！[/dim]")
        sys.exit(0)
    elif command == "/help" or command == "/h":
        render_help()
    elif command == "/clear":
        console.clear()
        render_banner()
    elif command == "/status":
        console.print(f"[dim]会话 ID: {session_id}[/dim]")
        console.print(f"[dim]工作目录: {workspace_dir or os.getcwd()}[/dim]")
        console.print(f"[dim]会话轮次: {turn} 轮[/dim]")
    elif command == "/sessions":
        try:
            sessions = sql_store.list_sessions() if sql_store else []
            known_ids = {item.get("session_id") for item in sessions}
            sessions.extend(
                item for sid, item in LOCAL_SESSIONS.items() if sid not in known_ids
            )
            if not sessions:
                console.print("[dim]没有已保存的会话。[/dim]")
                return
            if sql_store is None:
                console.print("[dim]以下会话只保存在当前进程，退出后无法恢复。[/dim]")
            for item in sessions:
                sid = str(item.get("session_id", ""))
                marker = "  当前会话" if sid == session_id else ""
                count = item.get("turn", len(item.get("history", [])) // 2)
                console.print(
                    f"[cyan]{escape(sid)}[/cyan]{marker}  "
                    f"[dim]{count} 轮[/dim]  "
                    f"{escape(str(item.get('workspace_dir', '')))}"
                )
        except Exception as exc:
            console.print(f"[red]读取会话列表失败: {exc}[/red]")
    elif command == "/resume-session":
        if len(parts) != 2:
            console.print("[yellow]用法: /resume-session <session_id>[/yellow]")
            return
        restored = _get_session_meta(sql_store, parts[1].lower())
        if restored is None:
            console.print(f"[yellow]没有找到会话 {parts[1]}。输入 /sessions 查看会话 ID。[/yellow]")
            return
        restored_id = restored.get("session_id", parts[1].lower())
        restored_history = _restore_history(restored.get("history", []))
        linked_tasks = _list_session_tasks(sql_store, restored_id)
        latest_task_turn = max(
            (_as_int(task.get("session_turn")) for task in linked_tasks),
            default=0,
        )

        resumable_tasks = [
            task for task in linked_tasks
            if task.get("status") in {"paused", "interrupted", "running", "failed"}
        ]
        actual_resumable_tasks = []
        for task in resumable_tasks:
            task_id = str(task.get("task_id", task.get("thread_id", "")))
            thread_id = task.get("thread_id", task_id)
            try:
                snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
                values = getattr(snapshot, "values", {}) or {}
                next_nodes = getattr(snapshot, "next", ())
            except Exception:
                # Checkpoint 暂时不可读时仍保留任务入口，让显式恢复给出具体错误。
                actual_resumable_tasks.append(task)
                continue

            if values and not next_nodes:
                terminal_status, final = _sync_terminal_task(
                    task_id, task, values, sql_store
                )
                if final:
                    _record_resumed_task_result(
                        task_id, task, final, sql_store, restored_id, restored_history
                    )
                console.print(
                    f"[dim]第 {task.get('session_turn', '?')} 轮的 checkpoint 已结束，"
                    f"任务状态已同步为 {terminal_status}。[/dim]"
                )
            else:
                actual_resumable_tasks.append(task)

        restored = _get_session_meta(sql_store, restored_id) or restored
        _render_session_history(restored_id, sql_store, restored)
        actual_resumable_tasks.sort(
            key=lambda task: (
                _as_int(task.get("session_turn")), str(task.get("created_at", ""))
            ),
            reverse=True,
        )
        if actual_resumable_tasks:
            pending_task = actual_resumable_tasks[0]
            pending_task_id = str(
                pending_task.get("task_id", pending_task.get("thread_id", ""))
            )
            pending_request = str(pending_task.get("user_request", "")).replace("\n", " ")
            if pending_request and (
                not restored_history
                or getattr(restored_history[-1], "content", None) != pending_request
            ):
                # 即使用户暂不续跑，恢复会话后继续提问也能看到这条未完成请求。
                restored_history.append(HumanMessage(content=pending_request))
            if len(pending_request) > 96:
                pending_request = pending_request[:96] + "..."
            console.print(
                f"[yellow]会话中有未完成任务 {pending_task_id} "
                f"（第 {pending_task.get('session_turn', '?')} 轮：{escape(pending_request)}）[/yellow]"
            )
            should_resume = Prompt.ask(
                "是否从 checkpoint 继续这轮任务？",
                choices=["y", "n"],
                default="n",
            )
            if should_resume == "y":
                _resume_task(
                    pending_task_id,
                    graph,
                    sql_store,
                    console,
                    restored_history,
                    restored_id,
                )
        else:
            console.print("[dim]会话记录已载入；继续输入即可在此会话中提问。[/dim]")

        saved_turn = _as_int(restored.get("turn"))
        return {
            "action": "resume_session",
            "session_id": restored_id,
            "messages_history": restored_history,
            "workspace_dir": restored.get("workspace_dir", workspace_dir or os.getcwd()),
            "turn": max(saved_turn, latest_task_turn, len(restored_history) // 2),
        }
    elif command == "/history":
        if len(parts) > 2:
            console.print("[yellow]用法: /history [session_id][/yellow]")
            return
        target_session = parts[1].lower() if len(parts) == 2 else session_id
        if not target_session:
            console.print("[yellow]当前没有活动会话。用法: /history <session_id>[/yellow]")
            return
        target_meta = _get_session_meta(sql_store, target_session)
        if target_meta is None:
            console.print(f"[yellow]没有找到会话 {target_session}。输入 /sessions 查看会话 ID。[/yellow]")
            return
        _render_session_history(target_session, sql_store, target_meta)
    elif command == "/session-tasks":
        if len(parts) > 2:
            console.print("[yellow]用法: /session-tasks [session_id][/yellow]")
            return
        target_session = parts[1].lower() if len(parts) == 2 else session_id
        if not target_session:
            console.print("[yellow]当前没有活动会话。用法: /session-tasks <session_id>[/yellow]")
            return
        _render_session_tasks(target_session, sql_store)
    elif command == "/tasks":
        try:
            tasks = sql_store.list_tasks() if sql_store else list(LOCAL_TASKS.values())
            if not tasks:
                console.print("[dim]没有已登记的任务。[/dim]")
                return
            if sql_store is None:
                console.print("[dim]以下任务只保存在当前进程中，退出后无法恢复。[/dim]")
            for task in tasks:
                request = task.get("user_request", "").replace("\n", " ")
                if len(request) > 72:
                    request = request[:72] + "..."
                console.print(
                    f"[cyan]{escape(str(task.get('task_id', '')))}[/cyan]  "
                    f"[dim]{escape(str(task.get('status', 'unknown')))}[/dim]  "
                    f"[dim]session={escape(str(task.get('session_id', 'legacy')))}[/dim]  "
                    f"{escape(request)}"
                )
        except Exception as exc:
            console.print(f"[red]读取任务列表失败: {exc}[/red]")
    elif command == "/resume":
        if len(parts) != 2:
            console.print("[yellow]用法: /resume <task_id>[/yellow]")
            return
        _resume_task(
            parts[1].lower(),
            graph,
            sql_store,
            console,
            messages_history if messages_history is not None else [],
            session_id,
        )
    elif command == "/setup":
        _setup_wizard()
    elif command in ("/eval", "/evaluate"):
        args = parts[1:]
        live = "--live" in args
        args = [arg for arg in args if arg != "--live"]
        valid_agents = {"supervisor", "explorer", "coder", "reviewer", "executor"}

        if len(args) > 1 or any(arg not in valid_agents for arg in args):
            console.print("[yellow]用法: /eval [supervisor|explorer|coder|reviewer|executor] [--live][/yellow]")
            return

        _run_eval(agent_filter=args[0] if args else "", live=live)
    else:
        console.print(f"[yellow]未知命令: {raw_cmd}[/yellow] 输入 /help 查看帮助")


def _setup_wizard():
    """交互式配置向导"""
    from rich.prompt import Prompt, Confirm
    from pathlib import Path

    console.print()
    console.print(Panel(
        "[bold]🔧 API 配置向导[/bold]",
        border_style="cyan",
    ))
    console.print()

    # API Key
    current_key = os.getenv("OPENAI_API_KEY", "")
    masked = current_key[:8] + "****" if len(current_key) > 8 else "(未设置)"
    console.print(f"当前 API Key: [dim]{masked}[/dim]")
    api_key = Prompt.ask("请输入 API Key", default=current_key or "", password=True)

    # Base URL
    current_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    console.print(f"\n当前 Base URL: [dim]{current_url}[/dim]")
    console.print("[dim]常用: 1=DeepSeek  2=OpenAI  3=自定义[/dim]")
    choice = Prompt.ask("选择 (1/2/3)", default="1")
    if choice == "1":
        base_url = "https://api.deepseek.com"
    elif choice == "2":
        base_url = "https://api.openai.com/v1"
    else:
        base_url = Prompt.ask("输入自定义 URL", default=current_url)

    # 写入 .env
    env_path = Path(".env")
    content = (
        f"OPENAI_API_KEY={api_key}\n"
        f"OPENAI_BASE_URL={base_url}\n"
        f"PG_PASSWORD=\n"
    )
    env_path.write_text(content, encoding="utf-8")
    console.print(f"\n[green]✅ 配置已保存到 {env_path}[/green]")

    # 测试连接
    if Confirm.ask("\n是否测试 API 连接?", default=True):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        try:
            from code_agent.model_factory import get_chat_model
            # 清除单例缓存，使用新配置
            import code_agent.model_factory as mf
            mf._chat_model = None
            model = get_chat_model()
            from langchain_core.messages import HumanMessage
            resp = model.invoke([HumanMessage(content="回复OK两个字母，不要其他内容")])
            console.print(f"[green]✅ 连接成功！模型: {model.model_name}[/green]")
        except Exception as e:
            console.print(f"[red]❌ 连接失败: {e}[/red]")
            console.print("[yellow]请检查 API Key 和 Base URL 是否正确[/yellow]")

    console.print()


def _run_eval(agent_filter: str = "", live: bool = False):
    """运行 Agent 评测；live=True 时调用真实 Agent。"""
    from tests.eval.runner import run_eval

    console.print()
    mode_text = "真实 LLM 模式 — 会调用模型和 Agent 工具" if live else "离线模拟模式 — 不调用 LLM"
    console.print(f"[bold cyan]正在运行 Agent 评测...[/bold cyan]")
    console.print(f"[dim]{mode_text}[/dim]")
    console.print()

    agents = [agent_filter] if agent_filter else None
    try:
        renderer = run_eval(mock=not live, agents=agents)
        renderer.render_summary()
    except Exception as e:
        console.print(f"[red]评测运行失败: {e}[/red]")

    console.print()
    if not live:
        console.print("[dim]💡 使用 /eval --live 可运行真实 LLM 评测 (需 API Key)[/dim]")


if __name__ == "__main__":
    main()
