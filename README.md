# SH Agent

> 基于 LangGraph 的 Multi-Agent 编码助手 — 5 个 AI Agent 协作完成编程任务

## 架构

```
用户输入 → Supervisor (任务拆解 + 调度)
              ├── Explorer   (搜索、阅读、理解代码)
              ├── Coder      (编写和修改代码)
              ├── Reviewer   (审查代码质量) ←→ Coder (最多 3 轮修复)
              └── Executor   (运行测试验证)
```

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 配置 API（首次运行会自动引导）
sh-agent

# 3. 输入你的编程问题
> 帮我分析 code_agent/graph.py 的代码结构
> 审查一下 cli.py
```

## 配置

编辑项目根目录的 `.env` 文件：

```bash
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1    # 或其他兼容端点
```

支持的 API 端点：

| 提供商 | BASE_URL |
|--------|----------|
| DeepSeek | `https://api.deepseek.com` |
| OpenAI | `https://api.openai.com/v1` |
| 其他 OpenAI 兼容 | 按提供商文档填写 |

`config/settings.yml` 中可调整模型名、温度、审查轮次等参数。

## 依赖

| 组件 | 用途 | 必选 |
|------|------|------|
| Redis | 断点续聊 + 工具缓存 | 否（降级为无状态模式） |
| PostgreSQL | 操作审计日志 | 否（降级为跳过） |
| ripgrep (rg) | 代码搜索加速 | 否（降级为 Python grep） |

## 技术栈

- **LangGraph** — Multi-Agent StateGraph 编排 + 条件路由
- **LangChain** — Agent 框架 + 工具注册
- **Redis** — Checkpoint 持久化
- **PostgreSQL** — 审计日志
- **Rich + prompt_toolkit** — 终端 UI

## 项目结构

```
sh-agent/
├── config/settings.yml          # 全局配置
├── code_agent/
│   ├── cli.py                   # CLI 入口
│   ├── graph.py                 # LangGraph 编排引擎
│   ├── state.py                 # 共享状态定义
│   ├── config.py                # 配置加载
│   ├── model_factory.py         # LLM 工厂（惰性初始化）
│   ├── agents/
│   │   ├── supervisor.py        # 调度器
│   │   ├── explorer.py          # 代码分析
│   │   ├── coder.py             # 代码编写
│   │   ├── reviewer.py          # 代码审查
│   │   └── executor.py          # 测试执行
│   ├── tools/                   # Agent 工具集
│   ├── middleware/              # 安全 + 日志中间件
│   ├── storage/                 # Redis + PostgreSQL
│   └── ui/                      # 终端渲染
├── pyproject.toml
└── .env.example
```

## 命令

REPL 内置命令：

| 命令 | 功能 |
|------|------|
| `/help` | 帮助信息 |
| `/setup` | 交互式配置向导 |
| `/clear` | 清屏 |
| `/status` | 查看会话状态 |
| `/quit` | 退出 |
