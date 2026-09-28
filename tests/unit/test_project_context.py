"""测试项目上下文自动检测"""
import os
import tempfile
import pytest
from code_agent.project_context import get_project_context, format_project_context


class TestProjectContext:
    """项目类型检测测试"""

    def test_python_project_pyproject(self):
        with tempfile.TemporaryDirectory() as td:
            (open(os.path.join(td, "pyproject.toml"), "w").close())
            os.makedirs(os.path.join(td, "src"))
            ctx = get_project_context(td)
            assert ctx["type"] == "python"
            assert ctx["build_system"] == "pyproject.toml"
            assert ctx["main_source"] == "src"

    def test_python_project_setup_py(self):
        with tempfile.TemporaryDirectory() as td:
            (open(os.path.join(td, "setup.py"), "w").close())
            ctx = get_project_context(td)
            assert ctx["type"] == "python"
            assert ctx["build_system"] == "setup.py"

    def test_node_project(self):
        with tempfile.TemporaryDirectory() as td:
            (open(os.path.join(td, "package.json"), "w").close())
            ctx = get_project_context(td)
            assert ctx["type"] == "node"
            assert ctx["build_system"] == "package.json"

    def test_go_project(self):
        with tempfile.TemporaryDirectory() as td:
            (open(os.path.join(td, "go.mod"), "w").close())
            ctx = get_project_context(td)
            assert ctx["type"] == "go"
            assert ctx["build_system"] == "go.mod"

    def test_rust_project(self):
        with tempfile.TemporaryDirectory() as td:
            (open(os.path.join(td, "Cargo.toml"), "w").close())
            ctx = get_project_context(td)
            assert ctx["type"] == "rust"
            assert ctx["build_system"] == "Cargo.toml"

    def test_unknown_project(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = get_project_context(td)
            assert ctx["type"] == "unknown"
            assert ctx["build_system"] is None

    def test_test_dir_detection(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "tests"))
            ctx = get_project_context(td)
            assert ctx["test_dir"] == "tests"

    def test_code_agent_main_source(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "code_agent"))
            ctx = get_project_context(td)
            assert ctx["main_source"] == "code_agent"

    def test_hidden_files_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, ".git"))
            (open(os.path.join(td, ".env"), "w").close())
            ctx = get_project_context(td)
            assert ".git" not in ctx["top_dirs"]
            assert ".env" not in ctx["top_files"]

    def test_format_context(self):
        ctx = {
            "type": "python",
            "name": "my-project",
            "top_files": ["README.md"],
            "top_dirs": ["src", "tests"],
            "build_system": "pyproject.toml",
            "test_dir": "tests",
            "main_source": "src",
        }
        formatted = format_project_context(ctx)
        assert "python" in formatted
        assert "my-project" in formatted
        assert "pyproject.toml" in formatted
        assert "src/" in formatted
        assert "tests/" in formatted
