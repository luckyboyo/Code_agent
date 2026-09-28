"""测试路径穿越防护"""
import os
import pytest
import tempfile
from code_agent.tools.file_tools import _safe_path


class TestSafePath:
    """_safe_path 路径穿越测试"""

    def test_normal_path(self):
        result = _safe_path("/tmp/ws", "src/main.py")
        expected = os.path.abspath(os.path.join("/tmp/ws", "src/main.py"))
        assert result == expected

    def test_subdirectory(self):
        result = _safe_path("/tmp/ws", "a/b/c/d.txt")
        expected = os.path.abspath(os.path.join("/tmp/ws", "a/b/c/d.txt"))
        assert result == expected

    def test_traversal_attack(self):
        with pytest.raises(ValueError, match="越权"):
            _safe_path("/tmp/ws", "../../../etc/passwd")

    def test_traversal_with_normal_prefix(self):
        with pytest.raises(ValueError, match="越权"):
            _safe_path("/tmp/ws", "src/../../../etc/shadow")

    def test_absolute_path_attack(self):
        """绝对路径穿越到 workspace 外应被拦截"""
        base = "/tmp/ws"
        full = os.path.abspath(os.path.join(base, "/etc/passwd"))
        if full.startswith(os.path.abspath(base)):
            # Windows 下 join /etc/passwd 可能仍在盘符内，跳过
            pytest.skip("platform-specific: absolute path not traversable")
        with pytest.raises(ValueError, match="越权"):
            _safe_path(base, "/etc/passwd")

    def test_dot_path(self):
        result = _safe_path("/tmp/ws", ".")
        expected = os.path.abspath("/tmp/ws")
        assert result == expected

    def test_empty_path(self):
        result = _safe_path("/tmp/ws", "")
        expected = os.path.abspath("/tmp/ws")
        assert result == expected

    def test_real_temp_directory(self):
        with tempfile.TemporaryDirectory() as td:
            result = _safe_path(td, "test.txt")
            assert result == os.path.join(td, "test.txt")

    def test_nested_traversal(self):
        with pytest.raises(ValueError, match="越权"):
            _safe_path("/tmp/ws", "a/../../b/../../c/../../../d")

    def test_result_is_within_base(self):
        """无论平台，结果必须在 base 内"""
        with tempfile.TemporaryDirectory() as td:
            result = _safe_path(td, "foo/bar/baz.py")
            assert result.startswith(os.path.abspath(td))
            assert os.path.normpath(result) == os.path.normpath(
                os.path.join(td, "foo/bar/baz.py"))
