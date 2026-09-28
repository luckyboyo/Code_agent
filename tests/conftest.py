import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as td:
        old = os.getcwd()
        os.chdir(td)
        try:
            yield td
        finally:
            os.chdir(old)


@pytest.fixture
def sample_state():
    return {
        "messages": [],
        "user_request": "",
        "workspace_dir": ".",
        "task_plan": "",
        "current_agent": "supervisor",
        "exploration_result": None,
        "relevant_files": None,
        "code_changes": None,
        "review_feedback": None,
        "review_approved": False,
        "test_result": None,
        "test_passed": False,
        "retry_count": 0,
        "max_retries": 3,
        "final_response": None,
        "task_complete": False,
    }
