"""配置加载模块"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "settings.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_settings = load_config()


def get_setting(*keys: str):
    """按路径读取配置，如 get_setting('redis', 'host')"""
    value = _settings
    for k in keys:
        value = value[k]
    return value


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)
