import os

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    """加载 config.yaml，返回嵌套字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str) -> None:
    """将配置字典序列化为 YAML，保存到指定路径。"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
