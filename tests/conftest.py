"""pytest 全局配置隔离：测试使用固定测试配置，不影响用户真实 config.yaml。

真实配置可能包含不同渠道/模型（如用户部署的远端配置），而测试硬编码了
openai-vision 等模型名，因此测试期间自动切换到 tests/test_config.yaml，
结束后恢复原 config.yaml。
"""
import shutil
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
TEST_CONFIG_PATH = Path(__file__).resolve().parent / "test_config.yaml"


@pytest.fixture(scope="session", autouse=True)
def isolate_test_config():
    original = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else None
    shutil.copy2(TEST_CONFIG_PATH, CONFIG_PATH)
    # 强制 config 单例重新加载
    import app.config as config_mod
    config_mod._config = None
    yield
    if original is not None:
        CONFIG_PATH.write_text(original, encoding="utf-8")
    else:
        CONFIG_PATH.unlink(missing_ok=True)
    config_mod._config = None
