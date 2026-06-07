# conftest.py — pytest 根配置
# 显式把项目根目录加入 sys.path，使 `import src.*` 在所有测试中可用。
# --import-mode=importlib 模式下 pytest 不自动注入根目录，必须在此手动处理。
import sys
from pathlib import Path

# 项目根目录（本文件所在目录）
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

