"""pytest 根配置：保证项目根在 sys.path 上，`src.*` 可导入。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
