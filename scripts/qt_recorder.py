"""薄封装：调用 data_ui.app.main，确保仓库根目录在 sys.path 中。

录制进程会进行相机采集与大量写盘（多线程/多进程），容易与控制进程争抢 CPU。
为了让“控制链路(≈90Hz)”更稳，这里默认把录制进程 nice 调低（对普通用户安全，不需要 root）。
"""

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # 数值越大优先级越低；经验值 10 基本不影响录制，但能明显减少对控制进程的抢占。
    os.nice(10)
except Exception:
    pass

from data_ui.app import main

if __name__ == "__main__":
    main()
