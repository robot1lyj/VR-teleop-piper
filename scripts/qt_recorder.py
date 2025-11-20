"""薄封装：调用 data_ui.app.main，确保仓库根目录在 sys.path 中。"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_ui.app import main

if __name__ == "__main__":
    main()
