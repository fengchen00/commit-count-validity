"""观察窗口与排除清单，对应 notes/decisions.md D2。"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 当前稿件版本目录（D22，S19-T-C0）：脚本的结果与图只写到这里；旧版本目录不再写入。
PAPER_DIR = Path("doc/v4.0")
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"

TZ = timezone(timedelta(hours=8))

WINDOW_END = {
    "SYS_A": datetime(2026, 9, 23, 21, 18, 8, tzinfo=TZ),
    "SYS_C": datetime(2026, 9, 29, 2, 20, 35, tzinfo=TZ),
}

EXCLUDED_SESSION_PREFIXES = [
    "00000000",  # 研究导出
    "00000000",  # 研究导出
    "00000000",  # 研究导出
    "00000000",  # 会话全量导出
    "00000000",  # 作者另一项研究，不读取
]


def excluded(session_id: str) -> bool:
    return any(session_id.startswith(p) for p in EXCLUDED_SESSION_PREFIXES)
