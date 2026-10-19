"""任务对齐与时钟计算。

每个任务（里程碑 + 任务号）计算：
  P_v  任务首次出现在已提交任务书中的时间
  I_v  首个带该任务号、且改动代码（非 docs/plans）的提交时间
  P_w  会话中首次写入该任务标题行的编辑（attribute_edits.py 的 plan_task_edits）
  I_w  按内容归属到 I_v 提交的代码编辑中最早的一次（attribute_edits.py 的 edit_attribution）
对照口径（*_timerule）：文件 f 在 (f 的上一次提交时间, 本次提交时间] 内的最早成功编辑，不看内容。
只用观察窗口内、未排除会话中、路径在本工作区内的 completed 编辑。

输出 data/derived/<系统>/tasks.csv，并在 stdout 打印汇总。
"""
import bisect
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import WINDOW_END, excluded  # noqa: E402

EDIT_TOOLS = {"edit_file_v2", "edit_file", "search_replace", "write"}
PLAN_DIR = "docs/plans/"


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def ts(s):
    return datetime.fromisoformat(s) if s else None


def build(sysname):
    d = Path("data/derived") / sysname
    commits = load(d / "commits.csv")
    for c in commits:
        c["t"] = ts(c["time"])
    by_hash = {c["hash"]: c for c in commits}
    cfiles = defaultdict(list)
    touches = defaultdict(list)
    for f in load(d / "commit_files.csv"):
        cfiles[f["hash"]].append(f["path"])
        touches[f["path"]].append(by_hash[f["hash"]]["t"])
    for p in touches:
        touches[p].sort()

    end = WINDOW_END[sysname]
    edits = defaultdict(list)
    n_edits = 0
    for e in csv.DictReader(open(d / "events.csv", encoding="utf-8")):
        if e["tool"] not in EDIT_TOOLS or e["status"] != "completed" or e["in_workspace"] != "1":
            continue
        if excluded(e["session"]) or not e["time"]:
            continue
        t = ts(e["time"])
        if t > end:
            continue
        edits[e["path"].lower()].append((t, float(e["uncert_s"] or 0), e["session"]))
        n_edits += 1
    for p in edits:
        edits[p].sort()

    def first_edit(path, commit_t):
        """path 在 (上一次提交, commit_t] 内的最早编辑。"""
        ts_list = touches.get(path, [])
        i = bisect.bisect_left(ts_list, commit_t)
        prev = ts_list[i - 1] if i > 0 else None
        for t, u, s in edits.get(path.lower(), []):
            if t > commit_t:
                break
            if prev is None or t > prev:
                return t, u, s
        return None

    impl = {}
    for c in commits:
        if c["label"] != "task" or int(c["n_code_files"]) == 0:
            continue
        for t in c["tasks"].split(";"):
            impl.setdefault((c["milestone"], t), c)

    attributed = defaultdict(list)
    for a in load(d / "edit_attribution.csv"):
        if a["state"] == "committed" and not a["path"].startswith(PLAN_DIR):
            t = ts(a["time"])
            if t <= end and not excluded(a["session"]):
                attributed[a["commit"]].append((t, float(a["uncert_s"] or 0), a["session"]))
    plan_first = {}
    for p in load(d / "plan_task_edits.csv"):
        t = ts(p["time"])
        if t > end or excluded(p["session"]):
            continue
        k = (p["milestone"], p["task"])
        if k not in plan_first or t < plan_first[k][0]:
            plan_first[k] = (t, float(p["uncert_s"] or 0), p["session"])

    rows = []
    for tb in load(d / "taskbook_tasks.csv"):
        key = (tb["milestone"], tb["task"])
        pv = by_hash[tb["first_commit"]]
        iv = impl.get(key)
        pw = plan_first.get(key)
        if pw and pw[0] > pv["t"]:
            pw = None
        pw_tr = first_edit(tb["file"], pv["t"])
        iw = min(attributed[iv["hash"]]) if iv and attributed.get(iv["hash"]) else None
        iw_tr = None
        if iv:
            cands = [first_edit(p, iv["t"]) for p in cfiles[iv["hash"]] if not p.startswith(PLAN_DIR)]
            cands = [x for x in cands if x]
            iw_tr = min(cands) if cands else None
        hrs = lambda a, b: round((b - a).total_seconds() / 3600, 3) if a and b else ""
        iso = lambda x: x[0].isoformat() if x else ""
        rows.append({
            "milestone": key[0], "task": key[1],
            "P_v": pv["time"], "P_v_hash": pv["hash"][:8],
            "I_v": iv["time"] if iv else "", "I_v_hash": iv["hash"][:8] if iv else "",
            "same_commit": int(bool(iv) and iv["hash"] == pv["hash"]),
            "P_w": iso(pw), "P_w_uncert_s": pw[1] if pw else "",
            "I_w": iso(iw), "I_w_uncert_s": iw[1] if iw else "",
            "lag_Iw_Iv_h": hrs(iw[0] if iw else None, iv["t"] if iv else None),
            "gap_Pv_Iv_h": hrs(pv["t"], iv["t"] if iv else None),
            "gap_Pw_Iw_h": hrs(pw[0] if pw else None, iw[0] if iw else None),
            "P_w_timerule": iso(pw_tr), "I_w_timerule": iso(iw_tr),
            "lag_Iw_Iv_h_timerule": hrs(iw_tr[0] if iw_tr else None, iv["t"] if iv else None),
        })
    with open(d / "tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{sysname}: tasks={len(rows)} with_impl={sum(1 for r in rows if r['I_v'])} "
          f"with_P_w={sum(1 for r in rows if r['P_w'])} with_I_w={sum(1 for r in rows if r['I_w'])} edits_used={n_edits}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        build(s)
