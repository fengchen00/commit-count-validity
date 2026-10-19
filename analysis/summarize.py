"""第一版汇总（R0）：RQ1 顺序判定对比、RQ2 时钟间隔。只输出计数与分位数，写入 <PAPER_DIR>/results/r0_summary.md。"""
import csv
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from config import PAPER_DIR, RESULTS_DIR  # noqa: E402

TOL_S = 60  # 工作区顺序判定的最小间隔；小于此值或落在时间不确定区间内记为“无法区分”


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def ts(s):
    return datetime.fromisoformat(s) if s else None


def git_order(r):
    if not r["I_v"]:
        return "无实现提交"
    if r["same_commit"] == "1":
        return "同一提交"
    return "计划在先" if ts(r["P_v"]) < ts(r["I_v"]) else "实现在先"


def ws_order(r):
    if not (r["P_w"] and r["I_w"]):
        return "缺工作区证据"
    gap = (ts(r["I_w"]) - ts(r["P_w"])).total_seconds()
    tol = max(TOL_S, float(r["P_w_uncert_s"] or 0) + float(r["I_w_uncert_s"] or 0))
    if abs(gap) <= tol:
        return "无法区分"
    return "计划在先" if gap > 0 else "实现在先"


def ms_order(tasks):
    """里程碑粒度：该里程碑任务首次进入任务书 vs 首个实现提交。"""
    out = {}
    by = defaultdict(list)
    for r in tasks:
        by[r["milestone"]].append(r)
    for ms, rs in by.items():
        pv = min(ts(r["P_v"]) for r in rs)
        ivs = [ts(r["I_v"]) for r in rs if r["I_v"]]
        if not ivs:
            out[ms] = "无实现提交"
            continue
        iv = min(ivs)
        out[ms] = "同一提交" if pv == iv else ("计划在先" if pv < iv else "实现在先")
    return out


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def fmt_h(x):
    if x is None:
        return "—"
    return f"{x * 60:.0f} 分钟" if x < 1 else f"{x:.1f} 小时"


def section(sysname):
    rows = load(Path("data/derived") / sysname / "tasks.csv")
    L = [f"## {sysname}", "", f"任务总数 {len(rows)}；有实现提交 {sum(1 for r in rows if r['I_v'])}。", ""]

    g = Counter(git_order(r) for r in rows)
    w = Counter(ws_order(r) for r in rows)
    m = Counter(ms_order(rows).values())
    cats = ["计划在先", "同一提交", "实现在先", "无法区分", "无实现提交", "缺工作区证据"]
    L += ["### RQ1 顺序判定：三种视角", "", "| 判定 | 仅 Git（任务粒度） | 仅 Git（里程碑粒度） | 工作区（会话编辑） |", "|---|---|---|---|"]
    for c in cats:
        if g[c] or w[c] or m[c]:
            L.append(f"| {c} | {g[c] or ''} | {m[c] or ''} | {w[c] or ''} |")
    L.append("")
    L.append("里程碑粒度的单位是里程碑，另外两列的单位是任务。")
    L.append("")

    both = [r for r in rows if git_order(r) not in ("无实现提交",) and ws_order(r) not in ("缺工作区证据",)]
    ct = Counter((git_order(r), ws_order(r)) for r in both)
    gcats = ["计划在先", "同一提交", "实现在先"]
    wcats = ["计划在先", "无法区分", "实现在先"]
    L += ["### RQ1 交叉表：仅 Git（任务粒度）× 工作区", "", "| Git ＼ 工作区 | " + " | ".join(wcats) + " | 合计 |", "|---" * (len(wcats) + 2) + "|"]
    for gc in gcats:
        vals = [ct[(gc, wc)] for wc in wcats]
        L.append(f"| {gc} | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)} |")
    agree = ct[("计划在先", "计划在先")] + ct[("实现在先", "实现在先")]
    resolved = sum(ct[(gc, wc)] for gc in ("计划在先", "实现在先") for wc in ("计划在先", "实现在先"))
    same_resolved = ct[("同一提交", "计划在先")] + ct[("同一提交", "实现在先")]
    L += ["", f"- 两边都给出明确顺序的任务 {resolved} 个，其中一致 {agree} 个。",
          f"- Git 判为“同一提交”（无法排序）的任务 {sum(ct[('同一提交', wc)] for wc in wcats)} 个，其中工作区能分出先后的 {same_resolved} 个。", ""]

    lag = [float(r["lag_Iw_Iv_h"]) for r in rows if r["lag_Iw_Iv_h"] != ""]
    lag_precise = [float(r["lag_Iw_Iv_h"]) for r in rows if r["lag_Iw_Iv_h"] != "" and float(r["I_w_uncert_s"] or 0) <= 60]
    gap_v = [float(r["gap_Pv_Iv_h"]) for r in rows if r["gap_Pv_Iv_h"] != ""]
    gap_w = [float(r["gap_Pw_Iw_h"]) for r in rows if r["gap_Pw_Iw_h"] != ""]
    L += ["### RQ2 时钟间隔", "", "| 间隔 | n | 中位数 | P75 | P90 | 最大 |", "|---|---|---|---|---|---|"]
    for name, xs in [("首次代码编辑 → 首个实现提交（全部）", lag), ("同上，仅时间精确的编辑", lag_precise),
                     ("任务书进 Git → 首个实现提交", gap_v), ("任务书首次编辑 → 首次代码编辑", gap_w)]:
        if xs:
            L.append(f"| {name} | {len(xs)} | {fmt_h(q(xs, .5))} | {fmt_h(q(xs, .75))} | {fmt_h(q(xs, .9))} | {fmt_h(max(xs))} |")
    over = lambda xs, h: sum(1 for x in xs if x > h)
    L += ["", f"- 首次代码编辑到提交超过 1 小时的任务 {over(lag, 1)} 个，超过 24 小时的 {over(lag, 24)} 个。", ""]
    return L


if __name__ == "__main__":
    out = (RESULTS_DIR / "r0_summary.md")
    L = ["# R0 第一版全量统计（以登记任务为单位的初步结果）", "",
         "> 本结果以任务书中登记、提交中带任务号的任务为单位，只反映规范流程内部的一致性，不代表真实工作到版本库的映射。"
         f"主分析单位已改为工作片段，见 `{PAPER_DIR.as_posix()}/research_design.md`。", "",
         "由 `analysis/extract_sessions.py`、`extract_git.py`、`align.py`、`summarize.py` 生成，数据为冻结快照 `original-snapshot-EXAMPLE`。",
         "口径见各脚本文件头说明。本文件只含汇总计数，由脚本覆盖生成，不要手改。", ""]
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        L += section(s)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
