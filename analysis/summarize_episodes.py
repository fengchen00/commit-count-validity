"""S01 覆盖与质量报告：写入 <PAPER_DIR>/results/s01_episodes.md。只含汇总计数，由脚本覆盖生成。"""
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def dur_fmt(m):
    return f"{m:.0f} 分钟" if m < 60 else f"{m / 60:.1f} 小时"


FEATURES = [
    ("f_edits_code", "改代码"), ("f_writes_plan", "写任务书"), ("f_no_edits", "无任何编辑"),
    ("f_task_ref", "请求中引用任务号或任务书"), ("f_customer_cue", "请求中有客户转述句式（高精度、低召回）"),
    ("f_change_cue", "请求中有改动措辞（改成、去掉等）"), ("f_commit_cmd", "执行过提交命令"), ("f_deploy_cmd", "执行过部署类命令"),
]


def section(s):
    d = Path("data/derived") / s
    eps = load(d / "episodes.csv")
    edits = load(d / "episode_edits.csv")
    sig = {(o["session"], str(o["seq"])) for o in map(json.loads, open(d / "edit_signatures.jsonl", encoding="utf-8"))}
    assigned = [e for e in edits if e["episode"] and e["reason"] == ""]
    n_code = sum(int(e["n_edits_code"]) for e in eps)
    n_all = sum(int(e["n_edits_code"]) + int(e["n_edits_plan"]) + int(e["n_edits_doc"]) + int(e["n_edits_outside"])
                + int(e["n_edits_other"]) for e in eps)
    assert n_all == len(assigned), (n_all, len(assigned))
    reasons = Counter(e["reason"] or "assigned" for e in edits)

    L = [f"## {s}", ""]
    L += [f"- 工作片段 {len(eps)} 个，来自 {len({e['session'] for e in eps})} 个主会话；"
          f"并入的接续消息 {sum(int(e['n_merged_msgs']) for e in eps)} 条；"
          f"会话首条即接续 {sum(int(e['continuation_start']) for e in eps)} 个，跨会话延续标记 {sum(int(e['cross_session_continuation']) for e in eps)} 个。",
          f"- 并入子代理的片段 {sum(1 for e in eps if int(e['n_subagents']) > 0)} 个。",
          f"- 范围内编辑：已归入片段 {reasons['assigned']} 次，失败编辑 {reasons['failed']} 次（单列）；"
          f"未能归入 {sum(v for k, v in reasons.items() if k not in ('assigned', 'failed'))} 次。", ""]

    weeks = Counter(datetime.fromisoformat(e["start"]).strftime("%G-W%V") for e in eps)
    L += ["### 按周", "", "| 周 | 片段数 |", "|---|---|"] + [f"| {w} | {n} |" for w, n in sorted(weeks.items())] + [""]

    n_ed = [int(e["n_edits_code"]) + int(e["n_edits_plan"]) + int(e["n_edits_doc"]) + int(e["n_edits_outside"])
            + int(e["n_edits_other"]) for e in eps]
    durs = [(datetime.fromisoformat(e["last_event"]) - datetime.fromisoformat(e["start"])).total_seconds() / 60 for e in eps]
    L += ["### 片段规模", "", "| 指标 | 中位数 | P75 | P90 | 最大 |", "|---|---|---|---|---|",
          f"| 编辑次数 | {q(n_ed, .5)} | {q(n_ed, .75)} | {q(n_ed, .9)} | {max(n_ed)} |",
          f"| 持续时间（起点到最后一个事件） | {dur_fmt(q(durs, .5))} | {dur_fmt(q(durs, .75))} | {dur_fmt(q(durs, .9))} | {dur_fmt(max(durs))} |", ""]

    with_sig = sum(1 for e in assigned if (e["session"], e["seq"]) in sig)
    from map_episodes import path_class  # noqa: E402  代码编辑口径与 map_episodes 相同
    ev_rows = load(d / "events.csv")
    ev_all = {(r["session"], r["seq"]): r for r in ev_rows}
    code_sig = sum(1 for e in assigned if (e["session"], e["seq"]) in sig
                   and path_class(ev_all[(e["session"], e["seq"])]["path"].replace("\\", "/"), ev_all[(e["session"], e["seq"])]["in_workspace"]) == "code")
    unc = {(r["session"], r["seq"]): (r["time_src"], float(r["uncert_s"] or 0)) for r in ev_rows}
    eu = [unc[(e["session"], e["seq"])] for e in assigned]
    assert len(eu) == len(assigned)
    src = Counter(t for t, _ in eu)
    L += ["### 数据质量", "",
          f"- 已归入编辑的时间来源：{'；'.join(f'{k} {v} 次（{v / len(eu):.0%}）' for k, v in src.most_common())}。",
          f"- 编辑时间不确定度：为 0 的 {sum(u == 0 for _, u in eu) / len(eu):.1%}，超过 1 分钟 {sum(u > 60 for _, u in eu) / len(eu):.1%}，"
          f"超过 10 分钟 {sum(u > 600 for _, u in eu) / len(eu):.1%}，超过 1 小时 {sum(u > 3600 for _, u in eu) / len(eu):.1%}。",
          f"- 片段起点时间：与另一片段起点相同（两条消息之间无带时间的动作）{sum(int(e['tied_start']) for e in eps)} 个；"
          f"起点不确定超过 10 分钟 {sum(int(e['start_time_uncertain']) for e in eps)} 个（只影响时长，不影响事件归属）。",
          f"- 已归入的编辑中，带写入内容、可按内容归属的 {with_sig} 次（{with_sig / len(assigned):.0%}），其余只能按时间与路径分析。",
          f"- 代码编辑 {n_code} 次，占已归入编辑的 {n_code / len(assigned):.0%}；其中无写入内容、不能按内容归属的 {n_code - code_sig} 次（{(n_code - code_sig) / n_code:.1%}；论文 3.4.2 节的分母为代码编辑）。", ""]

    L += ["### 片段特征（只作特征，不作分类）", "", "| 特征 | 片段数 | 占比 |", "|---|---|---|"]
    for k, name in FEATURES:
        n = sum(int(e[k]) for e in eps)
        L.append(f"| {name} | {n} | {n / len(eps):.0%} |")
    plan_only = sum(1 for e in eps if e["f_writes_plan"] == "1" and e["f_edits_code"] == "0")
    L += ["", f"- 只写任务书、不改代码的片段 {plan_only} 个；改代码的片段 {sum(int(e['f_edits_code']) for e in eps)} 个；"
          f"无任何编辑（讨论、查询、部署、只读）的片段 {sum(int(e['f_no_edits']) for e in eps)} 个。", ""]
    return L


if __name__ == "__main__":
    L = ["# S01 工作片段：覆盖与质量", "",
         "由 `analysis/extract_episodes.py` 与 `analysis/summarize_episodes.py` 生成；片段定义见 `notes/decisions.md` D4。只含汇总计数，由脚本覆盖生成，不要手改。", ""]
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        L += section(s)
    out = (RESULTS_DIR / "s01_episodes.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(out)
