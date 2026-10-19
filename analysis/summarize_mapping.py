"""R1 片段—版本库映射汇总：写入 <PAPER_DIR>/results/r1_mapping.md。只含汇总计数，由脚本覆盖生成。"""
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from patterns import PATTERNS  # noqa: E402

SUFFIX = os.environ.get("ATTR_SUFFIX", "")
MAIN = [p for p in PATTERNS if p != "无代码"]
LABELS = ["task", "milestone", "fix", "hotfix", "unlabeled"]
LABEL_ZH = {"task": "任务号", "milestone": "里程碑", "fix": "修复", "hotfix": "热修", "unlabeled": "无标签"}
LINK_ZH = {"request_task": "请求写明任务", "request_milestone": "请求写明里程碑", "commit_task": "进入登记任务的提交",
           "session_milestone": "同会话此前写过里程碑（弱）", "none": "未对应任何登记任务"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pct(a, b):
    return f"{a / b:.0%}" if b else "—"


def pattern_table(rows, col, title):
    n = len(rows)
    c = Counter(p for r in rows for p in r[col].split(";"))
    L = [f"**{title}**（n = {n}，可多标签，占比之和可超过 100%）", "", "| 模式 | 片段数 | 占比 |", "|---|---|---|"]
    L += [f"| {p} | {c[p]} | {pct(c[p], n)} |" for p in MAIN if c[p]]
    return L + [""], c


def section(s):
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    pats = load(d / f"episode_patterns{SUFFIX}.csv")
    tbk = {r["episode"]: r for r in load(d / "episode_taskbook.csv")}
    pairs = [p for p in load(d / f"episode_commit{SUFFIX}.csv") if p["source"] != "pathtime"]
    assert len(pats) == len(eps)
    code = [r for r in pats if r["evidence"]]
    ver = [r for r in code if r["evidence"] == "verified"]
    assert len(code) == sum(int(e["n_edits_code"]) > 0 for e in eps.values())
    assert all(r["patterns"] == "无代码" for r in pats if not r["evidence"])

    ev = Counter(r["evidence"] for r in code)
    L = [f"## {s}", "",
         f"- 工作片段 {len(pats)} 个，其中改代码 {len(code)} 个、无代码 {len(pats) - len(code)} 个（讨论、查询、部署、只改文档）。",
         f"- 改代码片段的证据层：全部可按内容核验 {ev['verified']}（{pct(ev['verified'], len(code))}），"
         f"部分可核验 {ev['partial']}，只能按路径—时间推定 {ev['inferred']}。", ""]

    L += ["### 候选模式", ""]
    t, cv = pattern_table(ver, "patterns", "主表：可核验片段")
    L += t
    t, _ = pattern_table(code, "patterns", "全部改代码片段（内容证据；证据不足者单列）")
    L += t
    t, cp = pattern_table(code, "patterns_pt", "对照：加入路径—时间推定")
    L += t
    _, cc = pattern_table(code, "patterns", "")
    diff = [(p, cc[p], cp[p]) for p in MAIN if abs(cp[p] - cc[p]) / len(code) > 0.05]
    if diff:
        L += ["加入推定后变化超过 5 个百分点的模式：" + "；".join(f"{p} {a}→{b}" for p, a, b in diff) + "。", ""]

    lab_of = defaultdict(Counter)
    for p in pairs:
        lab_of[p["episode"]][p["commit_label"]] += 1
    L += ["### 模式 × 提交类别（可核验片段；单元格为片段进入的提交数）", "",
          "| 模式 | " + " | ".join(LABEL_ZH[x] for x in LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for p in MAIN:
        rs = [r for r in ver if p in r["patterns"].split(";")]
        if not rs or p in ("缺失", "缺失-窗口末", "无新增内容", "证据不足"):
            continue
        c = Counter()
        for r in rs:
            c.update(lab_of[r["episode"]])
        L.append(f"| {p} | " + " | ".join(str(c[x]) for x in LABELS) + " |")
    L.append("")

    L += ["### 模式 × 片段特征（可核验片段）", "",
          "| 模式 | 片段数 | 代码编辑数中位数 | 请求字数中位数 | 客户转述句式 | 未对应登记任务 |", "|---|---|---|---|---|---|"]
    for p in MAIN:
        rs = [r for r in ver if p in r["patterns"].split(";")]
        if not rs:
            continue
        es = [eps[r["episode"]] for r in rs]
        L.append(f"| {p} | {len(rs)} | {median(int(e['n_edits_code']) for e in es):.0f} | "
                 f"{median(int(e['request_len']) for e in es):.0f} | {sum(int(e['f_customer_cue']) for e in es)} | "
                 f"{pct(sum(tbk[r['episode']]['link'] == 'none' for r in rs), len(rs))} |")
    L.append("")

    per_commit = Counter(p["commit"] for p in pairs)
    per_ep = Counter(p["episode"] for p in pairs)
    L += ["### 片段与提交的多对多程度（内容证据）", "",
          f"- 被片段内容触及的提交 {len(per_commit)} 个，平均每个提交含 {sum(per_commit.values()) / len(per_commit):.2f} 个片段，"
          f"含 ≥ 2 个片段的提交占 {pct(sum(v >= 2 for v in per_commit.values()), len(per_commit))}。",
          f"- 进入提交的片段 {len(per_ep)} 个，平均每个片段分散到 {sum(per_ep.values()) / len(per_ep):.2f} 个提交，"
          f"分散到 ≥ 2 个提交的占 {pct(sum(v >= 2 for v in per_ep.values()), len(per_ep))}。", ""]

    code_ids = {r["episode"] for r in code}
    lk = Counter(tbk[e]["link"] for e in code_ids)
    L += ["### 任务书层（改代码片段）", "", "| 与登记任务的对应 | 片段数 | 占比 |", "|---|---|---|"]
    L += [f"| {LINK_ZH[k]} | {lk[k]} | {pct(lk[k], len(code_ids))} |" for k in LINK_ZH]
    tk = [tbk[e] for e in code_ids if int(tbk[e]["n_tasks"])]
    L += ["", f"对应到具体任务的片段 {len(tk)} 个：任务书在片段开始前已在工作区写好的 "
          f"{sum(r['plan_written_before'] == '1' for r in tk)} 个，已进入版本库的 {sum(r['plan_committed_before'] == '1' for r in tk)} 个；"
          f"片段结束时版本库中已打勾的 {sum(r['ticked_git_by_end'] == '1' for r in tk)} 个。", ""]
    return L


if __name__ == "__main__":
    L = ["# R1 片段—版本库映射", "",
         "由 `analysis/map_episodes.py`、`patterns.py`、`taskbook_layer.py`、`summarize_mapping.py` 生成；"
         "映射规则见 `notes/decisions.md` D3、D3b，模式定义见 D5。自动模式只是候选，以人工核验结果为准。只含汇总，由脚本覆盖生成。", ""]
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        L += section(s)
    out = (RESULTS_DIR / f"r1_mapping{SUFFIX}.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(out)
