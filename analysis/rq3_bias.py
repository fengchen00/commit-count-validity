"""RQ3：以提交为单位的常见度量相对工作片段口径的偏差。写入 <PAPER_DIR>/results/r3_bias.md。

只用内容证据的（片段, 提交）对（逐次编辑或检查点差分），不用路径—时间推定。
1. 时间：用提交时间代表"工作何时完成"的误差 = 提交时间 − 片段最后事件时间（按片段—提交对，另按行数加权）。
2. 顺序：进入版本库的改代码片段两两比较，片段先后 vs 首个内容提交先后：一致 / 颠倒 / 同一提交（顺序丢失）。
3. 工作量：agent 新写入的行（逐次编辑内容，去掉重写既有内容）按最终去向的占比；
   按周统计提交数、改代码片段数、代码编辑数，给出 Spearman 相关。
"""
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from extract_episodes import path_class  # noqa: E402


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def ts(s):
    return datetime.fromisoformat(s)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def hfmt(h):
    return f"{round(h * 60)} 分钟" if abs(h) < 1 else f"{h:.1f} 小时"


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def section(s):
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    commits = {c["hash"]: c for c in load(d / "commits.csv")}
    pairs = [p for p in load(d / "episode_commit.csv") if p["source"] != "pathtime"]
    L = [f"## {s}", ""]

    lags, wl = [], []
    for p in pairs:
        h = (ts(p["commit_time"]) - ts(eps[p["episode"]]["last_event"])).total_seconds() / 3600
        lags.append(h)
        wl.append((h, int(p["n_lines"]) + int(p["cp_lines"])))
    n = len(lags)
    tot_w = sum(w for _, w in wl) or 1
    L += ["### 1 用提交时间代表工作完成时间的误差", "",
          f"（片段—提交对 {n} 个；误差 = 提交时间 − 片段最后事件时间；负值表示内容在片段结束前已提交，片段后续仍在工作）", "",
          "| 指标 | 值 |", "|---|---|",
          f"| 中位数 | {hfmt(q(lags, .5))} |", f"| P75 | {hfmt(q(lags, .75))} |", f"| P90 | {hfmt(q(lags, .9))} |",
          f"| 超过 1 小时 | {sum(h > 1 for h in lags) / n:.0%}（按行数加权 {sum(w for h, w in wl if h > 1) / tot_w:.0%}） |",
          f"| 超过 24 小时 | {sum(h > 24 for h in lags) / n:.0%}（按行数加权 {sum(w for h, w in wl if h > 24) / tot_w:.0%}） |", ""]
    last = {}
    for p in pairs:
        last[p["episode"]] = max(last.get(p["episode"], p["commit_time"]), p["commit_time"])
    el = [(ts(t) - ts(eps[e]["last_event"])).total_seconds() / 3600 for e, t in last.items()]
    m = len(el)
    L += [f"按片段（取每个片段最后一次内容提交，{m} 个片段）：中位数 {hfmt(q(el, .5))}，超过 1 小时 {sum(h > 1 for h in el) / m:.0%}，"
          f"超过 24 小时 {sum(h > 24 for h in el) / m:.0%}。", ""]

    first = {}
    for p in pairs:
        t = ts(p["commit_time"])
        if p["episode"] not in first or t < first[p["episode"]][0]:
            first[p["episode"]] = (t, p["commit"])
    mapped = sorted(first, key=lambda e: (eps[e]["start"], e))
    c = Counter()
    for a, b in combinations(mapped, 2):
        if first[a][1] == first[b][1]:
            c["tied"] += 1
        elif first[a][0] < first[b][0]:
            c["same"] += 1
        else:
            c["reversed"] += 1
    tp = sum(c.values())
    adj = Counter()
    for a, b in zip(mapped, mapped[1:]):
        adj["tied" if first[a][1] == first[b][1] else ("same" if first[a][0] < first[b][0] else "reversed")] += 1
    ta = sum(adj.values())
    L += ["### 2 按提交还原工作顺序", "",
          f"（进入版本库的改代码片段 {len(mapped)} 个；比较片段起点先后与各自首个内容提交先后）", "",
          "| 比较范围 | 一致 | 颠倒 | 同一提交（顺序丢失） |", "|---|---|---|---|",
          f"| 全部片段两两（{tp} 对） | {c['same'] / tp:.1%} | {c['reversed'] / tp:.1%} | {c['tied'] / tp:.1%} |",
          f"| 时间上相邻的片段（{ta} 对） | {adj['same'] / ta:.0%} | {adj['reversed'] / ta:.0%} | {adj['tied'] / ta:.0%} |", ""]

    lines = Counter()
    hit = 0
    for a in load(d / "edit_attribution.csv"):
        if path_class(a["path"], "1") == "code" and a["state"] != "no_new":
            lines[a["state"]] += int(a["n_new"])
            hit += int(a["n_hit"] or 0) if a["state"] == "committed" else 0
    tl = sum(lines.values())
    zh = {"committed": "进入提交", "not_committed": "未进入提交（被覆盖或撤销）",
          "after_last_commit": "此后该文件无提交", "untracked": "文件从未进入版本库"}
    L += ["### 3 agent 新写入代码行的去向（逐次编辑内容）", "",
          "| 去向 | 行数 | 占比 |", "|---|---|---|"]
    L += [f"| {zh[k]} | {lines[k]} | {lines[k] / tl:.0%} |" for k in zh if lines[k]]
    L += ["", f"按行命中（只计在归属提交的新增行中出现的行）：{hit / tl:.0%}（{hit}/{tl}）。",
          "注：同一行被多次写入会重复计数；进入提交的行中也包含后来被覆盖、又重新写入的中间版本。", ""]

    wk = lambda t: ts(t).strftime("%G-W%V")
    cw = Counter(wk(c["time"]) for c in commits.values())
    ew = Counter(wk(e["start"]) for e in eps.values() if int(e["n_edits_code"]) > 0)
    dw = Counter()
    for e in eps.values():
        dw[wk(e["start"])] += int(e["n_edits_code"])
    weeks = sorted(set(cw) | set(ew))
    L += ["### 4 按周：提交数与实际工作量", "", "| 周 | 提交数 | 改代码片段数 | 代码编辑数 | 每个改代码片段对应的提交数 |", "|---|---|---|---|---|"]
    L += [f"| {w} | {cw[w]} | {ew[w]} | {dw[w]} | {(cw[w] / ew[w]):.1f} |" if ew[w] else f"| {w} | {cw[w]} | 0 | {dw[w]} | — |" for w in weeks]
    if len(weeks) >= 3:
        L += ["", f"Spearman 相关（{len(weeks)} 周）：提交数 vs 改代码片段数 ρ = {spearman([cw[w] for w in weeks], [ew[w] for w in weeks]):.2f}；"
              f"提交数 vs 代码编辑数 ρ = {spearman([cw[w] for w in weeks], [dw[w] for w in weeks]):.2f}。周数少，只作描述。"]
    L.append("")
    return L


if __name__ == "__main__":
    L = ["# R3 以提交为单位的度量偏差", "",
         "由 `analysis/rq3_bias.py` 生成；只用内容证据的片段—提交对。只含汇总，由脚本覆盖生成。", ""]
    for s in ["SYS_A", "SYS_C"]:
        L += section(s)
    out = (RESULTS_DIR / "r3_bias.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(out)
