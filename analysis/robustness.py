"""v2.8 补充分析（S11-T20，口径见 notes/decisions.md D11）。

1 任务条目拆分（向细）  2 完整性与纯度  3 合并阈值  4 片段内提交与执行者、RQ2 加入片段内提交与会话内比较
5 按天的映射稳定性  6 按连通分量重抽
写入 <PAPER_DIR>/results/r8_robustness.md，关键数字写入 numbers.json 的 robustness 键（须在 render_tables 之后运行）。
"""
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import command_commits  # noqa: E402
from patterns import classify, prepare, unit_state  # noqa: E402

BOOT = 2000
SEED = "S11-T20"
CUE = re.compile(r"提交|commit|推送|push|入库", re.I)
NAME = {"SYS_A": "系统 A", "SYS_C": "系统 C"}
MERGE = ("合并-同会话", "合并-跨会话")
NUMS = (RESULTS_DIR / "numbers.json")


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def q(xs, p):
    xs = sorted(v for v in xs if v == v)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def pc(v):
    return "—" if v != v else f"{v:.0%}"


def ci(xs, signed=False):
    f = (lambda v: f"{v * 100:+.0f}") if signed else (lambda v: f"{v:.0%}")
    return f"{f(q(xs, .025))}–{f(q(xs, .975))}" if signed is False else f"{f(q(xs, .025))} 至 {f(q(xs, .975))}"


def has(labels, keys):
    return any(k in labels for k in keys)


def cluster_boot(items, cluster_of, tag):
    """按簇有放回重抽，返回重抽样本列表（每个样本是 items 的列表）。"""
    by = defaultdict(list)
    for it in items:
        by[cluster_of(it)].append(it)
    ks = sorted(by)
    rng = random.Random(SEED + tag)
    return [[it for k in rng.choices(ks, k=len(ks)) for it in by[k]] for _ in range(BOOT)]


def share(xs, pred):
    return sum(map(pred, xs)) / len(xs) if xs else float("nan")


def spearman(a, b):
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for x in set(v):
            m = v == x
            r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def logit(X, y):
    if y.min() == y.max():
        return None
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    else:
        return None
    return None if np.abs(b).max() > 20 else b


def ame(X, b, j):
    X1, X0 = X.copy(), X.copy()
    X1[:, j], X0[:, j] = 1.0, 0.0
    return float(np.mean(1 / (1 + np.exp(-X1 @ b)) - 1 / (1 + np.exp(-X0 @ b))))


def base(s):
    x = prepare(s)
    d = x["d"]
    pat = {r["episode"]: set(r["patterns"].split(";")) for r in load(d / "episode_patterns.csv")}
    ev = {r["episode"]: r["evidence"] for r in load(d / "episode_patterns.csv")}
    tb = {r["episode"]: int(r["ex_ante"] or 0) for r in load(d / "episode_taskbook.csv")}
    code = [e for e in x["eps"] if int(e["n_edits_code"]) > 0]
    lines = {}
    for p in load(d / "episode_commit.csv"):
        if p["source"] != "pathtime":
            lines[(p["episode"], p["commit"])] = max(int(p["n_lines"]), int(p["cp_lines"]))
    items = [{"e": e["episode"], "s": e["session"], "ver": ev[e["episode"]] == "verified", "labels": pat[e["episode"]],
              "reg": tb[e["episode"]], "n": int(e["n_edits_code"]), "cmd": int(e["n_commit_cmds"]) > 0,
              "cue": bool(CUE.search(e["request_text"])), "start": e["start"]} for e in code]
    return x, items, lines


def components(items, x):
    par = {}

    def f(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for it in items:
        par[f("E" + it["e"])] = f("S" + it["s"])
        for c in sorted(x["content"].get(it["e"], ())):
            par[f("E" + it["e"])] = f("C" + c)
    label = {}
    for a in sorted(par):
        label.setdefault(f(a), a)
    return {it["e"]: label[f("E" + it["e"])] for it in items}


# 1 任务条目拆分 --------------------------------------------------------------------------------
def refine(x, items):
    cm, content, key = x["cm"], x["content"], x["key"]
    res = {}
    for mode in ("episode", "task"):
        uc, meta = {}, {}
        for it in items:
            cs = content.get(it["e"], set())
            if not cs:
                continue
            groups = defaultdict(set)
            for c in cs:
                t = cm[c]
                g = (t["milestone"], t["tasks"]) if mode == "task" and t["label"] == "task" and t["milestone"] and t["tasks"] else ("", "")
                groups[g].add(c)
            for i, (_, gcs) in enumerate(sorted(groups.items())):
                u = f"{key(it['e'])}#{i}"
                uc[u], meta[u] = gcs, it
        eps_of = defaultdict(set)
        for u, cs in uc.items():
            for c in cs:
                eps_of[c].add(u)
        units = []
        for u, cs in uc.items():
            it = meta[u]
            st = unit_state([x["st"][it["e"]]])
            lab = set(classify(it["n"], u, cs, eps_of, cm, x["ctx"][it["e"]], x["flatten"][it["e"]], st))
            units.append({"s": it["s"], "ver": it["ver"], "labels": lab, "e": it["e"]})
        res[mode] = units
    return res


def per_episode(units):
    """片段层汇总：全部子单位一对一 / 任一子单位合并 / 子单位含拆分。"""
    by = defaultdict(list)
    for u in units:
        by[u["e"]].append(u)
    return [{"s": us[0]["s"], "ver": us[0]["ver"], "labels": ({"一对一"} if all("一对一" in u["labels"] for u in us) else set())
             | ({"合并-同会话"} if any(has(u["labels"], MERGE) for u in us) else set())
             | ({"拆分"} if any("拆分" in u["labels"] for u in us) else set())} for us in by.values()]


# 2 完整性与纯度 ---------------------------------------------------------------------------------
def purity(items, lines):
    tot_c, by_e = Counter(), defaultdict(dict)
    for (e, c), v in lines.items():
        tot_c[c] += v
        by_e[e][c] = v
    out = []
    for it in items:
        row = by_e.get(it["e"])
        if not row or sum(row.values()) == 0:
            continue
        c = max(row, key=row.get)
        out.append({"s": it["s"], "ver": it["ver"], "comp": row[c] / sum(row.values()),
                    "pur": row[c] / tot_c[c] if tot_c[c] else 1.0, "labels": it["labels"]})
    return out


# 3 合并阈值 ------------------------------------------------------------------------------------
def thresholds(x, items, lines):
    tot_c = Counter()
    for (e, c), v in lines.items():
        tot_c[c] += v
    out = {}
    for k in (1, 5, 20, "10%"):
        keep = (lambda e, c: lines[(e, c)] >= k) if k != "10%" else (lambda e, c: lines[(e, c)] >= 0.1 * tot_c[c])
        eps_of = defaultdict(set)
        for (e, c) in lines:
            if keep(e, c):
                eps_of[c].add(x["key"](e))
        labs = []
        for it in items:
            P = x["content"].get(it["e"], set())
            st = unit_state([x["st"][it["e"]]])
            own = {c: eps_of[c] | {x["key"](it["e"])} for c in P}
            labs.append((it, set(classify(it["n"], x["key"](it["e"]), P, own, x["cm"], x["ctx"][it["e"]], x["flatten"][it["e"]], st))))
        out[k] = labs
    return out


def run(s):
    x, items, lines = base(s)
    ver = [i for i in items if i["ver"]]
    comp_of = components(items, x)
    L = [f"## {NAME[s]}", ""]
    R = {}

    # 1
    rf = refine(x, items)
    L += ["### 1 任务条目拆分（向细，D11-1）", "",
          "| 单位 | 进入版本库的单位 n | 一对一 | 合并 | 拆分 | 可核验片段的单位 n | 一对一 | 合并 | 拆分 |", "|---|---|---|---|---|---|---|---|---|"]
    rf["task_ep"] = per_episode(rf["task"])
    for mode, nm in (("episode", "片段（原口径）"), ("task", "任务条目子单位"), ("task_ep", "片段（按子单位汇总）")):
        us = rf[mode]
        uv = [u for u in us if u["ver"]]
        cells = []
        for grp, tag in ((us, "all"), (uv, "ver")):
            bs = cluster_boot(grp, lambda u: u["s"], f"{s}-rf-{mode}-{tag}")
            row = [str(len(grp))]
            R[f"refine_{mode}_{tag}_n"] = len(grp)
            for keys in (("一对一",), MERGE, ("拆分",)):
                p = share(grp, lambda u: has(u["labels"], keys))
                row.append(f"{pc(p)}（{ci([share(b, lambda u: has(u['labels'], keys)) for b in bs])}）")
                R[f"refine_{mode}_{tag}_{keys[0][:2]}"] = round(p, 3)
            cells += row
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L += ["", "子单位按提交自身的任务编号划分，是对提交最有利的向细划分；括号内为按会话重抽的 95% 区间。"
          "按子单位汇总到片段：全部子单位一对一记为一对一，任一子单位合并记为合并，任一子单位拆分记为拆分。", ""]

    # 2
    pu = purity(items, lines)
    L += ["### 2 完整性与纯度（D11-2）", "", "| 分母 | n | 完整性中位数 | 纯度中位数 | 完整性 ≥ 0.9 | 纯度 ≥ 0.9 | 二者皆 ≥ 0.9 | 二者皆 ≥ 0.9 的 95% 区间 |",
          "|---|---|---|---|---|---|---|---|"]
    for grp, nm, tag in ((pu, "按内容进入版本库的代码修改片段", "all"), ([p for p in pu if p["ver"]], "其中可核验", "ver")):
        both = lambda p: p["comp"] >= 0.9 and p["pur"] >= 0.9
        bs = cluster_boot(grp, lambda p: p["s"], f"{s}-pu-{tag}")
        v = share(grp, both)
        R[f"pure90_{tag}"] = round(v, 3)
        R[f"comp_med_{tag}"] = round(q([p["comp"] for p in grp], .5), 2)
        R[f"pur_med_{tag}"] = round(q([p["pur"] for p in grp], .5), 2)
        L.append(f"| {nm} | {len(grp)} | {q([p['comp'] for p in grp], .5):.2f} | {q([p['pur'] for p in grp], .5):.2f} | "
                 f"{pc(share(grp, lambda p: p['comp'] >= 0.9))} | {pc(share(grp, lambda p: p['pur'] >= 0.9))} | {pc(v)} | "
                 f"{ci([share(b, both) for b in bs])} |")
    one = [p for p in pu if "一对一" in p["labels"]]
    L += ["", f"判为一对一的片段中二者皆 ≥ 0.9 的占 {pc(share(one, lambda p: p['comp'] >= 0.9 and p['pur'] >= 0.9))}（n = {len(one)}）。"
          "纯度只计会话可归属的内容，提交中来自无内容编辑或会话之外的行不计入分母。", ""]

    # 3
    th = thresholds(x, items, lines)
    L += ["### 3 合并判定阈值（D11-3）", "", "| 另一片段在该提交中的行数 | 可核验：一对一 | 可核验：合并 | 全部代码修改：一对一 | 全部代码修改：合并 |", "|---|---|---|---|---|"]
    for k, labs in th.items():
        lv = [lb for it, lb in labs if it["ver"]]
        la = [lb for _, lb in labs]
        L.append(f"| ≥ {k}{'' if k == '10%' else ' 行'}{'（现口径）' if k == 1 else ''} | {pc(share(lv, lambda l: '一对一' in l))} | "
                 f"{pc(share(lv, lambda l: has(l, MERGE)))} | {pc(share(la, lambda l: '一对一' in l))} | {pc(share(la, lambda l: has(l, MERGE)))} |")
        R[f"thr_{k}_ver_one"] = round(share(lv, lambda l: "一对一" in l), 3)
        R[f"thr_{k}_ver_merge"] = round(share(lv, lambda l: has(l, MERGE)), 3)
    L += ["", "“≥ 10%”指另一片段的行数不少于该提交可归属行数的 10%。", ""]

    # 4 片段内提交
    gold = command_commits(x["d"], x["cm"])
    sess_of = {e["episode"]: e["session"] for e in x["eps"]}
    L += ["### 4 片段内提交与提交执行者（D11-4）", "",
          f"代码修改片段 {len(items)} 个，片段内执行过提交命令的 {sum(i['cmd'] for i in items)} 个（{pc(share(items, lambda i: i['cmd']))}）；"
          f"请求含明确提交指令的 {sum(i['cue'] for i in items)} 个（{pc(share(items, lambda i: i['cue']))}），其中片段内执行了提交的 "
          f"{sum(i['cue'] and i['cmd'] for i in items)} 个。", ""]
    R["cmd_share_all"] = round(share(items, lambda i: i["cmd"]), 3)
    R["cue_share_all"] = round(share(items, lambda i: i["cue"]), 3)
    R["n_cmd_all"] = sum(i["cmd"] for i in items)
    R["n_cue_all"] = sum(i["cue"] for i in items)
    L += ["| 分母 | 组 | n | 一对一 | 合并 | 拆分 | 缺失 |", "|---|---|---|---|---|---|---|"]
    for grp, nm, tag in ((ver, "可核验", "ver"), (items, "全部代码修改", "all")):
        for flag, gname in ((True, "片段内提交"), (False, "片段内未提交")):
            g = [i for i in grp if i["cmd"] == flag]
            cells = [pc(share(g, lambda i, k=k: has(i["labels"], k))) for k in (("一对一",), MERGE, ("拆分",), ("缺失",))]
            L.append(f"| {nm} | {gname} | {len(g)} | " + " | ".join(cells) + " |")
            R[f"cmd{int(flag)}_{tag}_n"] = len(g)
            R[f"cmd{int(flag)}_{tag}_one"] = round(share(g, lambda i: "一对一" in i["labels"]), 3)
            R[f"cmd{int(flag)}_{tag}_merge"] = round(share(g, lambda i: has(i["labels"], MERGE)), 3)
        bs = cluster_boot(grp, lambda i: i["s"], f"{s}-cmd-{tag}")
        for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并")):
            diffs = [share([i for i in b if i["cmd"]], lambda i: has(i["labels"], keys)) - share([i for i in b if not i["cmd"]], lambda i: has(i["labels"], keys)) for b in bs]
            L.append(f"| {nm} | 差（提交 − 未提交）：{kn} | | {ci(diffs, True)} 个百分点 | | | |")
            R[f"cmd_diff_{tag}_{kn}"] = [round(q(diffs, .025), 3), round(q(diffs, .975), 3)]
    ex = Counter()
    ex_nocmd = Counter()
    for i in items:
        for c in x["content"].get(i["e"], ()):
            g = gold.get(c)
            k = "本片段" if g == i["e"] else ("同会话其他片段" if g and sess_of.get(g) == i["s"] else ("其他会话" if g else "无对应提交命令"))
            ex[k] += 1
            if not i["cmd"]:
                ex_nocmd[k] += 1
    L += ["", "| 内容所进入提交的执行者 | 全部片段—提交对 | 其中片段内未提交的片段—提交对 |", "|---|---|---|"]
    for k in ("本片段", "同会话其他片段", "其他会话", "无对应提交命令"):
        L.append(f"| {k} | {ex[k]}（{pc(ex[k] / sum(ex.values()))}） | {ex_nocmd[k]}（{pc(ex_nocmd[k] / max(1, sum(ex_nocmd.values())))}） |")
        R[f"exec_nocmd_{k}"] = round(ex_nocmd[k] / max(1, sum(ex_nocmd.values())), 3)
    L.append("")

    if sum(i["reg"] for i in ver) >= 20:
        L += rq2(s, ver, R)

    # 5 按天
    L += stability(s, x, items, R)

    # 6 按连通分量重抽
    nc = len(set(comp_of.values()))
    sizes = sorted(Counter(comp_of.values()).values(), reverse=True)
    L += ["### 6 按连通分量重抽（D11-6）", "",
          f"代码修改片段分布在 {len({i['s'] for i in items})} 个会话中（可核验片段在 {len({i['s'] for i in ver})} 个会话中）；"
          f"按同会话或共享内容提交连边后为 {nc} 个连通分量，最大的五个含 {', '.join(map(str, sizes[:5]))} 个片段。", "",
          "| 分母 | 模式 | 点估计 | 按会话重抽 | 按连通分量重抽 |", "|---|---|---|---|---|"]
    R["n_sessions_all"] = len({i["s"] for i in items})
    R["n_sessions_ver"] = len({i["s"] for i in ver})
    R["n_components"] = nc
    for grp, nm, tag in ((ver, "可核验", "ver"), (items, "全部代码修改", "all")):
        bs_s = cluster_boot(grp, lambda i: i["s"], f"{s}-cs-{tag}")
        bs_c = cluster_boot(grp, lambda i: comp_of[i["e"]], f"{s}-cc-{tag}")
        for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并"), (("拆分",), "拆分")):
            f = lambda b: share(b, lambda i: has(i["labels"], keys))
            L.append(f"| {nm} | {kn} | {pc(f(grp))} | {ci([f(b) for b in bs_s])} | {ci([f(b) for b in bs_c])} |")
            R[f"cc_{tag}_{kn}"] = [round(q([f(b) for b in bs_c], .025), 3), round(q([f(b) for b in bs_c], .975), 3)]
    L.append("")
    return L, R


def rq2(s, ver, R):
    """D10 回归加入片段内提交；会话内比较。"""
    def fit(xs, keys):
        X = np.array([[1.0, i["reg"], math.log1p(i["n"]), float(i["cmd"])] for i in xs])
        y = np.array([float(has(i["labels"], keys)) for i in xs])
        if X[:, 1].min() == X[:, 1].max() or X[:, 3].min() == X[:, 3].max():
            return None
        b = logit(X, y)
        return None if b is None else (ame(X, b, 1), ame(X, b, 3))

    def within(xs, keys):
        num = den = 0.0
        by = defaultdict(list)
        for i in xs:
            by[i["s"]].append(i)
        for g in by.values():
            g1, g0 = [i for i in g if i["reg"]], [i for i in g if not i["reg"]]
            if g1 and g0:
                w = len(g1) * len(g0) / len(g)
                num += w * (share(g0, lambda i: has(i["labels"], keys)) - share(g1, lambda i: has(i["labels"], keys)))
                den += w
        return (num / den, den) if den else (None, 0)

    bs = cluster_boot(ver, lambda i: i["s"], f"{s}-rq2")
    reg1, reg0 = [i for i in ver if i["reg"]], [i for i in ver if not i["reg"]]
    L = ["### 4b RQ2：加入片段内提交（系统内，可核验片段；D11-4）", "",
         f"片段内执行提交的比例：事前登记 {pc(share(reg1, lambda i: i['cmd']))}（n = {len(reg1)}），未登记 {pc(share(reg0, lambda i: i['cmd']))}（n = {len(reg0)}）。", "",
         "| 模式 | 登记 AME（未登记 − 登记；控制规模与片段内提交） | 95% 区间 | 片段内提交 AME（提交 − 未提交） | 95% 区间 | 会话内比较（未登记 − 登记） | 95% 区间 | 跳过 |",
         "|---|---|---|---|---|---|---|---|"]
    R["rq2_cmd_reg1"] = round(share(reg1, lambda i: i["cmd"]), 3)
    R["rq2_cmd_reg0"] = round(share(reg0, lambda i: i["cmd"]), 3)
    n_within = len({i["s"] for i in ver if any(j["reg"] for j in ver if j["s"] == i["s"]) and any(not j["reg"] for j in ver if j["s"] == i["s"])})
    for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并")):
        a = fit(ver, keys)
        w, _ = within(ver, keys)
        ar, ac, ww, skip = [], [], [], 0
        for b in bs:
            r = fit(b, keys)
            wv, _ = within(b, keys)
            if r is None or wv is None:
                skip += 1
                continue
            ar.append(-r[0])
            ac.append(r[1])
            ww.append(wv)
        L.append(f"| {kn} | {-a[0] * 100:+.0f} | {ci(ar, True)} | {a[1] * 100:+.0f} | {ci(ac, True)} | {w * 100:+.0f} | {ci(ww, True)} | {skip} |")
        R[f"rq2_{kn}"] = {"reg_ame": round(-a[0], 3), "reg_ci": [round(q(ar, .025), 3), round(q(ar, .975), 3)],
                          "cmd_ame": round(a[1], 3), "cmd_ci": [round(q(ac, .025), 3), round(q(ac, .975), 3)],
                          "within": round(w, 3), "within_ci": [round(q(ww, .025), 3), round(q(ww, .975), 3)]}
    R["rq2_within_sessions"] = n_within
    L += ["", f"逻辑回归：模式 ~ 截距 + 事前登记 + log(1 + 代码编辑数) + 片段内提交。会话内比较只用同时含登记与未登记片段的 {n_within} 个会话，"
          "层内差按 n₁n₀/(n₁+n₀) 加权。区间按会话重抽。", ""]
    return L


def stability(s, x, items, R):
    day = lambda t: t[:10]
    cw = Counter(day(c["time"]) for c in x["cm"].values())
    ew, dw, mw = Counter(), Counter(), Counter()
    for i in items:
        ew[day(i["start"])] += 1
        dw[day(i["start"])] += i["n"]
        mw[day(i["start"])] += i["cmd"]
    days = sorted(d for d in ew if ew[d] >= 3)
    c = np.array([cw[d] for d in days], float)
    e = np.array([ew[d] for d in days], float)
    n = np.array([dw[d] for d in days], float)
    m = np.array([mw[d] / ew[d] for d in days])
    r1, r2 = c / e, c / n * 100
    L = ["### 5 按天的映射稳定性（D11-5）", "",
         f"代码修改片段 ≥ 3 的日子 {len(days)} 天。每日提交数 / 代码修改片段：中位数 {np.median(r1):.1f}（四分位 {np.percentile(r1, 25):.1f}–{np.percentile(r1, 75):.1f}，"
         f"范围 {r1.min():.1f}–{r1.max():.1f}）；每百次代码编辑的提交数：中位数 {np.median(r2):.1f}（四分位 {np.percentile(r2, 25):.1f}–{np.percentile(r2, 75):.1f}，"
         f"范围 {r2.min():.1f}–{r2.max():.1f}）。", "",
         f"Spearman 相关：提交数与片段数 ρ = {spearman(c, e):.2f}，提交数与代码编辑数 ρ = {spearman(c, n):.2f}，"
         f"每片段提交数与当日片段内提交比例 ρ = {spearman(r1, m):.2f}。", ""]
    R.update({"days": len(days), "r1_med": round(float(np.median(r1)), 1), "r1_iqr": [round(float(np.percentile(r1, 25)), 1), round(float(np.percentile(r1, 75)), 1)],
              "r1_range": [round(float(r1.min()), 1), round(float(r1.max()), 1)], "rho_ce": round(spearman(c, e), 2), "rho_cn": round(spearman(c, n), 2),
              "rho_r1_m": round(spearman(r1, m), 2)})
    if len(days) >= 8:
        y = np.log1p(c)
        X0 = np.column_stack([np.ones(len(days)), np.log1p(e)])
        X1 = np.column_stack([X0, m])
        fit = lambda X: np.linalg.lstsq(X, y, rcond=None)[0]
        r2f = lambda X, b: 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b0, b1 = fit(X0), fit(X1)
        L += [f"log(1 + 提交数) ~ log(1 + 片段数)：斜率 {b0[1]:.2f}，R² = {r2f(X0, b0):.2f}；加入当日片段内提交比例：该项系数 {b1[2]:.2f}"
              f"（比例每增 0.1，提交数约乘以 {math.exp(b1[2] * 0.1):.2f}），R² = {r2f(X1, b1):.2f}。", ""]
        R.update({"ols_r2_base": round(r2f(X0, b0), 2), "ols_r2_cmd": round(r2f(X1, b1), 2), "ols_cmd_coef": round(float(b1[2]), 2),
                  "ols_cmd_mult10": round(math.exp(b1[2] * 0.1), 2)})
    return L


if __name__ == "__main__":
    out = ["# v2.8 补充分析（S11-T20）", "", "由 `analysis/robustness.py` 生成，口径见 `notes/decisions.md` D11。", ""]
    num = json.loads(NUMS.read_text(encoding="utf-8")) if NUMS.exists() else {}
    num["robustness"] = {}
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        L, R = run(s)
        out += L
        num["robustness"][s] = R
    (RESULTS_DIR / "r8_robustness.md").write_text("\n".join(out), encoding="utf-8")
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out))
