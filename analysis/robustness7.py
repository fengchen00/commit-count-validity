"""回应第七轮模拟审稿的 RQ3 补充分析（S18-T1，口径见 notes/decisions.md D18）。

1 事前分期（不含提交信息）  2 按片段归属提交与删失  3 阈值、门槛与系统 C 规则前日子  4 日对并列与编辑口径
写入 <PAPER_DIR>/results/r15_robustness7.md 与 numbers.json 的 robustness7 键（须在 boundary_labels.py score 之后运行）。
"""
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from robustness import NAME, base, q  # noqa: E402

BOOT, SEED = 2000, "S18"
NUMS = (RESULTS_DIR / "numbers.json")
RES = RESULTS_DIR
RULE_C = "2026-09-20"


def ts(s):
    return datetime.fromisoformat(s)


def bci(v):
    v = [x for x in v if x == x]
    return [q(v, .025), q(v, .975)] if v else [float("nan")] * 2


def load_ctx(s, works, samples=None):
    """works: {True/False: 两组每片段新工作项均值}；samples: {True/False: 各样本片段的新工作项数列表}（D26-4，两层重抽用）。"""
    x, items, _ = base(s)
    tref = {e["episode"]: int(e["f_task_ref"] or 0) for e in x["eps"]}
    E_of = defaultdict(set)
    for i in items:
        for c in x["content"].get(i["e"], ()):
            E_of[c].add(i["e"])
    share = {i["e"]: sum(1 / len(E_of[c]) for c in x["content"].get(i["e"], ())) for i in items}
    t_end = max(ts(c["time"]) for c in x["cm"].values())
    for i in items:
        i["tref"] = tref[i["e"]]
        i["share"] = share[i["e"]]
        i["w_hat"] = works[i["cmd"]]
        i["late"] = ts(i["start"]) > t_end - timedelta(hours=48)
    cday, ctask = defaultdict(int), defaultdict(int)
    for c in x["cm"].values():
        cday[c["time"][:10]] += 1
        ctask[c["time"][:10]] += c["label"] == "task"
    n_nocontent = sum(1 for c in x["cm"] if c not in E_of)
    return {"items": items, "cday": cday, "ctask": ctask, "n_commits": len(x["cm"]), "n_nocontent": n_nocontent,
            "works": works, "samples": samples}


def days(ctx, by, attr="time", min_eps=3, after=None, censor=False):
    g = defaultdict(list)
    for i in ctx["items"]:
        if censor and i["late"]:
            continue
        g[i["start"][:10]].append(i)
    out = []
    for d in sorted(g):
        if after and d < after:
            continue
        es = g[d]
        if len(es) < min_eps:
            continue
        c = ctx["cday"].get(d, 0) if attr == "time" else sum(i["share"] for i in es)
        cd = ctx["cday"].get(d, 0)
        gt = ctx["ctask"].get(d, 0) / cd if cd else float("nan")
        gv = gt if by == "git_task" else sum(bool(i[by]) for i in es) / len(es)
        if gv != gv:
            continue
        out.append({"d": d, "n": len(es), "w": sum(i["n"] for i in es), "k": sum(i["w_hat"] for i in es), "c": c, "g": gv, "_ctx": ctx,
                    "reg": sum(bool(i["reg"]) for i in es) / len(es), "cmd": sum(i["cmd"] for i in es) / len(es), "git_task": gt,
                    "n_reg": sum(bool(i["reg"]) for i in es), "n_cmd": sum(bool(i["cmd"]) for i in es),
                    "n_tref": sum(bool(i["tref"]) for i in es)})
    return out


def split(ds, hi, lo):
    if hi == "median":
        m = sorted(d["g"] for d in ds)[len(ds) // 2]
        return [d for d in ds if d["g"] >= m and d["g"] > 0], [d for d in ds if d["g"] < m or d["g"] == 0]
    return [d for d in ds if d["g"] >= hi - 1e-9], [d for d in ds if d["g"] <= lo + 1e-9]


def agg(ds, mm=None):
    """mm 给出时按重抽后的两组新工作项均值重算各日工作项数（效标样本层，D26-4）。"""
    n, w, c = (sum(d[f] for d in ds) for f in ("n", "w", "c"))
    k = sum(d["k"] for d in ds) if mm is None else sum(d["n_cmd"] * mm[True] + (d["n"] - d["n_cmd"]) * mm[False] for d in ds)
    return {"days": len(ds), "n": n, "c_ep": c / n, "c_100w": 100 * c / w if w else float("nan"), "c_work": c / k if k else float("nan")}


def draw_mm(ds, rng):
    """重抽效标样本，返回两组新工作项均值；无样本列表时返回 None（退回只重抽日的口径）。
    用独立于日重抽的随机流（rng.mm），使不依赖标注的各区间与只重抽日时逐位相同。"""
    sm = ds[0]["_ctx"]["samples"] if ds else None
    if not sm:
        return None
    r2 = getattr(rng, "mm", None)
    if r2 is None:
        r2 = rng.mm = random.Random(f"{SEED}|mm")
    return {f: (sum(r2.choice(sm[f]) for _ in sm[f]) / len(sm[f])) if sm[f] else ds[0]["_ctx"]["works"][f] for f in (True, False)}


def compare(ds, hi=2 / 3, lo=1 / 3, rng=None):
    H, Lo = split(ds, hi, lo)
    if not H or not Lo:
        return None
    a, b = agg(H), agg(Lo)
    r = {"hi": a, "lo": b, "ratio": {m: a[m] / b[m] if b[m] else float("nan") for m in ("c_ep", "c_100w", "c_work")}}
    bs = defaultdict(list)
    for _ in range(BOOT):
        mm = draw_mm(ds, rng)
        x, y = agg([rng.choice(H) for _ in H], mm), agg([rng.choice(Lo) for _ in Lo], mm)
        for m in ("c_ep", "c_100w", "c_work"):
            bs[m].append(x[m] / y[m] if y[m] else float("nan"))
    r["ci"] = {m: bci(v) for m, v in bs.items()}
    r["two_layer"] = draw_mm(ds, random.Random(0)) is not None
    return r


def compare_cells(ds, cells, rng, hi=2 / 3, lo=1 / 3):
    """D40：格均值外推。合并均值模型（compare / agg）把两组（片段内提交 / 未提交）的样本均值搬到每一天，隐含同一提交状态的片段在高、低日含同样多的工作；
    格均值模型改用 提交状态 × 事前登记日组 的样本均值（格为空时退回合并均值）。区间两层重抽 2,000 次：先在各格内重抽样本片段，再在高、低两组内重抽日子。
    另给只用高、低日内样本重估两组合并均值的结果（hl_only），用于区分"中间日样本"与"提交状态 × 日组交互"两种来源。"""
    H, Lo = split(ds, hi, lo)
    if not H or not Lo:
        return None
    pooled = ds[0]["_ctx"]["works"]
    mean = lambda v: sum(v) / len(v) if v else None
    reg_of = lambda d: "hi" if d in H else "lo"

    def kk(G, cm):
        return sum(d["n_cmd"] * (cm.get((True, reg_of(d))) or pooled[True]) + (d["n"] - d["n_cmd"]) * (cm.get((False, reg_of(d))) or pooled[False]) for d in G)

    def est(Hs, Ls, cm):
        kh, kl = kk(Hs, cm), kk(Ls, cm)
        ch, cl = sum(d["c"] for d in Hs), sum(d["c"] for d in Ls)
        return (ch / kh if kh else float("nan")), (cl / kl if kl else float("nan")), kh, kl
    cm0 = {(f, r): mean(cells[r][0 if f else 1]) for r in cells for f in (True, False)}
    a, b, kh, kl = est(H, Lo, cm0)
    bs = []
    for _ in range(BOOT):
        cm = {}
        for r in cells:
            for f in (True, False):
                v = cells[r][0 if f else 1]
                cm[(f, r)] = mean([rng.choice(v) for _ in v]) if v else None
        x, y, _, _ = est([rng.choice(H) for _ in H], [rng.choice(Lo) for _ in Lo], cm)
        bs.append(x / y if y else float("nan"))
    hl = {f: mean(cells["hi"][0 if f else 1] + cells["lo"][0 if f else 1]) or pooled[f] for f in (True, False)}
    a2, b2, _, _ = est(H, Lo, {(f, r): hl[f] for f in (True, False) for r in ("hi", "lo")})
    a0, b0, kh0, kl0 = est(H, Lo, {})
    return {"hi": {"days": len(H), "c": sum(d["c"] for d in H), "works": kh, "works_pooled": kh0, "c_work": a, "c_work_pooled": a0},
            "lo": {"days": len(Lo), "c": sum(d["c"] for d in Lo), "works": kl, "works_pooled": kl0, "c_work": b, "c_work_pooled": b0},
            "ratio": a / b if b else float("nan"), "ci": bci(bs), "ratio_pooled": a0 / b0 if b0 else float("nan"),
            "hl_only": {"means": [hl[True], hl[False]], "ratio": a2 / b2 if b2 else float("nan")},
            "cells": {r: {"n": [len(cells[r][0]), len(cells[r][1])], "mean": [mean(cells[r][0]), mean(cells[r][1])]} for r in cells}}


def fmt_cells(r):
    if r is None:
        return "—（某一类日子为空）"
    cn = {"hi": "高登记日", "mid": "中间日", "lo": "低登记日", "ineligible": "不合格日"}
    m = lambda v: f"{v:.2f}" if v is not None else "—"
    tab = "；".join(f"{cn[k]} {c['n'][0]} / {c['n'][1]}（{m(c['mean'][0])} / {m(c['mean'][1])}）" for k, c in r["cells"].items())
    return (f"样本 n（片段内提交 / 未提交）与新工作项均值按日组：{tab}。合并均值模型：高 / 低日工作项 {r['hi']['works_pooled']:.0f} / {r['lo']['works_pooled']:.0f}，"
            f"每项工作提交数 {r['hi']['c_work_pooled']:.2f} / {r['lo']['c_work_pooled']:.2f}，之比 {r['ratio_pooled']:.2f}；"
            f"格均值模型：工作项 {r['hi']['works']:.0f} / {r['lo']['works']:.0f}，每项工作提交数 {r['hi']['c_work']:.2f} / {r['lo']['c_work']:.2f}，"
            f"之比 {r['ratio']:.2f}（{r['ci'][0]:.2f}–{r['ci'][1]:.2f}）；只用高、低日内样本重估两组合并均值（{r['hl_only']['means'][0]:.2f} / {r['hl_only']['means'][1]:.2f}）：{r['hl_only']['ratio']:.2f}")


def groups(ds, rng):
    """表 9：高、中、低三类日子的汇总，每单位提交数的区间按组内日子重抽；日均量之比为高对低。"""
    H, Lo = split(ds, 2 / 3, 1 / 3)
    M = [d for d in ds if d not in H and d not in Lo]
    out = {}
    for name, G in (("hi", H), ("mid", M), ("lo", Lo)):
        if not G:
            continue
        a = agg(G)
        tot = {f: sum(d[f] for d in G) for f in ("n", "w", "c", "n_reg", "n_cmd", "n_tref")}
        a.update(tot, w_ep=tot["w"] / tot["n"], p_reg=tot["n_reg"] / tot["n"], p_cmd=tot["n_cmd"] / tot["n"],
                 p_tref=tot["n_tref"] / tot["n"])
        if len(G) >= 3:
            bs = defaultdict(list)
            for _ in range(BOOT):
                mm = draw_mm(ds, rng)
                x = [rng.choice(G) for _ in G]
                y = agg(x, mm)
                for m in ("c_ep", "c_100w", "c_work"):
                    bs[m].append(y[m])
                bs["w_ep"].append(sum(d["w"] for d in x) / sum(d["n"] for d in x))
            a["ci"] = {m: bci(v) for m, v in bs.items()}
        out[name] = a
    if H and Lo:
        dm = lambda G, f: sum(d[f] for d in G) / len(G)
        out["daily"] = {f: dm(H, f) / dm(Lo, f) for f in ("c", "n", "w", "k")}
        bs = defaultdict(list)
        for _ in range(BOOT):
            mm = draw_mm(ds, rng)
            x, y = [rng.choice(H) for _ in H], [rng.choice(Lo) for _ in Lo]
            kk = lambda G: sum(d["n_cmd"] * mm[True] + (d["n"] - d["n_cmd"]) * mm[False] for d in G) / len(G) if mm else dm(G, "k")
            for f in ("c", "n", "w", "k"):
                bs[f].append((kk(x) / kk(y) if f == "k" else dm(x, f) / dm(y, f)) if dm(y, f) else float("nan"))
        out["daily_ci"] = {f: bci(v) for f, v in bs.items()}
    return out


def fmt(r):
    if r is None:
        return "—（某一类日子为空）"
    f = lambda m: f"{r['ratio'][m]:.2f}（{r['ci'][m][0]:.2f}–{r['ci'][m][1]:.2f}）"
    return (f"高 {r['hi']['days']} 天 / 低 {r['lo']['days']} 天；每片段提交数之比 {f('c_ep')}，每百次代码编辑 {f('c_100w')}，"
            f"每项工作 {f('c_work')}")


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** .5 if va and vb else float("nan")


def ties(ds):
    out = {}
    for ref in ("n", "w", "k"):
        opp = tot = tie = 0
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = ds[i], ds[j]
                dc, dr = (a["c"] > b["c"]) - (a["c"] < b["c"]), (a[ref] > b[ref]) - (a[ref] < b[ref])
                if not dc or not dr:
                    tie += 1
                    continue
                opp += dc != dr
                tot += 1
        out[ref] = {"opp": opp, "tot": tot, "tie": tie}
    return out


def main():
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    bl = num["robustness6"]["boundary_labels"]
    R, L = {}, ["# 第七轮审稿补充分析：去循环分期、按片段归属提交、删失与敏感性（S18-T1，D18）", "",
                "由 `analysis/robustness7.py` 生成。每项工作的提交数以边界效标两组的每片段新工作项均值为分母；区间按两层重抽 2,000 次：先重抽效标样本片段重估两组均值，再按日重抽（D26-4，同时传播效标抽样与日间抽样的不确定性）。", ""]
    CTX = {}
    for s in ("SYS_A", "SYS_C"):
        rates = bl[f"{s}|final|rates"]
        nw = rates["new_works"]
        sm = rates.get("new_works_samples")
        CTX[s] = load_ctx(s, {True: nw[0], False: nw[1]}, {True: sm[0], False: sm[1]} if sm else None)
    for s in ("SYS_A", "SYS_C"):
        ctx, rng = CTX[s], random.Random(f"{SEED}|{s}")
        L += [f"## {NAME[s]}", ""]
        rows = [("事前登记比例分期（主口径）", dict(by="reg")),
                ("请求引用任务编号比例分期", dict(by="tref")),
                ("片段内提交比例分期（原口径，机制描述）", dict(by="cmd")),
                ("事前登记分期 · 按片段归属提交", dict(by="reg", attr="cohort")),
                ("事前登记分期 · 按片段归属提交 · 删去窗口末 48 小时", dict(by="reg", attr="cohort", censor=True)),
                ("片段内提交分期 · 按片段归属提交 · 删去窗口末 48 小时", dict(by="cmd", attr="cohort", censor=True)),
                ("事前登记分期 · 合格日门槛 2", dict(by="reg", min_eps=2)),
                ("事前登记分期 · 合格日门槛 5", dict(by="reg", min_eps=5)),
                ("仅用版本库：当日提交中带任务编号的比例分期（无提交的日子不计）", dict(by="git_task"))]
        if s == "SYS_C":
            rows += [("事前登记分期 · 只用规则进入版本库之后", dict(by="reg", after=RULE_C)),
                     ("片段内提交分期 · 只用规则进入版本库之后", dict(by="cmd", after=RULE_C))]
        for name, kw in rows:
            ds = days(ctx, **kw)
            r = compare(ds, rng=rng)
            R[f"{s}|{name}"] = r
            L.append(f"- {name}（{len(ds)} 个合格日）：{fmt(r)}。")
            if kw.get("by") == "reg" and len(kw) == 1:
                rm = compare(ds, "median", None, rng)
                R[f"{s}|{name}|median"] = rm
                L.append(f"  - 中位分割：{fmt(rm)}。")
                # D40：格均值外推（独立随机流，不影响上文任何区间）
                cells = bl[f"{s}|final|rates"].get("new_works_cells")
                if cells:
                    rc = compare_cells(ds, cells, random.Random(f"{SEED}|{s}|cells"))
                    R[f"{s}|{name}|cells"] = rc
                    L.append(f"  - 每项工作提交数的折算模型（D40）：{fmt_cells(rc)}。")
        ds = days(ctx, by="reg")
        R[f"{s}|reg_days"] = [(d["d"], round(d["g"], 3)) for d in ds]
        for by in ("reg", "cmd"):
            G = groups(days(ctx, by=by), rng)
            R[f"{s}|groups|{by}"] = G
            for k in ("hi", "mid", "lo"):
                if k in G:
                    a = G[k]
                    ci = lambda m: f"（{a['ci'][m][0]:.2f}–{a['ci'][m][1]:.2f}）" if "ci" in a else ""
                    L.append(f"- 表 9 · {by} · {k}：{a['days']} 天，片段 {a['n']}，代码编辑 {a['w']}，提交 {a['c']:.0f}；"
                             f"每片段提交 {a['c_ep']:.2f}{ci('c_ep')}，每百次编辑 {a['c_100w']:.1f}{ci('c_100w')}，"
                             f"每项工作 {a['c_work']:.2f}{ci('c_work')}，每片段编辑 {a['w_ep']:.0f}{ci('w_ep')}；"
                             f"事前登记 {a['p_reg']:.0%}，片段内提交 {a['p_cmd']:.0%}，引用任务编号 {a['p_tref']:.0%}。")
            if "daily" in G:
                dd, dc = G["daily"], G["daily_ci"]
                L.append(f"- 表 9 · {by} · 日均量之比（高对低）：" + "，".join(
                    f"{lab} {dd[f]:.2f}（{dc[f][0]:.2f}–{dc[f][1]:.2f}）"
                    for f, lab in (("c", "提交"), ("n", "片段"), ("w", "代码编辑"), ("k", "工作项"))) + "。")
        dg = days(ctx, by="git_task")
        rho = {k: spearman([d["git_task"] for d in dg], [d[k] for d in dg]) for k in ("reg", "cmd")}
        R[f"{s}|git_task_rho"] = rho
        L.append(f"- 仅用版本库的代理（当日提交中带任务编号的比例）与会话层指标的 Spearman 相关（{len(dg)} 天）："
                 f"与事前登记比例 {rho['reg']:.2f}，与片段内提交比例 {rho['cmd']:.2f}。")
        t = ties(days(ctx, by="cmd"))
        R[f"{s}|ties"] = t
        L.append(f"- 日对（片段内提交分期所用的合格日）：提交数与片段数排序相反 {t['n']['opp']}/{t['n']['tot']}（并列排除 {t['n']['tie']} 对），"
                 f"与代码编辑数 {t['w']['opp']}/{t['w']['tot']}（并列 {t['w']['tie']}），与工作项数 {t['k']['opp']}/{t['k']['tot']}（并列 {t['k']['tie']}）。")
        L.append(f"- 不含片段内容的提交 {ctx['n_nocontent']}/{ctx['n_commits']}，按片段归属时不计入。")
        L.append("")
    L += ["## 系统比较", ""]
    act = {s: sorted({i["start"][:10] for i in CTX[s]["items"]}) for s in CTX}
    per = lambda s, f, dd=None: sum(f(i) for i in CTX[s]["items"] if dd is None or i["start"][:10] >= dd) / len([d for d in act[s] if dd is None or d >= dd])
    allc = lambda s, dd=None: sum(v for d, v in CTX[s]["cday"].items() if dd is None or d >= dd) / len([d for d in act[s] if dd is None or d >= dd])
    sysr = {"commits_time": allc("SYS_A") / allc("SYS_C"),
            "commits_cohort": per("SYS_A", lambda i: i["share"]) / per("SYS_C", lambda i: i["share"]),
            "episodes": per("SYS_A", lambda i: 1) / per("SYS_C", lambda i: 1),
            "code_edits": per("SYS_A", lambda i: i["n"]) / per("SYS_C", lambda i: i["n"]),
            "works": per("SYS_A", lambda i: i["w_hat"]) / per("SYS_C", lambda i: i["w_hat"]),
            "C_after_rule_commits_time": allc("SYS_A") / allc("SYS_C", RULE_C),
            "C_after_rule_works": per("SYS_A", lambda i: i["w_hat"]) / per("SYS_C", lambda i: i["w_hat"], RULE_C),
            "active_days": [len(act["SYS_A"]), len(act["SYS_C"]), len([d for d in act["SYS_C"] if d >= RULE_C])]}
    R["sys"] = sysr
    L.append(f"- 分母为有代码修改片段的日子（系统 A {sysr['active_days'][0]} 天、系统 C {sysr['active_days'][1]} 天，其中规则进入版本库之后 {sysr['active_days'][2]} 天）。")
    L.append(f"- 系统 A 对系统 C 的日均量之比：提交（按提交时间）{sysr['commits_time']:.2f}，提交（按片段归属）{sysr['commits_cohort']:.2f}，"
             f"代码修改片段 {sysr['episodes']:.2f}，代码编辑 {sysr['code_edits']:.2f}，工作项 {sysr['works']:.2f}。")
    L.append(f"- 系统 C 只用规则进入版本库之后：提交 {sysr['C_after_rule_commits_time']:.2f}，工作项 {sysr['C_after_rule_works']:.2f}。")
    L.append("")
    # D32-3：一致子样本（剔除全部裁决片段）的两组新工作项均值代入，重算每项工作提交数之比；独立随机流，不影响上文任何区间
    if any(f"{s}|agree|rates" in bl for s in ("SYS_A", "SYS_C")):
        L += ["## 一致子样本：两组每片段新工作项均值只用两名编码人完全一致的样本片段（D32-3）", ""]
        for s in ("SYS_A", "SYS_C"):
            ra = bl.get(f"{s}|agree|rates")
            if not ra:
                continue
            nw, sm = ra["new_works"], ra["new_works_samples"]
            ctx2 = load_ctx(s, {True: nw[0], False: nw[1]}, {True: sm[0], False: sm[1]})
            for by, bn in (("reg", "事前登记比例分期（主口径）"), ("cmd", "片段内提交比例分期（机制描述）")):
                r = compare(days(ctx2, by=by), rng=random.Random(f"{SEED}|{s}|agree|{by}"))
                R[f"{s}|agree|{by}"] = r
                L.append(f"- {NAME[s]} · {bn}（样本 {ra['n_samples'][0]} + {ra['n_samples'][1]}，均值 {nw[0]:.2f} / {nw[1]:.2f}）：{fmt(r)}。")
        L.append("")
    # D35-6（B3）两端法：只用某一名标注人的原始标注；独立随机流
    if any(f"{s}|devonly|rates" in bl for s in ("SYS_A", "SYS_C")):
        L += ["## 两端法：两组每片段新工作项均值只用某一名标注人的原始标注（D35-6）", ""]
        for tag, tn in (("devonly", "只用开发者"), ("coderonly", "只用编码人甲")):
            for s in ("SYS_A", "SYS_C"):
                ra = bl.get(f"{s}|{tag}|rates")
                if not ra:
                    continue
                nw, sm = ra["new_works"], ra["new_works_samples"]
                ctx2 = load_ctx(s, {True: nw[0], False: nw[1]}, {True: sm[0], False: sm[1]})
                for by, bn in (("reg", "事前登记比例分期（主口径）"), ("cmd", "片段内提交比例分期（机制描述）")):
                    r = compare(days(ctx2, by=by), rng=random.Random(f"{SEED}|{s}|{tag}|{by}"))
                    R[f"{s}|{tag}|{by}"] = r
                    L.append(f"- {NAME[s]} · {tn} · {bn}（均值 {nw[0]:.2f} / {nw[1]:.2f}）：{fmt(r)}。")
        L.append("")
    (RES / "r15_robustness7.md").write_text("\n".join(L), encoding="utf-8")
    num["robustness7"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
