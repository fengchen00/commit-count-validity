"""v2.12 补充分析（S14-T1，口径见 notes/decisions.md D14）。

1 按天：非同源因变量、代码编辑数作工作量、增量 R² 与留一日交叉验证  2 反事实演示  3 两批合并的人工加权估计与确认率区间
4 跨会话延续  5 返工重写  6 不含片段内容的提交  7 可核验性 × 片段内提交
写入 <PAPER_DIR>/results/r11_robustness3.md 与 numbers.json 的 robustness3 键（须在 render_tables、holdout_agreement 之后运行）。
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import command_commits  # noqa: E402
from extract_episodes import path_class  # noqa: E402
from holdout_agreement import holdout_reference  # noqa: E402
from robustness import MERGE, NAME, base, has, load, pc, q, share  # noqa: E402
from sim_agreement import final_reference  # noqa: E402
from workitems import wilson  # noqa: E402

BOOT, SEED = 2000, "S14"
W_DRAWS = 4000
NUMS = (RESULTS_DIR / "numbers.json")
RES = RESULTS_DIR
SYSTEMS = ["SYS_A", "SYS_C"]


# 1 按天 ----------------------------------------------------------------------------------------
def r2(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def loo_r2(y, X):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        pred[i] = X[i] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def daily(s, x, items):
    day = lambda t: t[:10]
    gold = command_commits(x["d"], x["cm"])
    cw, gw = Counter(), Counter()
    for h, c in x["cm"].items():
        cw[day(c["time"])] += 1
        gw[day(c["time"])] += h in gold
    ew, nw, mw = Counter(), Counter(), Counter()
    for i in items:
        ew[day(i["start"])] += 1
        nw[day(i["start"])] += i["n"]
        mw[day(i["start"])] += i["cmd"]
    days = sorted(d for d in ew if ew[d] >= 3)
    c = np.array([cw[d] for d in days], float)
    g = np.array([gw[d] for d in days], float)
    o = c - g
    e = np.array([ew[d] for d in days], float)
    n = np.array([nw[d] for d in days], float)
    m = np.array([mw[d] / ew[d] for d in days])
    one = np.ones(len(days))
    X = {"片段数": np.column_stack([one, np.log1p(e)]), "片段数 + 片段内提交比例": np.column_stack([one, np.log1p(e), m]),
         "代码编辑数": np.column_stack([one, np.log1p(n)]), "代码编辑数 + 片段内提交比例": np.column_stack([one, np.log1p(n), m]),
         "片段数 + 代码编辑数": np.column_stack([one, np.log1p(e), np.log1p(n)]),
         "片段数 + 代码编辑数 + 片段内提交比例": np.column_stack([one, np.log1p(e), np.log1p(n), m])}
    ys = {"全部提交": np.log1p(c), "非片段内产生的提交": np.log1p(o)}
    out = {"days": len(days), "models": {}}
    for yn, y in ys.items():
        for xn, Xm in X.items():
            out["models"][f"{yn}|{xn}"] = {"r2": round(float(r2(y, Xm)), 2), "loo": round(float(loo_r2(y, Xm)), 2)}
    rng = random.Random(f"{SEED}|daily|{s}")
    inc = {yn: [] for yn in ys}
    for _ in range(BOOT):
        idx = [rng.randrange(len(days)) for _ in days]
        for yn, y in ys.items():
            yb = y[idx]
            if yb.std() == 0:
                continue
            inc[yn].append(r2(yb, X["代码编辑数 + 片段内提交比例"][idx]) - r2(yb, X["代码编辑数"][idx]))
    out["inc_ci"] = {yn: [round(q(v, .025), 2), round(q(v, .975), 2)] for yn, v in inc.items()}
    out["inc"] = {yn: round(float(r2(y, X["代码编辑数 + 片段内提交比例"]) - r2(y, X["代码编辑数"])), 2) for yn, y in ys.items()}
    vc, vg, vo = c.var(), g.var(), o.var()
    out["var"] = {"gold": round(float(vg / vc), 2), "other": round(float(vo / vc), 2), "cov2": round(float(2 * np.cov(g, o, bias=True)[0, 1] / vc), 2)}
    out["gold_share"] = round(float(g.sum() / c.sum()), 2)
    return out


# 2 反事实演示 -----------------------------------------------------------------------------------
def commit_share(x, items):
    """每个片段分得的提交数：Σ_{c ∈ P} 1 / 该提交所含片段数（只计内容证据）。"""
    eps_of = defaultdict(set)
    for e, cs in x["content"].items():
        for c in cs:
            eps_of[c].add(e)
    return {i["e"]: sum(1 / len(eps_of[c]) for c in x["content"].get(i["e"], ())) for i in items}


def counterfactual(s, x, items, p_from, p_to):
    sh = commit_share(x, items)

    def ratio(its):
        g1, g0 = [sh[i["e"]] for i in its if i["cmd"]], [sh[i["e"]] for i in its if not i["cmd"]]
        if not g1 or not g0:
            return None, None, None
        k1, k0 = sum(g1) / len(g1), sum(g0) / len(g0)
        return k1, k0, (p_to * k1 + (1 - p_to) * k0) / (p_from * k1 + (1 - p_from) * k0)
    k1, k0, r = ratio(items)
    by = defaultdict(list)
    for i in items:
        by[i["s"]].append(i)
    ks = sorted(by)
    rng = random.Random(f"{SEED}|cf|{s}")
    rs = []
    for _ in range(BOOT):
        v = ratio([i for k in rng.choices(ks, k=len(ks)) for i in by[k]])[2]
        if v is not None:
            rs.append(v)
    return {"k1": round(k1, 2), "k0": round(k0, 2), "ratio": round(r, 2), "ci": [round(q(rs, .025), 2), round(q(rs, .975), 2)],
            "p_from": p_from, "p_to": p_to}


# 3 两批合并的人工加权估计 -------------------------------------------------------------------------
def read_pop(p):
    pop = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\S+) \| (\d+)（(\d+) / (\d+)）", line)
        if m:
            pop[m.group(1)] = {"SYS_A": int(m.group(3)), "SYS_C": int(m.group(4))}
    return pop


def batch_draws(ref, strata, pop, syss, codes, tag, pred=None):
    """同 render_tables.weighted（D9）：分系统按层反加权；该系统某层无样本时借用两系统合并的该层样本。
    pred(参照行) 给定时代替"主或次模式命中 codes"的判定（D16-1）。"""
    def hit(e):
        if pred:
            return pred(ref[e])
        return ref[e]["main"] in codes or bool(set(codes) & set(ref[e]["sec"]))
    es_all = list(ref)
    k = [sum(ref[e]["main"] != "N0" and hit(e) for e in es_all), sum(ref[e]["main"] != "N0" and not hit(e) for e in es_all),
         sum(ref[e]["main"] == "N0" for e in es_all)]
    pr = [max(v / len(es_all), 1e-3) for v in k]
    cnt = []
    for h, ps in pop.items():
        N = sum(ps[s] for s in syss)
        if not N:
            continue
        es = [e for e, st in strata.items() if st == h and e.split("-")[0] in syss] or [e for e, st in strata.items() if st == h]
        if es:
            cnt.append((N, sum(ref[e]["main"] != "N0" and hit(e) for e in es), sum(ref[e]["main"] != "N0" and not hit(e) for e in es),
                        sum(ref[e]["main"] == "N0" for e in es), len(es)))
    point = sum(N * a / n for N, a, _, _, n in cnt) / sum(N * (a + b) / n for N, a, b, _, n in cnt)
    rng = random.Random(f"{SEED}|w|{tag}|{'+'.join(syss)}|{'+'.join(codes)}")
    draws = []
    for _ in range(W_DRAWS):
        nu = de = 0.0
        for N, a, b, z, _ in cnt:
            gs = [rng.gammavariate(v + p, 1) for v, p in zip((a, b, z), pr)]
            t = sum(gs)
            nu += N * gs[0] / t
            de += N * (gs[0] + gs[1]) / t
        draws.append(nu / de)
    return point, draws


W_ROWS = [("一对一", ("A1",)), ("合并（同会话或跨会话）", ("A2", "A3")), ("　同会话合并", ("A2",)), ("　跨会话合并", ("A3",)),
          ("拆分（逐条或非逐条）", ("A4", "A5")), ("　逐条任务拆分", ("A4",)), ("　非逐条拆分", ("A5",)), ("附带提交", ("A6",)),
          ("部分未提交", ("A8",)), ("缺失", ("A9",)), ("含未跟踪文件", ("A10",))]
AUTO_OF = {"A1": "一对一", "A2": "合并-同会话", "A3": "合并-跨会话", "A4": "拆分-逐条任务", "A5": "拆分-非逐条", "A6": "随他项提交",
           "A8": "部分未提交", "A9": "缺失", "A10": "含未跟踪文件"}


def pooled():
    orig = final_reference("sim2")
    hold = holdout_reference()
    if not orig or not hold:
        return None
    s1 = {r["episode"]: r["stratum"] for r in load("data/coding/sample_S03.csv")}
    s2 = {r["episode"]: r["stratum"] for r in load("data/coding/holdout/sample_S12.csv")}
    p1, p2 = read_pop(RES / "s03_sampling.md"), read_pop(RES / "s12_holdout_sampling.md")
    auto = {}
    for s in SYSTEMS:
        for r in load(Path("data/derived") / s / "episode_patterns.csv"):
            if r["evidence"]:
                auto.setdefault(s, []).append(set(r["patterns"].split(";")))
    out = {}
    for s in SYSTEMS:
        n1 = sum(e.startswith(s) for e in s1)
        n2 = sum(e.startswith(s) for e in s2)
        for name, codes in W_ROWS:
            a, da = batch_draws(orig, s1, p1, [s], codes, "orig")
            b, db = batch_draws(hold, s2, p2, [s], codes, "hold")
            w = n1 / (n1 + n2)
            mix = sorted(w * x + (1 - w) * y for x, y in zip(da, db))
            ci = lambda d: [round(d[int(.025 * W_DRAWS)], 3), round(d[int(.975 * W_DRAWS) - 1], 3)]
            au = share(auto[s], lambda l: any(AUTO_OF[c] in l for c in codes))
            out[f"{s}|{name}"] = {"orig": round(a, 3), "orig_ci": ci(sorted(da)), "hold": round(b, 3), "hold_ci": ci(sorted(db)),
                                  "pooled": round(w * a + (1 - w) * b, 3), "pooled_ci": ci(mix), "pooled_med": round(mix[W_DRAWS // 2], 3),
                                  "auto": round(au, 3), "n1": n1, "n2": n2}
    conf = {}
    ref = {**orig, **hold}
    autolab = {}
    for s in SYSTEMS:
        for r in load(Path("data/derived") / s / "episode_patterns.csv"):
            autolab[r["episode"]] = set(r["patterns"].split(";"))
    for c, an in AUTO_OF.items():
        aa = [e for e in ref if an in autolab[e]]
        k = sum(ref[e]["main"] == c or c in ref[e]["sec"] for e in aa)
        conf[c] = {"k": k, "n": len(aa), "wilson": [round(v, 3) for v in wilson(k, len(aa))] if aa else None}
    return out, conf


# 4 跨会话延续 -----------------------------------------------------------------------------------
def continuation(s, x, items):
    eps = {e["episode"]: e for e in x["eps"]}
    sess_start = {}
    for e in x["eps"]:
        sess_start[e["session"]] = min(sess_start.get(e["session"], e["start"]), e["start"])
    cont_sess = {e["session"] for e in x["eps"] if e["first_in_session"] == "1" and e["cross_session_continuation"] == "1"}
    eps_of = defaultdict(set)
    for e, cs in x["content"].items():
        for c in cs:
            eps_of[c].add(e)
    res = {}
    for grp, tag in (([i for i in items if i["ver"]], "ver"), (items, "all")):
        cross = [i for i in grp if "合并-跨会话" in i["labels"]]
        kept = 0
        for i in cross:
            partners = {o for c in x["content"].get(i["e"], ()) for o in eps_of[c]} - {i["e"]}
            other = [o for o in partners if eps[o]["session"] != i["s"]]
            linked = lambda o: (max((i["s"], eps[o]["session"]), key=lambda z: sess_start[z]) in cont_sess)
            kept += any(not linked(o) for o in other)
        res[tag] = {"n": len(grp), "cross": len(cross), "cross_after": kept, "cont_sessions": len(cont_sess)}
    return res


# 5 返工重写 -------------------------------------------------------------------------------------
def rewrite(s, x, items):
    d = x["d"]
    ep_of_edit = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_edits.csv") if r["episode"] and not r["reason"]}
    sig = {(o["session"], str(o["seq"])): set(o["lines"]) for o in map(json.loads, open(d / "edit_signatures.jsonl", encoding="utf-8"))}
    L = defaultdict(set)
    for a in load(d / "edit_attribution.csv"):
        ep = ep_of_edit.get((a["session"], a["seq"]))
        if ep and a["state"] == "committed" and path_class(a["path"].replace("\\", "/"), "1") == "code":
            L[(ep, a["commit"])] |= sig.get((a["session"], a["seq"]), set())
    src = {(p["episode"], p["commit"]): p["source"] for p in load(d / "episode_commit.csv") if p["source"] != "pathtime"}
    start = {e["episode"]: e["start"] for e in x["eps"]}
    eps_of = defaultdict(set)
    for e, cs in x["content"].items():
        for c in cs:
            eps_of[c].add(e)

    def overlap(e, o, c):
        """较晚片段在该提交中的特征行里，较早片段已写过的比例；较晚片段只有检查点证据时为 None。"""
        a, b = sorted((e, o), key=lambda z: start[z])
        if src.get((b, c)) not in ("edit", "both") or not L[(b, c)]:
            return None
        return len(L[(b, c)] & L[(a, c)]) / len(L[(b, c)])
    pairs = Counter()
    for c, es in eps_of.items():
        es = sorted(es, key=lambda z: start[z])
        for k, a in enumerate(es):
            for b in es[k + 1:]:
                v = overlap(a, b, c)
                pairs["unchecked" if v is None else ("full" if v == 1 else ("half" if v >= 0.5 else ("some" if v > 0 else "none")))] += 1
    res = {"pairs": dict(pairs)}
    for grp, tag in (([i for i in items if i["ver"]], "ver"), (items, "all")):
        merged = [i for i in grp if has(i["labels"], MERGE)]
        res[tag] = {"n": len(grp), "merged": len(merged)}
        for cut, nm in ((1.0, "after_full"), (0.5, "after_half")):
            kept = 0
            for i in merged:
                kept += any((v := overlap(i["e"], o, c)) is None or v < cut
                            for c in x["content"].get(i["e"], ()) for o in eps_of[c] - {i["e"]})
            res[tag][nm] = kept
    return res


# 6 不含片段内容的提交 -----------------------------------------------------------------------------
def empty_commits(s, x):
    d = x["d"]
    with_content = {c for cs in x["content"].values() for c in cs}
    edited = defaultdict(list)
    ep_start = {e["episode"]: e["start"] for e in x["eps"]}
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            edited[f["path"].replace("\\", "/").lower()].append(ep_start[f["episode"]])
    files = defaultdict(set)
    for f in load(d / "commit_files.csv"):
        files[f["hash"]].add(f["path"].replace("\\", "/").lower())
    gold = command_commits(d, x["cm"])
    empty = [h for h in x["cm"] if h not in with_content]
    touch = [h for h in empty if any(any(t < x["cm"][h]["time"] for t in edited.get(p, ())) for p in files[h])]
    code_any = [h for h in empty if any(path_class(p, "1") == "code" for p in files[h])]
    return {"commits": len(x["cm"]), "empty": len(empty), "empty_code": len(code_any), "empty_touch": len(touch),
            "empty_gold": sum(h in gold for h in empty), "touch_gold": sum(h in gold for h in touch)}


# 7 可核验性 × 片段内提交 --------------------------------------------------------------------------
def verif(items):
    out = {}
    for v in (True, False):
        for cmd in (True, False):
            g = [i for i in items if i["ver"] == v and i["cmd"] == cmd]
            out[f"{'ver' if v else 'nonver'}|{'cmd' if cmd else 'nocmd'}"] = {
                "n": len(g), "one": round(share(g, lambda i: "一对一" in i["labels"]), 3), "merge": round(share(g, lambda i: has(i["labels"], MERGE)), 3)}
    return out


def main():
    R = {}
    L = ["# v2.12 补充分析（S14-T1）", "", "由 `analysis/robustness3.py` 生成，口径见 `notes/decisions.md` D14。", ""]
    xs = {s: base(s) for s in SYSTEMS}
    cmd_share = {s: share(xs[s][1], lambda i: i["cmd"]) for s in SYSTEMS}

    L += ["## 1 按天：非同源因变量与代码编辑数（D14-2）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        D = daily(s, x, items)
        R[f"daily_{s}"] = D
        L += [f"### {NAME[s]}（{D['days']} 天）", "", "| 因变量 | 自变量 | R² | 留一日交叉验证 R² |", "|---|---|---|---|"]
        for k, v in D["models"].items():
            yn, xn = k.split("|")
            L.append(f"| log(1 + {yn}数) | {xn} | {v['r2']:.2f} | {v['loo']:.2f} |")
        L += ["", f"在代码编辑数之上加入片段内提交比例的增量 R²：全部提交 {D['inc']['全部提交']:.2f}（按日重抽 95% 区间 "
              f"{D['inc_ci']['全部提交'][0]:.2f}–{D['inc_ci']['全部提交'][1]:.2f}），非片段内产生的提交 {D['inc']['非片段内产生的提交']:.2f}"
              f"（{D['inc_ci']['非片段内产生的提交'][0]:.2f}–{D['inc_ci']['非片段内产生的提交'][1]:.2f}）。",
              f"这些日子的提交中可对上片段提交命令的占 {D['gold_share']:.0%}；每日提交数的方差分解：片段内产生 {D['var']['gold']:.2f}，"
              f"其余 {D['var']['other']:.2f}，协方差项 {D['var']['cov2']:.2f}。", ""]

    L += ["## 2 反事实演示（D14-3）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        C = counterfactual(s, x, items, round(cmd_share["SYS_C"], 2), round(cmd_share["SYS_A"], 2))
        R[f"cf_{s}"] = C
        L.append(f"- {NAME[s]}：片段内提交的片段平均分得 {C['k1']:.2f} 个提交，其余片段 {C['k0']:.2f} 个；"
                 f"片段数不变、片段内提交比例由 {C['p_from']:.0%} 变为 {C['p_to']:.0%} 时，含片段内容的提交数为原来的 {C['ratio']:.2f} 倍"
                 f"（按会话重抽 {C['ci'][0]:.2f}–{C['ci'][1]:.2f}）。")
    L += ["", "分得提交数 = Σ_{c ∈ P} 1 / 该提交所含片段数（只计内容证据）；假设两类片段的分得提交数不随比例变化。", ""]

    P = pooled()
    if P:
        W, conf = P
        R["pooled"], R["confirm"] = W, conf
        L += ["## 3 两批合并的人工加权估计（D14-4）", "",
              "原样本（审定参照）与留出样本（留出参照）各自按层反加权推回该系统的改代码片段，按样本量合并后验抽样；可多标签。"
              "自动一栏为全部代码修改片段中自动规则标出的比例。", "",
              "| 模式 | " + " | ".join(f"{NAME[s]}：自动 | 原样本 | 留出 | 合并" for s in SYSTEMS) + " |", "|---|" + "---|" * 8]
        for name, _ in W_ROWS:
            cells = []
            for s in SYSTEMS:
                v = W[f"{s}|{name}"]
                cells += [pc(v["auto"]), pc(v["orig"]), pc(v["hold"]), f"{pc(v['pooled'])}（{pc(v['pooled_ci'][0])}–{pc(v['pooled_ci'][1])}）"]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", "样本量：" + "；".join(f"{NAME[s]} 原样本 {W[f'{s}|一对一']['n1']}、留出 {W[f'{s}|一对一']['n2']}" for s in SYSTEMS) + "。", "",
              "自动规则确认率（两批合并；Wilson 95% 区间）：" + "；".join(
                  f"{AUTO_OF[c]} {v['k']}/{v['n']}（{pc(v['wilson'][0])}–{pc(v['wilson'][1])}）" for c, v in conf.items() if v["n"]) + "。", ""]

    L += ["## 4 跨会话延续（D14-5a）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        C = continuation(s, x, items)
        R[f"cont_{s}"] = C
        L.append(f"- {NAME[s]}：以延续语开头的会话 {C['all']['cont_sessions']} 个。跨会话合并：可核验 {C['ver']['cross']}/{C['ver']['n']} → 排除延续关系后 "
                 f"{C['ver']['cross_after']}/{C['ver']['n']}；全部代码修改 {C['all']['cross']}/{C['all']['n']} → {C['all']['cross_after']}/{C['all']['n']}。")
    L.append("")

    L += ["## 5 返工重写（D14-5b）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        C = rewrite(s, x, items)
        R[f"rewrite_{s}"] = C
        pr = C["pairs"]
        L.append(f"- {NAME[s]}：共享提交的片段对 {sum(pr.values())} 对；较晚片段在该提交中的特征行与较早片段的重合：全部重合 {pr.get('full', 0)}，"
                 f"≥ 50% {pr.get('half', 0)}，部分 {pr.get('some', 0)}，无 {pr.get('none', 0)}，较晚片段只有检查点证据、无法检验 {pr.get('unchecked', 0)}。"
                 f"排除全部重合 / ≥ 50% 重合的关系后，合并：可核验 {C['ver']['merged']} → {C['ver']['after_full']} / {C['ver']['after_half']}（n = {C['ver']['n']}）；"
                 f"全部代码修改 {C['all']['merged']} → {C['all']['after_full']} / {C['all']['after_half']}（n = {C['all']['n']}）。")
    L.append("")

    L += ["## 6 不含片段内容的提交（D14-5c）", ""]
    for s in SYSTEMS:
        x, _, _ = xs[s]
        C = empty_commits(s, x)
        R[f"empty_{s}"] = C
        L.append(f"- {NAME[s]}：{C['commits']} 个提交中 {C['empty']} 个不含片段内容；其中改动代码文件的 {C['empty_code']} 个，"
                 f"改动过此前有片段编辑的代码文件的 {C['empty_touch']} 个（可对上片段提交命令的 {C['touch_gold']} 个）。")
    L.append("")

    L += ["## 7 可核验性 × 片段内提交（D14-5d）", "", "| 系统 | 证据 | 组 | n | 一对一 | 合并 |", "|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        V = verif(xs[s][1])
        R[f"verif_{s}"] = V
        for k, v in V.items():
            ev, g = k.split("|")
            L.append(f"| {NAME[s]} | {'可核验' if ev == 'ver' else '不可核验'} | {'片段内提交' if g == 'cmd' else '片段内未提交'} | {v['n']} | "
                     f"{pc(v['one'])} | {pc(v['merge'])} |")
    L.append("")

    (RES / "r11_robustness3.md").write_text("\n".join(L), encoding="utf-8")
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    num["robustness3"] = R
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
