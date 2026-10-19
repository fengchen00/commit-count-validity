"""v3.0 度量效度分析（S17-T1，口径见 notes/decisions.md D17）。

1 提交规则  2 工作方式分期  3 下游后果  4 边界临界点
写入 <PAPER_DIR>/results/r14_robustness6.md 与 numbers.json 的 robustness6 键（须在 robustness5 之后运行）。
`build_units` 供 boundary_labels.py 按标注边界重算。
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_episodes import path_class  # noqa: E402
from extract_git import SYSTEMS as REPOS, git  # noqa: E402
from patterns import classify, prepare, unit_state  # noqa: E402
from robustness import MERGE, NAME, base, has, load, pc, q  # noqa: E402

BOOT, REPS, SEED = 2000, 200, "S17"
NUMS = (RESULTS_DIR / "numbers.json")
RES = RESULTS_DIR
SYSTEMS = ["SYS_A", "SYS_C"]
RULE = ".cursor/rules/workflow.mdc"
LEVELS = [0, .1, .2, .3, .4, .5, .6, .8, 1]
GRID = [0, .2, .4, .6, .8]
MULTI = [1, 2, 3, 4, 6, 8]
HI, LO = 2 / 3, 1 / 3


def bci(vals):
    return [q(vals, .025), q(vals, .975)]


def norm(p):
    return p.replace("\\", "/").lower()


# 1 提交规则 ---------------------------------------------------------------------------------------
def rule_history(s):
    repo = REPOS[s]
    log = [ln.split("|") for ln in git(repo, "log", "--format=%ad|%h", "--date=short", "--", RULE).splitlines() if ln.strip()]
    if not log:
        return None
    first, head = log[-1], log[0]
    has_rule = lambda h: "一条任务" in git(repo, "show", f"{h}:{RULE}")
    return {"first_date": first[0], "n_changes": len(log) - 1, "rule_first": has_rule(first[1]), "rule_head": has_rule(head[1]),
            "change_dates": [d for d, _ in log[:-1]]}


# 2 工作方式分期 ------------------------------------------------------------------------------------
def days_of(s, x, items):
    tref = {e["episode"]: int(e["f_task_ref"] or 0) for e in x["eps"]}
    by = defaultdict(list)
    for i in items:
        by[i["start"][:10]].append(i)
    cd = defaultdict(int)
    for c in x["cm"].values():
        cd[c["time"][:10]] += 1
    out = []
    for d in sorted(by):
        g = by[d]
        if len(g) < 3:
            continue
        out.append({"d": d, "n": len(g), "cmd": sum(i["cmd"] for i in g) / len(g), "w": sum(i["n"] for i in g), "c": cd.get(d, 0),
                    "reg": sum(bool(i["reg"]) for i in g), "tref": sum(tref[i["e"]] for i in g)})
    active = sorted({i["start"][:10] for i in items})
    return out, active, sum(cd.values())


def regime(v):
    return "hi" if v >= HI - 1e-9 else ("lo" if v <= LO + 1e-9 else "mid")


def regimes(s, days):
    rng = random.Random(f"{SEED}|reg|{s}")
    out = {}

    def agg(ds):
        n, w, c = sum(d["n"] for d in ds), sum(d["w"] for d in ds), sum(d["c"] for d in ds)
        return {"days": len(ds), "n": n, "w": w, "c": c, "c_per_ep": c / n, "c_per_100w": 100 * c / w if w else float("nan"),
                "w_per_ep": w / n, "reg": sum(d["reg"] for d in ds) / n, "tref": sum(d["tref"] for d in ds) / n,
                "day_c": c / len(ds), "day_n": n / len(ds), "day_w": w / len(ds)}
    grp = {k: [d for d in days if regime(d["cmd"]) == k] for k in ("hi", "mid", "lo")}
    for k, ds in grp.items():
        if not ds:
            continue
        out[k] = agg(ds)
        bs = [agg([rng.choice(ds) for _ in ds]) for _ in range(BOOT)]
        out[k]["ci"] = {m: bci([b[m] for b in bs]) for m in ("c_per_ep", "c_per_100w", "w_per_ep")}
    if grp["hi"] and grp["lo"]:
        rr = defaultdict(list)
        for _ in range(BOOT):
            a, b = agg([rng.choice(grp["hi"]) for _ in grp["hi"]]), agg([rng.choice(grp["lo"]) for _ in grp["lo"]])
            for m in ("c_per_ep", "c_per_100w", "w_per_ep", "day_c", "day_n", "day_w"):
                rr[m].append(a[m] / b[m] if b[m] else float("nan"))
        out["hi_lo"] = {m: {"ratio": out["hi"][m] / out["lo"][m] if out["lo"][m] else float("nan"), "ci": bci(v)} for m, v in rr.items()}
    return out


# 3 下游后果 ---------------------------------------------------------------------------------------
def discord(days):
    bins = {"<0.2": (0, .2), "0.2–0.5": (.2, .5), "≥0.5": (.5, 9)}
    out = {}
    for ref in ("w", "n"):
        cnt = {b: [0, 0] for b in bins}
        tot = [0, 0]
        for i in range(len(days)):
            for j in range(i + 1, len(days)):
                a, b = days[i], days[j]
                dc, dr = np.sign(a["c"] - b["c"]), np.sign(a[ref] - b[ref])
                if dc == 0 or dr == 0:
                    continue
                gap = abs(a["cmd"] - b["cmd"])
                k = next(k for k, (lo, hi) in bins.items() if lo <= gap < hi)
                cnt[k][0] += dc != dr
                cnt[k][1] += 1
                tot[0] += dc != dr
                tot[1] += 1
        out[ref] = {"all": tot, **cnt}
    return out


def adjacent_example(days):
    best = None
    for a, b in zip(days, days[1:]):
        if not (0.8 <= b["w"] / a["w"] <= 1.25) or not a["c"] or not b["c"]:
            continue
        r = max(a["c"], b["c"]) / min(a["c"], b["c"])
        if best is None or r > best["ratio"]:
            best = {"a": a, "b": b, "ratio": r}
    return best


# 4 边界临界点 -------------------------------------------------------------------------------------
def file_commits(s, x):
    """(片段, 代码文件) → 该片段在该文件上的已提交内容归属的提交集合；(片段, 代码文件) → 编辑次数。"""
    d = x["d"]
    ep_of_edit = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_edits.csv") if r["episode"] and not r["reason"]}
    fc = defaultdict(set)
    for a in load(d / "edit_attribution.csv"):
        ep = ep_of_edit.get((a["session"], a["seq"]))
        if ep and a["state"] == "committed" and a["commit"] and path_class(a["path"], "1") == "code":
            fc[(ep, norm(a["path"]))].add(a["commit"])
    for r in load(d / "checkpoint_deltas.csv"):
        if r["state"] == "committed" and r["commit"] and r["class"] == "code":
            fc[(r["episode"], norm(r["path"]))].add(r["commit"])
    fe = defaultdict(int)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            fe[(f["episode"], norm(f["path"]))] += int(f["n_edits"])
    files = defaultdict(set)
    for (e, p) in fe:
        files[e].add(p)
    return fc, fe, files


def ctx_of(s):
    x = prepare(s)
    _, items, _ = base(s)
    fc, fe, files = file_commits(s, x)
    code = {e["episode"]: e for e in x["eps"] if int(e["n_edits_code"]) > 0}
    by_s = defaultdict(list)
    for e in code.values():
        by_s[e["session"]].append(e)
    for es in by_s.values():
        es.sort(key=lambda e: (e["start"], int(e["msg_seq"])))
    return {"s": s, "x": x, "items": items, "fc": fc, "fe": fe, "files": files, "code": code, "by_s": by_s,
            "cmd": {i["e"]: i["cmd"] for i in items}, "ver": {i["e"]: i["ver"] for i in items}}


def build_units(C, groups, parts):
    """groups：片段组列表（同一单位的片段）；parts：片段 → 文件子集列表（该片段拆为若干子单位，只对单片段组生效）。
    返回 单位列表，每个单位含 uid、成员、标签、内容提交、是否含片段内提交。"""
    x, fc = C["x"], C["fc"]
    units = []
    for k, mem in enumerate(groups):
        sess = C["code"][mem[0]]["session"]
        subs = parts.get(mem[0]) if len(mem) == 1 else None
        if subs:
            e = mem[0]
            for j, fs in enumerate(subs):
                cs = set().union(*(fc.get((e, f), set()) for f in fs)) & x["content"].get(e, set())
                units.append({"uid": f"{sess}|g{k}.{j}", "mem": mem, "files": fs, "P": cs, "n": sum(C["fe"][(e, f)] for f in fs)})
        else:
            units.append({"uid": f"{sess}|g{k}", "mem": mem, "files": None, "P": set().union(*(x["content"].get(e, set()) for e in mem)),
                          "n": sum(int(C["code"][e]["n_edits_code"]) for e in mem)})
    eo = defaultdict(set)
    for u in units:
        for c in u["P"]:
            eo[c].add(u["uid"])
    for u in units:
        st = unit_state([x["st"][e] for e in u["mem"]])
        ctx = set().union(*(x["ctx"][e] for e in u["mem"]))
        fl = any(x["flatten"][e] for e in u["mem"])
        u["labels"] = set(classify(max(1, u["n"]), u["uid"], u["P"], eo, x["cm"], ctx, fl, st))
        u["cmd"] = any(C["cmd"][e] for e in u["mem"])
        u["share"] = sum(1 / len(eo[c]) for c in u["P"])
    return units


def metrics(units):
    n = len(units)
    g1, g0 = [u for u in units if u["cmd"]], [u for u in units if not u["cmd"]]
    mg = lambda g: sum(has(u["labels"], MERGE) for u in g) / len(g) if g else float("nan")
    k = lambda g, f: sum(f(u) for u in g) / len(g) if g else float("nan")
    k1s, k0s = k(g1, lambda u: u["share"]), k(g0, lambda u: u["share"])
    k1c, k0c = k(g1, lambda u: len(u["P"])), k(g0, lambda u: len(u["P"]))
    return {"n": n, "oto": sum("一对一" in u["labels"] for u in units) / n, "merge": mg(units), "split": sum("拆分" in u["labels"] for u in units) / n,
            "gap": mg(g0) - mg(g1), "ratio_share": k1s / k0s if k0s else float("nan"), "ratio_count": k1c / k0c if k0c else float("nan")}


def perturb(C, rho, sigma, rng, only_cmd_m=None):
    groups = []
    for es in C["by_s"].values():
        cur = [es[0]["episode"]]
        for e in es[1:]:
            if rng.random() < rho:
                cur.append(e["episode"])
            else:
                groups.append(cur)
                cur = [e["episode"]]
        groups.append(cur)
    parts = {}
    for g in groups:
        if len(g) != 1:
            continue
        e = g[0]
        fs = sorted(C["files"][e])
        if only_cmd_m is not None:
            m = min(only_cmd_m, len(fs)) if C["cmd"][e] else 1
        else:
            m = 2 if len(fs) >= 2 and rng.random() < sigma else 1
        if m < 2:
            continue
        rng.shuffle(fs)
        buckets = [fs[j::m] for j in range(m)]
        parts[e] = [set(b) for b in buckets if b]
    return build_units(C, groups, parts)


KEYS = ("oto", "merge", "split", "gap", "ratio_share", "ratio_count")


def curve(C, tag, fn):
    rng = random.Random(f"{SEED}|{C['s']}|{tag}")
    ms = [metrics(fn(rng)) for _ in range(REPS)]
    return {k: {"med": float(np.median([m[k] for m in ms])), "ci": [float(np.percentile([m[k] for m in ms], 2.5)), float(np.percentile([m[k] for m in ms], 97.5))]}
            for k in KEYS}


TIP = {"oto": ("≥", .5), "merge": ("<", .10), "gap": ("<", .20), "ratio_share": ("<", 2.0)}


def tipping(rows):
    out = {}
    for k, (op, th) in TIP.items():
        hit = [lv for lv, r in rows if (r[k]["med"] >= th if op == "≥" else r[k]["med"] < th)]
        out[k] = hit[0] if hit else None
    return out


def boundary(C):
    base_units = build_units(C, [[e["episode"]] for es in C["by_s"].values() for e in es], {})
    R = {"base": metrics(base_units)}
    R["O"] = [(lv, curve(C, f"O{lv}", lambda rng, lv=lv: perturb(C, lv, 0, rng))) for lv in LEVELS]
    R["U"] = [(lv, curve(C, f"U{lv}", lambda rng, lv=lv: perturb(C, 0, lv, rng))) for lv in LEVELS]
    R["U1"] = [(m, curve(C, f"U1{m}", lambda rng, m=m: perturb(C, 0, 0, rng, m))) for m in MULTI]
    R["grid"] = {f"{r}|{s}": curve(C, f"G{r}{s}", lambda rng, r=r, s=s: perturb(C, r, s, rng)) for r in GRID for s in GRID}
    R["tip_O"], R["tip_U"], R["tip_U1"] = tipping(R["O"]), tipping(R["U"]), tipping(R["U1"])
    R["multi_file"] = sum(len(C["files"][e]) >= 2 for e in C["code"]) / len(C["code"])
    R["multi_file_cmd"] = [sum(len(C["files"][e]) >= 2 for e in C["code"] if C["cmd"][e] == f) / max(1, sum(C["cmd"][e] == f for e in C["code"])) for f in (True, False)]
    R["adj_pairs"] = sum(len(es) - 1 for es in C["by_s"].values())
    return R


def main():
    R, L = {}, ["# v3.0 度量效度分析（S17-T1，D17）", "", "由 `analysis/robustness6.py` 生成。", ""]
    num = json.loads(NUMS.read_text(encoding="utf-8"))

    L += ["## 1 提交规则（D17-1）", ""]
    for s in SYSTEMS:
        h = rule_history(s)
        R[f"rule_{s}"] = h
        L.append(f"- {NAME[s]}：`{RULE}` 首次加入 {h['first_date']}，此后修改 {h['n_changes']} 次（{('、'.join(h['change_dates']) or '无')}）；"
                 f"首版含\"一条任务 = 一次提交\"规定：{'是' if h['rule_first'] else '否'}，窗口末版本：{'是' if h['rule_head'] else '否'}。" if h else f"- {NAME[s]}：无规则文件。")
    L.append("")

    L += ["## 2 工作方式分期（D17-2）", "",
          "| 系统 | 分期 | 天数 | 片段 | 代码编辑 | 提交 | 每片段提交数 | 每百次代码编辑提交数 | 每片段代码编辑数 | 事前登记 | 请求引用任务编号 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    DAYS = {}
    for s in SYSTEMS:
        x, items, _ = base(s)
        days, active, ncm = days_of(s, x, items)
        DAYS[s] = (days, active, ncm, len(items), sum(i["n"] for i in items))
        G = regimes(s, days)
        R[f"regime_{s}"] = G
        for k, kn in (("hi", "高（≥ 2/3）"), ("mid", "中"), ("lo", "低（≤ 1/3）")):
            if k not in G:
                continue
            v = G[k]
            f = lambda m, d=2: f"{v[m]:.{d}f}（{v['ci'][m][0]:.{d}f}–{v['ci'][m][1]:.{d}f}）"
            L.append(f"| {NAME[s]} | {kn} | {v['days']} | {v['n']} | {v['w']} | {v['c']} | {f('c_per_ep')} | {f('c_per_100w', 1)} | {f('w_per_ep', 0)} | {pc(v['reg'])} | {pc(v['tref'])} |")
    L += ["", "分期按当日片段内提交比例；合格日子为 ≥ 3 个代码修改片段的日子；区间按日重抽 2,000 次。提交按提交时间归日，片段按开始时间归日。", ""]
    for s in SYSTEMS:
        G = R[f"regime_{s}"]
        if "hi_lo" in G:
            H = G["hi_lo"]
            f = lambda m: f"{H[m]['ratio']:.2f}（{H[m]['ci'][0]:.2f}–{H[m]['ci'][1]:.2f}）"
            L.append(f"- {NAME[s]} 高 / 低两期之比：日均提交数 {f('day_c')}，日均片段数 {f('day_n')}，日均代码编辑数 {f('day_w')}；"
                     f"每片段提交数 {f('c_per_ep')}，每百次代码编辑提交数 {f('c_per_100w')}，每片段代码编辑数 {f('w_per_ep')}。")
    L.append("")

    L += ["## 3 下游后果（D17-3）", ""]
    for s in SYSTEMS:
        days = DAYS[s][0]
        D = discord(days)
        ex = adjacent_example(days)
        R[f"discord_{s}"], R[f"example_{s}"] = D, ex
        cell = lambda v: f"{v[0]}/{v[1]}（{v[0] / v[1]:.0%}）" if v[1] else "—"
        for ref, rn in (("w", "代码编辑数"), ("n", "片段数")):
            L.append(f"- {NAME[s]}（{len(days)} 天）：按提交数与按{rn}比较两日，方向相反的日对 {cell(D[ref]['all'])}；按两日片段内提交比例之差分组："
                     + "，".join(f"{b} {cell(D[ref][b])}" for b in ("<0.2", "0.2–0.5", "≥0.5")) + "。")
        if ex:
            a, b = ex["a"], ex["b"]
            L.append(f"  - 相邻合格日子中代码编辑数相差不超过 25%、提交数相差最大的一对：{a['d']}（片段 {a['n']}、片段内提交 {a['cmd']:.0%}、代码编辑 {a['w']}、提交 {a['c']}）"
                     f"与 {b['d']}（片段 {b['n']}、{b['cmd']:.0%}、{b['w']}、{b['c']}），提交数相差 {ex['ratio']:.1f} 倍。")
    A, Cc = DAYS["SYS_A"], DAYS["SYS_C"]
    per = lambda D, k: (D[2] if k == "c" else D[3] if k == "n" else D[4]) / len(D[1])
    R["sys_ratio"] = {k: per(A, k) / per(Cc, k) for k in ("c", "n", "w")}
    R["sys_active_days"] = [len(A[1]), len(Cc[1])]
    L.append(f"- 系统比较（分母为有代码修改片段的日子：系统 A {len(A[1])} 天、系统 C {len(Cc[1])} 天）：系统 A 对系统 C 的日均提交数之比 {R['sys_ratio']['c']:.2f}，"
             f"日均代码修改片段数之比 {R['sys_ratio']['n']:.2f}，日均代码编辑数之比 {R['sys_ratio']['w']:.2f}。")
    L.append("")

    L += ["## 4 边界临界点（D17-4）", ""]
    for s in SYSTEMS:
        C = ctx_of(s)
        B = boundary(C)
        R[f"boundary_{s}"] = B
        b = B["base"]
        L += [f"### {NAME[s]}", "",
              f"起点（片段）：{b['n']} 个代码修改单位，一对一 {pc(b['oto'])}、合并 {pc(b['merge'])}、拆分 {pc(b['split'])}；合并组间差（未提交 − 片段内提交）{b['gap'] * 100:+.0f} 个百分点；"
              f"每单位分得提交数之比（片段内提交 / 未提交）{b['ratio_share']:.2f}，各计 1 口径 {b['ratio_count']:.2f}。相邻代码修改片段对 {B['adj_pairs']} 个；"
              f"含 ≥ 2 个代码文件的片段 {pc(B['multi_file'])}（片段内提交 {pc(B['multi_file_cmd'][0])}、未提交 {pc(B['multi_file_cmd'][1])}）。", "",
              "| 扰动 | 水平 | 一对一 | 合并 | 拆分 | 合并组间差 | 每单位分得提交数之比 | 各计 1 口径之比 |", "|---|---|---|---|---|---|---|---|"]
        for kind, kn in (("O", "过度切分 ρ"), ("U", "切分不足 σ"), ("U1", "片段内提交片段按文件分为 m 份")):
            for lv, r in B[kind]:
                L.append(f"| {kn} | {lv} | {pc(r['oto']['med'])} | {pc(r['merge']['med'])} | {pc(r['split']['med'])} | {r['gap']['med'] * 100:+.0f} | "
                         f"{r['ratio_share']['med']:.2f}（{r['ratio_share']['ci'][0]:.2f}–{r['ratio_share']['ci'][1]:.2f}） | {r['ratio_count']['med']:.2f} |")
        tip = lambda t: "；".join(f"{k} {'未越过' if v is None else v}" for k, v in t.items())
        L += ["", f"临界点（200 次中位数首次越过：一对一 ≥ 50%、合并 < 10%、合并组间差 < 20 个百分点、每单位分得提交数之比 < 2）：过度切分 {tip(B['tip_O'])}；"
              f"切分不足 {tip(B['tip_U'])}；片段内提交片段分 m 份 {tip(B['tip_U1'])}。", "",
              "联合网格（ρ 行 × σ 列）：合并 / 一对一", "", "| ρ \\ σ | " + " | ".join(str(v) for v in GRID) + " |", "|---|" + "---|" * len(GRID)]
        for r in GRID:
            L.append(f"| {r} | " + " | ".join(f"{pc(B['grid'][f'{r}|{c}']['merge']['med'])} / {pc(B['grid'][f'{r}|{c}']['oto']['med'])}" for c in GRID) + " |")
        L.append("")
    L += ["过度切分：同一会话相邻代码修改片段之间的边界以概率 ρ 被移除；切分不足：含 ≥ 2 个代码文件的片段以概率 σ 按文件随机分为两个子单位，"
          "子单位的提交取该片段在这些文件上的已提交内容归属；片段内提交片段分 m 份：只把片段内提交的片段按文件随机分为 min(m, 文件数) 份。"
          "单位含任一片段内提交的片段即计入片段内提交组。每个水平 200 次，报告中位数（括号内为 2.5%–97.5% 分位）。", ""]

    (RES / "r14_robustness6.md").write_text("\n".join(L), encoding="utf-8")
    num["robustness6"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
