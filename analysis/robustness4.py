"""v2.13 补充分析（S15-T1，口径见 notes/decisions.md D15）。

1 合并的集中度  2 规模标准化的反事实  3 方差分解的比例基准  4 工作项的统一分母与替代判定
5 纯度下界  6 只有尾注的提交推定为片段内提交  7 人工估计的后验中位数  8 可核验片段的划分
写入 <PAPER_DIR>/results/r12_robustness4.md 与 numbers.json 的 robustness4 键（须在 robustness3 之后运行）。
"""
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import command_commits  # noqa: E402
from attribute_edits import commit_added_lines  # noqa: E402
from extract_episodes import path_class  # noqa: E402
from extract_git import SYSTEMS as REPOS, git  # noqa: E402
from robustness import MERGE, NAME, base, has, load, pc, q, share  # noqa: E402
from robustness3 import W_ROWS, pooled  # noqa: E402
from workitems import RULE_NAME, coded_decisions, group, units  # noqa: E402

BOOT, SEED = 2000, "S15"
NUMS = (RESULTS_DIR / "numbers.json")
RES = RESULTS_DIR
SYSTEMS = ["SYS_A", "SYS_C"]
TZ = timezone(timedelta(hours=8))


def ts(v):
    t = datetime.fromisoformat(v)
    return t if t.tzinfo else t.replace(tzinfo=TZ)


def eps_of_commit(x):
    out = defaultdict(set)
    for e, cs in x["content"].items():
        for c in cs:
            out[c].add(e)
    return out


# 1 合并的集中度 ---------------------------------------------------------------------------------
def concentration(x, items):
    eo = eps_of_commit(x)
    multi = sorted((c for c in eo if len(eo[c]) >= 2), key=lambda c: (-len(eo[c]), c))
    merged = [i for i in items if has(i["labels"], MERGE)]
    out = {"n": len(items), "merged": len(merged), "multi": len(multi), "multi_eps": len(set().union(*(eo[c] for c in multi))) if multi else 0,
           "sizes": [len(eo[c]) for c in multi[:5]], "drop": {}}
    for k in (1, 3, 5, 10):
        drop = set(multi[:k])
        still = sum(any(c not in drop and eo[c] - {i["e"]} for c in x["content"].get(i["e"], ())) for i in merged)
        out["drop"][k] = still
    top5 = set(multi[:5])
    out["via_top5"] = sum(any(c in top5 for c in x["content"].get(i["e"], ())) for i in merged)
    return out


# 2 规模标准化的反事实 ---------------------------------------------------------------------------
def std_counterfactual(s, x, items, p_from, p_to):
    eo = eps_of_commit(x)
    sh = {i["e"]: sum(1 / len(eo[c]) for c in x["content"].get(i["e"], ())) for i in items}
    cut = np.quantile([i["n"] for i in items], [.25, .5, .75])
    layer = {i["e"]: int(np.searchsorted(cut, i["n"], side="right")) for i in items}

    def ratio(its):
        by = defaultdict(list)
        for i in its:
            by[layer[i["e"]]].append(i)
        num = den = 0.0
        for g in by.values():
            g1, g0 = [sh[i["e"]] for i in g if i["cmd"]], [sh[i["e"]] for i in g if not i["cmd"]]
            if not g1 or not g0:
                continue
            k1, k0 = sum(g1) / len(g1), sum(g0) / len(g0)
            num += len(g) * (p_to * k1 + (1 - p_to) * k0)
            den += len(g) * (p_from * k1 + (1 - p_from) * k0)
        return num / den if den else None

    rows = []
    for h in range(4):
        g = [i for i in items if layer[i["e"]] == h]
        g1, g0 = [sh[i["e"]] for i in g if i["cmd"]], [sh[i["e"]] for i in g if not i["cmd"]]
        rows.append({"n": len(g), "n1": len(g1), "n0": len(g0), "k1": sum(g1) / len(g1) if g1 else None,
                     "k0": sum(g0) / len(g0) if g0 else None, "max_n": max(i["n"] for i in g) if g else 0})
    by = defaultdict(list)
    for i in items:
        by[i["s"]].append(i)
    ks = sorted(by)
    rng = random.Random(f"{SEED}|cf|{s}")
    rs = [v for v in (ratio([i for k in rng.choices(ks, k=len(ks)) for i in by[k]]) for _ in range(BOOT)) if v is not None]
    return {"ratio": ratio(items), "ci": [q(rs, .025), q(rs, .975)], "layers": rows}


# 3 方差分解的比例基准 ---------------------------------------------------------------------------
def variance(x, items):
    day = lambda t: t[:10]
    gold = command_commits(x["d"], x["cm"])
    cw, gw, ew = Counter(), Counter(), Counter()
    for h, c in x["cm"].items():
        cw[day(c["time"])] += 1
        gw[day(c["time"])] += h in gold
    for i in items:
        ew[day(i["start"])] += 1
    days = sorted(d for d in ew if ew[d] >= 3)
    c = np.array([cw[d] for d in days], float)
    g = np.array([gw[d] for d in days], float)
    o = c - g
    sh = g.sum() / c.sum()
    return {"days": len(days), "share": sh, "obs": [g.var() / c.var(), o.var() / c.var(), 2 * np.cov(g, o, bias=True)[0, 1] / c.var()],
            "base": [sh ** 2, (1 - sh) ** 2, 2 * sh * (1 - sh)], "cv": [g.std() / g.mean(), o.std() / o.mean(), c.std() / c.mean()],
            "corr": float(np.corrcoef(g, o)[0, 1])}


# 4 工作项的统一分母 -----------------------------------------------------------------------------
def workitem_denoms(x):
    code_files = defaultdict(set)
    for f in load(x["d"] / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"])
    coded = coded_decisions()
    out = {}
    for r in ("episode", "strict", "loose", "coded", "session"):
        if r == "coded" and not coded:
            continue
        wi = group(x["eps"], code_files, r, coded)
        us, eo = units(x, wi)
        mem = defaultdict(list)
        for e in x["eps"]:
            mem[wi[e["episode"]]].append(e["episode"])
        code = {u: v for u, v in us.items() if v["n_code"]}
        ver = {u: v for u, v in code.items() if v["evidence"] == "verified"}

        def majority(u):
            ms = [e for e in mem[u] if x["content"].get(e)]
            hit = sum(any(eo[c] - {u} for c in x["content"][e]) for e in ms)
            return bool(ms) and 2 * hit > len(ms)

        row = {}
        for tag, grp in (("all", code), ("ver", ver), ("mapped", {u: v for u, v in code.items() if v["commits"]})):
            n = len(grp)
            row[tag] = {"n": n, "merge": share(list(grp.values()), lambda v: "合并" in v["labels"]),
                        "same": share(list(grp.values()), lambda v: "合并-同会话" in v["labels"]),
                        "cross": share(list(grp.values()), lambda v: "合并-跨会话" in v["labels"]),
                        "majority": share(list(grp), majority)}
        out[r] = row
    return out


# 5 纯度下界 ------------------------------------------------------------------------------------
def purity_lb(s, x, items, lines):
    added = commit_added_lines(REPOS[s])
    tot = Counter()
    for (h, p), v in added.items():
        if path_class(p, "1") == "code":
            tot[h] += len(v)
    attr = Counter()
    by_e = defaultdict(dict)
    for (e, c), v in lines.items():
        attr[c] += v
        by_e[e][c] = v
    res = {"ver": [], "all": []}
    one_low = one_n = 0
    for it in items:
        row = by_e.get(it["e"])
        if not row or not sum(row.values()):
            continue
        c = max(row, key=row.get)
        lb = min(1.0, row[c] / tot[c]) if tot[c] else 1.0
        pur = row[c] / attr[c] if attr[c] else 1.0
        res["all"].append((pur, lb))
        if it["ver"]:
            res["ver"].append((pur, lb))
        if "一对一" in it["labels"]:
            one_n += 1
            one_low += bool(tot[c]) and attr[c] / tot[c] < 0.5
    out = {}
    for k, v in res.items():
        a = np.array(v)
        out[k] = {"n": len(v), "pur_med": float(np.median(a[:, 0])), "lb_med": float(np.median(a[:, 1])),
                  "pur_90": float((a[:, 0] >= .9).mean()), "lb_90": float((a[:, 1] >= .9).mean())}
    out["one_n"], out["one_low"] = one_n, one_low
    return out


# 6 只有尾注的提交 --------------------------------------------------------------------------------
def trailer_impute(s, x, items):
    gold = command_commits(x["d"], x["cm"])
    trailer = set()
    for blk in git(REPOS[s], "log", "--format=\x1e%H\x1f%B").split("\x1e")[1:]:
        h, b = blk.split("\x1f", 1)
        if "co-authored-by: cursor" in b.lower():
            trailer.add(h.strip())
    only = {h for h in trailer if h in x["cm"] and h not in gold}
    ep = {e["episode"]: e for e in x["eps"]}
    imputed = set()
    for i in items:
        if i["cmd"]:
            continue
        e = ep[i["e"]]
        lo = ts(e["start"])
        hi = ts(e["next_start"]) if e["next_start"] else datetime.max.replace(tzinfo=TZ)
        if any(h in only and lo <= ts(x["cm"][h]["time"]) < hi for h in x["content"].get(i["e"], ())):
            imputed.add(i["e"])
    wins = [(ts(e["start"]), ts(e["next_start"]) if e["next_start"] else ts(e["last_event"])) for e in x["eps"]]
    inside = sum(any(lo <= ts(x["cm"][h]["time"]) <= hi for lo, hi in wins) for h in only)
    out = {"only": len(only), "imputed": len(imputed), "inside_any": inside,
           "with_content": sum(any(h in cs for cs in x["content"].values()) for h in only)}
    for tag, grp in (("ver", [i for i in items if i["ver"]]), ("all", items)):
        for mode in ("obs", "imp"):
            cmd = (lambda i: i["cmd"]) if mode == "obs" else (lambda i: i["cmd"] or i["e"] in imputed)
            a, b = [i for i in grp if cmd(i)], [i for i in grp if not cmd(i)]
            out[f"{tag}|{mode}"] = {"n1": len(a), "n0": len(b),
                                    "one1": share(a, lambda i: "一对一" in i["labels"]), "one0": share(b, lambda i: "一对一" in i["labels"]),
                                    "mg1": share(a, lambda i: has(i["labels"], MERGE)), "mg0": share(b, lambda i: has(i["labels"], MERGE))}
        out[f"{tag}|imp_n"] = sum(i["e"] in imputed for i in grp)
    return out


# 9 并行会话 ----------------------------------------------------------------------------------------
def parallel(x, items):
    files_of = defaultdict(set)
    for f in load(x["d"] / "episode_files.csv"):
        if f["class"] == "code":
            files_of[f["episode"]].add(f["path"])
    ep = {e["episode"]: e for e in x["eps"]}
    spans = [(i["e"], i["s"], ts(ep[i["e"]]["start"]), ts(ep[i["e"]]["last_event"])) for i in items]
    ov = set()
    for a in spans:
        for b in spans:
            if a[1] < b[1] and a[2] < b[3] and b[2] < a[3] and files_of[a[0]] & files_of[b[0]]:
                ov |= {a[0], b[0]}
    cross = [i for i in items if "合并-跨会话" in i["labels"]]
    return {"overlap": len(ov), "cross": len(cross), "cross_ov": sum(i["e"] in ov for i in cross),
            "cross_ver": sum(i["ver"] for i in cross), "cross_ver_ov": sum(i["ver"] and i["e"] in ov for i in cross)}


# 8 可核验片段的划分 ------------------------------------------------------------------------------
def verified_split(x, items):
    ver = [i for i in items if i["ver"]]
    ent = [i for i in ver if x["content"].get(i["e"])]
    rest = Counter("缺失" if "缺失" in i["labels"] else ("无新增内容" if "无新增内容" in i["labels"] else "其他")
                   for i in ver if not x["content"].get(i["e"]))
    return {"ver": len(ver), "entered": len(ent), **rest}


def main():
    R, L = {}, ["# v2.13 补充分析（S15-T1，D15）", "", "由 `analysis/robustness4.py` 生成。", ""]
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    cf3 = num["robustness3"]
    xs = {s: base(s) for s in SYSTEMS}
    cmd_share = {s: share(xs[s][1], lambda i: i["cmd"]) for s in SYSTEMS}

    L += ["## 1 合并的集中度（D15-1）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        C = concentration(x, items)
        R[f"conc_{s}"] = C
        dr = "；".join(f"剔除最大 {k} 个后 {v} 个（{v / C['n']:.0%}）" for k, v in C["drop"].items())
        L.append(f"- {NAME[s]}：合并片段 {C['merged']}/{C['n']}（{C['merged'] / C['n']:.0%}）；含 ≥ 2 个片段的提交 {C['multi']} 个，涉及片段 {C['multi_eps']} 个；"
                 f"最大 5 个提交各含 {' / '.join(map(str, C['sizes']))} 个片段，合并片段中有 {C['via_top5']} 个（{C['via_top5'] / C['merged']:.0%}）涉及这 5 个提交。"
                 f"仍判为合并：{dr}。")
    L.append("")

    L += ["## 2 规模标准化的反事实（D15-2）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        C = std_counterfactual(s, x, items, round(cmd_share["SYS_C"], 2), round(cmd_share["SYS_A"], 2))
        R[f"cf_{s}"] = C
        lay = "；".join(f"第 {h + 1} 层（编辑数 ≤ {r['max_n']}，n = {r['n']}，片段内提交 {r['n1']}）k₁ = {r['k1']:.2f}、k₀ = {r['k0']:.2f}"
                       if r["k1"] is not None and r["k0"] is not None else f"第 {h + 1} 层 n = {r['n']}（一组为空）"
                       for h, r in enumerate(C["layers"]))
        old = cf3[f"cf_{s}"]
        L.append(f"- {NAME[s]}：按代码编辑数四分位分层后，{cmd_share['SYS_C']:.0%} → {cmd_share['SYS_A']:.0%} 的倍数为 {C['ratio']:.2f}"
                 f"（按会话重抽 {C['ci'][0]:.2f}–{C['ci'][1]:.2f}；未分层 {old['ratio']:.2f}）。{lay}。")
    L.append("")

    L += ["## 3 方差分解的比例基准（D15-3）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        V = variance(x, items)
        R[f"var_{s}"] = V
        f3 = lambda v: " / ".join(f"{a:.2f}" for a in v)
        L.append(f"- {NAME[s]}（{V['days']} 天）：片段内产生的提交占 {V['share']:.0%}。方差分解（片段内 / 其余 / 协方差项）观察 {f3(V['obs'])}，"
                 f"同比例基准 {f3(V['base'])}；变异系数 片段内 {V['cv'][0]:.2f}、其余 {V['cv'][1]:.2f}、全部 {V['cv'][2]:.2f}；两部分相关 {V['corr']:.2f}。")
    L.append("")

    L += ["## 4 工作项的统一分母与替代判定（D15-5）", "",
          "| 系统 | 口径 | 全部代码修改单位 n | 合并 | 同会话 | 跨会话 | 多数成员 | 可核验 n | 合并 | 进入版本库 n | 合并 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        x, _, _ = xs[s]
        W = workitem_denoms(x)
        R[f"wi_{s}"] = W
        for r, v in W.items():
            a, vv, m = v["all"], v["ver"], v["mapped"]
            L.append(f"| {NAME[s]} | {RULE_NAME[r]} | {a['n']} | {pc(a['merge'])} | {pc(a['same'])} | {pc(a['cross'])} | {pc(a['majority'])} | "
                     f"{vv['n']} | {pc(vv['merge'])} | {m['n']} | {pc(m['merge'])} |")
    L += ["", "只计同会话合并相当于假定全部跨会话合并都是同一项工作的延续，是合并的下界。多数成员：有内容的成员片段中过半数的某提交含其他单位的内容。", ""]

    L += ["## 5 纯度下界（D15-6）", ""]
    for s in SYSTEMS:
        x, items, lines = xs[s]
        P = purity_lb(s, x, items, lines)
        R[f"purity_{s}"] = P
        L.append(f"- {NAME[s]}：可核验（n = {P['ver']['n']}）纯度中位数 {P['ver']['pur_med']:.2f} → 下界 {P['ver']['lb_med']:.2f}，≥ 0.9 的比例 {pc(P['ver']['pur_90'])} → {pc(P['ver']['lb_90'])}；"
                 f"全部（n = {P['all']['n']}）中位数 {P['all']['pur_med']:.2f} → {P['all']['lb_med']:.2f}，≥ 0.9 {pc(P['all']['pur_90'])} → {pc(P['all']['lb_90'])}。"
                 f"判为一对一的 {P['one_n']} 个片段中，其提交有 ≥ 50% 新增特征行未归属任何片段的 {P['one_low']} 个。")
    L += ["", "下界以主提交中全部代码文件的新增特征行（≥ 15 字符）为分母；分子为片段归属到该提交的特征行数，封顶为 1。", ""]

    L += ["## 6 只有尾注的提交推定为片段内提交（D15-7）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        T = trailer_impute(s, x, items)
        R[f"trailer_{s}"] = T
        L.append(f"- {NAME[s]}：只有尾注、对不上命令的提交 {T['only']} 个，其中含片段内容的 {T['with_content']} 个、提交时间落在任一片段时间窗内的 {T['inside_any']} 个；据此推定为片段内提交的片段 {T['imputed']} 个（可核验 {T['ver|imp_n']} 个）。")
        for tag, tn in (("ver", "可核验"), ("all", "全部代码修改")):
            o, m = T[f"{tag}|obs"], T[f"{tag}|imp"]
            L.append(f"  - {tn}：一对一 片段内 {pc(o['one1'])} 对未提交 {pc(o['one0'])}（n = {o['n1']} / {o['n0']}）→ 推定后 {pc(m['one1'])} 对 {pc(m['one0'])}（{m['n1']} / {m['n0']}）；"
                     f"合并 {pc(o['mg1'])} 对 {pc(o['mg0'])} → {pc(m['mg1'])} 对 {pc(m['mg0'])}。")
    L.append("")

    L += ["## 7 人工估计的后验中位数（D15-8）", "", "| 模式 | " + " | ".join(f"{NAME[s]}：反加权点估计 | 后验中位数（95% 区间）" for s in SYSTEMS) + " |",
          "|---|" + "---|" * 4]
    P = pooled()
    if P:
        W, _ = P
        R["pooled_med"] = {k: {"med": v["pooled_med"], "ci": v["pooled_ci"], "point": v["pooled"]} for k, v in W.items()}
        for name, _ in W_ROWS:
            cells = []
            for s in SYSTEMS:
                v = W[f"{s}|{name}"]
                cells += [pc(v["pooled"]), f"{pc(v['pooled_med'])}（{pc(v['pooled_ci'][0])}–{pc(v['pooled_ci'][1])}）"]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "可多标签：主模式或次模式命中即计入。", ""]

    L += ["## 8 可核验片段的划分（D15-9）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        V = verified_split(x, items)
        R[f"versplit_{s}"] = V
        L.append(f"- {NAME[s]}：可核验 {V['ver']} = 进入版本库 {V['entered']} + 缺失 {V.get('缺失', 0)} + 无新增内容 {V.get('无新增内容', 0)}"
                 + (f" + 其他 {V['其他']}" if V.get("其他") else "") + "。")
    L.append("")

    L += ["## 9 并行会话（D15-10）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        V = parallel(x, items)
        R[f"parallel_{s}"] = V
        L.append(f"- {NAME[s]}：与其他会话时间重叠（片段开始至最后事件）且编辑同一代码文件的代码修改片段 {V['overlap']} 个；"
                 f"跨会话合并的 {V['cross']} 个片段中 {V['cross_ov']} 个属于此类（可核验 {V['cross_ver']} 个中 {V['cross_ver_ov']} 个）。")
    L.append("")

    (RES / "r12_robustness4.md").write_text("\n".join(L), encoding="utf-8")
    num["robustness4"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
