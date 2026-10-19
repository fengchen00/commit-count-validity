"""v2.14 补充分析（S16-T1，口径见 notes/decisions.md D16）。

1 人工估计的子类互斥口径  2 合并子类的优先级与组合下界  3 反事实的会计情景变体  4 按天零模型与区间
5 表 5 补全  6 证据层与样板行  7 会话内置换检验  8 图 7 读数
写入 <PAPER_DIR>/results/r13_robustness5.md 与 numbers.json 的 robustness5 键（须在 robustness4 之后运行）。
"""
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import command_commits  # noqa: E402
from attribute_edits import commit_added_lines  # noqa: E402
from extract_episodes import path_class  # noqa: E402
from extract_git import SYSTEMS as REPOS  # noqa: E402
from holdout_agreement import holdout_reference  # noqa: E402
from robustness import MERGE, NAME, base, cluster_boot, has, load, pc, q, share  # noqa: E402
from robustness3 import RES, W_DRAWS, batch_draws, read_pop  # noqa: E402
from robustness4 import concentration, eps_of_commit  # noqa: E402
from sim_agreement import final_reference  # noqa: E402
from workitems import RULE_NAME, coded_decisions, group, units  # noqa: E402

BOOT, SEED = 2000, "S16"
NUMS = (RESULTS_DIR / "numbers.json")
SYSTEMS = ["SYS_A", "SYS_C"]


def bci(vals):
    return [q(vals, .025), q(vals, .975)]


# 1 人工估计的子类互斥口径 ------------------------------------------------------------------------
def sub(r, grp, default):
    if r["main"] in grp:
        return r["main"]
    hs = [c for c in grp if c in r["sec"]]
    return hs[0] if len(hs) == 1 else (default if hs else None)


H_ROWS = [("拆分", lambda r: sub(r, ("A4", "A5"), "A5") is not None), ("　逐条任务拆分（互斥）", lambda r: sub(r, ("A4", "A5"), "A5") == "A4"),
          ("　非逐条拆分（互斥）", lambda r: sub(r, ("A4", "A5"), "A5") == "A5"),
          ("合并", lambda r: sub(r, ("A2", "A3"), "A3") is not None), ("　同会话合并（互斥）", lambda r: sub(r, ("A2", "A3"), "A3") == "A2"),
          ("　跨会话合并（互斥）", lambda r: sub(r, ("A2", "A3"), "A3") == "A3"), ("一对一", lambda r: r["main"] == "A1" or "A1" in r["sec"])]


def human_exclusive():
    orig, hold = final_reference("sim2"), holdout_reference()
    if not orig or not hold:
        return None
    s1 = {r["episode"]: r["stratum"] for r in load("data/coding/sample_S03.csv")}
    s2 = {r["episode"]: r["stratum"] for r in load("data/coding/holdout/sample_S12.csv")}
    p1, p2 = read_pop(RES / "s03_sampling.md"), read_pop(RES / "s12_holdout_sampling.md")
    both = lambda ref, a, b: sum((ref[e]["main"] in (a, b) or bool({a, b} & ref[e]["sec"])) and
                                 len({a, b} & ({ref[e]["main"]} | ref[e]["sec"])) == 2 for e in ref)
    out = {"overlap": {"split": [both(orig, "A4", "A5"), both(hold, "A4", "A5")], "merge": [both(orig, "A2", "A3"), both(hold, "A2", "A3")]}}
    for s in SYSTEMS:
        n1, n2 = sum(e.startswith(s) for e in s1), sum(e.startswith(s) for e in s2)
        w = n1 / (n1 + n2)
        for name, pred in H_ROWS:
            tag = name.strip("　")
            a, da = batch_draws(orig, s1, p1, [s], (tag,), "orig16", pred)
            b, db = batch_draws(hold, s2, p2, [s], (tag,), "hold16", pred)
            mix = sorted(w * u + (1 - w) * v for u, v in zip(da, db))
            out[f"{s}|{name}"] = {"orig": a, "hold": b, "point": w * a + (1 - w) * b, "med": mix[W_DRAWS // 2],
                                  "ci": [mix[int(.025 * W_DRAWS)], mix[int(.975 * W_DRAWS) - 1]]}
    return out


# 2 合并子类的优先级与组合下界 ----------------------------------------------------------------------
def merge_bounds(s, x, items):
    code_files = defaultdict(set)
    for f in load(x["d"] / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"])
    C = concentration(x, items)
    eo_ep = eps_of_commit(x)
    multi = sorted((c for c in eo_ep if len(eo_ep[c]) >= 2), key=lambda c: (-len(eo_ep[c]), c))
    top5 = set(multi[:5])
    coded = coded_decisions()
    out = {"top5_sizes": C["sizes"]}
    for r in ("episode", "strict", "loose", "coded", "session"):
        if r == "coded" and not coded:
            continue
        wi = group(x["eps"], code_files, r, coded)
        us, eo = units(x, wi)
        code = {u: v for u, v in us.items() if v["n_code"]}
        sess = lambda u: u.split("|")[0]

        def merged(u, drop=frozenset(), same_only=False):
            return any(any(o != u and (not same_only or sess(o) == sess(u)) for o in eo[c]) for c in code[u]["commits"] if c not in drop)
        n = len(code)
        row = {"n": n, "label_same": sum("合并-同会话" in v["labels"] for v in code.values()) / n,
               "label_cross": sum("合并-跨会话" in v["labels"] for v in code.values()) / n,
               "any": sum(merged(u) for u in code) / n, "same_partner": sum(merged(u, same_only=True) for u in code) / n,
               "both_kinds": sum(merged(u, same_only=True) and "合并-跨会话" in code[u]["labels"] for u in code) / n,
               "drop5": sum(merged(u, top5) for u in code) / n, "same_drop5": sum(merged(u, top5, True) for u in code) / n}
        out[r] = row
    return out


# 3 反事实的会计情景变体 ----------------------------------------------------------------------------
def accounting(s, x, items, p_from, p_to):
    eo = eps_of_commit(x)
    cm = x["cm"]
    k_share = {i["e"]: sum(1 / len(eo[c]) for c in x["content"].get(i["e"], ())) for i in items}
    k_count = {i["e"]: len(x["content"].get(i["e"], ())) for i in items}
    cut = np.quantile([i["n"] for i in items], [.25, .5, .75])
    lay_edit = {i["e"]: int(np.searchsorted(cut, i["n"], side="right")) for i in items}

    def ntask(e):
        return len({(cm[c]["milestone"], cm[c]["tasks"]) for c in x["content"].get(e, ())
                    if cm[c]["label"] == "task" and cm[c]["milestone"] and cm[c]["tasks"]})
    lay_task = {i["e"]: min(3, (lambda t: 0 if t == 0 else 1 if t == 1 else 2 if t <= 3 else 3)(ntask(i["e"]))) for i in items}
    lay_none = {i["e"]: 0 for i in items}

    def ratio(its, k, lay):
        by = defaultdict(list)
        for i in its:
            by[lay[i["e"]]].append(i)
        num = den = 0.0
        for g in by.values():
            g1, g0 = [k[i["e"]] for i in g if i["cmd"]], [k[i["e"]] for i in g if not i["cmd"]]
            if not g1 or not g0:
                continue
            k1, k0 = sum(g1) / len(g1), sum(g0) / len(g0)
            num += len(g) * (p_to * k1 + (1 - p_to) * k0)
            den += len(g) * (p_from * k1 + (1 - p_from) * k0)
        return num / den if den else None

    by = defaultdict(list)
    for i in items:
        by[i["s"]].append(i)
    ks = sorted(by)
    out = {}
    for kn, k in (("share", k_share), ("count", k_count)):
        for ln, lay in (("none", lay_none), ("edit", lay_edit), ("task", lay_task)):
            rng = random.Random(f"{SEED}|acc|{s}|{kn}|{ln}")
            rs = [v for v in (ratio([i for c in rng.choices(ks, k=len(ks)) for i in by[c]], k, lay) for _ in range(BOOT)) if v is not None]
            out[f"{kn}|{ln}"] = {"ratio": ratio(items, k, lay), "ci": bci(rs)}
    rows = []
    for h in range(4):
        g = [i for i in items if lay_task[i["e"]] == h]
        g1, g0 = [k_share[i["e"]] for i in g if i["cmd"]], [k_share[i["e"]] for i in g if not i["cmd"]]
        rows.append({"n": len(g), "n1": len(g1), "k1": sum(g1) / len(g1) if g1 else None, "k0": sum(g0) / len(g0) if g0 else None})
    out["task_layers"] = rows
    out["k1_count"] = float(np.mean([k_count[i["e"]] for i in items if i["cmd"]]))
    out["k0_count"] = float(np.mean([k_count[i["e"]] for i in items if not i["cmd"]]))
    return out


# 4 按天零模型与区间 --------------------------------------------------------------------------------
def daily_null(s, x, items):
    day = lambda t: t[:10]
    eo = eps_of_commit(x)
    sh = {i["e"]: sum(1 / len(eo[c]) for c in x["content"].get(i["e"], ())) for i in items}
    by_d = defaultdict(list)
    for i in items:
        by_d[day(i["start"])].append(i)
    days = sorted(d for d, g in by_d.items() if len(g) >= 3)
    cv = lambda v: float(np.std(v) / np.mean(v)) if np.mean(v) else float("nan")
    iqr = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))
    obs = np.array([sum(sh[i["e"]] for i in by_d[d]) / len(by_d[d]) for d in days])
    pool = {True: [sh[i["e"]] for i in items if i["cmd"]], False: [sh[i["e"]] for i in items if not i["cmd"]]}
    allp = [sh[i["e"]] for i in items]
    rng = random.Random(f"{SEED}|null|{s}")
    res = {"N1": [], "N0": []}
    for _ in range(BOOT):
        r1 = [sum(rng.choice(pool[i["cmd"]]) for i in by_d[d]) / len(by_d[d]) for d in days]
        r0 = [sum(rng.choice(allp) for _ in by_d[d]) / len(by_d[d]) for d in days]
        res["N1"].append((cv(r1), iqr(r1)))
        res["N0"].append((cv(r0), iqr(r0)))
    out = {"days": len(days), "obs_cv": cv(obs), "obs_iqr": iqr(obs), "obs_med": float(np.median(obs))}
    for k, v in res.items():
        a = np.array(v)
        out[k] = {"cv_ci": bci(a[:, 0]), "iqr_ci": bci(a[:, 1]), "p_cv": float((a[:, 0] >= out["obs_cv"]).mean()),
                  "p_iqr": float((a[:, 1] >= out["obs_iqr"]).mean())}
    # D15-3 的变异系数与相关系数：按日重抽
    gold = command_commits(x["d"], x["cm"])
    cw, gw = Counter(), Counter()
    for h, c in x["cm"].items():
        cw[day(c["time"])] += 1
        gw[day(c["time"])] += h in gold
    c = np.array([cw[d] for d in days], float)
    g = np.array([gw[d] for d in days], float)
    o = c - g
    stat = lambda idx: (cv(g[idx]), cv(o[idx]), cv(c[idx]), float(np.corrcoef(g[idx], o[idx])[0, 1]) if g[idx].std() and o[idx].std() else float("nan"))
    rng = random.Random(f"{SEED}|daycv|{s}")
    bs = np.array([stat([rng.randrange(len(days)) for _ in days]) for _ in range(BOOT)])
    out["cv_obs"] = list(stat(list(range(len(days)))))
    out["cv_ci"] = [bci(bs[:, j]) for j in range(4)]
    return out


# 5 表 5 补全 --------------------------------------------------------------------------------------
T5 = [("一对一", ("一对一",)), ("合并", MERGE), ("拆分", ("拆分",)), ("逐条", ("拆分-逐条任务",)), ("非逐条", ("拆分-非逐条",)), ("缺失", ("缺失",))]


def table5(s, x, items):
    out = {}
    for tag, grp in (("ver", [i for i in items if i["ver"]]), ("all", items)):
        for flag in (True, False):
            g = [i for i in grp if i["cmd"] == flag]
            bs = cluster_boot(g, lambda i: i["s"], f"{SEED}-t5-{s}-{tag}-{flag}")
            out[f"{tag}|{int(flag)}"] = {"n": len(g), "sess": len({i["s"] for i in g}),
                                         **{k: [share(g, lambda i: has(i["labels"], ks)), bci([share(b, lambda i: has(i["labels"], ks)) for b in bs])]
                                            for k, ks in T5}}
    eo = eps_of_commit(x)
    cmd = {i["e"]: i["cmd"] for i in items}
    mixed = [c for c, es in eo.items() if {cmd[e] for e in es if e in cmd} == {True, False}]
    out["mixed_commits"] = len(mixed)
    out["mixed_eps"] = len({e for c in mixed for e in eo[c]})
    gold = command_commits(x["d"], x["cm"])
    files = defaultdict(set)
    for f in load(x["d"] / "commit_files.csv"):
        files[f["hash"]].add(f["path"].replace("\\", "/").lower())
    code_files = defaultdict(set)
    for f in load(x["d"] / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"].replace("\\", "/").lower())
    miss = []
    for i in items:
        if i["ver"] and i["cmd"] and "缺失" in i["labels"]:
            own = [h for h, e in gold.items() if e == i["e"]]
            miss.append({"n": i["n"], "own_commits": len(own), "own_touch": sum(bool(files[h] & code_files[i["e"]]) for h in own),
                         "files": len(code_files[i["e"]])})
    out["cmd_missing"] = miss
    return out


# 6 证据层与样板行 ----------------------------------------------------------------------------------
def tiers(x, items):
    ev = {r["episode"]: r["evidence"] for r in load(x["d"] / "episode_patterns.csv")}
    return dict(Counter(ev[i["e"]] for i in items))


def boilerplate(s, x, items):
    d = x["d"]
    added = commit_added_lines(REPOS[s])
    cnt = defaultdict(set)
    for (h, p), v in added.items():
        for ln in v:
            cnt[ln].add(h)
    ncom = {ln: len(v) for ln, v in cnt.items()}
    ep_of_edit = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_edits.csv") if r["episode"] and not r["reason"]}
    sig = {}
    for line in open(d / "edit_signatures.jsonl", encoding="utf-8"):
        o = json.loads(line)
        sig[(o["session"], str(o["seq"]))] = {v for v in o["lines"] if len(v) >= 15}
    matched = defaultdict(set)
    for a in load(d / "edit_attribution.csv"):
        ep = ep_of_edit.get((a["session"], a["seq"]))
        p = a["path"].replace("\\", "/")
        if ep and a["state"] == "committed" and path_class(p, "1") == "code":
            matched[(ep, a["commit"])] |= sig.get((a["session"], a["seq"]), set()) & added.get((a["commit"], p.lower()), set())
    src = {(r["episode"], r["commit"]): r["source"] for r in load(d / "episode_commit.csv") if r["source"] != "pathtime"}
    code_eps = {i["e"] for i in items}
    out = {"edit_only_pairs": sum(v == "edit" for (e, _), v in src.items() if e in code_eps)}
    for k in (3, 5, 10):
        keep = defaultdict(set)
        dropped = 0
        for (e, c), v in src.items():
            if e not in code_eps:
                continue
            if v == "edit" and not any(ncom.get(ln, 0) < k for ln in matched.get((e, c), ())):
                dropped += 1
                continue
            keep[e].add(c)
        eo = defaultdict(set)
        for e, cs in keep.items():
            for c in cs:
                eo[c].add(e)
        mg = lambda i: any(eo[c] - {i["e"]} for c in keep.get(i["e"], ()))
        out[k] = {"dropped": dropped, "merge_all": share(items, mg), "merge_ver": share([i for i in items if i["ver"]], mg)}
    out["base_all"] = share(items, lambda i: has(i["labels"], MERGE))
    out["base_ver"] = share([i for i in items if i["ver"]], lambda i: has(i["labels"], MERGE))
    return out


# 7 会话内置换检验 ----------------------------------------------------------------------------------
def perm_test(s, grp, flag, keys, tag):
    by = defaultdict(list)
    for i in grp:
        by[i["s"]].append(i)
    mixed = [g for g in by.values() if len({flag(i) for i in g}) == 2]
    pool = [i for g in mixed for i in g]

    def stat(assign):
        a = [i for i in pool if assign[i["e"]]]
        b = [i for i in pool if not assign[i["e"]]]
        return share(a, lambda i: has(i["labels"], keys)) - share(b, lambda i: has(i["labels"], keys))
    obs = stat({i["e"]: flag(i) for i in pool})
    rng = random.Random(f"{SEED}|perm|{s}|{tag}")
    hits = 0
    for _ in range(BOOT):
        assign = {}
        for g in mixed:
            lab = [flag(i) for i in g]
            rng.shuffle(lab)
            assign.update({i["e"]: v for i, v in zip(g, lab)})
        hits += abs(stat(assign)) >= abs(obs) - 1e-12
    return {"sessions": len(mixed), "n": len(pool), "diff": obs, "p": (hits + 1) / (BOOT + 1)}


# 8 图 7 读数 --------------------------------------------------------------------------------------
def lag_read(s):
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    pairs = [p for p in load(d / "episode_commit.csv") if p["source"] != "pathtime"]
    h = lambda t, e: (datetime.fromisoformat(t) - datetime.fromisoformat(eps[e]["last_event"])).total_seconds() / 3600
    lags = [h(p["commit_time"], p["episode"]) for p in pairs]
    last = {}
    for p in pairs:
        last[p["episode"]] = max(last.get(p["episode"], p["commit_time"]), p["commit_time"])
    el = [h(t, e) for e, t in last.items()]
    return {"pairs_gt1h": share(lags, lambda v: v > 1), "last_gt1h": share(el, lambda v: v > 1), "n_pairs": len(lags), "n_eps": len(el)}


def main():
    R, L = {}, ["# v2.14 补充分析（S16-T1，D16）", "", "由 `analysis/robustness5.py` 生成。", ""]
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    xs = {s: base(s) for s in SYSTEMS}
    cmd_share = {s: share(xs[s][1], lambda i: i["cmd"]) for s in SYSTEMS}

    H = human_exclusive()
    if H:
        R["human"] = H
        ov = H["overlap"]
        L += ["## 1 人工估计的子类互斥口径（D16-1）", "",
              f"两类拆分同时标注（主模式一类、次模式另一类或次模式含两类）的片段：原样本 {ov['split'][0]} 个、留出 {ov['split'][1]} 个；"
              f"两类合并同时标注：{ov['merge'][0]} 个、{ov['merge'][1]} 个。", "",
              "| 模式 | " + " | ".join(f"{NAME[s]}：原样本 | 留出 | 加权点估计 | 混合后验中位数（95% 区间）" for s in SYSTEMS) + " |", "|---|" + "---|" * 8]
        for name, _ in H_ROWS:
            cells = []
            for s in SYSTEMS:
                v = H[f"{s}|{name}"]
                cells += [pc(v["orig"]), pc(v["hold"]), pc(v["point"]), f"{pc(v['med'])}（{pc(v['ci'][0])}–{pc(v['ci'][1])}）"]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", "子类按主模式互斥：主模式在该组时取主模式，否则取次模式中该组的一类，次模式同时含两类时计入非逐条 / 跨会话。"
              "加权点估计 = 两批反加权点估计按样本量加权。", ""]

    L += ["## 2 合并子类的优先级与组合下界（D16-2）", "",
          "| 系统 | 单位 | n | 合并（任一共享） | 标为同会话 | 标为跨会话 | 有同会话单位共享 | 其中也有跨会话共享 | 剔除最大 5 个提交 | 有同会话单位共享且剔除最大 5 个 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        M = merge_bounds(s, x, items)
        R[f"mb_{s}"] = M
        for r, v in M.items():
            if r == "top5_sizes":
                continue
            L.append(f"| {NAME[s]} | {RULE_NAME[r]} | {v['n']} | {pc(v['any'])} | {pc(v['label_same'])} | {pc(v['label_cross'])} | {pc(v['same_partner'])} | "
                     f"{pc(v['both_kinds'])} | {pc(v['drop5'])} | {pc(v['same_drop5'])} |")
    L += ["", "分母为全部代码修改单位。标为同会话 / 跨会话按 `patterns.classify` 的优先级：其他单位全部来自本会话记为同会话，否则记为跨会话，二者因此互斥。"
          "“有同会话单位共享”不受优先级影响。最大 5 个提交按片段层所含片段数排序（D15-1）。", ""]

    L += ["## 3 反事实的会计情景（D16-3）", "",
          "| 系统 | 每片段提交数口径 | 不分层 | 按代码编辑数四分位 | 按任务编号数（0 / 1 / 2–3 / ≥ 4） |", "|---|---|---|---|---|"]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        A = accounting(s, x, items, round(cmd_share["SYS_C"], 2), round(cmd_share["SYS_A"], 2))
        R[f"acc_{s}"] = A
        for kn, kname in (("share", "平分（每个提交按所含片段数平分）"), ("count", "各计 1（含该片段内容即计 1）")):
            cells = [f"{A[f'{kn}|{ln}']['ratio']:.2f}（{A[f'{kn}|{ln}']['ci'][0]:.2f}–{A[f'{kn}|{ln}']['ci'][1]:.2f}）" for ln in ("none", "edit", "task")]
            L.append(f"| {NAME[s]} | {kname} | " + " | ".join(cells) + " |")
        lay = "；".join(f"第 {h + 1} 层 n = {r['n']}（片段内提交 {r['n1']}）k₁ = {r['k1']:.2f}、k₀ = {r['k0']:.2f}"
                       if r["k1"] is not None and r["k0"] is not None else f"第 {h + 1} 层 n = {r['n']}（一组为空）" for h, r in enumerate(A["task_layers"]))
        L += ["", f"{NAME[s]} 按任务编号数分层（平分口径）：{lay}。各计 1 口径下 k₁ = {A['k1_count']:.2f}、k₀ = {A['k0_count']:.2f}。", ""]
    L += [f"情景：片段数不变、各层两类片段的每片段提交数不随比例变化，片段内提交比例由系统 C 的 {cmd_share['SYS_C']:.0%} 变为系统 A 的 {cmd_share['SYS_A']:.0%}；"
          "区间按会话重抽 2,000 次。任务编号数取片段 P 中不同（里程碑, 任务）的个数，来自提交自身的标签。", ""]

    L += ["## 4 按天零模型与区间（D16-4）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        D = daily_null(s, x, items)
        R[f"null_{s}"] = D
        f2 = lambda v: f"{v[0]:.2f}–{v[1]:.2f}"
        L.append(f"- {NAME[s]}（{D['days']} 天）：每日分得提交数之和 / 代码修改片段数，中位数 {D['obs_med']:.2f}，变异系数 {D['obs_cv']:.2f}，四分位距 {D['obs_iqr']:.2f}。"
                 f"零模型 N1（保持每日片段数与片段内提交状态）变异系数 95% 范围 {f2(D['N1']['cv_ci'])}（观察值不低于它的比例 p = {D['N1']['p_cv']:.2f}），"
                 f"四分位距 {f2(D['N1']['iqr_ci'])}（p = {D['N1']['p_iqr']:.2f}）；N0（不分状态）变异系数 {f2(D['N0']['cv_ci'])}（p = {D['N0']['p_cv']:.2f}）。"
                 f"D15-3 的变异系数（片段内 / 其余 / 全部）{' / '.join(f'{v:.2f}' for v in D['cv_obs'][:3])}，按日重抽区间 "
                 f"{' / '.join(f2(v) for v in D['cv_ci'][:3])}；两部分相关 {D['cv_obs'][3]:.2f}（{f2(D['cv_ci'][3])}）。")
    L.append("")

    L += ["## 5 表 5 补全（D16-5）", "", "| 系统·分母 | 组 | n（会话） | 一对一 | 合并 | 拆分 | 逐条 | 非逐条 | 缺失 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        T = table5(s, x, items)
        R[f"t5_{s}"] = T
        for tag, tn in (("ver", "可核验"), ("all", "全部代码修改")):
            for flag, gn in ((1, "片段内提交"), (0, "片段内未提交")):
                v = T[f"{tag}|{flag}"]
                cells = [f"{pc(v[k][0])}（{pc(v[k][1][0])}–{pc(v[k][1][1])}）" for k, _ in T5]
                L.append(f"| {NAME[s]}·{tn} | {gn} | {v['n']}（{v['sess']}） | " + " | ".join(cells) + " |")
    for s in SYSTEMS:
        T = R[f"t5_{s}"]
        ms = "；".join(f"代码编辑 {m['n']} 次、编辑代码文件 {m['files']} 个、本片段命令产生提交 {m['own_commits']} 个（改动其代码文件的 {m['own_touch']} 个）"
                       for m in T["cmd_missing"]) or "无"
        L.append(f"\n- {NAME[s]}：同时含两组片段内容的提交 {T['mixed_commits']} 个（涉及片段 {T['mixed_eps']} 个）。可核验、片段内提交且判为缺失的片段 {len(T['cmd_missing'])} 个：{ms}。")
    L.append("")

    L += ["## 6 证据层与样板行（D16-6）", ""]
    for s in SYSTEMS:
        x, items, _ = xs[s]
        tv = tiers(x, items)
        B = boilerplate(s, x, items)
        R[f"tiers_{s}"], R[f"boiler_{s}"] = tv, B
        bl = "；".join(f"k = {k}：去除 {B[k]['dropped']} 对，合并 可核验 {pc(B[k]['merge_ver'])}、全部 {pc(B[k]['merge_all'])}" for k in (3, 5, 10))
        L.append(f"- {NAME[s]}：可核验 {tv.get('verified', 0)}、部分可核验 {tv.get('partial', 0)}、推定 {tv.get('inferred', 0)}。"
                 f"只由逐次编辑证据支撑的片段—提交对 {B['edit_only_pairs']} 对；样板行过滤（在 ≥ k 个提交中出现的特征行）后，{bl}（主口径 {pc(B['base_ver'])}、{pc(B['base_all'])}）。")
    L.append("")

    L += ["## 7 会话内置换检验（D16-7）", ""]
    xa, ia, _ = xs["SYS_A"]
    va = [i for i in ia if i["ver"]]
    for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并")):
        P = perm_test("SYS_A", va, lambda i: not i["reg"], keys, f"reg-{kn}")
        R[f"perm_A_reg_{kn}"] = P
        L.append(f"- 系统 A 可核验片段，事前登记的会话内比较（未登记 − 登记，只用同时含两类的 {P['sessions']} 个会话、{P['n']} 个片段）：{kn}差 {P['diff'] * 100:+.0f} 个百分点，置换 p = {P['p']:.3f}。")
    xc, ic, _ = xs["SYS_C"]
    for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并")):
        P = perm_test("SYS_C", ic, lambda i: i["cmd"], keys, f"cmd-{kn}")
        R[f"perm_C_cmd_{kn}"] = P
        L.append(f"- 系统 C 全部代码修改片段，片段内提交与否（只用同时含两类的 {P['sessions']} 个会话、{P['n']} 个片段）：{kn}差 {P['diff'] * 100:+.0f} 个百分点，置换 p = {P['p']:.3f}。")
    for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并")):
        P = perm_test("SYS_A", ia, lambda i: i["cmd"], keys, f"cmdA-{kn}")
        R[f"perm_A_cmd_{kn}"] = P
        L.append(f"- 系统 A 全部代码修改片段，片段内提交与否（会话内置换，{P['sessions']} 个会话、{P['n']} 个片段）：{kn}差 {P['diff'] * 100:+.0f} 个百分点，置换 p = {P['p']:.3f}。")
    L += ["", "会话内置换：在同时含两类片段的会话内打乱标签 2,000 次，统计量为两组比例之差，双侧 p =（|置换差| ≥ |观察差| 的次数 + 1）/ 2,001。", ""]

    L += ["## 8 图 7 读数（D16-8）", ""]
    for s in SYSTEMS:
        G = lag_read(s)
        R[f"lag_{s}"] = G
        L.append(f"- {NAME[s]}：片段—提交对 {G['n_pairs']} 个中晚于 1 小时 {pc(G['pairs_gt1h'])}（实线）；片段末次内容提交 {G['n_eps']} 个中晚于 1 小时 {pc(G['last_gt1h'])}（虚线）。")
    L.append("")

    (RES / "r13_robustness5.md").write_text("\n".join(L), encoding="utf-8")
    num["robustness5"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
