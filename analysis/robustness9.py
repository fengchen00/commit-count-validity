"""分期比较的时间结构与按日计数模型（S18-T4，口径见 notes/decisions.md D21）。

1 事前登记分期高、低两类合格日的精确置换检验
2 时间聚集：游程检验与按 ISO 周分块重抽
3 按日泊松回归：日提交数对日片段数、日代码编辑数与事前登记比例的弹性
4 事前登记比例与每片段提交数的偏相关（控制每片段代码编辑数）
写入 <PAPER_DIR>/results/r17_robustness9.md 与 numbers.json 的 robustness9 键（须在 robustness7.py 之后运行）。
"""
import datetime as dt
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from robustness7 import BOOT, NUMS, RES, agg, bci, days, load_ctx, spearman, split  # noqa: E402

SEED = "S18d"


def ratios(H, Lo):
    a, b = agg(H), agg(Lo)
    return a["c_ep"] / b["c_ep"], a["c_100w"] / b["c_100w"]


def permutation(H, Lo):
    obs = ratios(H, Lo)
    pool = H + Lo
    cnt, tot = [0, 0], 0
    for idx in itertools.combinations(range(len(pool)), len(H)):
        s = set(idx)
        r = ratios([pool[i] for i in s], [pool[i] for i in range(len(pool)) if i not in s])
        tot += 1
        for k in (0, 1):
            cnt[k] += abs(math.log(r[k])) >= abs(math.log(obs[k])) - 1e-12
    return {"obs": obs, "p": [c / tot for c in cnt], "n_perm": tot}


def runs(seq):
    return 1 + sum(a != b for a, b in zip(seq, seq[1:]))


def time_adjacent(ds, rng, n_mc=20000):
    """D32 B2：保留时间邻近结构的零分布。ds 为全部合格日（含中间一类），按日期排序；
    置换暴露向量 g（事前登记比例），结果不动，每次按固定阈值 2/3、1/3 重新分组并算 |log 比|（每片段、每百次代码编辑）。
    (a) 循环移位：g 相对日期整体平移 k = 1..n-1；(b) ISO 周内置换：只在同一周的日子之间交换 g，蒙特卡洛。p 按 (k+1)/(B+1)。"""
    ds = sorted(ds, key=lambda d: d["d"])
    n = len(ds)
    g = [d["g"] for d in ds]

    def stat(gs):
        H = [d for d, x in zip(ds, gs) if x >= 2 / 3 - 1e-9]
        Lo = [d for d, x in zip(ds, gs) if x <= 1 / 3 + 1e-9]
        if not H or not Lo:
            return None
        r = ratios(H, Lo)
        return [abs(math.log(r[0])), abs(math.log(r[1]))]

    obs = stat(g)
    cyc = [s for s in (stat(g[k:] + g[:k]) for k in range(1, n)) if s is not None]
    ex = [sum(s[i] >= obs[i] - 1e-12 for s in cyc) for i in (0, 1)]
    wk = defaultdict(list)
    for i, d in enumerate(ds):
        wk[dt.date.fromisoformat(d["d"]).isocalendar()[:2]].append(i)
    cnt, tot = [0, 0], 0
    for _ in range(n_mc):
        gs = list(g)
        for idx in wk.values():
            vals = [g[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                gs[i] = v
        s = stat(gs)
        if s is None:
            continue
        tot += 1
        for i in (0, 1):
            cnt[i] += s[i] >= obs[i] - 1e-12
    return {"n_days": n, "obs": obs, "cyclic_valid": len(cyc), "cyclic_exceed": ex, "p_cyclic": [(e + 1) / (len(cyc) + 1) for e in ex],
            "n_weeks": len(wk), "week_sizes": [len(v) for v in wk.values()], "n_mc": n_mc, "within_valid": tot,
            "within_exceed": cnt, "p_within": [(c + 1) / (tot + 1) for c in cnt]}


def time_structure(H, Lo, rng):
    lab = sorted([(d["d"], "H") for d in H] + [(d["d"], "L") for d in Lo])
    seq = [x for _, x in lab]
    r_obs = runs(seq)
    n, k = len(seq), len(H)
    le, tot = 0, 0
    for idx in itertools.combinations(range(n), k):
        s = set(idx)
        tot += 1
        le += runs(["H" if i in s else "L" for i in range(n)]) <= r_obs
    wk = defaultdict(lambda: ([], []))
    for d in H:
        wk[dt.date.fromisoformat(d["d"]).isocalendar()[:2]][0].append(d)
    for d in Lo:
        wk[dt.date.fromisoformat(d["d"]).isocalendar()[:2]][1].append(d)
    W = list(wk.values())
    bs = [[], []]
    for _ in range(BOOT):
        pick = [rng.choice(W) for _ in W]
        h = [d for w in pick for d in w[0]]
        lo = [d for w in pick for d in w[1]]
        if h and lo:
            r = ratios(h, lo)
            bs[0].append(r[0])
            bs[1].append(r[1])
    return {"sequence": "".join(seq), "runs": r_obs, "p_runs_le": le / tot, "n_weeks": len(W),
            "weeks": [[len(a), len(b)] for a, b in W], "block_ci": [bci(bs[0]), bci(bs[1])], "block_valid": len(bs[0])}


def poisson(X, y, iters=100):
    b = np.zeros(X.shape[1])
    b[0] = math.log(max(y.mean(), 1e-9))
    for _ in range(iters):
        mu = np.exp(X @ b)
        z = X @ b + (y - mu) / mu
        W = mu
        nb = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        if np.max(np.abs(nb - b)) < 1e-10:
            b = nb
            break
        b = nb
    return b


def count_model(ds, rng, trend=False):
    def design(sub):
        cols = [np.ones(len(sub)), np.log([d["n"] for d in sub]), np.log([d["w"] for d in sub]), np.array([d["g"] for d in sub])]
        if trend:
            base = dt.date.fromisoformat(ds[0]["d"])
            cols.append(np.array([(dt.date.fromisoformat(d["d"]) - base).days / 7 for d in sub]))
        return np.column_stack(cols), np.array([d["c"] for d in sub], dtype=float)
    X, y = design(ds)
    b = poisson(X, y)
    mu = np.exp(X @ b)
    disp = float(((y - mu) ** 2 / mu).sum() / (len(y) - X.shape[1]))
    bs = []
    for _ in range(BOOT):
        sub = [rng.choice(ds) for _ in ds]
        Xs, ys = design(sub)
        try:
            bs.append(poisson(Xs, ys))
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    names = ["截距", "log 日片段数", "log 日代码编辑数", "事前登记比例"] + (["每周趋势"] if trend else [])
    lodo = []
    for j in range(len(ds)):
        Xj, yj = design(ds[:j] + ds[j + 1:])
        lodo.append(poisson(Xj, yj))
    lodo = np.array(lodo)
    return {"coef": dict(zip(names, map(float, b))), "ci": {nm: bci(list(bs[:, j])) for j, nm in enumerate(names)},
            "lodo": {nm: [float(lodo[:, j].min()), float(lodo[:, j].max())] for j, nm in enumerate(names)},
            "dispersion": disp, "n_boot": len(bs), "n_days": len(ds)}


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


def partial(ds, rng):
    def pc(sub):
        x, y, z = (np.array(rank(v)) for v in ([d["g"] for d in sub], [d["c"] / d["n"] for d in sub], [d["w"] / d["n"] for d in sub]))
        Z = np.column_stack([np.ones(len(sub)), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])
    bs = [pc([rng.choice(ds) for _ in ds]) for _ in range(BOOT)]
    bs = [v for v in bs if v == v]
    return {"rho": pc(ds), "ci": bci(bs),
            "rho_g_size": spearman([d["g"] for d in ds], [d["w"] / d["n"] for d in ds])}


def load_csv(p):
    import csv
    return list(csv.DictReader(open(p, encoding="utf-8")))


def commit_types(ctx, H, Lo):
    """提交按内容分类：含代码文件（mixed / impl）与只改任务书（plan）；是否含任一片段的内容。"""
    cm = load_csv(Path("data/derived/SYS_A/commits.csv"))
    content = set()
    for p in load_csv(Path("data/derived/SYS_A/episode_commit.csv")):
        if p["source"] != "pathtime":
            content.add(p["commit"])
    tot = defaultdict(int)
    for c in cm:
        tot[(c["kind"] == "plan", c["hash"] in content)] += 1
    hd, ld = {d["d"] for d in H}, {d["d"] for d in Lo}
    grp = {"H": defaultdict(int), "L": defaultdict(int)}
    code_day = defaultdict(int)
    for c in cm:
        day = c["time"][:10]
        if c["kind"] != "plan":
            code_day[day] += 1
        for g, dd in (("H", hd), ("L", ld)):
            if day in dd:
                grp[g]["all"] += 1
                grp[g]["plan"] += c["kind"] == "plan"
                grp[g]["nocontent"] += c["hash"] not in content
    return {"n": len(cm), "plan": sum(v for (p, _), v in tot.items() if p), "nocontent": sum(v for (_, h), v in tot.items() if not h),
            "nocontent_plan": tot[(True, False)], "nocontent_code": tot[(False, False)],
            "by_group": {g: dict(v) for g, v in grp.items()}}, code_day


def with_code_commits(ds, code_day):
    return [dict(d, c=code_day.get(d["d"], 0)) for d in ds]


def episode_model(ctx, rng):
    """片段层：片段按内容分得的提交数（平分口径）对 log 代码编辑数与事前登记的泊松回归，按会话整块重抽。"""
    its = [i for i in ctx["items"] if i["n"] > 0]

    def design(sub):
        X = np.column_stack([np.ones(len(sub)), np.log([i["n"] for i in sub]), np.array([float(bool(i["reg"])) for i in sub])])
        return X, np.array([i["share"] for i in sub])
    X, y = design(its)
    b = poisson(X, y)
    by_s = defaultdict(list)
    for i in its:
        by_s[i["s"]].append(i)
    S = list(by_s.values())
    bs = []
    for _ in range(BOOT):
        sub = [i for g in (rng.choice(S) for _ in S) for i in g]
        try:
            bs.append(poisson(*design(sub)))
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    names = ["截距", "log 代码编辑数", "事前登记"]
    return {"coef": dict(zip(names, map(float, b))), "ci": {nm: bci(list(bs[:, j])) for j, nm in enumerate(names)},
            "n_eps": len(its), "n_sessions": len(S), "n_boot": len(bs)}


def system_compare(rng):
    """系统比较：分子与分母都只计有代码修改片段的日子；各单位的 A / C 日均量之比按日重抽（两个系统各自重抽）。"""
    from robustness8 import UNITS, unit_counts
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    bl = num["robustness6"]["boundary_labels"]
    per = {}
    for s in ("SYS_A", "SYS_C"):
        nw = bl[f"{s}|final|rates"]["new_works"]
        ctx = load_ctx(s, {True: nw[0], False: nw[1]})
        unit_counts(s, ctx)
        dd = defaultdict(lambda: defaultdict(float))
        for i in ctx["items"]:
            for k, _ in UNITS:
                dd[i["start"][:10]][k] += i["u"][k]
        for d in dd:
            dd[d]["commits"] = ctx["cday"].get(d, 0)
        per[s] = {"days": [dict(v) for v in dd.values()], "commits_all_window": ctx["n_commits"]}
    keys = ["commits"] + [k for k, _ in UNITS]
    mean = lambda ds, k: sum(d[k] for d in ds) / len(ds)
    out = {}
    for k in keys:
        a, c = per["SYS_A"]["days"], per["SYS_C"]["days"]
        est = mean(a, k) / mean(c, k)
        bs = []
        for _ in range(BOOT):
            x = [rng.choice(a) for _ in a]
            z = [rng.choice(c) for _ in c]
            if mean(z, k):
                bs.append(mean(x, k) / mean(z, k))
        out[k] = {"ratio": est, "ci": bci(bs)}
    out["n_days"] = {s: len(per[s]["days"]) for s in per}
    out["commits_on_active"] = {s: sum(d["commits"] for d in per[s]["days"]) for s in per}
    out["commits_all"] = {s: per[s]["commits_all_window"] for s in per}
    return out


def maxstat(ctx, rng, n_perm=200000):
    """D27-6：同一数据上的多重比较族的 max-statistic 置换（Westfall–Young 单步）。

    可交换单位为合格日（min_eps=3、无窗口末删失、无规则日过滤，即主口径的 23 天）。置换对象是各日的
    暴露向量（事前登记比例 reg、任务编号比例 tref、片段内提交比例 cmd）整体，结果（日提交数 c、
    按片段归属的日提交数、日片段数 n、日编辑数 w、各工作单位日计数）固定不动。族内每个变体是
    "分期变量 × 归属口径 × 分组方式 × 工作单位"的一个组合，统计量为 |log(高/低)|。族只含能在同一日集
    上算的变体；合格日门槛 2/5、删失、规则日、仅版本库（git_task，日集不同）四类变体不入族，另列为
    不同数据上的敏感性。头条（reg × 按时间 × 2/3–1/3 × 每片段）的校正 p = P(max_j T_j^π ≥ T_头条^obs)。
    """
    from robustness8 import UNITS, unit_counts
    unit_counts("SYS_A", ctx)
    base_days = days(ctx, by="reg")
    coh_days = {d["d"]: d["c"] for d in days(ctx, by="reg", attr="cohort")}
    unit_day = {k: defaultdict(float) for k, _ in UNITS}
    for i in ctx["items"]:
        for k, _ in UNITS:
            unit_day[k][i["start"][:10]] += i["u"][k]
    D = [dict(d=d["d"], n=d["n"], w=d["w"], c=d["c"], c_coh=coh_days.get(d["d"], float("nan")),
              x={"reg": d["reg"], "tref": d["tref"] if "tref" in d else d["n_tref"] / d["n"], "cmd": d["cmd"]},
              u={k: unit_day[k][d["d"]] for k, _ in UNITS}) for d in base_days]
    n = len(D)

    def stat(days_, by, attr, sp, unit):
        g = [dd["x"][by] for dd in days_]
        if sp == "median":
            m = sorted(g)[n // 2]
            H = [dd for dd, gv in zip(days_, g) if gv >= m and gv > 0]
            L = [dd for dd, gv in zip(days_, g) if gv < m or gv == 0]
        else:
            H = [dd for dd, gv in zip(days_, g) if gv >= 2 / 3 - 1e-9]
            L = [dd for dd, gv in zip(days_, g) if gv <= 1 / 3 + 1e-9]
        if not H or not L:
            return float("nan"), (len(H), len(L))
        cf = (lambda dd: dd["c"]) if attr == "time" else (lambda dd: dd["c_coh"])
        den = (lambda G: sum(dd["n"] for dd in G)) if unit == "episodes" else \
              (lambda G: sum(dd["w"] for dd in G) / 100) if unit == "edits100" else (lambda G: sum(dd["u"][unit] for dd in G))
        a, b = sum(map(cf, H)) / den(H), sum(map(cf, L)) / den(L)
        if not a or not b or a != a or b != b:
            return float("nan"), (len(H), len(L))
        return abs(math.log(a / b)), (len(H), len(L))

    family = []
    for by in ("reg", "tref", "cmd"):
        for attr in ("time", "cohort"):
            for sp in ("tertile", "median"):
                for unit in ("episodes", "edits100") + tuple(k for k, _ in UNITS if k not in ("episodes", "edits")):
                    family.append((by, attr, sp, unit))
    obs = {}
    for v in family:
        t, hl = stat(D, *v)
        obs[v] = (t, hl)
    head = ("reg", "time", "tertile", "episodes")
    t_head = obs[head][0]
    valid = [v for v in family if obs[v][0] == obs[v][0]]
    hi = valid.index(head)
    xs = [dd["x"] for dd in D]
    # T[b, j]：第 b 次置换下第 j 个变体的统计量；不可算（某组为空）记 -inf，不参与取最大、p 值计 1
    T = np.full((n_perm, len(valid)), -np.inf)
    for b in range(n_perm):
        perm = xs[:]
        rng.shuffle(perm)
        P = [dict(dd, x=px) for dd, px in zip(D, perm)]
        for j, v in enumerate(valid):
            t = stat(P, *v)[0]
            if t == t:
                T[b, j] = t
    t_obs = np.array([obs[v][0] for v in valid])
    exceed_each = (T >= t_obs[None, :] - 1e-12).sum(axis=0)
    p_raw = exceed_each / n_perm
    exceed_raw = int(exceed_each[hi])
    exceed_head = int((T.max(axis=1) >= t_head - 1e-12).sum())
    # 单步 min-P（D28-1）：每个变体在自身置换分布中的 p 值 p_bj = #{b': T_b'j >= T_bj} / n_perm，
    # 校正 p = P(min_j p_bj <= p_头条^obs)。对各单位不同尺度的统计量作了标准化，回应"跨单位是否可比"。
    Pm = np.ones_like(T)
    for j in range(len(valid)):
        col = T[:, j]
        ok = np.isfinite(col)
        if ok.any():
            srt = np.sort(col[ok])
            # 大于等于的个数 = n_ok - (严格小于的个数)
            ge = ok.sum() - np.searchsorted(srt, col[ok] - 1e-12, side="left")
            Pm[ok, j] = ge / n_perm
    p_head_obs = p_raw[hi]
    exceed_minp = int((Pm.min(axis=1) <= p_head_obs + 1e-12).sum())

    def ci(k):
        q = k / n_perm
        h = 1.96 * math.sqrt(max(q * (1 - q), 0) / n_perm)
        return [max(0.0, q - h), min(1.0, q + h)]
    per_variant = [{"by": v[0], "attr": v[1], "split": v[2], "unit": v[3], "t_obs": obs[v][0],
                    "ratio": math.exp(obs[v][0]), "n_hi_lo": list(obs[v][1]), "p_raw": float(p_raw[j])} for j, v in enumerate(valid)]
    return {"n_days": n, "n_perm": n_perm, "family_size": len(valid), "family_defined": len(family),
            "headline": {"by": head[0], "attr": head[1], "split": head[2], "unit": head[3], "t_obs": t_head, "ratio": math.exp(t_head)},
            "p_raw_exposure_perm": exceed_raw / n_perm, "p_raw_exceed": exceed_raw, "p_raw_ci95": ci(exceed_raw),
            "p_maxstat": exceed_head / n_perm, "p_maxstat_ci95": ci(exceed_head),
            "p_minp": exceed_minp / n_perm, "p_minp_ci95": ci(exceed_minp),
            "per_variant": per_variant, "n_ratio_gt1": sum(v["ratio"] > 1 for v in per_variant),
            "n_praw_lt05": sum(v["p_raw"] < 0.05 for v in per_variant),
            "excluded_variants": ["合格日门槛 2", "合格日门槛 5", "删去窗口末 48 小时（两种分期）", "只用规则进入版本库之后（两种分期）", "仅用版本库 git_task"]}


def main():
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    bl = num["robustness6"]["boundary_labels"]
    nw = bl["SYS_A|final|rates"]["new_works"]
    ctx = load_ctx("SYS_A", {True: nw[0], False: nw[1]})
    ds = days(ctx, by="reg")
    H, Lo = split(ds, 2 / 3, 1 / 3)
    ct, code_day = commit_types(ctx, H, Lo)
    dsc = with_code_commits(ds, code_day)
    Hc, Loc = split(dsc, 2 / 3, 1 / 3)
    R = {"perm": permutation(H, Lo), "time": time_structure(H, Lo, random.Random(f"{SEED}|wk")),
         "model": count_model(ds, random.Random(f"{SEED}|m")), "model_trend": count_model(ds, random.Random(f"{SEED}|mt"), trend=True),
         "model_hilo": count_model(H + Lo, random.Random(f"{SEED}|mh")),
         "partial": partial(ds, random.Random(f"{SEED}|pc")),
         "commit_types": ct, "code_only": {"ratios": ratios(Hc, Loc), "model": count_model(dsc, random.Random(f"{SEED}|mc"))},
         "episode_model": episode_model(ctx, random.Random(f"{SEED}|ep")),
         "model_cohort": count_model(days(ctx, by="reg", attr="cohort"), random.Random(f"{SEED}|mco")),
         "system": system_compare(random.Random(f"{SEED}|sys"))}
    p, t, pc = R["perm"], R["time"], R["partial"]
    L = ["# 第八轮审稿补充分析：分期比较的时间结构与按日计数模型（S18-T4，D21）", "", "由 `analysis/robustness9.py` 生成。系统 A 合格日。", "",
         "## 1 精确置换检验（高 9 天 / 低 7 天）", "",
         f"- 每片段提交数之比 {p['obs'][0]:.2f}，双侧 p = {p['p'][0]:.4f}；每百次代码编辑 {p['obs'][1]:.2f}，p = {p['p'][1]:.3f}（全部 {p['n_perm']} 种分组）。", "",
         "## 2 时间聚集", "",
         f"- 按日期排列的高 / 低序列：{t['sequence']}；游程数 {t['runs']}，全部分组中游程数不多于此的比例 {t['p_runs_le']:.3f}。",
         f"- 按 ISO 周分块重抽（{t['n_weeks']} 周，每周高 / 低天数 {t['weeks']}；有效重抽 {t['block_valid']} 次）：每片段之比区间 {t['block_ci'][0][0]:.2f}–{t['block_ci'][0][1]:.2f}，"
         f"每百次代码编辑 {t['block_ci'][1][0]:.2f}–{t['block_ci'][1][1]:.2f}。", "",
         "## 3 按日泊松回归（日提交数）", ""]
    for key, name in (("model", "主模型"), ("model_trend", "加每周趋势"), ("model_hilo", "只用高、低两类日子"),
                      ("model_cohort", "按片段归属提交（不含 436 个无片段内容的提交）")):
        m = R[key]
        L.append(f"- {name}（{m['n_days']} 天，离散度 {m['dispersion']:.1f}，有效重抽 {m['n_boot']} 次）：" + "；".join(
            f"{k} {v:.2f}（{m['ci'][k][0]:.2f}–{m['ci'][k][1]:.2f}；逐日删除 {m['lodo'][k][0]:.2f}–{m['lodo'][k][1]:.2f}）"
            for k, v in m["coef"].items() if k != "截距") + "。")
    L += ["", "## 4 偏相关", "",
          f"- 当日事前登记比例与每片段提交数的 Spearman 偏相关（控制每片段代码编辑数）= {pc['rho']:.2f}（{pc['ci'][0]:.2f}–{pc['ci'][1]:.2f}）；"
          f"事前登记比例与每片段代码编辑数的 ρ = {pc['rho_g_size']:.2f}。", ""]
    ct, co, em, sy = R["commit_types"], R["code_only"], R["episode_model"], R["system"]
    m = co["model"]
    L += ["## 5 提交的构成", "",
          f"- 系统 A 共 {ct['n']} 个提交，其中只改任务书的 {ct['plan']} 个；不含任何片段内容的 {ct['nocontent']} 个（只改任务书 {ct['nocontent_plan']}、含代码文件 {ct['nocontent_code']}）。",
          f"- 高 / 低两类日子的提交：{ct['by_group']['H'].get('all', 0)} / {ct['by_group']['L'].get('all', 0)}，其中只改任务书 {ct['by_group']['H'].get('plan', 0)} / {ct['by_group']['L'].get('plan', 0)}，"
          f"不含片段内容 {ct['by_group']['H'].get('nocontent', 0)} / {ct['by_group']['L'].get('nocontent', 0)}。",
          f"- 只计含代码文件的提交：每片段提交数之比 {co['ratios'][0]:.2f}，每百次代码编辑 {co['ratios'][1]:.2f}；按日泊松回归 " + "；".join(
              f"{k} {v:.2f}（{m['ci'][k][0]:.2f}–{m['ci'][k][1]:.2f}）" for k, v in m["coef"].items() if k != "截距") + "。", "",
          "## 6 片段层模型", "",
          f"- {em['n_eps']} 个代码修改片段、{em['n_sessions']} 个会话；片段按内容分得的提交数（平分口径）的泊松回归，按会话整块重抽 {em['n_boot']} 次：" + "；".join(
              f"{k} {v:.2f}（{em['ci'][k][0]:.2f}–{em['ci'][k][1]:.2f}）" for k, v in em["coef"].items() if k != "截距") + "。", "",
          "## 7 系统比较（分子与分母都只计有代码修改片段的日子）", "",
          f"- 天数 A {sy['n_days']['SYS_A']}、C {sy['n_days']['SYS_C']}；这些日子上的提交 A {sy['commits_on_active']['SYS_A']:.0f} / 全窗口 {sy['commits_all']['SYS_A']}，"
          f"C {sy['commits_on_active']['SYS_C']:.0f} / {sy['commits_all']['SYS_C']}。"]
    for k, v in sy.items():
        if isinstance(v, dict) and "ratio" in v:
            L.append(f"  - {k}：{v['ratio']:.2f}（{v['ci'][0]:.2f}–{v['ci'][1]:.2f}）")
    L.append("")
    # D26-5：比较次数与头条 p 值的最坏情形调整。同一"高对低"比较在各种口径与单位下重复计算，属敏感性而非独立假设，
    # 但按最保守的口径把它们全部计入 Bonferroni 族。
    r7 = num.get("robustness7", {})
    rows13 = [k for k, v in r7.items() if k.startswith("SYS_A|") and "|agree|" not in k and isinstance(v, dict) and "ratio" in v]  # D32-3 一致子样本行不入族
    m13 = 3 * len(rows13)          # 表 13：每行 3 个指标（每片段、每百次编辑、每项工作）
    m14 = 9                        # 表 14：九种工作单位的分期比较
    m_models = sum(len([k for k in R[key]["coef"] if k != "截距"]) for key in ("model", "model_trend", "model_hilo", "model_cohort", "episode_model")) + \
        len([k for k in R["code_only"]["model"]["coef"] if k != "截距"])
    p0 = p["p"][0]
    fam = {"perm_stats": 2, "table13": m13, "table14": m14, "models": m_models}
    m_same = 2 + m13 + m14
    m_all = m_same + m_models
    R["multiplicity"] = {"p_raw": p0, "family": fam, "m_same_comparison": m_same, "m_all": m_all,
                         "p_bonf_same": min(1.0, p0 * m_same), "p_bonf_all": min(1.0, p0 * m_all)}
    L += ["## 8 比较次数与多重比较调整（D26-5）", "",
          f"- 头条置换检验 p = {p0:.4f} 为事后检验（正文 2.4.1 节）。同一高 / 低比较的重复口径：置换检验统计量 2 个，表 13 {len(rows13)} 行 × 3 个指标 = {m13} 个，"
          f"表 14 九种单位 {m14} 个，合计 {m_same} 次；另有本文件第 3、5、6 节计数模型的系数 {m_models} 个。",
          f"- 最坏情形 Bonferroni 调整：按同一比较的 {m_same} 次口径，p × m = {min(1.0, p0 * m_same):.3f}；把全部 {m_all} 次都计入，p × m = {min(1.0, p0 * m_all):.3f}。"
          "这些口径高度相关，Bonferroni 过于保守，只作上界；结论应以效应量与区间为主。", ""]
    # D27-6：max-statistic 置换（Westfall–Young 单步），把同一日集上的全部变体作为一个族，校正头条 p。
    ms = maxstat(ctx, random.Random(f"{SEED}|maxstat"))
    R["maxstat"] = ms
    top = sorted(ms["per_variant"], key=lambda v: -v["t_obs"])[:5]
    L += ["## 9 主检验的置换 p 与族校正（D27-6、D28-1）", "",
          f"- 可交换单位为 {ms['n_days']} 个合格日；置换各日的三个分期变量（事前登记、任务编号、片段内提交）整体，结果不动；{ms['n_perm']} 次置换。"
          "D28-1 起这是论文的主检验口径：原始 p、max-statistic 与 min-P 三者在同一批置换上算；第 1 节固定标签的 16 天精确枚举只作记录。",
          f"- 族：分期变量 3 × 归属口径 2（按时间 / 按片段归属）× 分组 2（三分位 / 中位）× 分母 9（每片段、每百次编辑、七种其他工作单位），"
          f"定义 {ms['family_defined']} 个变体，其中 {ms['family_size']} 个两组非空可算；{ms['n_ratio_gt1']} 个的高 / 低之比大于 1，{ms['n_praw_lt05']} 个的原始 p < 0.05。不在同一日集上的变体（{'；'.join(ms['excluded_variants'])}）不入族，另作敏感性。",
          f"- 头条（事前登记 × 按时间 × 三分位 × 每片段）：|log 比| = {ms['headline']['t_obs']:.3f}（比 {ms['headline']['ratio']:.2f}）；"
          f"原始 p = {ms['p_raw_exposure_perm']:.4f}（{ms['p_raw_exceed']} / {ms['n_perm']}；与第 1 节不同：第 1 节固定高 / 低标签在 16 天上重新分组，这里在 23 天上置换整条暴露向量）；"
          f"max-statistic 校正 p = {ms['p_maxstat']:.3f}（蒙特卡洛 95% 区间 {ms['p_maxstat_ci95'][0]:.3f}–{ms['p_maxstat_ci95'][1]:.3f}）；"
          f"min-P 校正 p = {ms['p_minp']:.4f}（{ms['p_minp_ci95'][0]:.4f}–{ms['p_minp_ci95'][1]:.4f}）。"
          "两种校正的差别来自统计量尺度：max-statistic 直接比较 |log 比|，日计数少的工作单位比值波动大，常占据族内最大值，对头条偏保守；min-P 先把各变体换成各自置换分布中的 p 值再取最小，尺度一致。",
          "- 族内 |log 比| 最大的五个变体：" + "；".join(f"{v['by']}×{v['attr']}×{v['split']}×{v['unit']} 比 {v['ratio']:.2f}（原始 p {v['p_raw']:.3f}）" for v in top) + "。", ""]
    # D32 B2：保留时间邻近结构的两种零分布（循环移位、ISO 周内置换），作为第 9 节完全置换的对照；独立随机流，不改上文任何数字。
    ta = time_adjacent(ds, random.Random(f"{SEED}|ta"))
    R["time_adjacent"] = ta
    L += ["## 10 保留时间结构的置换（D32 B2）", "",
          f"- 对象与第 9 节相同：{ta['n_days']} 个合格日按日期排列，置换事前登记比例这一暴露向量，提交数与工作量不动，按固定阈值重新分组后算高 / 低之比。"
          f"观察值 |log 比|：每片段 {ta['obs'][0]:.3f}（比 {math.exp(ta['obs'][0]):.2f}），每百次代码编辑 {ta['obs'][1]:.3f}（比 {math.exp(ta['obs'][1]):.2f}）。",
          f"- 循环移位（暴露向量相对日期整体平移 1–{ta['n_days'] - 1} 日，保留其自相关结构；两组非空的移位 {ta['cyclic_valid']} 个）："
          f"|log 比| 不小于观察值的移位数 每片段 {ta['cyclic_exceed'][0]}，每百次编辑 {ta['cyclic_exceed'][1]}；按 (k+1)/(B+1) 计 p = {ta['p_cyclic'][0]:.3f} 与 {ta['p_cyclic'][1]:.3f}。",
          f"- ISO 周内置换（只在同一周的日子之间交换暴露值，{ta['n_weeks']} 周，每周天数 {ta['week_sizes']}；蒙特卡洛 {ta['n_mc']} 次，两组非空 {ta['within_valid']} 次）："
          f"p = {ta['p_within'][0]:.4f}（每片段）与 {ta['p_within'][1]:.3f}（每百次编辑）。",
          "- 两种零分布可取的重排都少于第 9 节的完全置换：循环移位只有 22 个候选，p 的分辨率为 1/23；周内置换保留周间差异，只打乱周内顺序。", ""]
    (RES / "r17_robustness9.md").write_text("\n".join(L), encoding="utf-8")
    num["robustness9"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
