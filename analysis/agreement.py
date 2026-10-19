"""编码一致性的统一报告（S19-T-E2，D26-5）：Cohen κ 连同观察一致率 P_o、两名编码人的边缘分布与 95% 自助区间。

独立盲审（2026-09-30）指出全文八个 κ 无一给区间，其中 n = 20 的 κ = 0.71 两位小数是虚假精度；粗类 κ 低于细类 κ
是边缘分布偏斜所致的 κ 悖论，须连边缘分布一起报告。各脚本在打印 κ 处改调本模块。
"""
import random
from collections import Counter

BOOT = 2000


def cohen(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    cats = {v for p in pairs for v in p}
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), po


def kappa_stats(a, b, seed="κ", boot=BOOT, clusters=None):
    """a、b 为两名编码人的标签序列（None 跳过）。返回 k、po、n、ci、marg_a、marg_b、pe。
    clusters 给出时（与 a、b 等长，如会话标识），自助区间按簇整块重抽（D32：相邻片段对共享片段并嵌套于会话，按对重抽会低估不确定性），
    另返回 ci_cluster 与 n_clusters；ci 仍为按项重抽的区间，便于对照。"""
    idx = [i for i, (x, y) in enumerate(zip(a, b)) if x is not None and y is not None]
    pairs = [(a[i], b[i]) for i in idx]
    n = len(pairs)
    k, po = cohen(pairs)
    rng = random.Random(f"{seed}|{n}")
    ks = []
    for _ in range(boot):
        s = [rng.choice(pairs) for _ in pairs]
        kk, _ = cohen(s)
        if kk == kk:
            ks.append(kk)
    ks.sort()
    pct = lambda v: [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]] if v else [float("nan")] * 2
    ci = pct(ks)
    cats = sorted({v for p in pairs for v in p}, key=str)
    ma, mb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ma[c] / n) * (mb[c] / n) for c in cats) if n else float("nan")
    out = {"k": k, "po": po, "pe": pe, "n": n, "ci": ci,
           "marg_a": {str(c): ma[c] for c in cats}, "marg_b": {str(c): mb[c] for c in cats}}
    if clusters is not None:
        groups = {}
        for i in idx:
            groups.setdefault(clusters[i], []).append((a[i], b[i]))
        gl = list(groups.values())
        rng2 = random.Random(f"{seed}|cluster|{len(gl)}")
        kc = []
        for _ in range(boot):
            s = [p for g in (rng2.choice(gl) for _ in gl) for p in g]
            kk, _ = cohen(s)
            if kk == kk:
                kc.append(kk)
        kc.sort()
        out["ci_cluster"], out["n_clusters"] = pct(kc), len(gl)
    return out


def fmt(st, name_a="甲", name_b="乙"):
    """一行文字：κ = 0.71（95% 区间 0.4–0.9），P_o = 75%，P_e = 0.14，n = 20；边缘分布 甲 {…} / 乙 {…}。"""
    if not st["n"]:
        return "—"
    m = lambda d: "、".join(f"{c} {v}" for c, v in d.items())
    return (f"κ = {st['k']:.2f}（95% 区间 {st['ci'][0]:.2f}–{st['ci'][1]:.2f}），P_o = {st['po']:.0%}，P_e = {st['pe']:.2f}，n = {st['n']}；"
            f"边缘分布 {name_a}：{m(st['marg_a'])}；{name_b}：{m(st['marg_b'])}")
