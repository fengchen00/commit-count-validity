"""RQ2 修订（S10-T5）：事前登记状态与映射模式。写入 <PAPER_DIR>/results/r7_rq2.md。

事前登记只用片段开始时已知的信息：请求写明任务或里程碑，或同一会话此前的请求提到过里程碑；
不使用"内容进入登记任务提交"（那依赖映射结果）。逐条任务拆分的判据要求提交带任务编号，
附带提交的判据使用上下文里程碑，两者与登记状态部分由定义相关，单独标出。
不确定性：以会话为单位有放回重抽（片段嵌套在会话中、同会话片段共享提交）。
"""
import csv
import math
import random
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

BOOT = 2000
SEED = "S10-rq2"
ADJ_SEED = "S11-T13"
PATS = [("一对一", ["一对一"]), ("合并（并集）", ["合并-同会话", "合并-跨会话"]), ("非逐条拆分", ["拆分-非逐条"]),
        ("部分未提交", ["部分未提交"]), ("缺失", ["缺失"]), ("含未跟踪文件", ["含未跟踪文件"]),
        ("逐条任务拆分†", ["拆分-逐条任务"]), ("附带提交†", ["随他项提交"])]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run(s):
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    tb = {r["episode"]: r for r in load(d / "episode_taskbook.csv")}
    rows = [r for r in load(d / "episode_patterns.csv") if r["evidence"] == "verified"]
    data = [{"s": eps[r["episode"]]["session"], "reg": int(tb[r["episode"]]["ex_ante"]),
             "old_unreg": tb[r["episode"]]["link"] == "none", "n": int(eps[r["episode"]]["n_edits_code"]),
             "labels": set(r["patterns"].split(";"))} for r in rows]
    by_s = defaultdict(list)
    for x in data:
        by_s[x["s"]].append(x)
    sess = sorted(by_s)
    rng = random.Random(SEED + s)
    boots = [[x for k in rng.choices(sess, k=len(sess)) for x in by_s[k]] for _ in range(BOOT)]

    def share(xs, labs, reg):
        g = [x for x in xs if x["reg"] == reg]
        return sum(any(l in x["labels"] for l in labs) for x in g) / len(g) if g else float("nan")

    n1 = sum(x["reg"] for x in data)
    L = [f"## {s}", "", f"可核验代码修改片段 {len(data)} 个（{len(sess)} 个会话）：事前登记 {n1} 个，未登记 {len(data) - n1} 个；"
         f"原口径（含内容进入登记任务提交）未登记 {sum(x['old_unreg'] for x in data)} 个。", "",
         "| 模式 | 事前登记 | 事前未登记 | 差（未登记 − 登记） | 差的 95% 区间（按会话重抽） |", "|---|---|---|---|---|"]
    for name, labs in PATS:
        a, b = share(data, labs, 1), share(data, labs, 0)
        diffs = sorted(share(bs, labs, 0) - share(bs, labs, 1) for bs in boots)
        diffs = [v for v in diffs if v == v]
        L.append(f"| {name} | {a:.0%} | {b:.0%} | {b - a:+.0%} | {q(diffs, .025):+.0%} 至 {q(diffs, .975):+.0%} |")
    m1 = q([x["n"] for x in data if x["reg"]], .5)
    m0 = q([x["n"] for x in data if not x["reg"]], .5)
    L += ["", f"代码编辑次数中位数：事前登记 {m1}，未登记 {m0}。† 与登记状态部分由定义相关，不作为经验关联解读。", ""]
    if n1 >= 20:
        L += adjusted(s, data, by_s, sess)
    return L


def logit_ame(xs, labs):
    """y ~ 1 + reg + log(1 + n)，牛顿法；返回登记的平均边际效应。不收敛或完全分离时返回 None。"""
    X = np.array([[1.0, x["reg"], math.log1p(x["n"])] for x in xs])
    y = np.array([float(any(l in x["labels"] for l in labs)) for x in xs])
    if y.min() == y.max() or X[:, 1].min() == X[:, 1].max():
        return None
    b = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(3)
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    else:
        return None
    if np.abs(b).max() > 20:
        return None
    X1, X0 = X.copy(), X.copy()
    X1[:, 1], X0[:, 1] = 1.0, 0.0
    return float(np.mean(1 / (1 + np.exp(-X1 @ b)) - 1 / (1 + np.exp(-X0 @ b)))), b


def mh_diff(xs, labs, cuts):
    """按代码编辑数三分位分层，层内（登记 − 未登记）比例差按 n1·n0/(n1+n0) 加权。"""
    num, den = 0.0, 0.0
    for lo, hi in cuts:
        g = [x for x in xs if lo <= x["n"] <= hi]
        g1, g0 = [x for x in g if x["reg"]], [x for x in g if not x["reg"]]
        if not g1 or not g0:
            continue
        w = len(g1) * len(g0) / (len(g1) + len(g0))
        p1 = sum(any(l in x["labels"] for l in labs) for x in g1) / len(g1)
        p0 = sum(any(l in x["labels"] for l in labs) for x in g0) / len(g0)
        num += w * (p1 - p0)
        den += w
    return num / den if den else None


def adjusted(s, data, by_s, sess):
    ns = sorted(x["n"] for x in data)
    t1, t2 = ns[len(ns) // 3], ns[2 * len(ns) // 3]
    cuts = [(0, t1), (t1 + 1, t2), (t2 + 1, 10 ** 9)]
    rng = random.Random(ADJ_SEED + s)
    boots = [[x for k in rng.choices(sess, k=len(sess)) for x in by_s[k]] for _ in range(BOOT)]
    L = ["### 控制片段规模（系统内，未登记 − 登记；D10）", "",
         f"规模分层按代码编辑数三分位：≤ {t1}、{t1 + 1}–{t2}、> {t2}。区间按会话重抽 {BOOT} 次。", "",
         "| 模式 | 未调整差 | 逻辑回归平均边际效应 | 95% 区间 | 规模分层加权差 | 95% 区间 | log(1+编辑数) 系数 | 跳过次数 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, labs in PATS[:2]:
        g1, g0 = [x for x in data if x["reg"]], [x for x in data if not x["reg"]]
        raw = (sum(any(l in x["labels"] for l in labs) for x in g0) / len(g0)
               - sum(any(l in x["labels"] for l in labs) for x in g1) / len(g1))
        ame, b = logit_ame(data, labs)
        mh = mh_diff(data, labs, cuts)
        ab, mb, skip = [], [], 0
        for bs in boots:
            r = logit_ame(bs, labs)
            m = mh_diff(bs, labs, cuts)
            if r is None or m is None:
                skip += 1
                continue
            ab.append(-r[0])
            mb.append(-m)
        L.append(f"| {name} | {raw:+.0%} | {-ame:+.0%} | {q(ab, .025):+.0%} 至 {q(ab, .975):+.0%} | "
                 f"{-mh:+.0%} | {q(mb, .025):+.0%} 至 {q(mb, .975):+.0%} | {b[2]:+.2f} | {skip} |")
    return L + ["", "逻辑回归：模式 ~ 截距 + 事前登记 + log(1 + 代码编辑数)；平均边际效应为全部片段分别设为未登记与登记时预测概率之差的均值。", ""]


if __name__ == "__main__":
    out = ["# RQ2 修订：事前登记状态与映射模式（S10-T5）", "", "由 `analysis/rq2_factors.py` 生成。", ""]
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        out += run(s)
    (RESULTS_DIR / "r7_rq2.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
