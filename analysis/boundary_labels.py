"""开发者视角的工作边界效标（S17-T2，口径见 notes/decisions.md D17-5）。

sample  ：系统 A 按片段内提交与否各抽 40 个、系统 C 各抽 10 个代码修改片段（种子 S17），
          生成证据包 data/coding/boundary/packets/B###.md、分册 part1–4.md 与答案 key.json（仅存本地）。
repack  ：沿用 v1 的样本与编号，按 D19 生成 v2 证据包（第 4 节只列事前信息），输出 data/coding/boundary_v2/。
版本参数 v1 / v2（默认 v2）决定读写的目录、结果文件与 numbers.json 键（D19）。v1 证据包第 4 节含由提交推出的任务，只作对照。
disagree：读取两名编码人的编码表，列出分歧，生成裁决材料 adjudicate.md。
score   ：一致性、按标注边界重算映射模式（robustness6.build_units 的推广），写入 <PAPER_DIR>/results/r14_boundary.md
          与 numbers.json 的 robustness6.boundary_labels 键。
编码表（CSV，UTF-8）：packet,prev_same,next_same,n_works,groups,cross,conf,note
  prev_same / next_same：1 = 同一项工作，0 = 不同，NA = 无相邻片段；n_works：本片段所含可独立验收的工作项数；
  groups：n_works > 1 时按文件分组，如 F1+F2|F3（无法按文件分开时填"不可分"）；cross：X1–X5 之一或"无"；conf：1–3。
"""
import csv
import json
import os
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from build_packets import Sys, hm, quote, redact  # noqa: E402
from patterns import classify, unit_state  # noqa: E402
from robustness import MERGE, NAME, has, pc  # noqa: E402
from robustness6 import ctx_of, days_of, norm, regime  # noqa: E402
from extract_git import parse_tasks  # noqa: E402
from taskbook_layer import REF  # noqa: E402
import agreement  # noqa: E402

VER = next((a for a in sys.argv[2:] if a in ("v1", "v2")), "v2")
OUT = Path("data/coding/boundary" + ("" if VER == "v1" else "_v2"))
HUMAN = os.environ.get("CODING_HUMAN", "0") == "1"  # D29：真人编码表在 data/coding/human/boundary_v2/
SIM = Path(("data/coding/human/boundary" if HUMAN else "data/coding/sim/boundary") + ("" if VER == "v1" else "_v2"))
NUM_KEY = "boundary_labels" + ("_v1" if VER == "v1" else "")
RES_NAME = "r14_boundary" + ("_v1" if VER == "v1" else "") + ".md"
SEED = "S17"
PER = {"SYS_A": 40, "SYS_C": 10}
PARTS = 4
NUMS = (RESULTS_DIR / "numbers.json")
RES = RESULTS_DIR
REQ_TL, REQ_NB, REQ_X, TL_MAX = 160, 600, 300, 40


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def one_line(t, n):
    t = redact(" ".join((t or "").split()))
    return (t[:n] + "…") if len(t) > n else (t or "（空）")


# 抽样与证据包 ------------------------------------------------------------------------------------
def neighbours(C):
    nb = {}
    for es in C["by_s"].values():
        for i, e in enumerate(es):
            nb[e["episode"]] = (es[i - 1]["episode"] if i else None, es[i + 1]["episode"] if i + 1 < len(es) else None)
    return nb


def candidates(C, e):
    """其他会话中、本片段开始前 48 小时内结束、改过相同代码文件的代码修改片段，至多 5 个（最近的在前）。"""
    t0 = datetime.fromisoformat(C["code"][e]["start"])
    mine = C["files"][e]
    out = []
    for o, oe in C["code"].items():
        if oe["session"] == C["code"][e]["session"]:
            continue
        t1 = datetime.fromisoformat(oe["last_event"])
        if 0 <= (t0 - t1).total_seconds() <= 48 * 3600 and C["files"][o] & mine:
            out.append((t1, o))
    return [o for _, o in sorted(out, reverse=True)[:5]]


def fname(p):
    return p.split("/")[-1]


def exante(S, sess, i):
    """片段开始前可得的任务信息（D19）：请求中写明的登记任务；本片段及同会话此前请求中出现的、有任务书的里程碑。"""
    ms_all = {m for m, _ in S.tb}
    req_t, ms = set(), set()
    for m, tl in REF.findall(sess[i]["request_text"]):
        if m in ms_all:
            ms.add(m)
        for n in parse_tasks(tl):
            if (m, f"T{n}") in S.tb:
                req_t.add((m, f"T{n}"))
    for x in sess[:i]:
        ms |= {m for m, _ in REF.findall(x["request_text"]) if m in ms_all}
    return req_t, ms


def packet(S, C, pid, e, nb, cands, fmap=None):
    ep = S.eps[e]
    sess = S.by_sess[ep["session"]]
    i = next(k for k, x in enumerate(sess) if x["episode"] == e)
    prev_e, next_e = nb[e]
    fmap = fmap or {f"F{k + 1}": p for k, p in enumerate(sorted(C["files"][e], key=lambda p: (-C["fe"][(e, p)], p)))}
    L = [f"# 边界证据包 {pid}", "",
         f"- 系统：{'A' if S.s == 'SYS_A' else 'C'}；本片段是所在会话第 {i + 1}/{len(sess)} 条开发者消息；开始 {hm(ep['start'])}，最后事件 {hm(ep['last_event'])}。",
         f"- 代码编辑 {ep['n_edits_code']} 次，改动代码文件 {len(fmap)} 个；任务书编辑 {ep['n_edits_plan']} 次；并入的接续消息 {ep['n_merged_msgs']} 条。", ""]
    L += ["## 1 本片段的请求", "", quote(ep["request_text"][:1500]), ""]
    L += ["## 2 会话时间线（只列开发者消息；★ 为本片段，◀ 为前一代码修改片段，▶ 为后一代码修改片段）", ""]
    lo, hi = max(0, i - TL_MAX // 2), min(len(sess), i + TL_MAX // 2)
    if lo:
        L.append(f"- （此前另有 {lo} 条消息未列出）")
    for x in sess[lo:hi]:
        mark = "★" if x["episode"] == e else "◀" if x["episode"] == prev_e else "▶" if x["episode"] == next_e else "·"
        code = f"改代码 {x['n_edits_code']} 次" if int(x["n_edits_code"]) else "未改代码"
        L.append(f"- {mark} {hm(x['start'])}（{code}）{one_line(x['request_text'], REQ_TL)}")
    if hi < len(sess):
        L.append(f"- （此后另有 {len(sess) - hi} 条消息未列出）")
    L += ["", "## 3 本片段改动的代码文件（编号供分组使用）", "", "| 编号 | 文件 | 编辑次数 | 前一片段也改过 | 后一片段也改过 |", "|---|---|---|---|---|"]
    fp, fn = (C["files"][prev_e] if prev_e else set()), (C["files"][next_e] if next_e else set())
    for k, p in fmap.items():
        L.append(f"| {k} | `{redact(p)}` | {C['fe'][(e, p)]} | {'是' if p in fp else ''} | {'是' if p in fn else ''} |")
    plans = sorted({f["path"] for f in S.plan_files.get(e, [])})
    if plans:
        L += ["", "本片段写过的任务书：" + "、".join(f"`{fname(p)}`" for p in plans)]
    L += ["", "## 4 相关任务书条目", ""]
    if VER == "v1":
        tk = S.tbk[e]
        tasks = [t.split("-", 1) for t in tk["tasks"].split(";") if t]
        mss = tk["milestones"] or ""
        L.append(f"- 本片段及同会话此前请求中出现的里程碑：{mss or '无'}")
        if tasks:
            L.append("- 本片段对应到的登记任务：" + "；".join(f"{ms}-{t} {one_line(S.title(ms, t), 50)}" for ms, t in tasks))
    else:
        req_t, ms_set = exante(S, sess, i)
        mss = ";".join(sorted(ms_set))
        L.append(f"- 本片段及同会话此前请求中出现的里程碑：{mss or '无'}")
        if req_t:
            L.append("- 本片段请求中写明的登记任务：" + "；".join(f"{ms}-{t} {one_line(S.title(ms, t), 50)}" for ms, t in sorted(req_t)))
    for ms in sorted({m for m in mss.split(";") if m})[:3]:
        ts = sorted((t for (m, t) in S.tb if m == ms), key=lambda t: int(t[1:]) if t[1:].isdigit() else 999)
        if ts:
            L.append(f"- {ms} 任务书的全部条目：" + "；".join(f"{t} {one_line(S.title(ms, t), 40)}" for t in ts[:15]) + ("等" if len(ts) > 15 else ""))
    L.append("")
    for tag, x in (("前一代码修改片段 ◀", prev_e), ("后一代码修改片段 ▶", next_e)):
        L += [f"## {'5' if tag.startswith('前') else '6'} {tag}", ""]
        if not x:
            L += ["- （无：本片段是会话中" + ("第一个" if tag.startswith("前") else "最后一个") + "代码修改片段）", ""]
            continue
        xe = S.eps[x]
        a, b = (xe, ep) if tag.startswith("前") else (ep, xe)
        gap = (datetime.fromisoformat(b["start"]) - datetime.fromisoformat(a["last_event"])).total_seconds() / 60
        shared = sorted(C["files"][x] & C["files"][e])
        L += [f"- 开始 {hm(xe['start'])}；与本片段间隔 {gap:.0f} 分钟；代码编辑 {xe['n_edits_code']} 次；"
              f"与本片段共同改动的代码文件：{('、'.join(k for k, p in fmap.items() if p in shared)) or '无'}；其余文件："
              + ("、".join(f"`{fname(p)}`" for p in sorted(C['files'][x] - set(shared))[:8]) or "无"), "", quote(xe["request_text"][:REQ_NB]), ""]
    L += ["## 7 其他会话中此前 48 小时内改过相同代码文件的片段", ""]
    if not cands:
        L.append("- （无）")
    for k, o in enumerate(cands, 1):
        oe = S.eps[o]
        shared = sorted(C["files"][o] & C["files"][e])
        L.append(f"- X{k}：{hm(oe['start'])}—{hm(oe['last_event'])}，共同改动 {('、'.join(kk for kk, p in fmap.items() if p in shared))}；"
                 f"请求：{one_line(oe['request_text'], REQ_X)}")
    L += ["", "## 8 待判问题（对照 `<PAPER_DIR>/boundary_manual.md`）", "",
          "- Q1 与前一代码修改片段 ◀ 是否同一项工作（1 / 0 / NA）", "- Q2 与后一代码修改片段 ▶ 是否同一项工作（1 / 0 / NA）",
          "- Q3 本片段含几项可独立验收的工作；多于 1 项时按文件编号分组（如 F1,F2|F3）", "- Q4 是否延续第 7 节中的某一片段（X1–X5 或 无）",
          "- Q5 把握（1 猜测 / 2 较有把握 / 3 证据明确）", ""]
    return "\n".join(L), fmap


def sample():
    rng = random.Random(SEED)
    picked = []
    for s, k in PER.items():
        C = ctx_of(s)
        for flag in (True, False):
            pool = sorted(e for e in C["code"] if C["cmd"][e] == flag)
            picked += [(s, e, flag, len(pool)) for e in rng.sample(pool, min(k, len(pool)))]
    rng.shuffle(picked)
    OUT.joinpath("packets").mkdir(parents=True, exist_ok=True)
    key, parts = {}, defaultdict(list)
    cache = {}
    for n, (s, e, flag, popn) in enumerate(picked, 1):
        if s not in cache:
            S = Sys(s)
            S.plan_files = defaultdict(list)
            for f in csv.DictReader(open(S_path(s) / "episode_files.csv", encoding="utf-8")):
                if f["class"] == "plan":
                    S.plan_files[f["episode"]].append(f)
            C = ctx_of(s)
            cache[s] = (S, C, neighbours(C))
        S, C, nb = cache[s]
        pid = f"B{n:03d}"
        cands = candidates(C, e)
        text, fmap = packet(S, C, pid, e, nb, cands)
        (OUT / "packets" / f"{pid}.md").write_text(text, encoding="utf-8")
        parts[(n - 1) % PARTS].append(text)
        key[pid] = {"system": s, "episode": e, "cmd": flag, "pop": popn, "n_stratum": PER[s], "prev": nb[e][0], "next": nb[e][1],
                    "cands": cands, "files": fmap}
    for k, texts in parts.items():
        (OUT / f"part{k + 1}.md").write_text("\n\n---\n\n".join(texts), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print("packets:", len(key), {p: sum(v["system"] == p.split("|")[0] and v["cmd"] == (p.split("|")[1] == "1") for v in key.values())
                                 for p in ("SYS_A|1", "SYS_A|0", "SYS_C|1", "SYS_C|0")})


def repack():
    """v2 证据包（D19）：样本、编号与文件编号沿用 v1 的 key.json。"""
    assert VER == "v2"
    key = json.loads((Path("data/coding/boundary") / "key.json").read_text(encoding="utf-8"))
    OUT.joinpath("packets").mkdir(parents=True, exist_ok=True)
    parts, cache = defaultdict(list), {}
    for n, pid in enumerate(sorted(key), 1):
        k = key[pid]
        s, e = k["system"], k["episode"]
        if s not in cache:
            S = Sys(s)
            S.plan_files = defaultdict(list)
            for f in csv.DictReader(open(S_path(s) / "episode_files.csv", encoding="utf-8")):
                if f["class"] == "plan":
                    S.plan_files[f["episode"]].append(f)
            C = ctx_of(s)
            cache[s] = (S, C, neighbours(C))
        S, C, nb = cache[s]
        assert set(k["files"].values()) == C["files"][e], pid
        text, _ = packet(S, C, pid, e, nb, k["cands"], k["files"])
        (OUT / "packets" / f"{pid}.md").write_text(text, encoding="utf-8")
        parts[(n - 1) % PARTS].append(text)
    for i, texts in parts.items():
        (OUT / f"part{i + 1}.md").write_text("\n\n---\n\n".join(texts), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print("v2 packets:", len(key))


def S_path(s):
    return Path("data/derived") / s


# 编码表 -----------------------------------------------------------------------------------------
def read_coder(tag):
    rows = {}
    for f in sorted(SIM.glob(f"{tag}_part*.csv")):
        for r in load(f):
            rows[r["packet"].strip()] = r
    return rows


def parse(r, key):
    g = lambda k: (r.get(k) or "").strip()
    bit = lambda v: None if v.upper() in ("NA", "") else int(v)
    n = int(g("n_works") or 1)
    groups = [set(x.strip() for x in grp.replace(",", "+").split("+") if x.strip()) for grp in g("groups").split("|") if grp.strip()] if n > 1 else []
    valid = set(key["files"])
    groups = [grp & valid for grp in groups if grp & valid]
    if n > 1 and len(groups) < 2:
        groups = []
    cross = g("cross")
    return {"prev": bit(g("prev_same")) if key["prev"] else None, "next": bit(g("next_same")) if key["next"] else None, "n": max(1, n),
            "groups": groups, "cross": cross if cross in {f"X{k + 1}" for k in range(len(key["cands"]))} else None, "conf": int(g("conf") or 0)}


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan"), 0
    cats = sorted({v for p in pairs for v in p}, key=str)
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), n


QS = [("prev", "Q1 与前一片段同一工作"), ("next", "Q2 与后一片段同一工作"), ("multi", "Q3 含多项工作"), ("cross", "Q4 延续其他会话")]


def view(p, q):
    return (p["n"] > 1) if q == "multi" else (p["cross"] is not None) if q == "cross" else p[q]


def disagree():
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    c1, c2 = read_coder("coder1"), read_coder("coder2")
    L = ["# 边界编码裁决材料", "", "以下片段两名编码人（甲、乙）的判断不一致。请阅读对应证据包（data/coding/boundary/packets/<编号>.md）后给出最终判断。", ""]
    n = 0
    for pid in sorted(key):
        a, b = parse(c1[pid], key[pid]), parse(c2[pid], key[pid])
        diff = [qn for q, qn in QS if view(a, q) != view(b, q)] + (["Q3 分组"] if a["n"] > 1 and b["n"] > 1 and a["groups"] != b["groups"] else [])
        if diff:
            n += 1
            fmt = lambda r: f"prev_same={r['prev_same']}，next_same={r['next_same']}，n_works={r['n_works']}，groups={r['groups'] or '—'}，cross={r['cross']}，note={r.get('note', '')}"
            L += [f"## {pid}", "", f"- 分歧：{'、'.join(diff)}", f"- 甲：{fmt(c1[pid])}", f"- 乙：{fmt(c2[pid])}", ""]
    (OUT / "adjudicate.md").write_text("\n".join(L), encoding="utf-8")
    print("disagreements:", n)


# 按标注边界重算 -----------------------------------------------------------------------------------
def build(C, specs):
    """specs：单位列表，每个单位是 [(片段, 文件集合或 None), ...]。"""
    x, fc = C["x"], C["fc"]
    units = []
    for k, mem in enumerate(specs):
        sess = C["code"][mem[0][0]]["session"]
        P, n = set(), 0
        for e, fs in mem:
            if fs is None:
                P |= x["content"].get(e, set())
                n += int(C["code"][e]["n_edits_code"])
            else:
                P |= set().union(*(fc.get((e, f), set()) for f in fs)) & x["content"].get(e, set())
                n += sum(C["fe"][(e, f)] for f in fs)
        units.append({"uid": f"{sess}|d{k}", "mem": mem, "P": P, "n": n})
    eo = defaultdict(set)
    for u in units:
        for c in u["P"]:
            eo[c].add(u["uid"])
    for u in units:
        es = sorted({e for e, _ in u["mem"]})
        st = unit_state([x["st"][e] for e in es])
        u["labels"] = set(classify(max(1, u["n"]), u["uid"], u["P"], eo, x["cm"], set().union(*(x["ctx"][e] for e in es)),
                                   any(x["flatten"][e] for e in es), st))
        u["cmd"] = any(C["cmd"][e] for e in es)
        u["share"] = sum(1 / len(eo[c]) for c in u["P"])
    return units


def dev_units(C, lab, key_s):
    """lab：样本片段 → 解析后的标注；返回 单位列表与 片段 → 其所在单位下标列表。"""
    par = {}

    def f(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def u(a, b):
        par[f(a)] = f(b)
    pieces = {}
    for e in C["code"]:
        if e in lab and len(lab[e]["groups"]) >= 2:
            fmap = key_s[e]["files"]
            pieces[e] = [(e, frozenset(fmap[g] for g in grp)) for grp in lab[e]["groups"]]
            covered = set().union(*(p[1] for p in pieces[e]))
            rest = C["files"][e] - covered
            if rest:
                pieces[e][0] = (e, pieces[e][0][1] | rest)
        else:
            pieces[e] = [(e, None)]
        for p in pieces[e]:
            f(p)

    def attach(e, other):
        """把 e 的相应子单位与 other 的相应子单位合并：取与对方共享文件的子单位，否则取第一个。"""
        def pick(a, b):
            ps = pieces[a]
            if len(ps) == 1:
                return ps[0]
            for p in ps:
                if p[1] & C["files"][b]:
                    return p
            return ps[0]
        u(pick(e, other), pick(other, e))
    for e, v in lab.items():
        k = key_s[e]
        if v["prev"] == 1 and k["prev"]:
            attach(e, k["prev"])
        if v["next"] == 1 and k["next"]:
            attach(e, k["next"])
        if v["cross"]:
            attach(e, k["cands"][int(v["cross"][1:]) - 1])
    grp = defaultdict(list)
    for e in C["code"]:
        for p in pieces[e]:
            grp[f(p)].append((p[0], None if p[1] is None else set(p[1])))
    specs = list(grp.values())
    units = build(C, specs)
    where = defaultdict(list)
    for i, un in enumerate(units):
        for e, _ in un["mem"]:
            where[e].append(i)
    return units, where


def summarize(C, units, where, sample_eps, w):
    """样本片段的加权模式比例：片段 e 的权重 w[e] 平均分给其所在的单位；另报两组每单位分得提交数。"""
    acc = defaultdict(float)
    tot = 0.0
    k = {True: [0.0, 0.0], False: [0.0, 0.0]}
    for e in sample_eps:
        idx = sorted(set(where[e]))
        for i in idx:
            un = units[i]
            wt = w[e] / len(idx)
            tot += wt
            acc["oto"] += wt * ("一对一" in un["labels"])
            acc["merge"] += wt * has(un["labels"], MERGE)
            acc["split"] += wt * ("拆分" in un["labels"])
            k[un["cmd"]][0] += wt * un["share"]
            k[un["cmd"]][1] += wt
    out = {m: acc[m] / tot for m in ("oto", "merge", "split")}
    k1, k0 = (k[True][0] / k[True][1] if k[True][1] else float("nan")), (k[False][0] / k[False][1] if k[False][1] else float("nan"))
    out.update({"k1": k1, "k0": k0, "ratio": k1 / k0 if k0 else float("nan")})
    return out


def resolve(a, b, adj, mode):
    """mode：final（一致取一致，分歧取裁决）/ merge（分歧取更多合并）/ split（分歧取更多拆分）。"""
    if mode == "final":
        return {q: (a[q] if a[q] == b[q] else adj[q]) for q in ("prev", "next", "cross")} | (
            {"n": a["n"], "groups": a["groups"]} if (a["n"] > 1) == (b["n"] > 1) and a["groups"] == b["groups"] else {"n": adj["n"], "groups": adj["groups"]})
    more = mode == "merge"
    pick = lambda x, y: (max(x, y) if more else min(x, y)) if x is not None and y is not None else (x if x is not None else y)
    cross = (a["cross"] or b["cross"]) if more else (a["cross"] if a["cross"] == b["cross"] else None)
    sa, sb = len(a["groups"]), len(b["groups"])
    g = (a if (sa <= sb) == more else b)
    return {"prev": pick(a["prev"], b["prev"]), "next": pick(a["next"], b["next"]), "cross": cross, "n": g["n"], "groups": g["groups"]}


def new_works(r):
    return max(0, r["n"] - (1 if r["prev"] == 1 or r["cross"] else 0))


def days_by_work(key, F, R, subset=None, tag="days_by_work", title="## 4 按工作项折算的日与系统比较（样本组均值外推）"):
    """按日折算工作项：每个代码修改片段计入其所在组（片段内提交与否）的样本新工作项均值；
    比较高 / 低两类日子每项工作的提交数、日对排序方向与系统比较。区间同时重抽日子与各组样本片段。
    subset 给出时只用其中的样本片段估计组均值（D32 一致子样本）。"""
    rng = random.Random(f"{SEED}|days")
    L = [title, ""]
    D, SM = {}, {}
    use = set(subset) if subset is not None else set(key)
    for s in ("SYS_A", "SYS_C"):
        C = ctx_of(s)
        days, active, tot = days_of(s, C["x"], C["items"])
        D[s] = (days, active, tot, C["items"])
        SM[s] = {f: [new_works(F[p]) for p in key if p in use and key[p]["system"] == s and key[p]["cmd"] == f] for f in (True, False)}

    def est(s, mm):
        days, active, tot, items = D[s]
        wk = lambda d: d["n"] * (d["cmd"] * mm[True] + (1 - d["cmd"]) * mm[False])
        hi = [d for d in days if regime(d["cmd"]) == "hi"]
        lo = [d for d in days if regime(d["cmd"]) == "lo"]
        per = lambda ds: sum(d["c"] for d in ds) / sum(wk(d) for d in ds) if ds and sum(wk(d) for d in ds) else float("nan")
        works_all = sum(mm[True] if i["cmd"] else mm[False] for i in items)
        return {"hi": per(hi), "lo": per(lo), "hi_lo": per(hi) / per(lo) if per(lo) else float("nan"),
                "day_works": works_all / len(active), "day_c": tot / len(active), "wk": {d["d"]: wk(d) for d in days}}
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    point = {s: est(s, {f: mean(SM[s][f]) for f in (True, False)}) for s in D}
    boot = defaultdict(list)
    for _ in range(1000):
        r = {}
        for s in D:
            mm = {f: mean([rng.choice(SM[s][f]) for _ in SM[s][f]]) for f in (True, False)}
            days = D[s][0]
            keep = D[s]
            D[s] = ([rng.choice(days) for _ in days], keep[1], keep[2], keep[3])
            r[s] = est(s, mm)
            D[s] = keep
            boot[f"{s}|hi_lo"].append(r[s]["hi_lo"])
        boot["sys_works"].append(r["SYS_A"]["day_works"] / r["SYS_C"]["day_works"])
    ci = {k: [float(sorted(v)[25]), float(sorted(v)[974])] for k, v in boot.items()}
    out = {}
    for s in D:
        days = D[s][0]
        wk = point[s]["wk"]
        opp = tot = 0
        for i in range(len(days)):
            for j in range(i + 1, len(days)):
                a, b = days[i], days[j]
                dc, dw = (a["c"] > b["c"]) - (a["c"] < b["c"]), (wk[a["d"]] > wk[b["d"]]) - (wk[a["d"]] < wk[b["d"]])
                if dc and dw:
                    opp += dc != dw
                    tot += 1
        out[s] = {"c_per_work_hi": point[s]["hi"], "c_per_work_lo": point[s]["lo"], "hi_lo": point[s]["hi_lo"], "hi_lo_ci": ci[f"{s}|hi_lo"],
                  "discord": [opp, tot], "day_works": point[s]["day_works"]}
        L.append(f"- {NAME[s]}：高 / 低两类日子每项工作的提交数 {point[s]['hi']:.2f} / {point[s]['lo']:.2f}，之比 {point[s]['hi_lo']:.2f}"
                 f"（{ci[f'{s}|hi_lo'][0]:.2f}–{ci[f'{s}|hi_lo'][1]:.2f}）；按提交数与按工作项数给两日排序方向相反的日对 {opp}/{tot}（{opp / tot:.0%}）。")
    sw = point["SYS_A"]["day_works"] / point["SYS_C"]["day_works"]
    out["sys_works"] = {"ratio": sw, "ci": ci["sys_works"]}
    L.append(f"- 系统比较：系统 A 对系统 C 的日均工作项数之比 {sw:.2f}（{ci['sys_works'][0]:.2f}–{ci['sys_works'][1]:.2f}）。")
    L += ["", "每日工作项数 = 当日片段内提交片段数 × 该组样本每片段新工作项均值 + 未提交片段数 × 相应均值；只外推组均值，不区分日子。"
          "区间同时按日与按组内样本片段重抽 1,000 次。", ""]
    R[tag] = out
    return L


def agree_subsample(key, A, B, FIN, need, R):
    """D32：剔除全部经裁决的片段，只用两名编码人四问（及分组）完全一致的样本片段重算 §2 的边界指标、§3 的模式与每项工作提交数之比、§4 的按日折算。
    权重按子样本内各层片段数重新反加权。用于回应"分歧由第二作者裁决可能带入偏倚"。"""
    AG = [p for p in key if p not in need]
    F = FIN["final"]  # 一致片段上 final == 编码人 1 == 编码人 2
    L = ["## 6 一致子样本：剔除全部裁决片段（D32）", "",
         f"只用两名编码人完全一致的 {len(AG)}/{len(key)} 个样本片段（{NAME['SYS_A']} {sum(key[p]['system'] == 'SYS_A' for p in AG)}，{NAME['SYS_C']} {sum(key[p]['system'] == 'SYS_C' for p in AG)}），"
         "不含任何第二作者裁决的判断；按子样本内各层片段数反加权。", "",
         "| 系统 | ρ̂ | σ̂ | 每片段新工作项数：片段内提交 / 未提交 | 一对一 | 合并 | 拆分 | 每项工作提交数之比（机制分期） |", "|---|---|---|---|---|---|---|---|"]
    for s in ("SYS_A", "SYS_C"):
        ps = [p for p in AG if key[p]["system"] == s]
        if not ps or not all(any(key[p]["cmd"] == f for p in ps) for f in (True, False)):
            L.append(f"| {NAME[s]} | — | — | — | — | — | — | — |")
            continue
        w = {p: key[p]["pop"] / sum(1 for q in ps if key[q]["cmd"] == key[p]["cmd"]) for p in ps}
        wp = [(w[p], F[p]["prev"]) for p in ps if F[p]["prev"] is not None]
        rho = sum(a * b for a, b in wp) / sum(a for a, _ in wp) if wp else float("nan")
        sig = sum(w[p] * (F[p]["n"] > 1) for p in ps) / sum(w.values())
        nn = [sum(new_works(F[p]) for p in ps if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in ps)) for f in (True, False)]
        nws = [[new_works(F[p]) for p in ps if key[p]["cmd"] == f] for f in (True, False)]
        C = ctx_of(s)
        eps = [key[p]["episode"] for p in ps]
        we = {key[p]["episode"]: w[p] for p in ps}
        key_s = {key[p]["episode"]: key[p] for p in ps}
        base_units, base_where = dev_units(C, {}, key_s)
        un, wh = dev_units(C, {key[p]["episode"]: F[p] for p in ps}, key_s)
        r0, r1 = summarize(C, base_units, base_where, eps, we), summarize(C, un, wh, eps, we)
        pw = (r0["k1"] / nn[0]) / (r0["k0"] / nn[1]) if nn[0] and nn[1] and r0["k0"] else float("nan")
        rng = random.Random(f"{SEED}|agree|{s}")
        boot = defaultdict(list)
        for _ in range(1000):
            bs = [rng.choice([p for p in ps if key[p]["cmd"] == f]) for f in (True, False) for _ in range(sum(key[q]["cmd"] == f for q in ps))]
            beps = [key[p]["episode"] for p in bs]
            b1, b0 = summarize(C, un, wh, beps, we), summarize(C, base_units, base_where, beps, we)
            bn = [sum(new_works(F[p]) for p in bs if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in bs)) for f in (True, False)]
            for m in ("oto", "merge", "split"):
                boot[m].append(b1[m])
            boot["per_work"].append((b0["k1"] / bn[0]) / (b0["k0"] / bn[1]) if bn[0] and bn[1] and b0["k0"] else float("nan"))
        ci = {m: [float(sorted(v)[25]), float(sorted(v)[974])] for m, v in boot.items()}
        R[f"{s}|agree|rates"] = {"rho": rho, "sigma": sig, "new_works": nn, "new_works_samples": nws, "n_samples": [len(nws[0]), len(nws[1])], "n_prev": len(wp)}
        R[f"{s}|agree|recalc"] = {"base": r0, "dev": r1, "per_work": pw, "ci": ci}
        L.append(f"| {NAME[s]} | {pc(rho)}（n = {len(wp)}） | {pc(sig)} | {nn[0]:.2f} / {nn[1]:.2f} | {pc(r1['oto'])}（{pc(ci['oto'][0])}–{pc(ci['oto'][1])}） | "
                 f"{pc(r1['merge'])}（{pc(ci['merge'][0])}–{pc(ci['merge'][1])}） | {pc(r1['split'])}（{pc(ci['split'][0])}–{pc(ci['split'][1])}） | "
                 f"{pw:.2f}（{ci['per_work'][0]:.2f}–{ci['per_work'][1]:.2f}） |")
    L += ["", "一对一、合并、拆分为按标注边界重算的模式比例（与 §3 裁决口径同一算法）；最后一列为片段原口径的每单位提交数之比除以两组每片段新工作项数之比（与 §3 末行同一算法）；括号内为样本片段重抽 1,000 次的 95% 区间。", ""]
    L += days_by_work(key, F, R, subset=AG, tag="days_by_work_agree", title="### 6.1 一致子样本的按日折算（机制分期）")
    return L


def score():
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    c1, c2 = read_coder("coder1"), read_coder("coder2")
    adj_rows = read_coder("adjudicate")
    miss = [p for p in key if p not in c1 or p not in c2]
    assert not miss, f"coder files incomplete: {miss[:5]}"
    A = {p: parse(c1[p], key[p]) for p in key}
    B = {p: parse(c2[p], key[p]) for p in key}
    ADJ = {p: parse(adj_rows[p], key[p]) for p in adj_rows}
    R, L = {}, (["# 开发者视角的工作边界效标（S17-T2，D17-5；真人编码，D29）", "",
                 "由 `analysis/boundary_labels.py` 生成。编码人 1 为开发者（事后回看），编码人 2 为独立编码人甲；分歧由第二作者裁定（D30-1）。", ""] if HUMAN else
                ["# 开发者视角的工作边界效标（S17-T2，D17-5；AI 模拟编码）", "",
                "由 `analysis/boundary_labels.py` 生成。编码人与裁决人均为不同的语言模型，扮演事后回看的开发者；结果只说明流程与量级，不是真实效标。", ""])
    L += ["## 1 一致性", "", "| 问题 | 可比片段 | 一致 | Cohen κ（按片段重抽） | κ 区间（按会话整块重抽，D32） |", "|---|---|---|---|---|"]
    full = []
    sess = {}
    for s in ("SYS_A", "SYS_C"):
        sess.update({e["episode"]: e["session"] for e in ctx_of(s)["x"]["eps"]})
    clus = [sess.get(key[p]["episode"], key[p]["episode"]) for p in sorted(key)]
    for q, qn in QS:
        a = [view(A[p], q) for p in sorted(key)]
        b = [view(B[p], q) for p in sorted(key)]
        kk, n = kappa(a, b)
        ag = sum(x == y for x, y in zip(a, b) if x is not None and y is not None)
        st = agreement.kappa_stats(a, b, f"bl|{q}", clusters=clus)
        R[f"kappa_{q}"] = {"k": kk, "n": n, "agree": ag, "stats": st}
        L.append(f"| {qn} | {n} | {ag}（{ag / n:.0%}） | {kk:.2f}（{st['ci'][0]:.2f}–{st['ci'][1]:.2f}） | {st['ci_cluster'][0]:.2f}–{st['ci_cluster'][1]:.2f}（{st['n_clusters']} 个会话） |"
                 if n else f"| {qn} | 0 | — | — | — |")
        full.append(f"- {qn}：{agreement.fmt(st, '开发者', '独立编码人')}。")
    # 四问联合一致（D26-5）：决定重组单位的是四项判断的联合
    joint = [p for p in key if all(view(A[p], q) == view(B[p], q) for q, _ in QS)]
    R["joint_agree"] = {"k": len(joint), "n": len(key)}
    L += ["", "括号内为 κ 的 95% 自助区间（按片段重抽 2,000 次，D26-5）。", "",
          f"四个问题全部一致的片段 {len(joint)}/{len(key)}（{len(joint) / len(key):.0%}）；决定重组单位的是四项判断的联合。", ""] + full
    need = [p for p in key if any(view(A[p], q) != view(B[p], q) for q, _ in QS) or (A[p]["n"] > 1 and B[p]["n"] > 1 and A[p]["groups"] != B[p]["groups"])]
    lack = [p for p in need if p not in ADJ]
    R["n_disagree"], R["n_adjudicated"] = len(need), len(need) - len(lack)
    # D35-7（A6）：分歧的两种来源分开计数——四问任一不一致；四问全一致但两人都判多项工作、分组不同
    n_q = sum(1 for p in key if any(view(A[p], q) != view(B[p], q) for q, _ in QS))
    n_g = len(need) - n_q
    R["n_disagree_questions"], R["n_disagree_grouping_only"] = n_q, n_g
    L += ["", f"至少一项分歧的片段 {len(need)} 个（四问任一不一致 {n_q} 个；四问全一致、但两人都判为多项工作且分组不同 {n_g} 个），已裁决 {len(need) - len(lack)} 个；"
          f"其余 {len(key) - len(need)} 个片段两人完全一致。", ""]
    for p in lack:
        ADJ[p] = A[p]

    L += ["## 2 边界指标（按抽样层反加权）", ""]
    FIN = {m: {p: resolve(A[p], B[p], ADJ.get(p, A[p]), m) for p in key} for m in ("final", "merge", "split")}
    L += ["| 系统 | 口径 | ρ̂ 前一边界判为同一工作 | 含多项工作（σ̂） | 每片段工作项数：片段内提交 / 未提交 | 每片段新工作项数：片段内提交 / 未提交 | 延续其他会话 |",
          "|---|---|---|---|---|---|---|"]
    cell_lines = []
    for s in ("SYS_A", "SYS_C"):
        ps = [p for p in key if key[p]["system"] == s]
        w = {p: key[p]["pop"] / sum(1 for q in ps if key[q]["cmd"] == key[p]["cmd"]) for p in ps}
        # D40：样本片段所在日子的事前登记日组（高 / 中 / 低；不足 3 个片段的日子为不合格），供 robustness7 做格均值外推
        C = ctx_of(s)
        dys, _, _ = days_of(s, C["x"], C["items"])
        dayreg = {d["d"]: regime(d["reg"] / d["n"]) for d in dys}
        ep_day = {i["e"]: i["start"][:10] for i in C["items"]}
        cell_of = {p: dayreg.get(ep_day.get(key[p]["episode"]), "ineligible") for p in ps}
        for m, mn in (("final", "裁决"), ("merge", "分歧取更多合并"), ("split", "分歧取更多拆分")):
            F = FIN[m]
            cells = {r: [[new_works(F[p]) for p in ps if key[p]["cmd"] == f and cell_of[p] == r] for f in (True, False)]
                     for r in ("hi", "mid", "lo", "ineligible")}
            if m == "final":
                mean = lambda v: f"{sum(v) / len(v):.2f}" if v else "—"
                cell_lines.append(f"- {NAME[s]}：" + "；".join(
                    f"{rn}日 片段内提交 n = {len(cells[r][0])}（均值 {mean(cells[r][0])}）/ 未提交 n = {len(cells[r][1])}（{mean(cells[r][1])}）"
                    for r, rn in (("hi", "高登记"), ("mid", "中间"), ("lo", "低登记"), ("ineligible", "不合格"))) + "。")
            wp = [(w[p], F[p]["prev"]) for p in ps if F[p]["prev"] is not None]
            rho = sum(a * b for a, b in wp) / sum(a for a, _ in wp) if wp else float("nan")
            sig = sum(w[p] * (F[p]["n"] > 1) for p in ps) / sum(w.values())
            nw = [sum(F[p]["n"] for p in ps if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in ps)) for f in (True, False)]
            cr = sum(w[p] * (F[p]["cross"] is not None) for p in ps) / sum(w.values())
            nn = [sum(new_works(F[p]) for p in ps if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in ps)) for f in (True, False)]
            # D26-4：保留每个样本片段的新工作项数，供 robustness7 做效标样本 → 日的两层重抽
            nws = [[new_works(F[p]) for p in ps if key[p]["cmd"] == f] for f in (True, False)]
            R[f"{s}|{m}|rates"] = {"rho": rho, "sigma": sig, "works": nw, "new_works": nn, "cross": cr, "n_prev": len(wp),
                                   "new_works_samples": nws, "n_samples": [len(nws[0]), len(nws[1])], "new_works_cells": cells}
            L.append(f"| {NAME[s]} | {mn} | {pc(rho)}（n = {len(wp)}） | {pc(sig)} | {nw[0]:.2f} / {nw[1]:.2f} | {nn[0]:.2f} / {nn[1]:.2f} | {pc(cr)} |")
    L += ["", "ρ̂ 的分母为有前一代码修改片段的样本片段；各比例按系统内两层（片段内提交与否）的总体规模反加权；每片段工作项数为未加权组内均值；"
          "新工作项数为工作项数减去延续前一片段或其他会话的一项。", "",
          "样本片段的新工作项数按 提交状态 × 所在日的事前登记日组（≥ 2/3 为高、≤ 1/3 为低；不足 3 个片段的日子为不合格）列表（裁决口径，D40；"
          "抽样只按提交状态分层、层内对日子随机，格均值是各格总体均值的无偏估计；robustness7 以此做格均值外推）：", ""] + cell_lines + [""]

    L += ["## 3 按标注边界重算的映射模式（样本片段，按层反加权）", "",
          "| 系统 | 口径 | 一对一 | 合并 | 拆分 | 每单位分得提交数：片段内提交 / 未提交 | 之比 |", "|---|---|---|---|---|---|---|"]
    for s in ("SYS_A", "SYS_C"):
        C = ctx_of(s)
        ps = [p for p in key if key[p]["system"] == s]
        eps = [key[p]["episode"] for p in ps]
        w = {key[p]["episode"]: key[p]["pop"] / sum(1 for q in ps if key[q]["cmd"] == key[p]["cmd"]) for p in ps}
        key_s = {key[p]["episode"]: key[p] for p in ps}
        base_units, base_where = dev_units(C, {}, key_s)
        rows = [("片段（原口径）", summarize(C, base_units, base_where, eps, w))]
        for m, mn in (("final", "开发者边界·裁决"), ("merge", "开发者边界·分歧取更多合并"), ("split", "开发者边界·分歧取更多拆分")):
            lab = {key[p]["episode"]: FIN[m][p] for p in ps}
            un, wh = dev_units(C, lab, key_s)
            rows.append((mn, summarize(C, un, wh, eps, w)))
        rng = random.Random(f"{SEED}|bl|{s}")
        boot = defaultdict(list)
        un, wh = dev_units(C, {key[p]["episode"]: FIN["final"][p] for p in ps}, key_s)

        def per_work(r0, sp, m="final"):
            nw = [sum(new_works(FIN[m][p]) for p in sp if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in sp)) for f in (True, False)]
            return (r0["k1"] / nw[0]) / (r0["k0"] / nw[1]) if nw[0] and nw[1] and r0["k0"] else float("nan")
        for _ in range(1000):
            bs = [rng.choice([p for p in ps if key[p]["cmd"] == f]) for f in (True, False) for _ in range(sum(key[q]["cmd"] == f for q in ps))]
            beps = [key[p]["episode"] for p in bs]
            r1 = summarize(C, un, wh, beps, w)
            r0 = summarize(C, base_units, base_where, beps, w)
            for m in ("oto", "merge", "split", "ratio"):
                boot[m].append(r1[m] - r0[m])
                boot[m + "_dev"].append(r1[m])
            boot["base_ratio"].append(r0["ratio"])
            boot["per_work"].append(per_work(r0, bs))
        R[f"{s}|recalc"] = {n: r for n, r in rows}
        R[f"{s}|per_work"] = per_work(rows[0][1], ps)
        R[f"{s}|per_work_modes"] = {m: per_work(rows[0][1], ps, m) for m in ("final", "merge", "split")}
        R[f"{s}|recalc_ci"] = {m: [float(sorted(v)[25]), float(sorted(v)[974])] for m, v in boot.items()}
        for n, r in rows:
            L.append(f"| {NAME[s]} | {n} | {pc(r['oto'])} | {pc(r['merge'])} | {pc(r['split'])} | {r['k1']:.2f} / {r['k0']:.2f} | {r['ratio']:.2f} |")
        ci = R[f"{s}|recalc_ci"]
        L.append(f"| {NAME[s]} | 裁决口径的 95% 区间（样本片段重抽 1,000 次） | {pc(ci['oto_dev'][0])}–{pc(ci['oto_dev'][1])} | {pc(ci['merge_dev'][0])}–{pc(ci['merge_dev'][1])} | "
                 f"{pc(ci['split_dev'][0])}–{pc(ci['split_dev'][1])} | | {ci['ratio_dev'][0]:.2f}–{ci['ratio_dev'][1]:.2f} |")
        L.append(f"| {NAME[s]} | 片段原口径的每单位提交数之比，除以两组每片段新工作项数之比 | | | | | {R[f'{s}|per_work']:.2f}"
                 f"（{ci['per_work'][0]:.2f}–{ci['per_work'][1]:.2f}）；分歧取更多合并 / 更多拆分：{R[f'{s}|per_work_modes']['merge']:.2f} / {R[f'{s}|per_work_modes']['split']:.2f} |")
    L += ["", "重算时样本片段按标注与前后相邻片段、跨会话延续片段合并，按文件分组拆分（邻接关系挂在与邻片段共享文件的子单位上，否则挂在第一个子单位）；"
          "非样本片段保持原单位。片段的权重平均分给其所在的单位。", ""]
    L += days_by_work(key, FIN["final"], R)
    L += circularity(key, FIN["final"], R)
    L += agree_subsample(key, A, B, FIN, need, R)
    L += items_vs_works(key, A, B, FIN["final"], R)
    L += two_ends(key, A, B, R)
    (RES / RES_NAME).write_text("\n".join(L), encoding="utf-8")
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    num.setdefault("robustness6", {})[NUM_KEY] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


def two_ends(key, A, B, R):
    """D35-6（B3）两端法：不经裁决，分别只用开发者（编码人 1）与编码人甲（编码人 2）的原始标注，重算两组每片段新工作项数、
    样本片段上的每项工作提交数之比与按日折算（机制分期）；事前登记分期由 robustness7 读取 `*|devonly|rates` / `*|coderonly|rates` 计算。
    回应导师与审稿人"一项工作的判断主观、结论是否依赖某一个人"的意见。"""
    L = ["## 8 两端法：只用某一名标注人的原始标注（D35-6）", "",
         "| 系统 | 标注人 | 每片段新工作项数：片段内提交 / 未提交 | 样本片段上每项工作提交数之比 |", "|---|---|---|---|"]
    for tag, F, tn in (("devonly", A, "开发者"), ("coderonly", B, "编码人甲")):
        for s in ("SYS_A", "SYS_C"):
            ps = [p for p in key if key[p]["system"] == s]
            w = {key[p]["episode"]: key[p]["pop"] / sum(1 for q in ps if key[q]["cmd"] == key[p]["cmd"]) for p in ps}
            key_s = {key[p]["episode"]: key[p] for p in ps}
            C = ctx_of(s)
            eps = [key[p]["episode"] for p in ps]
            base_units, base_where = dev_units(C, {}, key_s)
            r0 = summarize(C, base_units, base_where, eps, w)
            nn = [sum(new_works(F[p]) for p in ps if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in ps)) for f in (True, False)]
            nws = [[new_works(F[p]) for p in ps if key[p]["cmd"] == f] for f in (True, False)]
            pw = (r0["k1"] / nn[0]) / (r0["k0"] / nn[1]) if nn[0] and nn[1] and r0["k0"] else float("nan")
            rng = random.Random(f"{SEED}|{tag}|{s}")
            boot = []
            for _ in range(1000):
                bs = [rng.choice([p for p in ps if key[p]["cmd"] == f]) for f in (True, False) for _ in range(sum(key[q]["cmd"] == f for q in ps))]
                b0 = summarize(C, base_units, base_where, [key[p]["episode"] for p in bs], w)
                bn = [sum(new_works(F[p]) for p in bs if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in bs)) for f in (True, False)]
                boot.append((b0["k1"] / bn[0]) / (b0["k0"] / bn[1]) if bn[0] and bn[1] and b0["k0"] else float("nan"))
            v = sorted(x for x in boot if x == x)
            ci = [float(v[25]), float(v[974])] if len(v) > 975 else [float("nan")] * 2
            R[f"{s}|{tag}|rates"] = {"new_works": nn, "new_works_samples": nws, "n_samples": [len(nws[0]), len(nws[1])]}
            R[f"{s}|{tag}|per_work"] = {"ratio": pw, "ci": ci}
            L.append(f"| {NAME[s]} | {tn} | {nn[0]:.2f} / {nn[1]:.2f} | {pw:.2f}（{ci[0]:.2f}–{ci[1]:.2f}） |")
    L += ["", "与 §3 末行同一算法：片段原口径的每单位提交数之比除以该标注人给出的两组每片段新工作项数之比；括号内为样本片段重抽 1,000 次的 95% 区间。"
          "两行之间的差别就是裁决可能影响的范围：裁决口径的估计若落在两端之间或与两端区间重叠，结论不依赖裁决。", ""]
    for tag, tn in (("devonly", "只用开发者标注"), ("coderonly", "只用编码人甲标注")):
        F = A if tag == "devonly" else B
        L += days_by_work(key, F, R, tag=f"days_by_work_{tag}", title=f"### 8.{1 if tag == 'devonly' else 2} {tn}的按日折算（机制分期）")
    return L


def circularity(key, F, R):
    """效标与提交是否不独立（D19-4）：工作项数与片段内提交命令数、与证据包第 4 节所列任务数的关系（系统 A 片段内提交组）。"""
    from robustness7 import spearman
    ncc = {}
    for s in ("SYS_A", "SYS_C"):
        ncc.update({r["episode"]: int(r["n_commit_cmds"]) for r in load(S_path(s) / "episodes.csv")})
    rows = []
    for p, k in key.items():
        t = (OUT / "packets" / f"{p}.md").read_text(encoding="utf-8")
        line = next((l for l in t.split("\n") if l.startswith("- 本片段对应到的登记任务：") or l.startswith("- 本片段请求中写明的登记任务：")), "")
        nt = line.count("；") + 1 if line else 0
        rows.append({"s": k["system"], "cmd": k["cmd"], "n": F[p]["n"], "ncc": ncc[k["episode"]], "nt": nt})
    g = [r for r in rows if r["s"] == "SYS_A" and r["cmd"]]
    lt = [r for r in g if r["nt"]]
    out = {"rho_ncc": spearman([r["n"] for r in g], [r["ncc"] for r in g]),
           "rho_nt": spearman([r["n"] for r in g], [r["nt"] for r in g]),
           "n_listed": len(lt), "eq_listed": sum(r["n"] == r["nt"] for r in lt),
           "mean_n_listed": sum(r["n"] for r in lt) / len(lt) if lt else float("nan"),
           "mean_n_unlisted": sum(r["n"] for r in g if not r["nt"]) / max(1, len(g) - len(lt)),
           "eq_ncc": sum(r["n"] == r["ncc"] for r in g), "n": len(g)}
    R["circularity"] = out
    R["final_n"] = {p: F[p]["n"] for p in key}
    L = ["## 5 效标与提交的独立性（系统 A 片段内提交组，D19）", "",
         f"- 工作项数与片段内提交命令数的 Spearman ρ = {out['rho_ncc']:.2f}，二者相等的片段 {out['eq_ncc']}/{out['n']}。"]
    if lt:
        L.append(f"- 证据包第 4 节列出任务的片段 {out['n_listed']}/{out['n']}；其工作项数均值 {out['mean_n_listed']:.2f}，未列出的 {out['mean_n_unlisted']:.2f}；"
                 f"工作项数等于所列任务数的 {out['eq_listed']}/{out['n_listed']}；工作项数与所列任务数的 ρ = {out['rho_nt']:.2f}。")
    else:
        L.append(f"- 证据包第 4 节列出任务的片段 0/{out['n']}。")
    if VER == "v2":
        v1 = json.loads(NUMS.read_text(encoding="utf-8")).get("robustness6", {}).get("boundary_labels_v1", {}).get("final_n")
        if v1:
            leak = set()
            for p in key:
                t = (Path("data/coding/boundary/packets") / f"{p}.md").read_text(encoding="utf-8")
                if any(l.startswith("- 本片段对应到的登记任务：") for l in t.split("\n")):
                    leak.add(p)
            cmp = {}
            for name, ps in (("leak", sorted(leak)), ("clean", sorted(set(key) - leak))):
                a, b = [v1[p] for p in ps], [F[p]["n"] for p in ps]
                cmp[name] = {"n": len(ps), "v1": sum(a) / len(a), "v2": sum(b) / len(b), "same": sum(x == y for x, y in zip(a, b)),
                             "lower": sum(y < x for x, y in zip(a, b))}
            R["v1_vs_v2"] = cmp
            k, c = cmp["leak"], cmp["clean"]
            L.append(f"- 第一版证据包列出由提交推出的任务的 {k['n']} 个样本：每片段工作项数第一版 {k['v1']:.2f}、本版 {k['v2']:.2f}，"
                     f"本版较低的 {k['lower']} 个、相同的 {k['same']} 个；其余 {c['n']} 个样本：第一版 {c['v1']:.2f}、本版 {c['v2']:.2f}，相同的 {c['same']} 个。")
    return L + [""]


def items_vs_works(key, A, B, F, R):
    """D32 B8：同 100 个样本片段上，请求中列举的事项数（robustness8 的 ITEM 规则，至少 1）与标注的工作项数（开发者、独立编码人、裁决口径）的一致性。
    报告 Spearman ρ、相等比例、列举数 > / < 工作项数的片段数；按系统与片段内提交分组。"""
    from robustness7 import spearman
    from robustness8 import ITEM
    txt = {}
    for s in ("SYS_A", "SYS_C"):
        txt.update({e["episode"]: e["request_text"] for e in ctx_of(s)["x"]["eps"]})
    ps = sorted(key)
    it = {p: max(1, len(ITEM.findall(txt[key[p]["episode"]]))) for p in ps}
    out = {}
    L = ["## 7 工作项数与请求中列举的事项数（D32 B8）", "",
         "| 片段 | n | 口径 | Spearman ρ | 相等 | 列举数多于工作项数 | 列举数少于工作项数 | 列举数均值 / 工作项数均值 |", "|---|---|---|---|---|---|---|---|"]
    groups = [("全部", ps)] + [(f"{NAME[s]}·{'片段内提交' if f else '未提交'}", [p for p in ps if key[p]["system"] == s and key[p]["cmd"] == f]) for s in ("SYS_A", "SYS_C") for f in (True, False)]
    for gn, gp in groups:
        for cn, src in (("开发者", A), ("独立编码人", B), ("裁决", F)):
            x, y = [it[p] for p in gp], [src[p]["n"] for p in gp]
            r = {"n": len(gp), "rho": spearman(x, y) if len(gp) > 2 else float("nan"), "eq": sum(a == b for a, b in zip(x, y)),
                 "more": sum(a > b for a, b in zip(x, y)), "less": sum(a < b for a, b in zip(x, y)),
                 "mean_items": sum(x) / len(x), "mean_works": sum(y) / len(y)}
            out[f"{gn}|{cn}"] = r
            L.append(f"| {gn} | {r['n']} | {cn} | {r['rho']:.2f} | {r['eq']}（{pc(r['eq'] / r['n'])}） | {r['more']} | {r['less']} | {r['mean_items']:.2f} / {r['mean_works']:.2f} |")
    out["items_dist"] = dict(Counter(it.values()))
    # 行内编号变体（非冻结口径，只作对照）：编号不在行首、以空白或标点隔开的 "2、" "3." 也计
    INL = re.compile(r"(?:^|[\s，。；,;])(\d{1,2}|[一二三四五六七八九十])\s*[、.．)）:：]")
    it2 = {p: max(1, len(INL.findall(txt[key[p]["episode"]]))) for p in ps}
    x, y = [it2[p] for p in ps], [F[p]["n"] for p in ps]
    out["inline_variant|裁决"] = {"n": len(ps), "rho": spearman(x, y), "eq": sum(a == b for a, b in zip(x, y)), "more": sum(a > b for a, b in zip(x, y)),
                                  "less": sum(a < b for a, b in zip(x, y)), "mean_items": sum(x) / len(x), "mean_works": sum(y) / len(y), "dist": dict(Counter(x))}
    R["items_vs_works"] = out
    v = out["inline_variant|裁决"]
    L += ["", "列举事项数为请求中以编号开头的行数（与 r16 的“请求中列举的事项”同一规则，至少 1）；工作项数为编码表的“含几项工作”（含 1 项）。"
          f"列举事项数的分布：{'，'.join(f'{k} 项 {v_} 个' for k, v_ in sorted(out['items_dist'].items()))}。",
          f"对照（非冻结口径）：若把行内编号（如“1、…，2、…”）也计入，100 个样本的列举数分布为 {'，'.join(f'{k} 项 {v_} 个' for k, v_ in sorted(v['dist'].items()))}，"
          f"与裁决口径工作项数的 Spearman ρ = {v['rho']:.2f}，相等 {v['eq']}/100，列举数多于 / 少于工作项数 {v['more']} / {v['less']}，均值 {v['mean_items']:.2f} / {v['mean_works']:.2f}。", ""]
    return L


if __name__ == "__main__":
    {"sample": sample, "repack": repack, "disagree": disagree, "score": score}[sys.argv[1]]()
