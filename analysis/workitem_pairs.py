"""工作项边界校验（S10-T2）。

sample：从两个系统同一会话内相邻的代码修改片段对中，按规则判定分三层（strict 合并 / 仅 loose 合并 / 不合并）
        各抽 20 对，生成盲评材料 data/coding/workitem_pairs/pairs.md（不含规则判定）与答案 key.csv。
score ：读取 data/coding/workitem_pairs/coder*.csv（pair,same），计算编码人一致性与各规则相对共识的精确率、召回率，
        写入 <PAPER_DIR>/results/r5_workitem_check.md。
"""
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import agreement  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from workitems import wilson  # noqa: E402

OUT = Path("data/coding/workitem_pairs")
HUMAN = os.environ.get("CODING_HUMAN", "0") == "1"  # D29
CODER_DIR = Path("data/coding/human/workitem_pairs") if HUMAN else OUT  # 编码表所在目录（key 仍在 OUT）
SEED = "S10-20260929"
PER = 20


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pairs(s):
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    wi = {r["episode"]: r for r in load(d / "workitems.csv")}
    files = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            files[f["episode"]].add(f["path"])
    by_s = defaultdict(list)
    for e in eps.values():
        if int(e["n_edits_code"]):
            by_s[e["session"]].append(e)
    out = []
    for es in by_s.values():
        es.sort(key=lambda e: (e["start"], int(e["msg_seq"])))
        for a, b in zip(es, es[1:]):
            st = "strict" if wi[a["episode"]]["wi_strict"] == wi[b["episode"]]["wi_strict"] else (
                "loose" if wi[a["episode"]]["wi_loose"] == wi[b["episode"]]["wi_loose"] else "none")
            gap = (datetime.fromisoformat(b["start"]) - datetime.fromisoformat(a["last_event"])).total_seconds() / 60
            out.append({"a": a, "b": b, "stratum": st, "gap": gap, "fa": files[a["episode"]], "fb": files[b["episode"]]})
    return out


def short(paths):
    names = sorted({p.replace("\\", "/").split("/")[-1] for p in paths})
    return "、".join(names[:8]) + ("等" if len(names) > 8 else "")


def sample(full=False):
    allp = pairs("SYS_A") + pairs("SYS_C")
    rng = random.Random(SEED)
    picked = []
    if full:
        picked = list(allp)
    else:
        for st in ("strict", "loose", "none"):
            pool = [p for p in allp if p["stratum"] == st]
            picked += rng.sample(pool, min(PER, len(pool)))
    rng.shuffle(picked)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_all" if full else ""
    L = ["# 相邻工作片段对：是否属于同一项工作", "",
         "每对是同一会话中先后相邻的两个代码修改片段（中间可能隔着不改代码的消息）。给出两条开发者请求原文（截断至 400 字）、"
         "各自改动的代码文件名，以及前一片段结束到后一片段开始的间隔。", ""]
    with open(OUT / f"key{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "system", "a", "b", "stratum", "gap_min", "share_file"])
        for i, p in enumerate(picked, 1):
            pid = f"Q{i:03d}" if full else f"P{i:02d}"
            w.writerow([pid, p["a"]["system"], p["a"]["episode"], p["b"]["episode"], p["stratum"], f"{p['gap']:.0f}",
                        int(bool(p["fa"] & p["fb"]))])
            ta = p["a"]["request_text"][:400].replace("\n", " ")
            tb = p["b"]["request_text"][:400].replace("\n", " ")
            L += [f"## {pid}", "", f"- 间隔：{p['gap']:.0f} 分钟", f"- 前一片段请求：{ta}", f"- 前一片段改动的代码文件：{short(p['fa'])}",
                  f"- 后一片段请求：{tb}", f"- 后一片段改动的代码文件：{short(p['fb'])}", ""]
    (OUT / f"pairs{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print("pairs:", len(picked), {st: sum(p["stratum"] == st for p in picked) for st in ("strict", "loose", "none")},
          "population:", {st: sum(p["stratum"] == st for p in allp) for st in ("strict", "loose", "none")})


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score():
    key = {r["pair"]: r for r in load(OUT / "key.csv")}
    coders = [OUT / "coder1.csv", OUT / "coder2.csv"]
    ans = [{r["pair"]: int(r["same"]) for r in load(c)} for c in coders]
    ids = sorted(key)
    assert all(set(a) == set(ids) for a in ans), "coder files incomplete"
    c1, c2 = [ans[0][i] for i in ids], [ans[1][i] for i in ids]
    agree = sum(a == b for a, b in zip(c1, c2))
    cons = {i: a for i, a, b in zip(ids, c1, c2) if a == b}
    L = ["# 工作项边界校验（S10-T2，AI 模拟编码）", "",
         f"由 `analysis/workitem_pairs.py` 生成。{len(ids)} 对相邻代码修改片段（种子 `{SEED}`，三层各 {PER} 对），两名编码人独立盲评"
         "（不知分层与规则判定）。以两人一致的判断为参照，不一致的对不计入精确率与召回率。", "",
         f"- 编码人一致：{agree}/{len(ids)}（{agree / len(ids):.0%}），Cohen κ = {kappa(c1, c2):.2f}",
             f"- 完整报告（D26-5）：{agreement.fmt(agreement.kappa_stats(c1, c2, 'wp|sample'), '编码人 1', '编码人 2')}",
         f"- 参照（两人一致）：{len(cons)} 对，其中判为同一项工作 {sum(cons.values())} 对", "",
         "| 分层（规则判定） | 对数 | 参照判为同一项工作 |", "|---|---|---|"]
    for st in ("strict", "loose", "none"):
        ids_st = [i for i in cons if key[i]["stratum"] == st]
        k = sum(cons[i] for i in ids_st)
        lo, hi = wilson(k, len(ids_st))
        L.append(f"| {st} | {len(ids_st)} | {k}（{k / max(1, len(ids_st)):.0%}，{lo:.0%}–{hi:.0%}） |")
    pop = {st: sum(p["stratum"] == st for p in pairs("SYS_A") + pairs("SYS_C")) for st in ("strict", "loose", "none")}
    n_st = {st: sum(key[i]["stratum"] == st for i in cons) for st in pop}
    wt = {i: pop[key[i]["stratum"]] / n_st[key[i]["stratum"]] for i in cons}
    L += ["", f"总体相邻代码片段对：strict {pop['strict']}、仅 loose {pop['loose']}、不合并 {pop['none']}。"
          "精确率与召回率按各层总体规模加权（括号内为未加权计数）。", "",
          "| 规则 | 加权精确率 | 加权召回率 |", "|---|---|---|"]
    for rule, joined in (("strict", {"strict"}), ("loose", {"strict", "loose"})):
        pred = {i: key[i]["stratum"] in joined for i in cons}
        tp = sum(wt[i] for i in cons if pred[i] and cons[i])
        fp = sum(wt[i] for i in cons if pred[i] and not cons[i])
        fn = sum(wt[i] for i in cons if not pred[i] and cons[i])
        ctp = sum(1 for i in cons if pred[i] and cons[i])
        cfp = sum(1 for i in cons if pred[i] and not cons[i])
        cfn = sum(1 for i in cons if not pred[i] and cons[i])
        L.append(f"| {rule} | {tp / (tp + fp):.0%}（{ctp}/{ctp + cfp}） | {tp / (tp + fn):.0%}（{ctp}/{ctp + cfn}） |")
    L.append("")
    (RESULTS_DIR / "r5_workitem_check.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def score_full():
    key = {r["pair"]: r for r in load(OUT / "key_all.csv")}
    ans = []
    for i in (1, 2):
        a = {}
        for f in sorted(CODER_DIR.glob(f"coder{i}_all_*.csv")):
            a.update({r["pair"]: int(r["same"]) for r in load(f)})
        ans.append(a)
    ids = sorted(key)
    assert all(set(a) == set(ids) for a in ans), "coder files incomplete"
    c1, c2 = [ans[0][i] for i in ids], [ans[1][i] for i in ids]
    sess = {}
    for s in ("SYS_A", "SYS_C"):
        sess.update({e["episode"]: e["session"] for e in load(Path("data/derived") / s / "episodes.csv")})
    clus = [sess.get(key[i]["a"], key[i]["a"]) for i in ids]
    st = agreement.kappa_stats(c1, c2, "wp|full", clusters=clus)
    dis = [i for i, a, b in zip(ids, c1, c2) if a != b]
    adj = {r["pair"]: int(r["same"]) for r in load(CODER_DIR / "adjudicate_all.csv")} if (CODER_DIR / "adjudicate_all.csv").exists() else {}
    final = {i: (a if a == b else adj.get(i)) for i, a, b in zip(ids, c1, c2)}
    L = ["# 相邻代码片段对的全量编码（S10-T2，" + ("真人编码：甲、乙，分歧由第二作者裁定（D30-1）；D29" if HUMAN else "AI 模拟编码") + "）", "",
         f"由 `analysis/workitem_pairs.py` 生成。两个系统全部 {len(ids)} 对相邻代码修改片段，两名编码人各自独立判断是否属于同一项工作；"
         + ("分歧由第二作者在答案匿名的情况下裁决。" if HUMAN else "分歧由第三人裁决。"), "",
         f"- 编码人一致：{len(ids) - len(dis)}/{len(ids)}（{1 - len(dis) / len(ids):.0%}），Cohen κ = {kappa(c1, c2):.2f}；"
         f"分歧 {len(dis)} 对，已裁决 {sum(i in adj for i in dis)} 对。",
         f"- 完整报告（D26-5）：{agreement.fmt(st, '编码人 1', '编码人 2')}。",
         f"- 按会话整块重抽的 κ 95% 区间（D32；{st['n_clusters']} 个会话）：{st['ci_cluster'][0]:.2f}–{st['ci_cluster'][1]:.2f}。"]
    done = {i: v for i, v in final.items() if v is not None}
    L.append(f"- 最终判为同一项工作：{sum(done.values())}/{len(done)}（{sum(done.values()) / len(done):.0%}）。")
    for s in ("SYS_A", "SYS_C"):
        ii = [i for i in done if key[i]["system"] == s]
        L.append(f"  - {s}：{sum(done[i] for i in ii)}/{len(ii)}")
    L += ["", "| 规则 | 精确率 | 召回率 |", "|---|---|---|"]
    for rule, joined in (("strict", {"strict"}), ("loose", {"strict", "loose"})):
        tp = sum(1 for i in done if key[i]["stratum"] in joined and done[i])
        fp = sum(1 for i in done if key[i]["stratum"] in joined and not done[i])
        fn = sum(1 for i in done if key[i]["stratum"] not in joined and done[i])
        L.append(f"| {rule} | {tp}/{tp + fp}（{tp / (tp + fp):.0%}） | {tp}/{tp + fn}（{tp / (tp + fn):.0%}） |")
    L.append("")
    (RESULTS_DIR / "r5_workitem_coding.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("DISAGREE:", ",".join(dis))


if __name__ == "__main__":
    {"sample": sample, "full": lambda: sample(True), "score": score, "score_full": score_full}[sys.argv[1]]()
