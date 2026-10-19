"""对齐方法的评估（S19-T-C2，D25）：在智能体提交命令确定的参照上报告召回率、一致率与混淆矩阵，
并与"该文件的下一次提交"的按时间归属基线对照。正文不称精确率或金标准。

参照：片段 E 中执行的提交命令产生提交 C，且 C 改动了 E 编辑过的代码文件（agent_commits.commit_sources）。
一致率限定在参照提交上：方法给出的对 (E', C) 分严格一致（E' = E）、可解释（同会话且 E' 开始不晚于 C）、可疑（其余）。
基线：片段中每次代码编辑按时间归到该文件的下一次提交，不看内容。

写入 <PAPER_DIR>/results/r18_align_eval.md 与 numbers.json 的 align_eval 键（须在 render_tables.py 之后运行）。
"""
import csv
import json
import random
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import commit_sources  # noqa: E402
from config import RESULTS_DIR  # noqa: E402
from workitems import wilson  # noqa: E402

NUMS = RESULTS_DIR / "numbers.json"
SYS_NAME = {"SYS_A": "系统 A", "SYS_C": "系统 B"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pc(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n}（{k / n:.0%}，{lo:.0%}–{hi:.0%}）" if n else "0/0"


def baseline_pairs(d, eps, commits):
    """按时间归属：片段中每次代码编辑 → 该文件时间不早于编辑时间的第一个提交。"""
    by_path = defaultdict(list)
    for f in load(d / "commit_files.csv"):
        by_path[f["path"].lower()].append((commits[f["hash"]]["time"], f["hash"]))
    for p in by_path:
        by_path[p].sort()
    code_paths = {(f["episode"], f["path"].lower()) for f in load(d / "episode_files.csv") if f["class"] == "code"}
    ep_of = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_events.csv")}
    pairs = defaultdict(set)
    for e in load(d / "events.csv"):
        ep = ep_of.get((e["session"], e["seq"]))
        if not ep or e["tool"] != "edit_file_v2" or e["status"] != "completed" or e["in_workspace"] != "1" or not e["time"]:
            continue
        p = e["path"].replace("\\", "/").lower()
        if (ep, p) not in code_paths:
            continue
        cands = by_path.get(p, [])
        i = bisect_left(cands, (e["time"], ""))
        if i < len(cands):
            pairs[ep].add(cands[i][1])
    return pairs


def evaluate(pairs, gold, rel, eps, commits, verified):
    """pairs: {episode: set(commit)}；gold: {commit: episode}；rel: [(commit, episode)] 参照关系。"""
    eps_of = defaultdict(set)
    for ep, cs in pairs.items():
        for c in cs:
            eps_of[c].add(ep)
    out = {"n_rel": len(rel), "recall_k": sum(c in pairs.get(ep, ()) for c, ep in rel)}
    rv = [(c, ep) for c, ep in rel if ep in verified]
    out["n_rel_verified"] = len(rv)
    out["recall_verified_k"] = sum(c in pairs.get(ep, ()) for c, ep in rv)
    strict = explain = suspect = suspect_other_sess = suspect_later = 0
    for c, e_gold in gold.items():
        for e in eps_of.get(c, ()):
            if e == e_gold:
                strict += 1
            elif eps[e]["session"] == eps[e_gold]["session"] and eps[e]["start"] <= commits[c]["time"]:
                explain += 1
            else:
                suspect += 1
                if eps[e]["session"] != eps[e_gold]["session"]:
                    suspect_other_sess += 1
                else:
                    suspect_later += 1
    out.update({"strict": strict, "explain": explain, "suspect": suspect, "suspect_other_session": suspect_other_sess,
                "suspect_later_same_session": suspect_later, "n_pairs_on_gold": strict + explain + suspect,
                "n_pairs_total": sum(len(v) for v in pairs.values()),
                "pairs_per_episode_median": median(len(v) for v in pairs.values()) if pairs else 0,
                "n_episodes_mapped": len(pairs)})
    out["fn"] = out["n_rel"] - out["recall_k"]
    # D26-5：按会话聚类重抽的 95% 区间（召回率、可核验召回率、严格一致率），与表 6、表 7 的区间构造一致
    rng = random.Random("D26|align_eval")
    by_rel = defaultdict(list)
    for c, ep in rel:
        by_rel[eps[ep]["session"]].append((c, ep, c in pairs.get(ep, ()), ep in verified))
    by_gold = defaultdict(list)
    for c, e_gold in gold.items():
        for e in eps_of.get(c, ()):
            by_gold[eps[e_gold]["session"]].append(e == e_gold)
    ks = sorted(by_rel)
    kg = sorted(by_gold)
    bs = defaultdict(list)
    for _ in range(2000):
        smp = [x for k in rng.choices(ks, k=len(ks)) for x in by_rel[k]]
        if smp:
            bs["recall"].append(sum(x[2] for x in smp) / len(smp))
            v = [x for x in smp if x[3]]
            if v:
                bs["recall_verified"].append(sum(x[2] for x in v) / len(v))
        sg = [x for k in rng.choices(kg, k=len(kg)) for x in by_gold[k]]
        if sg:
            bs["precision_strict"].append(sum(sg) / len(sg))
    qq = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[min(len(v) - 1, int(0.975 * len(v)))]] if v else [float("nan")] * 2
    out["cluster_ci"] = {k: qq(v) for k, v in bs.items()}
    out["n_sessions_rel"] = len(ks)
    return out


def compare_pairs(d, content, base, files_of_c):
    """两种方法给出的对集合之差；基线独有的对按该片段在该提交所改文件上的编辑的内容状态分类。"""
    cp = {(e, c) for e, cs in content.items() for c in cs}
    bp = {(e, c) for e, cs in base.items() for c in cs}
    ep_of = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_events.csv")}
    attr = defaultdict(list)  # (episode, path) -> [(state, commit)]
    for a in load(d / "edit_attribution.csv"):
        ep = ep_of.get((a["session"], a["seq"]))
        if ep:
            attr[(ep, a["path"].lower())].append((a["state"], a["commit"]))
    cls = {"elsewhere": 0, "not_committed": 0, "no_content": 0}
    for e, c in bp - cp:
        states = [x for p in files_of_c[c] for x in attr.get((e, p), [])]
        if any(st == "committed" and h != c for st, h in states):
            cls["elsewhere"] += 1
        elif states:
            cls["not_committed"] += 1
        else:
            cls["no_content"] += 1
    return {"both": len(cp & bp), "content_only": len(cp - bp), "baseline_only": len(bp - cp), "baseline_only_by_state": cls}


def run(s):
    d = Path("data/derived") / s
    commits, gold, _ = commit_sources(s)
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    verified = {r["episode"] for r in load(d / "episode_patterns.csv") if r["evidence"] == "verified"}
    files_of_c = defaultdict(set)
    for f in load(d / "commit_files.csv"):
        files_of_c[f["hash"]].add(f["path"].lower())
    code_files = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"].lower())
    rel = [(c, ep) for c, ep in gold.items() if code_files[ep] & files_of_c[c]]

    content = defaultdict(set)
    for p in load(d / "episode_commit.csv"):
        if p["source"] != "pathtime":
            content[p["episode"]].add(p["commit"])
    base = baseline_pairs(d, eps, commits)
    res = {"content": evaluate(content, gold, rel, eps, commits, verified),
           "baseline_time": evaluate(base, gold, rel, eps, commits, verified),
           "n_gold_commits": len(gold), "pairs": compare_pairs(d, content, base, files_of_c)}

    L = [f"## {SYS_NAME[s]}", "",
         f"参照关系（片段执行命令产生提交，且提交改动了片段编辑过的代码文件）{len(rel)} 组；参照提交 {len(gold)} 个。", "",
         "| 指标 | 内容对齐（本文方法） | 按时间归属（该文件的下一次提交） |", "|---|---|---|"]
    m, b = res["content"], res["baseline_time"]
    cc = lambda r, k: f"；按会话重抽 {r['cluster_ci'][k][0]:.0%}–{r['cluster_ci'][k][1]:.0%}" if k in r["cluster_ci"] else ""
    L += [f"| 召回率（全部参照关系） | {pc(m['recall_k'], m['n_rel'])}{cc(m, 'recall')} | {pc(b['recall_k'], b['n_rel'])}{cc(b, 'recall')} |",
          f"| 召回率（片段可核验） | {pc(m['recall_verified_k'], m['n_rel_verified'])}{cc(m, 'recall_verified')} | {pc(b['recall_verified_k'], b['n_rel_verified'])}{cc(b, 'recall_verified')} |",
          f"| 参照提交上的片段—提交对 | {m['n_pairs_on_gold']} | {b['n_pairs_on_gold']} |",
          f"| 　严格正确（E′ = E） | {m['strict']} | {b['strict']} |",
          f"| 　可解释（同会话、更早片段） | {m['explain']} | {b['explain']} |",
          f"| 　可疑（其他会话 / 同会话更晚） | {m['suspect']}（{m['suspect_other_session']} / {m['suspect_later_same_session']}） | {b['suspect']}（{b['suspect_other_session']} / {b['suspect_later_same_session']}） |",
          f"| 严格一致率（原称精确率，D30） | {pc(m['strict'], m['n_pairs_on_gold'])}{cc(m, 'precision_strict')} | {pc(b['strict'], b['n_pairs_on_gold'])}{cc(b, 'precision_strict')} |",
          f"| 宽松一致率（严格 + 可解释） | {pc(m['strict'] + m['explain'], m['n_pairs_on_gold'])} | {pc(b['strict'] + b['explain'], b['n_pairs_on_gold'])} |",
          f"| 全部片段—提交对数 / 有对的片段数 | {m['n_pairs_total']} / {m['n_episodes_mapped']} | {b['n_pairs_total']} / {b['n_episodes_mapped']} |",
          f"| 每个有对片段的对数中位数 | {m['pairs_per_episode_median']:g} | {b['pairs_per_episode_median']:g} |", "",
          "混淆矩阵（参照提交 × 片段）：TP = 被映射的参照关系，FN = 未映射的参照关系，FP = 可疑对；可解释对单列。", "",
          "| 方法 | TP | FN | FP（可疑） | 可解释 |", "|---|---|---|---|---|",
          f"| 内容对齐 | {m['recall_k']} | {m['fn']} | {m['suspect']} | {m['explain']} |",
          f"| 按时间归属 | {b['recall_k']} | {b['fn']} | {b['suspect']} | {b['explain']} |", ""]
    q = res["pairs"]
    k = q["baseline_only_by_state"]
    L += [f"两种方法的全部片段—提交对：共有 {q['both']} 对，只有内容对齐给出 {q['content_only']} 对，只有基线给出 {q['baseline_only']} 对。"
          f"基线独有的对中，该片段在该提交所改文件上的编辑按内容被归到了其他提交的 {k['elsewhere']} 对（基线归错了提交），"
          f"按内容判定未进入任何提交（被覆盖、撤销或未提交）的 {k['not_committed']} 对（基线把不存在的对应关系记为存在），"
          f"没有内容可判的 {k['no_content']} 对。", ""]
    return L, res


if __name__ == "__main__":
    out = ["# 对齐方法的评估：召回率、参照一致率与按时间归属的基线（S19-T-C2，D25）", "",
           "由 `analysis/align_eval.py` 生成。参照只知道谁执行了提交，不知道提交还含有哪些片段的内容，"
           "因此严格一致率是下界；宽松一致率把同会话更早片段的内容一并提交视为正常合并。区间先给 Wilson 95%，分号后为按会话聚类重抽 2,000 次的 95% 区间（D26-5；正文表 5 用后者）。", ""]
    num = json.loads(NUMS.read_text(encoding="utf-8")) if NUMS.exists() else {}  # 单独运行核心流程时可无 numbers.json
    num["align_eval"] = {}
    for s in ["SYS_A", "SYS_C"]:
        L, res = run(s)
        out += L
        num["align_eval"][s] = res
    (RESULTS_DIR / "r18_align_eval.md").write_text("\n".join(out), encoding="utf-8")
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out))
