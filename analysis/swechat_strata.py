"""D35-6（B5/B6）与 D36-5：SWE-chat 复制的三项补充——逐仓库对照表与按仓库聚类的区间；按提交触发方式分层的每单位提交数之比（事后、探索性）；
按 SWE-chat 附录 D.3 的"能无歧义归属"标准在会话层切分，报告被其保留与被其排除两侧的片段映射模式（对照）。

B5 逐仓库：基准、V1 Write 去继承、V2 first-parent 三口径的逐仓库映射模式（V3 是基准的子集，不另列）；
   五个仓库合计比例按仓库整块重抽 2,000 次的百分位区间（只有 5 个簇，区间粗糙，只说明量级）。
B6 分层：按会话中智能体提交由谁执行分三层——只由不改代码的片段执行（单独一条提交指令）、只由改过代码的片段执行（片段内提交）、
   两者都有（混合）；另列没有可对上的智能体提交的会话。每层报告代码修改片段数、代码编辑数、按内容归属到本层片段的提交数、
   每片段与每百次代码编辑的提交数，以及"单独指令"对"片段内提交"两层之比（按会话整块重抽 2,000 次）；并报各层的一对一 / 共享 / 拆分比例。
   这是 RQ3 调节变量在公开语料中唯一可得的代理：提交由单独指令触发还是在工作片段内触发；它不是事前定义的工作方式，只作探索。
写入 <PAPER_DIR>/results/r25_swechat_strata.md 与 numbers.json 的 swechat_strata 键。不改 r21 与 swechat 键。
"""
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR  # noqa: E402
import swechat_replicate as sw  # noqa: E402

BOOT, SEED = 2000, "S25-swechat"
NUMS = RESULTS_DIR / "numbers.json"
PAT = (("一对一", ("one_to_one",)), ("共享", ("shared", "split+shared")), ("拆分", ("split", "split+shared")), ("缺失", ("missing",)))


def q(v, p):
    v = sorted(x for x in v if x == x)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def pc(x):
    return "—" if x != x else f"{x:.0%}"


def per_repo_table(num):
    sw_n = num["swechat"]
    L = ["## 1 逐仓库对照（B5）", "",
         "| 仓库 | 口径 | 代码修改片段 | 一对一 | 共享 | 拆分 | 缺失 |", "|---|---|---|---|---|---|---|"]
    out = {}
    sets = [("基准", sw_n["repos"]), ("V1 Write 去继承", sw_n["variants_per_repo"].get("v1_write_deinherit", [])),
            ("V2 first-parent", sw_n["variants_per_repo"].get("v2_first_parent", []))]
    by_repo = defaultdict(dict)
    for name, rows in sets:
        for r in rows:
            by_repo[r["repo"]][name] = r
    for repo, d in by_repo.items():
        for name, _ in sets:
            r = d.get(name)
            if not r:
                continue
            p, n = r["patterns"], r["code_episodes"]
            vals = {k: sum(p.get(x, 0) for x in ks) / n if n else float("nan") for k, ks in PAT}
            out.setdefault(repo, {})[name] = {"n": n, **vals}
            L.append(f"| {repo} | {name} | {n} | " + " | ".join(pc(vals[k]) for k, _ in PAT) + " |")
    # 按仓库整块重抽的区间（基准口径）
    base = sw_n["repos"]
    rng = random.Random(SEED + "|repo")
    bs = defaultdict(list)
    for _ in range(BOOT):
        pick = [rng.choice(base) for _ in base]
        n = sum(r["code_episodes"] for r in pick)
        for k, ks in PAT:
            bs[k].append(sum(sum(r["patterns"].get(x, 0) for x in ks) for r in pick) / n if n else float("nan"))
    ci = {k: [q(v, .025), q(v, .975)] for k, v in bs.items()}
    n = sum(r["code_episodes"] for r in base)
    pooled = {k: sum(sum(r["patterns"].get(x, 0) for x in ks) for r in base) / n for k, ks in PAT}
    out["pooled_repo_cluster_ci"] = {k: {"p": pooled[k], "ci": ci[k]} for k, _ in PAT}
    L += ["", "基准口径五仓库合计（按仓库整块重抽 2,000 次的 95% 区间；只有 5 个簇，区间粗糙，只说明量级）：" +
          "；".join(f"{k} {pc(pooled[k])}（{pc(ci[k][0])}–{pc(ci[k][1])}）" for k, _ in PAT) + "。", ""]
    return out, L


def strata(all_eps):
    """all_eps：[(repo, episodes, ref_commits, content_pairs)]。"""
    # 会话层：智能体提交的执行片段是否改过代码
    sess = {}
    for repo, episodes, ref_commits, content_pairs in all_eps:
        exec_by = defaultdict(lambda: {"code": 0, "nocode": 0})
        for h, i in ref_commits.items():
            exec_by[episodes[i]["session"]]["code" if episodes[i]["code_edits"] else "nocode"] += 1
        c_of = defaultdict(set)
        for i, h in content_pairs:
            c_of[i].add(h)
        for i, ep in enumerate(episodes):
            if not ep["code_edits"]:
                continue
            s = (repo, ep["session"])
            sess.setdefault(s, {"eps": [], "exec": exec_by[ep["session"]]})
            sess[s]["eps"].append({"w": ep["code_edits"], "C": c_of.get(i, set()), "pattern": ep["pattern"]})

    def layer(e):
        if e["code"] and not e["nocode"]:
            return "片段内提交"
        if e["nocode"] and not e["code"]:
            return "单独提交指令"
        if e["code"] and e["nocode"]:
            return "混合"
        return "无可对上的智能体提交"
    groups = defaultdict(list)
    for s, d in sess.items():
        groups[layer(d["exec"])].append(d)

    def agg(G):
        eps = [e for d in G for e in d["eps"]]
        n = len(eps)
        w = sum(e["w"] for e in eps)
        C = len({c for e in eps for c in e["C"]})
        pats = Counter(e["pattern"] for e in eps)
        return {"sessions": len(G), "n": n, "w": w, "C": C, "c_ep": C / n if n else float("nan"), "c_100w": 100 * C / w if w else float("nan"),
                "pat": {k: sum(pats.get(x, 0) for x in ks) / n if n else float("nan") for k, ks in PAT}}
    order = ["单独提交指令", "片段内提交", "混合", "无可对上的智能体提交"]
    out = {g: agg(groups[g]) for g in order if groups.get(g)}
    L = ["## 2 按提交触发方式分层（B6；事后、探索性）", "",
         "| 层（会话中智能体提交的执行片段） | 会话 | 代码修改片段 | 代码编辑 | 按内容归属的提交 | 每片段提交数 | 每百次代码编辑提交数 | 一对一 | 共享 | 拆分 | 缺失 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in order:
        a = out.get(g)
        if not a:
            continue
        L.append(f"| {g} | {a['sessions']} | {a['n']} | {a['w']:,} | {a['C']} | {a['c_ep']:.2f} | {a['c_100w']:.2f} | " +
                 " | ".join(pc(a["pat"][k]) for k, _ in PAT) + " |")
    # 两层之比：单独提交指令 对 片段内提交，按会话整块重抽
    A, B = groups.get("单独提交指令", []), groups.get("片段内提交", [])
    ratio = {}
    if A and B:
        rng = random.Random(SEED + "|strata")
        bs = defaultdict(list)
        for _ in range(BOOT):
            x, y = agg([rng.choice(A) for _ in A]), agg([rng.choice(B) for _ in B])
            for m in ("c_ep", "c_100w"):
                bs[m].append(x[m] / y[m] if y[m] else float("nan"))
        a, b = out["单独提交指令"], out["片段内提交"]
        ratio = {m: {"ratio": a[m] / b[m], "ci": [q(bs[m], .025), q(bs[m], .975)]} for m in ("c_ep", "c_100w")}
        L += ["", f"单独提交指令层对片段内提交层：每片段提交数之比 {ratio['c_ep']['ratio']:.2f}（{ratio['c_ep']['ci'][0]:.2f}–{ratio['c_ep']['ci'][1]:.2f}），"
              f"每百次代码编辑提交数之比 {ratio['c_100w']['ratio']:.2f}（{ratio['c_100w']['ci'][0]:.2f}–{ratio['c_100w']['ci'][1]:.2f}）；区间为两层内各自按会话整块重抽 2,000 次的百分位法。"]
    out["ratio_standalone_over_inepisode"] = ratio
    L += ["", "分层只用会话内可对上的智能体 `git commit` 回显哈希判断执行片段是否改过代码；手工提交无回显，不参与分层。"
          "提交数为按内容归属到本层片段的不同提交数（同一提交被两层片段共享时两层各计一次）。"
          "这一分层是提交方式的代理而非事前定义的工作方式，且与提交是否在工作片段内执行本身相关（机制性），只作探索。", ""]
    return out, L


def ambiguity_split(all_eps):
    """D36-5：按 SWE-chat 附录 D.3 的"能无歧义归属到提交的会话"标准切分（会话层），报告两侧的片段映射模式。

    严格口径：会话的代码修改片段按内容只进入恰好一个提交，且该提交不含其他会话的内容（一会话—一提交）；
    宽松口径：会话进入的每个提交都不含其他会话的内容（会话可跨多个提交，但没有跨会话共享）；
    无提交：会话的代码修改片段没有任何内容进入提交；其余为有歧义。
    SWE-chat 原文的标准依赖 Entire 的检查点—提交链接，这里用本文的内容归属近似，只作对照。
    """
    sess = {}
    for repo, episodes, ref_commits, content_pairs in all_eps:
        c_of = defaultdict(set)
        for i, h in content_pairs:
            c_of[i].add(h)
        sess_of_commit = defaultdict(set)
        for i, hs in c_of.items():
            for h in hs:
                sess_of_commit[h].add(episodes[i]["session"])
        for i, ep in enumerate(episodes):
            if not ep["code_edits"]:
                continue
            s = (repo, ep["session"])
            d = sess.setdefault(s, {"eps": [], "C": set(), "shared_other": False})
            d["eps"].append({"pattern": ep["pattern"]})
            for h in c_of.get(i, set()):
                d["C"].add(h)
                if len(sess_of_commit[h] - {ep["session"]}) > 0:
                    d["shared_other"] = True

    def cls(d):
        if not d["C"]:
            return "无提交"
        if not d["shared_other"] and len(d["C"]) == 1:
            return "无歧义（严格）"
        if not d["shared_other"]:
            return "无歧义（宽松）"
        return "有歧义"
    groups = defaultdict(list)
    for d in sess.values():
        groups[cls(d)].append(d)
    total_s = len(sess)

    def agg(G):
        eps = [e for d in G for e in d["eps"]]
        n = len(eps)
        pats = Counter(e["pattern"] for e in eps)
        return {"sessions": len(G), "n": n, "pat": {k: sum(pats.get(x, 0) for x in ks) / n if n else float("nan") for k, ks in PAT}}
    order = ["无歧义（严格）", "无歧义（宽松）", "有歧义", "无提交"]
    out = {g: agg(groups[g]) for g in order if groups.get(g)}
    # SWE-chat 式的保留集合 = 严格 + 宽松；被排除 = 有歧义 + 无提交
    out["保留（严格+宽松）"] = agg(groups["无歧义（严格）"] + groups["无歧义（宽松）"])
    out["排除（有歧义+无提交）"] = agg(groups["有歧义"] + groups["无提交"])
    out["total_sessions"] = total_s
    L = ["## 3 按 SWE-chat 的无歧义归属标准切分（D36-5；对照）", "",
         "| 会话类别 | 会话数（占比） | 代码修改片段 | 一对一 | 共享 | 拆分 | 缺失 |", "|---|---|---|---|---|---|---|"]
    for g in order + ["保留（严格+宽松）", "排除（有歧义+无提交）"]:
        a = out.get(g)
        if not a:
            continue
        L.append(f"| {g} | {a['sessions']}（{a['sessions'] / total_s:.0%}） | {a['n']} | " + " | ".join(pc(a["pat"][k]) for k, _ in PAT) + " |")
    L += ["", "会话层分类：严格 = 会话的代码修改片段按内容只进入一个提交且该提交不含其他会话的内容；宽松 = 会话进入的每个提交都不含其他会话的内容；"
          "有歧义 = 至少一个提交含其他会话的内容；无提交 = 没有内容进入任何提交。SWE-chat 原文（附录 D.3）以 Entire 的检查点—提交链接判定"
          "\"能无歧义归属\"的会话（v1 为 48.6%，v2 为 47.7%），这里用本文的内容归属近似，分母为有代码修改片段的 Claude Code 会话。", ""]
    return out, L


def main():
    t0 = time.time()
    repos = sw.REPOS
    canon = lambda r: sw.ALIASES.get(r, r)
    wanted = [r for r in list(sw.REPOS) + list(sw.ALIASES) if canon(r) in repos]
    ss = pq.read_table(sw.DATA / "sessions.parquet", columns=["session_id", "repo_id", "agent"]).to_pandas()
    keep_s = set(ss[ss.agent == sw.AGENT].session_id)
    conv = sw.load_conversations(wanted)
    conv = conv[conv.session_id.isin(keep_s)].copy()
    conv["repo_id"] = conv.repo_id.map(canon)
    links_raw = sw.entire_links_table(wanted)
    links = defaultdict(lambda: defaultdict(set))
    for rid, d in links_raw.items():
        for s, shas in d.items():
            links[canon(rid)][s] |= shas
    all_eps = []
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    base_by_repo = {r["repo"]: r for r in num["swechat"]["repos"]}
    for r in repos:
        kp = {}
        out = sw.replicate_repo(r, conv, links, R=sw.Repo(sw.MIRRORS / (r.replace("/", "__") + ".git")), keep=kp)
        # 与已写入 numbers.json 的基准结果逐项核对，保证本脚本复算的中间结果与论文数字同源
        assert out["patterns"] == base_by_repo[r]["patterns"] and out["code_episodes"] == base_by_repo[r]["code_episodes"], r
        all_eps.append((r, kp["episodes"], kp["ref_commits"], kp["content_pairs"]))
        print(f"[{r}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    L = ["# SWE-chat 复制的补充：逐仓库对照与按提交触发方式分层（D35-6）", "",
         "由 `analysis/swechat_strata.py` 生成；基准口径的片段级结果与 `swechat_replicate.py` 复算一致（脚本断言）。", ""]
    R = {}
    R["per_repo"], L1 = per_repo_table(num)
    R["strata"], L2 = strata(all_eps)
    R["ambiguity"], L3 = ambiguity_split(all_eps)
    L += L1 + L2 + L3
    (RESULTS_DIR / "r25_swechat_strata.md").write_text("\n".join(L), encoding="utf-8")
    num["swechat_strata"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
