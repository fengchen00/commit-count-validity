"""v2.11 补充稳健性（S13-T1，口径见 notes/decisions.md D13）。

1 多重比较（Holm）  2 提交执行者对照  3 特征行长度与条数  4 删除型改动
写入 <PAPER_DIR>/results/r10_robustness2.md，关键数字写入 numbers.json 的 robustness2 键（须在 robustness 之后运行）。
用法：python analysis/robustness2.py [--no-prepare]   （--no-prepare 跳过第 3 节的重跑）
"""
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import command_commits  # noqa: E402
from config import excluded, RESULTS_DIR  # noqa: E402
from extract_episodes import path_class  # noqa: E402
from extract_git import SYSTEMS as REPOS, git  # noqa: E402
from extract_sessions import EDIT_TOOLS, SYSTEMS as SESS, build_sequence, load_session, parse_params, signature_lines  # noqa: E402
from robustness import MERGE, NAME, ame, base, has, logit, pc, q, share  # noqa: E402

BOOT = 2000
SEED = "S13"
NUMS = (RESULTS_DIR / "numbers.json")
VARIANTS = [("", "最短 15 字符、每次编辑至多 80 条（主口径）", {}), ("_s20", "最短 20 字符", {"ATTR_SIG_MIN": "20"}),
            ("_s25", "最短 25 字符", {"ATTR_SIG_MIN": "25"}), ("_c40", "每次编辑至多 40 条", {"ATTR_SIG_CAP": "40"})]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def boot(items, tag):
    by = defaultdict(list)
    for it in items:
        by[it["s"]].append(it)
    ks = sorted(by)
    rng = random.Random(f"{SEED}|{tag}")
    return [[it for k in rng.choices(ks, k=len(ks)) for it in by[k]] for _ in range(BOOT)]


def pval(draws):
    d = [v for v in draws if v is not None and v == v]
    lo, hi = sum(v <= 0 for v in d) / len(d), sum(v >= 0 for v in d) / len(d)
    return min(1.0, 2 * min(lo, hi))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


# 1 多重比较 ------------------------------------------------------------------------------------
def family(s):
    x, items, _ = base(s)
    ver = [i for i in items if i["ver"]]

    def cmd_diff(xs, keys):
        a, b = [i for i in xs if i["cmd"]], [i for i in xs if not i["cmd"]]
        return share(a, lambda i: has(i["labels"], keys)) - share(b, lambda i: has(i["labels"], keys)) if a and b else None

    def reg_ame(xs, keys, with_cmd):
        cols = [lambda i: 1.0, lambda i: float(i["reg"]), lambda i: math.log1p(i["n"])] + ([lambda i: float(i["cmd"])] if with_cmd else [])
        X = np.array([[f(i) for f in cols] for i in xs])
        y = np.array([float(has(i["labels"], keys)) for i in xs])
        if X[:, 1].min() == X[:, 1].max() or (with_cmd and X[:, 3].min() == X[:, 3].max()):
            return None
        b = logit(X, y)
        if b is None:
            return None
        return (-ame(X, b, 1), ame(X, b, 3)) if with_cmd else (-ame(X, b, 1),)

    def within(xs, keys):
        num = den = 0.0
        by = defaultdict(list)
        for i in xs:
            by[i["s"]].append(i)
        for g in by.values():
            g1, g0 = [i for i in g if i["reg"]], [i for i in g if not i["reg"]]
            if g1 and g0:
                w = len(g1) * len(g0) / len(g)
                num += w * (share(g0, lambda i: has(i["labels"], keys)) - share(g1, lambda i: has(i["labels"], keys)))
                den += w
        return num / den if den else None

    stats = []
    for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并")):
        stats += [(f"片段内提交 − 未提交（可核验）：{kn}", ver, lambda xs, k=keys: cmd_diff(xs, k)),
                  (f"片段内提交 − 未提交（全部代码修改）：{kn}", items, lambda xs, k=keys: cmd_diff(xs, k)),
                  (f"登记 AME（控制规模）：{kn}", ver, lambda xs, k=keys: (reg_ame(xs, k, False) or (None,))[0]),
                  (f"登记 AME（控制规模与片段内提交）：{kn}", ver, lambda xs, k=keys: (reg_ame(xs, k, True) or (None, None))[0]),
                  (f"片段内提交 AME（控制规模与登记）：{kn}", ver, lambda xs, k=keys: (reg_ame(xs, k, True) or (None, None))[1]),
                  (f"会话内比较（未登记 − 登记）：{kn}", ver, lambda xs, k=keys: within(xs, k))]
    bs = {"ver": boot(ver, f"{s}|ver"), "all": boot(items, f"{s}|all")}
    rows = []
    for name, xs, f in stats:
        draws = [f(b) for b in bs["ver" if xs is ver else "all"]]
        rows.append({"name": name, "est": f(xs), "p": pval(draws), "ci": [q([d for d in draws if d is not None], .025),
                                                                         q([d for d in draws if d is not None], .975)],
                     "skip": sum(d is None for d in draws)})
    for r, a in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = a
    return rows


# 2 提交执行者 ----------------------------------------------------------------------------------
def executors(s):
    x, items, lines = base(s)
    cm = x["cm"]
    gold = command_commits(x["d"], cm)
    trailer = set()
    for blk in git(REPOS[s], "log", "--format=\x1e%H\x1f%B").split("\x1e")[1:]:
        h, b = blk.split("\x1f", 1)
        if "co-authored-by: cursor" in b.lower():
            trailer.add(h)
    cls = {h: "智能体提交命令" if h in gold else ("只有协作尾注" if h in trailer else "二者皆无") for h in cm}
    eps_of = defaultdict(set)
    by_e = defaultdict(dict)
    for (e, c), v in lines.items():
        eps_of[c].add(e)
        by_e[e][c] = v
    C = {k: {"commits": 0, "with_content": 0, "multi": 0, "eps": 0, "one": 0, "merge": 0} for k in ("智能体提交命令", "只有协作尾注", "二者皆无")}
    for h, k in cls.items():
        C[k]["commits"] += 1
        if eps_of[h]:
            C[k]["with_content"] += 1
            C[k]["multi"] += len(eps_of[h]) >= 2
    for it in items:
        row = by_e.get(it["e"])
        if not row or not sum(row.values()):
            continue
        k = cls[max(row, key=row.get)]
        C[k]["eps"] += 1
        C[k]["one"] += "一对一" in it["labels"]
        C[k]["merge"] += has(it["labels"], MERGE)
    return C


# 3 特征行 --------------------------------------------------------------------------------------
def prepare():
    for suf, _, env in VARIANTS[1:]:
        e = {**os.environ, "PYTHONIOENCODING": "utf-8", "ATTR_SUFFIX": suf, **env}
        for sc in ["attribute_edits.py", "checkpoint_deltas.py", "map_episodes.py", "patterns.py"]:
            subprocess.run([sys.executable, f"analysis/{sc}"], env=e, check=True, stdout=subprocess.DEVNULL)


def sig_variant(s, suf):
    rows = load(Path("data/derived") / s / f"episode_patterns{suf}.csv")
    code = [r for r in rows if r["evidence"]]
    ver = [r for r in code if r["evidence"] == "verified"]
    lab = lambda r: set(r["patterns"].split(";"))
    main = {r["episode"]: r["patterns"] for r in load(Path("data/derived") / s / "episode_patterns.csv")}
    out = {"n_ver": len(ver), "n_code": len(code), "changed": sum(main.get(r["episode"]) != r["patterns"] for r in code)}
    for grp, tag in ((ver, "ver"), (code, "all")):
        for keys, kn in ((("一对一",), "一对一"), (MERGE, "合并"), (("拆分",), "拆分"), (("缺失",), "缺失")):
            out[f"{tag}_{kn}"] = share(grp, lambda r: has(lab(r), keys))
    return out


# 4 删除型改动 ----------------------------------------------------------------------------------
NOBODY = "只有前后内容标识"


def deletions(s):
    d = Path("data/derived") / s
    ev = {(e["session"], e["seq"]): e for e in load(d / "events.csv")}
    ep_of = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_events.csv")}
    attr = {(a["session"], a["seq"]): a["state"] for a in load(d / "edit_attribution.csv")}
    code_keys = {k for k, e in ev.items() if k in ep_of and e["tool"] in EDIT_TOOLS and e["status"] == "completed"
                 and path_class(e["path"].replace("\\", "/"), e["in_workspace"]) == "code"}
    cat = {}
    with_ids = 0
    cfg = SESS[s]
    for cdir in sorted(cfg["composer"].iterdir()):
        cid = cdir.name
        if excluded(cid):
            continue
        head, cd, bubbles = load_session(cdir)
        ordered, loose, _ = build_sequence(cd, bubbles)
        for seq, x in enumerate(ordered + loose):
            k = (cid, str(seq))
            if k not in code_keys:
                continue
            tool = x["v"].get("toolFormerData") or {}
            if (a := attr.get(k)) is not None:
                cat[k] = "重写既有内容" if a == "no_new" else "有新写入行"
                continue
            if signature_lines(tool):
                cat[k] = "有新写入行"
                continue
            p = parse_params(tool)
            new = p.get("streamingContent") or p.get("contents") or p.get("new_string") or ""
            ad = tool.get("additionalData") or {}
            if isinstance(ad, str):
                try:
                    ad = json.loads(ad)
                except json.JSONDecodeError:
                    ad = {}
            pd = (ad.get("precomputedDiff") or {}) if isinstance(ad, dict) else {}
            removed = any(y.get("type") == "removed" for y in pd.get("lines", []))
            if tool.get("name") == "delete_file":
                cat[k] = "删除文件"
            elif not new.strip() and ((p.get("old_string") or "").strip() or removed):
                cat[k] = "只删除"
            elif new.strip():
                cat[k] = "只有短行"
            else:
                cat[k] = NOBODY
                try:
                    res = json.loads(tool.get("result") or "{}")
                except json.JSONDecodeError:
                    res = {}
                with_ids += isinstance(res, dict) and bool(res.get("beforeContentId")) and bool(res.get("afterContentId"))
    missing = code_keys - set(cat)
    for k in missing:
        cat[k] = NOBODY
    by_ep = defaultdict(list)
    for k, c in cat.items():
        by_ep[ep_of[k]].append(c)
    pat = {r["episode"]: r for r in load(d / "episode_patterns.csv")}
    only_del = [e for e, cs in by_ep.items() if all(c in ("只删除", "删除文件", "重写既有内容") for c in cs)
                and any(c in ("只删除", "删除文件") for c in cs)]
    nobody_eps = [e for e, cs in by_ep.items() if all(c in (NOBODY, "删除文件") for c in cs)]
    miss = [e for e in by_ep if pat[e]["evidence"] == "verified" and "缺失" in pat[e]["patterns"].split(";")]
    miss_del = [e for e in miss if "删除文件" in by_ep[e]]
    return {"edits": Counter(cat.values()), "n_edits": len(cat), "unmatched": len(missing), "with_ids": with_ids,
            "nobody_eps": len(nobody_eps), "miss_ver": len(miss), "miss_ver_with_delete": len(miss_del), "nobody_evidence": Counter(pat[e]["evidence"] or "无" for e in nobody_eps),
            "only_del_eps": len(only_del), "only_del_evidence": Counter(pat[e]["evidence"] or "无" for e in only_del),
            "n_code_eps": len(by_ep)}


def main():
    if "--no-prepare" not in sys.argv:
        prepare()
    R = {}
    L = ["# v2.11 补充稳健性（S13-T1）", "", "由 `analysis/robustness2.py` 生成，口径见 `notes/decisions.md` D13。", ""]

    rows = family("SYS_A")
    R["holm"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "ci"} | {"ci": [round(r["ci"][0], 3), round(r["ci"][1], 3)]} for r in rows]
    L += ["## 1 多重比较（系统 A，4.2 节的 12 个比较；D13-1）", "",
          f"按会话重抽 {BOOT} 次（种子 `{SEED}`）；双侧自助 p 值；Holm 校正。区间为本次重抽的 95% 百分位区间，与正文区间的种子不同。", "",
          "| 比较 | 点估计（百分点） | 95% 区间 | p | Holm 校正 p | 跳过 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['est'] * 100:+.0f} | {r['ci'][0] * 100:+.0f} 至 {r['ci'][1] * 100:+.0f} | {r['p']:.3f} | {r['p_holm']:.3f} | {r['skip']} |")
    sig = [r["name"] for r in rows if r["p_holm"] < 0.05]
    R["holm_sig"] = sig
    L += ["", f"Holm 校正后 p < 0.05 的比较 {len(sig)} 个：" + "；".join(sig) + "。", ""]

    L += ["## 2 提交执行者对照（D13-2）", ""]
    for s in ("SYS_A", "SYS_C"):
        C = executors(s)
        R[f"exec_{s}"] = C
        L += [f"### {NAME[s]}", "", "| 提交来源 | 提交数 | 含片段内容的提交 | 其中含 ≥ 2 个片段 | 以此类为主提交的片段 | 其中一对一 | 其中合并 |",
              "|---|---|---|---|---|---|---|"]
        for k, v in C.items():
            L.append(f"| {k} | {v['commits']} | {v['with_content']} | {v['multi']}（{pc(v['multi'] / v['with_content']) if v['with_content'] else '—'}） | "
                     f"{v['eps']} | {pc(v['one'] / v['eps']) if v['eps'] else '—'} | {pc(v['merge'] / v['eps']) if v['eps'] else '—'} |")
        L.append("")
    L += ["主提交 = 片段可归属行数最多的提交（D11-2）；片段为按内容进入版本库的代码修改片段。", ""]

    L += ["## 3 特征行长度与条数（D13-3）", ""]
    for s in ("SYS_A", "SYS_C"):
        L += [f"### {NAME[s]}", "", "| 口径 | 可核验 n | 可核验：一对一 | 合并 | 拆分 | 缺失 | 全部代码修改 n | 一对一 | 合并 | 拆分 | 模式集合有变的片段 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for suf, nm, _ in VARIANTS:
            v = sig_variant(s, suf)
            R[f"sig_{s}{suf or '_main'}"] = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}
            L.append(f"| {nm} | {v['n_ver']} | {pc(v['ver_一对一'])} | {pc(v['ver_合并'])} | {pc(v['ver_拆分'])} | {pc(v['ver_缺失'])} | "
                     f"{v['n_code']} | {pc(v['all_一对一'])} | {pc(v['all_合并'])} | {pc(v['all_拆分'])} | {v['changed'] if suf else '—'} |")
        L.append("")

    L += ["## 4 删除型改动（D13-4）", ""]
    for s in ("SYS_A", "SYS_C"):
        D = deletions(s)
        R[f"del_{s}"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in D.items()}
        ev_txt = lambda c: ("（" + "，".join(f"{k} {v}" for k, v in c.most_common()) + "）") if c else ""
        L += [f"### {NAME[s]}", "", f"代码编辑 {D['n_edits']} 次（未能在原始导出中定位 {D['unmatched']} 次）：" +
              "，".join(f"{k} {v}（{v / D['n_edits']:.0%}）" for k, v in D["edits"].most_common()) + "。",
              f"其中“{NOBODY}”的编辑中 {D['with_ids']} 次在工具结果里只有编辑前后的内容标识，正文不在导出中，无法判断是否为删除。",
              f"代码修改片段 {D['n_code_eps']} 个中，全部代码编辑都是只删除、删除文件或重写既有内容（且至少一次删除）的 {D['only_del_eps']} 个{ev_txt(D['only_del_evidence'])}；"
              f"全部代码编辑都是删除文件或只有内容标识的 {D['nobody_eps']} 个{ev_txt(D['nobody_evidence'])}。",
              f"可核验且含缺失的片段 {D['miss_ver']} 个，其中含删除文件编辑的 {D['miss_ver_with_delete']} 个。", ""]

    (RESULTS_DIR / "r10_robustness2.md").write_text("\n".join(L), encoding="utf-8")
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    num["robustness2"] = R
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
