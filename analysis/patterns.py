"""片段候选模式（S02-T2，定义见 notes/decisions.md D5）。

证据层（只看代码编辑）：
  verified  每次代码编辑都能按内容判定（有逐次内容，或本片段有检查点差分）
  partial   至少一部分能按内容判定
  inferred  全部只能按路径—时间推定
模式只用内容映射（逐次编辑 + 检查点差分）计算；pathtime 映射另算一套 *_pt 列，只作敏感性对照。

输出 data/derived/<系统>/episode_patterns.csv。
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_episodes import path_class  # noqa: E402
from extract_git import plan_milestone  # noqa: E402

SUFFIX = os.environ.get("ATTR_SUFFIX", "")
SOURCES = {"all": {"edit", "checkpoint", "both"}, "edit": {"edit", "both"}, "checkpoint": {"checkpoint", "both"}}[
    os.environ.get("PAIR_SOURCES", "all")]
OUT_SUFFIX = SUFFIX + os.environ.get("PATTERN_SUFFIX", "")
MS_REF = re.compile(r"(?<![A-Za-z0-9])M(\d+(?:\.\d+)?[a-z]?)(?![0-9])")
TASKED = {"task", "milestone", "fix"}
PATTERNS = ["一对一", "合并-同会话", "合并-跨会话", "拆分", "拆分-逐条任务", "拆分-非逐条", "随他项提交", "顺序压平", "部分未提交", "含未跟踪文件", "缺失", "缺失-窗口末", "无新增内容", "证据不足", "无代码"]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def classify(n_code, uid, P, eps_of, cm, ctx, flatten, st):
    """uid = "会话|单位"；eps_of[提交] 为含其内容的单位 uid 集合。单位可以是片段，也可以是工作项。"""
    if n_code == 0:
        return ["无代码"]
    ev = st["evidence"]
    if not P:
        if ev != "verified":
            return ["证据不足"]
        if not st["new_content"]:
            return ["无新增内容"]
        return ["缺失", "缺失-窗口末"] if st["only_after_last"] else ["缺失"]
    labels = []
    session = uid.split("|")[0]
    others = set().union(*(eps_of[c] for c in P)) - {uid}
    if len(P) == 1 and not others:
        labels.append("一对一")
    if others:
        same = all(o.split("|")[0] == session for o in others)
        labels.append("合并-同会话" if same else "合并-跨会话")
    if len(P) >= 2:
        labels.append("拆分")
        keys = [(cm[c]["milestone"], cm[c]["tasks"]) for c in P]
        per_task = all(cm[c]["label"] == "task" for c in P) and len(set(keys)) == len(keys)
        labels.append("拆分-逐条任务" if per_task else "拆分-非逐条")
    if any(cm[c]["label"] in TASKED and cm[c]["milestone"] and cm[c]["milestone"] not in ctx for c in P):
        labels.append("随他项提交")
    if flatten:
        labels.append("顺序压平")
    if st["uncommitted_new"]:
        labels.append("部分未提交")
    if st["untracked"]:
        labels.append("含未跟踪文件")
    return labels


def prepare(sysname):
    """每个片段的分类输入：内容映射、上下文里程碑、顺序压平、代码状态。"""
    d = Path("data/derived") / sysname
    eps = load(d / "episodes.csv")
    cm = {c["hash"]: c for c in load(d / "commits.csv")}
    pairs = load(d / f"episode_commit{SUFFIX}.csv")
    states = {r["episode"]: r for r in load(d / f"episode_code_states{SUFFIX}.csv")}
    sess_of = {e["episode"]: e["session"] for e in eps}
    key = lambda ep: f"{sess_of[ep]}|{ep}"

    content, withpt = defaultdict(set), defaultdict(set)
    for p in pairs:
        withpt[p["episode"]].add(p["commit"])
        if p["source"] in SOURCES:
            content[p["episode"]].add(p["commit"])
    # 上下文里程碑：本片段及同会话此前片段的请求中出现的 M 号，加上本片段写过的任务书所属里程碑
    ep_plans = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "plan" and plan_milestone(f["path"]):
            ep_plans[f["episode"]].add(plan_milestone(f["path"]))
    ctx, seen = {}, defaultdict(set)
    for e in sorted(eps, key=lambda e: (e["session"], e["start"], int(e["msg_seq"]))):
        seen[e["session"]] |= {"M" + m for m in MS_REF.findall(e["request_text"])}
        ctx[e["episode"]] = seen[e["session"]] | ep_plans[e["episode"]]

    # 顺序压平：片段内先写任务书/文档，后改代码，且两者进入同一提交
    ep_of_edit = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_edits.csv") if r["episode"] and not r["reason"]}
    doc_commits = defaultdict(set)
    for a in load(d / f"edit_attribution{SUFFIX}.csv"):
        ep = ep_of_edit.get((a["session"], a["seq"]))
        if ep and a["state"] == "committed" and path_class(a["path"], "1") in ("plan", "doc"):
            doc_commits[ep].add(a["commit"])
    for r in load(d / f"checkpoint_deltas{SUFFIX}.csv"):
        if r["state"] == "committed" and r["class"] in ("plan", "doc"):
            doc_commits[r["episode"]].add(r["commit"])
    first = defaultdict(dict)
    for f in load(d / "episode_files.csv"):
        c = "doc" if f["class"] in ("plan", "doc") else f["class"]
        if c in ("doc", "code"):
            first[f["episode"]][c] = min(first[f["episode"]].get(c, f["first"]), f["first"])

    st_of, flat_of = {}, {}
    for e in eps:
        s = states[e["episode"]]
        n = {k: int(v) for k, v in s.items() if k != "episode"}
        # D3d：检查点只对其差分中出现的文件作证，其余文件仍看逐次编辑
        cp = n["has_checkpoint_delta"]
        blind = n["ucov_no_content"] if cp else n["no_content"]
        judged = n["n_edits_code"] - blind
        evidence = "verified" if blind == 0 else ("partial" if judged else "inferred")
        new_content = (n["committed"] + n["not_committed"] + n["after_last_commit"] + n["untracked"]
                       + n["cp_committed"] + n["cp_not_committed"] + n["cp_after_last_commit"] + n["cp_untracked"])
        if cp:
            unc = n["cp_not_committed"] + n["cp_after_last_commit"] + n["ucov_not_committed"] + n["ucov_after_last_commit"]
            untr = n["cp_untracked"] + n["ucov_untracked"]
            after = n["cp_after_last_commit"] + n["ucov_after_last_commit"]
        else:
            unc, untr = n["not_committed"] + n["after_last_commit"], n["untracked"]
            after = n["after_last_commit"]
        st_of[e["episode"]] = {"evidence": evidence, "new_content": new_content, "uncommitted_new": unc, "untracked": untr,
                               "after": after, "n_code": n["n_edits_code"]}
        k = e["episode"]
        fl = first[e["episode"]]
        flat_of[k] = bool(doc_commits[k] & content.get(k, set())) and "doc" in fl and "code" in fl and fl["doc"] < fl["code"]
    return {"d": d, "eps": eps, "cm": cm, "content": content, "withpt": withpt, "ctx": ctx, "flatten": flat_of,
            "st": st_of, "key": key}


def unit_state(sts):
    """若干片段合成一个单位时的代码状态：证据取最弱，计数相加。"""
    rank = {"verified": 0, "partial": 1, "inferred": 2}
    code = [s for s in sts if s["n_code"]]
    ev = max((s["evidence"] for s in code), key=rank.get) if code else ""
    tot = {k: sum(s[k] for s in sts) for k in ("new_content", "uncommitted_new", "untracked", "after")}
    return dict(tot, evidence=ev, only_after_last=tot["after"] > 0 and tot["after"] == tot["uncommitted_new"] + tot["untracked"])


def run(sysname):
    x = prepare(sysname)
    d, eps, cm, content, withpt, ctx, key = x["d"], x["eps"], x["cm"], x["content"], x["withpt"], x["ctx"], x["key"]
    eps_of_c, eps_of_pt = defaultdict(set), defaultdict(set)
    for ep, cs in content.items():
        for c in cs:
            eps_of_c[c].add(key(ep))
    for ep, cs in withpt.items():
        for c in cs:
            eps_of_pt[c].add(key(ep))
    rows, cnt, cnt_pt, ev_cnt = [], Counter(), Counter(), Counter()
    for e in eps:
        k = e["episode"]
        st = unit_state([x["st"][k]])
        evidence = st["evidence"]
        n_code, flatten = int(e["n_edits_code"]), x["flatten"][k]
        labels = classify(n_code, key(k), content.get(k, set()), eps_of_c, cm, ctx[k], flatten, st)
        st_pt = dict(st, evidence="verified")
        labels_pt = classify(n_code, key(k), withpt.get(k, set()), eps_of_pt, cm, ctx[k], flatten, st_pt)
        assert labels, e["episode"]
        rows.append({"episode": e["episode"], "evidence": evidence if int(e["n_edits_code"]) else "",
                     "n_commits": len(content.get(k, ())), "n_commits_pt": len(withpt.get(k, ())),
                     "patterns": ";".join(labels), "patterns_pt": ";".join(labels_pt)})
        cnt.update(labels)
        cnt_pt.update(labels_pt)
        if int(e["n_edits_code"]):
            ev_cnt[evidence] += 1

    with open(d / f"episode_patterns{OUT_SUFFIX}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{sysname}{OUT_SUFFIX}: episodes={len(eps)} evidence={dict(ev_cnt)}")
    print("  content :", {p: cnt[p] for p in PATTERNS})
    print("  +pathtime:", {p: cnt_pt[p] for p in PATTERNS})
    nc = Counter(min(r["n_commits"], 5) for r in rows if r["n_commits"])
    print("  commits per mapped episode (5=5+):", dict(sorted(nc.items())))
    ver = [r for r in rows if r["evidence"] == "verified"]
    vc = Counter(p for r in ver for p in r["patterns"].split(";"))
    print(f"  verified only (n={len(ver)}):", {p: vc[p] for p in PATTERNS if vc[p]})
    return rows


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        run(s)
