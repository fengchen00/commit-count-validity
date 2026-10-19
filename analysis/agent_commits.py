"""提交来源与对齐方法的外部验证（S10-T3）。写入 <PAPER_DIR>/results/r6_validation.md。

1. 提交来源：会话中智能体执行的提交命令（按 -m 标题与时间对上提交）、提交正文中的 Cursor 协作尾注、两者皆无。
2. 参照：智能体在片段 E 中执行提交命令产生提交 C，这一关系不依赖内容对齐。对 E 修改过、且 C 也改动了的代码文件，
   检验内容对齐是否把 E 映射到 C（召回），以及 C 的内容是否被归到 E 或同会话更早的片段（会话一致性）。
3. 不含任何片段内容的提交的构成；进入 10 个及以上提交的片段的构成。
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from extract_git import SYSTEMS, git  # noqa: E402
from workitems import wilson  # noqa: E402

CMD = re.compile(r"git\s+commit")
MSG = re.compile(r"""-m\s+(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S+))""")
LABEL = {"task": "任务编号", "milestone": "里程碑", "fix": "里程碑修复", "hotfix": "热修", "unlabeled": "无标签"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n}（{k / n:.0%}，{lo:.0%}–{hi:.0%}）" if n else "0/0"


def command_commits(d, cm):
    """提交 → 执行其提交命令的片段（按 -m 标题与时间窗口对上）。"""
    ep_of = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_events.csv")}
    by_subj = defaultdict(list)
    for h, c in cm.items():
        by_subj[c["subject"].strip()].append(h)
    gold = {}
    for e in load(d / "events.csv"):
        if not CMD.search(e["command"] or ""):
            continue
        m = MSG.search(e["command"])
        ep = ep_of.get((e["session"], e["seq"]))
        if not m or not ep:
            continue
        subj = next(g for g in m.groups() if g is not None).split("\\n")[0].strip()
        t = datetime.fromisoformat(e["time"])
        hs = sorted((h for h in by_subj.get(subj, []) if timedelta(minutes=-2) <= datetime.fromisoformat(cm[h]["time"]) - t <= timedelta(hours=2)),
                    key=lambda h: cm[h]["time"])
        if hs and hs[0] not in gold:
            gold[hs[0]] = ep
    return gold


def commit_sources(s):
    """提交来源（D24-2）：{hash: 提交行}、可对上智能体提交命令的提交 → 片段、带 Cursor 协作尾注的提交。"""
    d = Path("data/derived") / s
    cm = {c["hash"]: c for c in load(d / "commits.csv")}
    body = {}
    for blk in git(SYSTEMS[s], "log", "--format=\x1e%H\x1f%B").split("\x1e")[1:]:
        h, b = blk.split("\x1f", 1)
        body[h] = b.lower()
    trailer = {h for h in cm if "co-authored-by: cursor" in body.get(h, "")}
    return cm, command_commits(d, cm), trailer


def run(s):
    d = Path("data/derived") / s
    cm, gold, trailer = commit_sources(s)
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}

    pairs = [p for p in load(d / "episode_commit.csv") if p["source"] != "pathtime"]
    content = defaultdict(set)
    eps_of = defaultdict(set)
    for p in pairs:
        content[p["episode"]].add(p["commit"])
        eps_of[p["commit"]].add(p["episode"])
    files_of_c = defaultdict(set)
    for f in load(d / "commit_files.csv"):
        files_of_c[f["hash"]].add(f["path"].lower())
    code_files = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"].lower())
    pat = {r["episode"]: r for r in load(d / "episode_patterns.csv")}

    n = len(cm)
    agent = set(gold) | trailer
    L = [f"## {s}", "", "### 提交来源", "",
         f"- 提交 {n} 个：会话中可对上智能体提交命令的 {len(gold)} 个，带 Cursor 协作尾注的 {len(trailer)} 个，"
         f"二者至少其一的 {len(agent)} 个（{len(agent) / n:.0%}），二者皆无的 {n - len(agent)} 个。",
         f"- 二者皆有的 {len(set(gold) & trailer)} 个；只有命令的 {len(set(gold) - trailer)} 个；只有尾注的 {len(trailer - set(gold))} 个。", ""]

    rel = [(h, ep) for h, ep in gold.items() if code_files[ep] & files_of_c[h]]
    rec = sum(h in content[ep] for h, ep in rel)
    ver = [(h, ep) for h, ep in rel if pat[ep]["evidence"] == "verified"]
    rec_v = sum(h in content[ep] for h, ep in ver)
    consistent = 0
    attributed = [(h, ep) for h, ep in gold.items() if eps_of[h]]
    for h, ep in attributed:
        sess, t = eps[ep]["session"], datetime.fromisoformat(cm[h]["time"])
        consistent += any(eps[o]["session"] == sess and datetime.fromisoformat(eps[o]["start"]) <= t for o in eps_of[h])
    L += ["### 以智能体提交命令为参照", "",
          f"- 参照关系（片段 E 中执行命令产生提交 C）{len(gold)} 个；其中 E 修改过且 C 也改动了的代码文件非空的 {len(rel)} 个。",
          f"- 召回：内容对齐把 E 映射到 C 的 {pct(rec, len(rel))}；E 为可核验片段时 {pct(rec_v, len(ver))}。",
          f"- 会话一致性：C 的内容被归属给某些片段时（{len(attributed)} 个），其中含 E 或同会话更早片段的 {pct(consistent, len(attributed))}。", ""]

    no_ep = [h for h in cm if not eps_of[h]]
    lab = Counter(cm[h]["label"] for h in no_ep)
    kind = Counter(cm[h]["kind"] for h in no_ep)
    src = Counter("命令" if h in gold else ("尾注" if h in trailer else "皆无") for h in no_ep)
    code_touch = sum(int(cm[h]["n_code_files"] or 0) > 0 for h in no_ep)
    L += ["### 不含任何片段内容的提交", "",
          f"- 共 {len(no_ep)} 个（{len(no_ep) / n:.0%}）。来源：" + "，".join(f"{k} {v}" for k, v in src.most_common()) + "。",
          "- 标题类别：" + "，".join(f"{LABEL.get(k, k)} {v}" for k, v in lab.most_common()) + "。",
          "- 改动类别：" + "，".join(f"{k} {v}" for k, v in kind.most_common()) + f"；改动了代码文件的 {code_touch} 个。", ""]

    hi = [ep for ep, cs in content.items() if len(cs) >= 10]
    hc = [(ep, h) for ep in hi for h in content[ep]]
    same_sess_agent = sum(h in gold and eps[gold[h]]["session"] == eps[ep]["session"] for ep, h in hc)
    hl = Counter(cm[h]["label"] for _, h in hc)
    L += ["### 进入 10 个及以上提交的片段", "",
          f"- 片段 {len(hi)} 个，片段—提交对 {len(hc)} 个；其中提交由同一会话的智能体命令产生的 {pct(same_sess_agent, len(hc))}。",
          "- 这些提交的标题类别：" + "，".join(f"{LABEL.get(k, k)} {v}" for k, v in hl.most_common()) + "。", ""]

    times = sorted(datetime.fromisoformat(c["time"]) for c in cm.values())
    L += ["### 观察窗口", "", f"- 首个提交至最后一个提交 {(times[-1] - times[0]).days + 1} 天；主会话 "
          f"{len({e['session'] for e in eps.values()})} 个。", ""]
    return L


if __name__ == "__main__":
    out = ["# 提交来源与对齐方法的外部验证（S10-T3）", "",
           "由 `analysis/agent_commits.py` 生成。只统计数量；提交标题与命令文本不写出。", ""]
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        out += run(s)
    (RESULTS_DIR / "r6_validation.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
