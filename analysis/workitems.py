"""工作项层面的映射模式（S10-T1）。写入 <PAPER_DIR>/results/r5_workitems.md。

工作片段按"一条开发者消息"切分，粒度可能细于一项工作。本脚本把同一会话中相邻的代码修改片段
按三条从严到宽的规则聚合为工作项，再用与片段相同的判据（patterns.classify）重算映射模式：
  strict   相邻代码片段改动了同一代码文件，间隔不超过 30 分钟，且后一条请求很短或含修正/返工措辞
  loose    相邻代码片段改动了同一代码文件（不限间隔）
  session  同一会话的全部代码片段为一个工作项（合并程度的上限）
相邻指同一会话内按时间排序的代码修改片段之间，不涉及代码的片段不打断相邻关系。
输出 data/derived/<系统>/workitems.csv（片段 → 各规则下的工作项编号）。
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from patterns import classify, prepare, unit_state  # noqa: E402

RULES = ["episode", "strict", "loose", "coded", "coded_hi", "session"]
RULE_NAME = {"episode": "片段（原口径）", "strict": "工作项·严格", "loose": "工作项·宽松", "coded": "工作项·编码",
             "coded_hi": "工作项·编码（分歧计合并）", "session": "工作项·整会话"}
import os
HUMAN = os.environ.get("CODING_HUMAN", "0") == "1"  # D29 / D30：真人编码表在 data/coding/human/workitem_pairs/，key 仍在 data/coding/workitem_pairs/
KEY_DIR = Path("data/coding/workitem_pairs")
CODING = Path("data/coding/human/workitem_pairs") if HUMAN else KEY_DIR
GAP_MIN = 30
SHORT = 40
REWORK = re.compile(r"不对|不行|不是|还是|仍然|依然|没有(?:变|生效|改|反应)|没变|没生效|报错|出错|错了|有问题|问题|怎么|为什么|"
                    r"再(?:改|调|试|看)|重新|回退|撤销|恢复|不要|别|太(?:大|小|宽|窄|高|低|长|短)|对齐|位置|样式|显示|还有")
SHOW = ["一对一", "合并-同会话", "合并-跨会话", "合并", "拆分", "拆分-逐条任务", "拆分-非逐条", "随他项提交", "部分未提交",
        "含未跟踪文件", "缺失"]
SHOW_NAME = {"合并-同会话": "同会话合并", "合并-跨会话": "跨会话合并", "合并": "合并（并集）", "拆分-逐条任务": "逐条任务拆分",
             "拆分-非逐条": "非逐条拆分", "随他项提交": "附带提交"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def coded_decisions():
    """相邻代码片段对的编码判断：(前片段, 后片段) → (共识或裁决, 分歧计合并)。无编码文件时返回 None。"""
    key = KEY_DIR / "key_all.csv"
    files = [sorted(CODING.glob(f"coder{i}_all_*.csv")) for i in (1, 2)]
    if not key.exists() or not all(files):
        return None
    ans = []
    for fs in files:
        a = {}
        for f in fs:
            a.update({r["pair"]: int(r["same"]) for r in load(f)})
        ans.append(a)
    adj = {}
    if (CODING / "adjudicate_all.csv").exists():
        adj = {r["pair"]: int(r["same"]) for r in load(CODING / "adjudicate_all.csv")}
    out = {}
    for r in load(key):
        p = r["pair"]
        if p not in ans[0] or p not in ans[1]:
            return None
        a, b = ans[0][p], ans[1][p]
        out[(r["a"], r["b"])] = (a if a == b else adj.get(p, 0), max(a, b))
    return out


def group(eps, code_files, rule, coded=None):
    """返回 片段 → 工作项编号（会话|编号）。不涉及代码的片段各自成项。"""
    wi = {}
    by_s = defaultdict(list)
    for e in eps:
        by_s[e["session"]].append(e)
    for s, es in by_s.items():
        es.sort(key=lambda e: (e["start"], int(e["msg_seq"])))
        n = 0
        cur, cur_files, prev = None, set(), None
        for e in es:
            k = e["episode"]
            if not int(e["n_edits_code"]):
                wi[k] = f"{s}|n{k}"
                continue
            f = code_files[k]
            join = False
            if cur is not None and rule != "episode":
                if rule == "session":
                    join = True
                elif rule in ("coded", "coded_hi"):
                    join = bool(coded[(prev["episode"], k)][0 if rule == "coded" else 1])
                else:
                    share = bool(f & cur_files)
                    if rule == "loose":
                        join = share
                    else:
                        gap = (datetime.fromisoformat(e["start"]) - datetime.fromisoformat(prev["last_event"])).total_seconds() / 60
                        cue = int(e["request_len"] or 0) <= SHORT or bool(REWORK.search(e["request_text"][:200]))
                        join = share and gap <= GAP_MIN and cue
            if join:
                cur_files |= f
            else:
                n += 1
                cur, cur_files = f"{s}|w{n}", set(f)
            wi[k] = cur
            prev = e
    return wi


def units(x, wi):
    eps, content = x["eps"], x["content"]
    mem = defaultdict(list)
    for e in eps:
        mem[wi[e["episode"]]].append(e)
    commits_of = {u: set().union(*(content.get(e["episode"], set()) for e in es)) for u, es in mem.items()}
    eps_of = defaultdict(set)
    for u, cs in commits_of.items():
        for c in cs:
            eps_of[c].add(u)
    out = {}
    for u, es in mem.items():
        n_code = sum(int(e["n_edits_code"]) for e in es)
        st = unit_state([x["st"][e["episode"]] for e in es])
        ctx = set().union(*(x["ctx"][e["episode"]] for e in es))
        flatten = any(x["flatten"][e["episode"]] for e in es)
        labels = classify(n_code, u, commits_of[u], eps_of, x["cm"], ctx, flatten, st)
        if "合并-同会话" in labels or "合并-跨会话" in labels:
            labels.append("合并")
        out[u] = {"n_eps": len(es), "n_code": n_code, "evidence": st["evidence"], "labels": labels,
                  "commits": commits_of[u]}
    return out, eps_of


def run(sysname):
    x = prepare(sysname)
    d = x["d"]
    code_files = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"])
    coded = coded_decisions()
    rules = [r for r in RULES if coded or not r.startswith("coded")]
    maps = {r: group(x["eps"], code_files, r, coded) for r in rules}
    with open(d / "workitems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + [f"wi_{r}" for r in rules])
        for e in x["eps"]:
            w.writerow([e["episode"]] + [maps[r][e["episode"]] for r in rules])

    L = [f"## {sysname}", ""]
    stats = {}
    for r in rules:
        us, eps_of = units(x, maps[r])
        code = [u for u in us.values() if u["n_code"]]
        ver = [u for u in code if u["evidence"] == "verified"]
        mapped = [u for u in code if u["commits"]]
        size = Counter(min(u["n_eps"], 4) for u in code)
        mc = sum(len(v) >= 2 for v in eps_of.values())
        stats[r] = {"n_code": len(code), "ver": ver, "mapped": mapped, "size": size, "commits": len(eps_of), "multi": mc}
    L += ["### 工作项规模", "", "| 口径 | 代码修改单位数 | 可核验单位数 | 含 1 / 2 / 3 / ≥4 个片段的单位 | 含多个单位的提交 |",
          "|---|---|---|---|---|"]
    for r in rules:
        s = stats[r]
        sz = " / ".join(str(s["size"][k]) for k in (1, 2, 3, 4))
        L.append(f"| {RULE_NAME[r]} | {s['n_code']} | {len(s['ver'])} | {sz} | {s['multi']}/{s['commits']}（{s['multi'] / s['commits']:.0%}） |")

    def table(title, key, pats):
        T = ["", title, "", "| 模式 | " + " | ".join(f"{RULE_NAME[r]}（n = {len(stats[r][key])}）" for r in rules) + " |",
             "|---|" + "---|" * len(rules)]
        for p in pats:
            cells = []
            for r in rules:
                us = stats[r][key]
                k = sum(p in u["labels"] for u in us)
                lo, hi = wilson(k, len(us))
                cells.append(f"{k / len(us):.0%}（{lo:.0%}–{hi:.0%}）" if us else "—")
            T.append(f"| {SHOW_NAME.get(p, p)} | " + " | ".join(cells) + " |")
        return T

    L += table("### 可核验单位的模式分布（可多标签；括号内为 Wilson 95% 置信区间）", "ver", SHOW)
    L += table("### 进入版本库的单位（任意证据层，至少一个内容提交）", "mapped", ["一对一", "合并", "拆分", "随他项提交"])
    L.append("")
    return L


if __name__ == "__main__":
    out = ["# 工作项层面的映射模式（S10-T1）", "",
           "由 `analysis/workitems.py` 生成。聚合规则见脚本说明；判据与片段口径相同（`patterns.classify`）。"
           "工作项的证据层取其片段中最弱的一层；合并（并集）指同会话或跨会话合并至少一项。", ""]
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        out += run(s)
    (RESULTS_DIR / "r5_workitems.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
