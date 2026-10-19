"""S03-T2 证据包：每个抽样片段一页 Markdown（仅存本地 data/coding/，不进版本库）。

packets/<片段>.md        作者版：含自动候选模式与证据层（放在末尾，标明"自动候选，仅供参考"）
packets_blind/<片段>.md  盲评版：不含自动候选模式与证据层，其余相同
内容全部是事实性记录（时间、文件、提交、任务书），不写结论。
脱敏：IPv4、密码参数、长令牌串替换为占位符。请求原文保留（判断需要）。
"""
import csv
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_episodes import COMMIT_CMD, DEPLOY_CMD, EDIT_TOOLS, path_class  # noqa: E402
from extract_git import SYSTEMS as REPOS, TASK_LINE  # noqa: E402
from patterns import PATTERNS  # noqa: E402

OUT, OUT_BLIND = Path("data/coding/packets"), Path("data/coding/packets_blind")
MAX_FILES, MAX_CMDS, MAX_COMMIT_FILES, CMD_LEN = 80, 40, 15, 160
UNDO_CMD = re.compile(r"git\s+(checkout|restore|reset|revert|stash)\b|\bgit\s+add\b")
REDACT = [(re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
          (re.compile(r"(?i)(pass(?:word|wd)?\s*[=:]\s*|sshpass\s+-p\s*|-p\s*')\S+"), r"\1<密码>"),
          (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "<令牌>")]
STATE_ZH = {"committed": "进入提交", "not_committed": "未进入提交", "after_last_commit": "此后该文件无提交",
            "untracked": "文件从未进入版本库", "no_new": "重写既有内容", "no_content": "无写入内容"}
LABEL_ZH = {"task": "任务", "milestone": "里程碑", "fix": "修复", "hotfix": "热修", "unlabeled": "无标签"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def redact(s):
    for rx, rep in REDACT:
        s = rx.sub(rep, s)
    return s


def hm(s):
    return datetime.fromisoformat(s).strftime("%m-%d %H:%M") if s else "—"


def lag(a, b):
    h = (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600
    sign = "+" if h >= 0 else "−"
    h = abs(h)
    return f"{sign}{h * 60:.0f} 分钟" if h < 1 else f"{sign}{h:.1f} 小时"


def quote(text):
    text = redact(text.strip()) or "（空）"
    return "\n".join("> " + ln for ln in text.splitlines())


class Sys:
    def __init__(self, s):
        d = Path("data/derived") / s
        self.s = s
        self.eps = {e["episode"]: e for e in load(d / "episodes.csv")}
        self.by_sess = defaultdict(list)
        for e in sorted(self.eps.values(), key=lambda e: (e["start"], int(e["msg_seq"]))):
            self.by_sess[e["session"]].append(e)
        self.ev = {(e["session"], e["seq"]): e for e in load(d / "events.csv") if e["kind"] == "tool"}
        self.ep_events = defaultdict(list)
        for r in load(d / "episode_events.csv"):
            self.ep_events[r["episode"]].append(self.ev[(r["session"], r["seq"])])
        self.attr = {(a["session"], a["seq"]): a for a in load(d / "edit_attribution.csv")}
        self.cp = defaultdict(dict)
        for r in load(d / "checkpoint_deltas.csv"):
            self.cp[r["episode"]][r["path"]] = r
        self.pairs = defaultdict(list)
        self.eps_of_commit = defaultdict(set)
        for p in load(d / "episode_commit.csv"):
            self.pairs[p["episode"]].append(p)
            if p["source"] != "pathtime":
                self.eps_of_commit[p["commit"]].add(p["episode"])
        self.commits = {c["hash"]: c for c in load(d / "commits.csv")}
        self.commit_list = sorted(self.commits.values(), key=lambda c: c["time"])
        self.cfiles = defaultdict(list)
        for f in load(d / "commit_files.csv"):
            self.cfiles[f["hash"]].append(f["path"])
        self.tbk = {r["episode"]: r for r in load(d / "episode_taskbook.csv")}
        self.tb = {(r["milestone"], r["task"]): r for r in load(d / "taskbook_tasks.csv")}
        self.pat = {r["episode"]: r for r in load(d / "episode_patterns.csv")}
        self.titles = {}

    def title(self, ms, task):
        r = self.tb.get((ms, task))
        if not r:
            return ""
        if r["file"] not in self.titles:
            out = subprocess.run(["git", "-C", str(REPOS[self.s]), "show", f"HEAD:{r['file']}"], capture_output=True)
            lines = out.stdout.decode("utf-8", "replace").splitlines() if out.returncode == 0 else []
            self.titles[r["file"]] = {f"T{m.group('n')}": re.sub(r"[#*`|\[\]xX ]{0,8}", "", ln.strip(), count=1).strip("*| ")
                                      for ln in lines if (m := TASK_LINE.match(ln))}
        return self.titles[r["file"]].get(task, "")


def packet(S, ep_id, blind):
    e = S.eps[ep_id]
    sess = S.by_sess[e["session"]]
    i = next(k for k, x in enumerate(sess) if x["episode"] == ep_id)
    prev_e, next_e = (sess[i - 1] if i > 0 else None), (sess[i + 1] if i + 1 < len(sess) else None)
    fact, quoted = [], []

    def D(x):
        quoted.append(x)
        return x

    fact += [f"# 证据包 {ep_id}", "",
             f"- 系统：{S.s}；会话内第 {i + 1}/{len(sess)} 个片段；起点 {hm(e['start'])}，最后事件 {hm(e['last_event'])}，"
             f"下一片段起点 {hm(e['next_start'])}。",
             f"- 代码编辑 {e['n_edits_code']} 次，任务书编辑 {e['n_edits_plan']} 次，其他文档 {e['n_edits_doc']} 次，"
             f"工作区外 {e['n_edits_outside']} 次；终端命令 {e['n_terminal']} 条（提交 {e['n_commit_cmds']}、部署类 {e['n_deploy_cmds']}）；"
             f"向开发者提问 {e['n_ask']} 次；子代理 {e['n_subagents']} 个；并入的接续消息 {e['n_merged_msgs']} 条。", ""]

    fact += ["## 1 请求与上下文", ""]
    blocks = [("上一片段的请求", prev_e), ("本片段的请求", e), ("下一片段的请求", next_e)]
    for name, x in blocks:
        fact.append(f"**{name}**" + (f"（{hm(x['start'])}）" if x else "") + "：")
        fact.append("")
        q = quote(x["request_text"]) if x else "> （无）"
        quoted.append(q)
        fact += [q, ""]
    if e["cross_session_continuation"] == "1":
        fact += ["本片段是会话首条消息，且含接续线索（如“继续”“上次”）：可能接着另一个会话的工作。", ""]

    fact += ["## 2 片段时间线", "", "### 2.1 编辑（按文件汇总）", "",
             "| 文件 | 类别 | 次数 | 首次—末次 | 逐次编辑的内容去向 | 检查点差分 |", "|---|---|---|---|---|---|"]
    files = defaultdict(list)
    for ev in S.ep_events[ep_id]:
        if ev["tool"] in EDIT_TOOLS and ev["status"] == "completed":
            files[ev["path"].replace("\\", "/")].append(ev)
    order = sorted(files, key=lambda p: min(x["time"] for x in files[p]))
    for p in order[:MAX_FILES]:
        evs = files[p]
        st = Counter()
        for ev in evs:
            a = S.attr.get((ev["session"], ev["seq"]))
            if a is None:
                st["无写入内容"] += 1
            elif a["state"] == "committed":
                st[f"进入 {a['commit'][:7]}"] += 1
            else:
                st[STATE_ZH[a["state"]]] += 1
        cp = S.cp[ep_id].get(p)
        cps = (f"{cp['n_delta']} 行；" + (f"进入 {cp['commit'][:7]}" if cp["state"] == "committed" else STATE_ZH[cp["state"]])) if cp else "—"
        times = sorted(x["time"] for x in evs)
        fact.append(f"| `{D(p)}` | {path_class(p, evs[0]['in_workspace'])} | {len(evs)} | {hm(times[0])}—{hm(times[-1])} | "
                    f"{'；'.join(f'{k} ×{v}' for k, v in st.most_common())} | {cps} |")
    if len(order) > MAX_FILES:
        fact.append(f"| 另 {len(order) - MAX_FILES} 个文件 | | {sum(len(files[p]) for p in order[MAX_FILES:])} | | | |")
    if not order:
        fact.append("| （无编辑） | | | | | |")
    has_cp = bool(e["checkpoint"] and e["next_checkpoint"])
    nocp = ("本片段有检查点差分（只统计本片段编辑过的文件）；“检查点差分”一栏为“—”表示检查点里没有看到该文件的差异。"
            "检查点可能漏记，不能据“—”认为该文件在本片段没有新写入的内容。"
            if has_cp else "本片段没有检查点差分（会话最后一个片段，或消息无检查点）；“—”表示没有采集到，不代表没有变化。")
    fact += ["", nocp, ""]

    cmds = [ev for ev in S.ep_events[ep_id] if ev["tool"].startswith("run_terminal")]
    key_cmds = [ev for ev in cmds if COMMIT_CMD.search(ev["command"]) or DEPLOY_CMD.search(ev["command"])
                or UNDO_CMD.search(ev["command"])]
    fact += ["### 2.2 终端命令（只列提交、暂存、恢复撤销与部署上传类）", ""]
    for ev in sorted(key_cmds, key=lambda x: x["time"])[:MAX_CMDS]:
        c = redact(ev["command"].replace("\n", " ⏎ "))
        fact.append(f"- {hm(ev['time'])} `{D((c[:CMD_LEN] + '…') if len(c) > CMD_LEN else c)}`")
    if len(key_cmds) > MAX_CMDS:
        fact.append(f"- 另 {len(key_cmds) - MAX_CMDS} 条")
    fact += [f"- 其他终端命令 {len(cmds) - len(key_cmds)} 条（未列出）" if cmds else "- （无终端命令）", ""]
    asks = sorted(ev["time"] for ev in S.ep_events[ep_id] if ev["tool"] == "ask_question")
    fact += ["### 2.3 向开发者提问", "", ("- 时间：" + "、".join(hm(t) for t in asks)) if asks else "- （无）", ""]

    fact += ["## 3 相关提交", "", "### 3.1 与本片段新增内容重合的提交", ""]
    content = [p for p in S.pairs[ep_id] if p["source"] != "pathtime"]
    src_zh = {"edit": "逐次编辑", "checkpoint": "检查点差分", "both": "逐次编辑+检查点差分"}
    for p in sorted(content, key=lambda p: p["commit_time"]):
        c = S.commits[p["commit"]]
        others = sorted(S.eps_of_commit[p["commit"]] - {ep_id})
        fl = S.cfiles[p["commit"]]
        other_txt = "；".join(f"{o}（{'同会话' if S.eps[o]['session'] == e['session'] else '其他会话'}）" for o in others) or "无"
        cpf = sorted(path for path, r in S.cp[ep_id].items() if r["commit"] == p["commit"])
        fact += [f"- `{p['commit'][:7]}` {hm(c['time'])}【{LABEL_ZH[c['label']]}】{D(redact(c['subject']))}",
                 f"  - 重合依据：{src_zh[p['source']]}；重合编辑 {p['n_edits']} 次 / {p['n_lines']} 行，检查点文件 {p['cp_files']} 个 / {p['cp_lines']} 行",
                 *([f"  - 检查点重合的文件：" + "、".join(f"`{D(x)}`" for x in cpf)] if cpf else []),
                 f"  - 提交时间距本片段最后事件：{lag(e['last_event'], c['time'])}",
                 f"  - 该提交还含以下片段的内容：{other_txt}",
                 f"  - 改动文件 {len(fl)} 个：" + "、".join(f"`{D(x)}`" for x in fl[:MAX_COMMIT_FILES]) + ("等" if len(fl) > MAX_COMMIT_FILES else "")]
    if not content:
        fact.append("- （无）")
    pt = [p for p in S.pairs[ep_id] if p["source"] == "pathtime"]
    fact += ["", "### 3.2 只能按“文件 + 时间”推测的候选提交（无内容证据）", ""]
    for p in sorted(pt, key=lambda p: p["commit_time"]):
        c = S.commits[p["commit"]]
        fl = S.cfiles[p["commit"]]
        fact.append(f"- `{p['commit'][:7]}` {hm(c['time'])}【{LABEL_ZH[c['label']]}】{D(redact(c['subject']))}（无内容编辑 {p['pt_edits']} 次指向此提交）")
        fact.append(f"  - 改动文件 {len(fl)} 个：" + "、".join(f"`{D(x)}`" for x in fl[:MAX_COMMIT_FILES]) + ("等" if len(fl) > MAX_COMMIT_FILES else ""))
    if not pt:
        fact.append("- （无）")
    t0, t1 = e["start"], e["next_start"] or e["last_event"]
    during = [c for c in S.commit_list if t0 <= c["time"] <= t1]
    fact += ["", f"### 3.3 片段期间（{hm(t0)} 至 {hm(t1)}）版本库中的全部提交", ""]
    fact += [f"- `{c['hash'][:7]}` {hm(c['time'])}【{LABEL_ZH[c['label']]}】{D(redact(c['subject']))}" for c in during[:MAX_CMDS]] or ["- （无）"]
    if len(during) > MAX_CMDS:
        fact.append(f"- 另 {len(during) - MAX_CMDS} 个")
    fact.append("")

    tk = S.tbk[ep_id]
    fact += ["## 4 相关任务书", ""]
    fact.append(f"- 请求或同会话此前请求中出现的里程碑：{tk['milestones'] or '无'}")
    tasks = [t.split("-", 1) for t in tk["tasks"].split(";") if t]
    if tasks:
        fact += ["", "| 任务 | 标题 | 工作区首次写入 | 首次进入版本库 | 版本库中打勾 |", "|---|---|---|---|---|"]
        for ms, t in tasks:
            r = S.tb[(ms, t)]
            fact.append(f"| {ms}-{t} | {D(redact(S.title(ms, t))[:60])} | 见时间线 | {hm(r['first_time'])} | {hm(r['tick_time'])} |")
    else:
        fact.append("- 未对应到具体登记任务。")
    fact.append("")

    fact += ["## 5 待判问题（对照编码手册填写编码表）", "",
             "1. 主模式：本片段的工作以什么方式进入版本库？（手册第 2 节，填一个代码）",
             "2. 次模式：还有哪些方式同时成立？（可多填）",
             "3. 成因线索：是什么导致了这种方式？（手册第 3 节，可多填）",
             "4. 置信度：1 = 猜测，2 = 较有把握，3 = 证据明确。",
             "5. 证据是否足以判断？不足时写明缺什么。", ""]

    body = "\n".join(fact)
    factual = body
    for q in sorted(quoted, key=len, reverse=True):
        if q:
            factual = factual.replace(q, "")
    leaked = [p for p in PATTERNS if p in factual]
    assert not leaked, (ep_id, leaked)
    if blind:
        return body
    pr = S.pat[ep_id]
    return body + "\n".join(["---", "", "## 附：自动候选（仅供参考，不是结论；盲评版不含此节）", "",
                             f"- 证据层：{pr['evidence'] or '无代码编辑'}", f"- 候选模式（内容证据）：{pr['patterns']}",
                             f"- 候选模式（加入路径—时间推定）：{pr['patterns_pt']}", ""])


def main():
    global OUT, OUT_BLIND
    root = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    if root:
        OUT, OUT_BLIND = Path(root) / "packets", Path(root) / "packets_blind"
    sample = load(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--sample=")), "data/coding/sample_S03.csv"))
    systems = {s: Sys(s) for s in {r["system"] for r in sample}}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_BLIND.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    changed = 0
    for r in sample:
        S = systems[r["system"]]
        for path, text in [(OUT / f"{r['episode']}.md", packet(S, r["episode"], blind=False)),
                           (OUT_BLIND / f"{r['episode']}.md", packet(S, r["episode"], blind=True))]:
            if path.parent == OUT_BLIND:
                assert "自动候选" not in text and "证据层" not in text
            if path.exists() and path.read_text(encoding="utf-8") == text:
                continue
            changed += 1
            if path.exists() and not force:
                continue
            path.write_text(text, encoding="utf-8")
    print(f"packets: {len(sample)}; differing from existing: {changed}" + ("" if force or not changed else " (not overwritten; use --force)"))


if __name__ == "__main__":
    main()
