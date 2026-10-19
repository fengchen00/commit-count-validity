"""S27 / D38（B9）：SWE-chat 样本片段的盲评证据包（每片段一页 Markdown，只存本地 data/coding/swechat/packets_blind/）。

内容全部是事实性记录：请求原文与会话消息摘要、编辑时间线与新写入特征行的去向、本片段与同会话其他片段的 git 命令、
含本片段写入行的提交（逐文件覆盖与重合行样例、还含哪些片段的内容）、其后改过同一文件但不含这些行的提交、片段期间的提交、
其他会话 48 小时内改过相同文件的片段。不含自动模式、组、触发层与任何判定文字（脚本断言这些词不出现在非引文部分）。
作者与甲看同一份。方法口径与 swechat_replicate.py 相同（θ、ℓ、m 见论文 3.4.3 节）。

用法：python analysis/swechat_packets.py [--sample data/coding/swechat/sample_S27.csv] [--force]
"""
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import swechat_replicate as sw  # noqa: E402
from swechat_sample import SAMPLE, OUT_DIR, SHORT, ep_id, load_all  # noqa: E402

OUT = OUT_DIR / "packets_blind"
MAX_FILES, MAX_LINES_HIT, MAX_LINES_MISS, MAX_OTHER_EPS, MAX_DURING, MAX_NEAR = 40, 8, 5, 8, 25, 3
REQ_LEN, SUMMARY_LEN, REPLY_LEN, CMD_LEN, WINDOW_H = 4000, 90, 800, 200, 48
REDACT = [(re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
          (re.compile(r"(?i)(pass(?:word|wd)?\s*[=:]\s*|sshpass\s+-p\s*|-p\s*')\S+"), r"\1<密码>"),
          (re.compile(r"(?i)\b(sk|ghp|gho|xox[abp]|AKIA)[A-Za-z0-9_\-]{16,}\b"), "<令牌>")]
STATUS_ZH = {"committed": "进入 {h}", "no_new": "无新写入行", "no_later_commit": "此后该文件无提交", "not_in_commit": "其后提交未覆盖这些行",
             "untracked": "文件从未进入版本库", "no_text": "无写入内容"}
# 不得出现在非引文部分的词（自动判定与分层的名称）
FORBIDDEN = ["一对一", "共享", "拆分", "缺失", "证据不足", "单独提交指令", "片段内提交", "混合", "无可对上", "保留", "排除", "有歧义", "无歧义",
             "one_to_one", "shared", "split", "missing", "insufficient"]
GIT_RE = re.compile(r"\bgit\b")


def redact(s):
    for rx, rep in REDACT:
        s = rx.sub(rep, s)
    return s


def ts(x):
    return datetime.fromtimestamp(x, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if x == x else "—"


def lag(a, b):
    h = (b - a) / 3600
    sign = "+" if h >= 0 else "−"
    h = abs(h)
    return f"{sign}{h * 60:.0f} 分钟" if h < 1 else (f"{sign}{h:.1f} 小时" if h < 48 else f"{sign}{h / 24:.1f} 天")


def quote(text, limit=REQ_LEN):
    text = redact((text or "").strip()) or "（空）"
    if len(text) > limit:
        text = text[:limit] + f"\n…（截断，原文共 {len(text)} 字符）"
    return "\n".join("> " + ln for ln in text.splitlines())


def short(text, n=SUMMARY_LEN):
    t = redact(" ".join((text or "").split()))
    return t[:n] + ("…" if len(t) > n else "")


class RepoCtx:
    """一个仓库的会话一侧 + 版本库一侧，供多个证据包复用。"""

    def __init__(self, repo, kp, conv):
        self.repo, self.R = repo, kp["R"]
        self.eps = kp["episodes"]
        self.ref_commits = kp["ref_commits"]                 # commit -> 执行片段 i
        self.by_sess = defaultdict(list)
        for i, ep in enumerate(self.eps):
            self.by_sess[ep["session"]].append(i)
        self.k_of = {}                                       # i -> 会话内序号
        for s, idxs in self.by_sess.items():
            for k, i in enumerate(idxs, 1):
                self.k_of[i] = k
        self.sharers = defaultdict(set)
        for i, ep in enumerate(self.eps):
            for h in ep.get("P", ()):
                self.sharers[h].add(i)
        self.file_edits = defaultdict(list)                  # rel -> [(t, i)]
        for i, ep in enumerate(self.eps):
            for e in ep["edits"]:
                if e.get("code") and e.get("rel"):
                    self.file_edits[e["rel"]].append((e["t"], i))
        rows = conv[conv.repo_id == repo]
        self.prompt = {}                                     # (session, ts) -> 全文
        self.bash = defaultdict(list)                        # session -> [(ts, command, tool_call_id)]
        self.reply = defaultdict(list)                       # session -> [(ts, content)]
        self.results = {}
        for r in rows.itertuples(index=False):
            if r.turn_type == "user_prompt" and pd.notna(r.ts):
                self.prompt[(r.session_id, r.ts)] = r.content or ""
            elif r.turn_type == "tool_use" and r.tool_name == "Bash":
                try:
                    cmd = json.loads(r.tool_input_json).get("command", "") if isinstance(r.tool_input_json, str) else ""
                except Exception:
                    cmd = ""
                self.bash[r.session_id].append((r.ts, cmd or "", r.tool_call_id))
            elif r.turn_type == "tool_result":
                self.results[r.tool_call_id] = r.content or ""
            elif r.turn_type == "assistant_response" and pd.notna(r.ts):
                self.reply[r.session_id].append((r.ts, r.content or ""))
        self.last_commit_t = max(self.R.time.values())
        self._meta = {}

    def meta(self, h):
        if h not in self._meta:
            out = sw.git(self.R.path, "show", "-s", "--format=%an%x01%s", h)
            if "\x01" not in out:
                raise RuntimeError(f"git show 失败：{self.repo} {h}（镜像不可读？）")
            an, _, subj = out.strip().partition("\x01")
            self._meta[h] = (an, subj)
        return self._meta[h]

    def added_lines(self, h, path, limit):
        """提交 h 在 path 上的全部 + 行（不限长度），供新写入特征行为 0 的文件核对。"""
        raw = sw.git(self.R.path, "show", "-U0", "--format=", "--no-renames", h, "--", path, binary=True)
        out = []
        for ln in raw.split(b"\n"):
            if ln.startswith(b"+") and not ln.startswith(b"+++"):
                s = ln[1:].decode("utf-8", "replace").rstrip()
                if s.strip():
                    out.append(s)
        return out[:limit], len(out)

    def ep_label(self, i, cur_session, Q=lambda x: x):
        ep = self.eps[i]
        where = f"本会话第 {self.k_of[i]} 条" if ep["session"] == cur_session else f"其他会话 …{ep['session'][-8:]} 第 {self.k_of[i]} 条"
        return f"{where}（{ts(ep['start'])}，“{Q(short(ep['prompt'], 60))}”）"

    def end_of(self, i):
        idxs = self.by_sess[self.eps[i]["session"]]
        k = idxs.index(i)
        return self.eps[idxs[k + 1]]["start"] if k + 1 < len(idxs) else None


def packet(C, i, ep_name):
    R, ep = C.R, C.eps[i]
    s, t0 = ep["session"], ep["start"]
    idxs = C.by_sess[s]
    k, n_s = C.k_of[i], len(idxs)
    t1 = C.end_of(i)
    code_edits = [e for e in ep["edits"] if e.get("code")]
    fact, quoted = [], []

    def Q(x):
        quoted.append(x)
        return x

    fact += [f"# 证据包 {ep_name}", "",
             f"- 仓库：{C.repo}（公开仓库；镜像中最后一次提交 {ts(C.last_commit_t)}）。会话 …{s[-8:]} 共 {n_s} 条开发者消息，本片段为第 {k} 条。",
             f"- 片段起点 {ts(t0)}；下一条消息 {ts(t1) if t1 else '（无，会话最后一条）'}；起点距镜像最后提交 {lag(t0, C.last_commit_t)}。",
             f"- 编辑调用 {len(ep['edits'])} 次，其中代码文件 {len(code_edits)} 次；本片段内智能体执行 `git commit` {len(ep['commit_cmds'])} 次。", ""]

    # ---- 1 请求与上下文
    fact += ["## 1 请求与上下文", ""]
    prev_i = idxs[idxs.index(i) - 1] if idxs.index(i) > 0 else None
    next_i = idxs[idxs.index(i) + 1] if idxs.index(i) + 1 < n_s else None
    for name, j in (("上一条开发者消息", prev_i), ("本片段的请求", i), ("下一条开发者消息", next_i)):
        fact.append(f"**{name}**" + (f"（{ts(C.eps[j]['start'])}）" if j is not None else "") + "：")
        fact.append("")
        q = quote(C.prompt.get((s, C.eps[j]["start"]), C.eps[j]["prompt"])) if j is not None else "> （无）"
        fact += [Q(q), ""]
    replies = [(t, c) for t, c in C.reply.get(s, []) if t >= t0 and (t1 is None or t < t1)]
    if replies:
        fact += ["**智能体对本条的最后回复（摘要）**：", "", Q(quote(replies[-1][1], REPLY_LEN)), ""]
    fact += ["**本会话全部开发者消息**（★ 为本片段；“提交命令”列为该条消息下智能体执行 `git commit` 的次数）：", "",
             "| # | 时间 | 消息摘要 | 改代码 | 提交命令 |", "|---|---|---|---|---|"]
    for kk, j in enumerate(idxs, 1):
        e2 = C.eps[j]
        fact.append(f"| {'★ ' if j == i else ''}{kk} | {ts(e2['start'])} | {Q(short(C.prompt.get((s, e2['start']), e2['prompt'])))} | "
                    f"{'是' if e2['code_edits'] else '否'} | {len(e2['commit_cmds']) or ''} |")
    fact.append("")

    # ---- 2 时间线
    fact += ["## 2 片段时间线", "", "### 2.1 代码文件的编辑（按文件汇总）", "",
             "新写入特征行 = 本片段写入、且不在编辑前版本库版本中的行（≥ 15 字符，每次编辑至多 80 行）。去向按这些行其后首次被哪个提交覆盖 ≥ 50% 判。", "",
             "| 文件 | 工具 | 次数 | 首次—末次 | 新写入特征行 | 去向 |", "|---|---|---|---|---|---|"]
    files = defaultdict(list)
    for e in code_edits:
        files[e["rel"]].append(e)
    order = sorted(files, key=lambda p: min(x["t"] for x in files[p]))
    N_of = {}                                                 # rel -> 本片段在该文件的新写入行集合（最后一次编辑口径合并）
    for p in order:
        N = set()
        for e in files[p]:
            if e.get("text"):
                S = set(sw.sig_lines(e["text"]))
                old = set(sw.sig_lines(e["old"])) if e.get("old") else set()
                V = R.version_before(p, e["t"]) if not e.get("untracked") else None
                N |= S - old - (V or set())
        N_of[p] = N
    for p in order[:MAX_FILES]:
        evs = files[p]
        st = defaultdict(int)
        for e in evs:
            st[STATUS_ZH[e["status"]].format(h=(e.get("commit") or "")[:7])] += 1
        times = sorted(x["t"] for x in evs)
        tools = "/".join(sorted({e["tool"] for e in evs}))
        fact.append(f"| `{Q(p)}` | {tools} | {len(evs)} | {ts(times[0])[5:16]}—{ts(times[-1])[5:16]} | {sum(e.get('n_new', 0) for e in evs)} | "
                    f"{'；'.join(f'{kk} ×{v}' for kk, v in sorted(st.items(), key=lambda x: -x[1]))} |")
    if len(order) > MAX_FILES:
        fact.append(f"| 另 {len(order) - MAX_FILES} 个文件 | | {sum(len(files[p]) for p in order[MAX_FILES:])} | | | |")
    if not order:
        fact.append("| （无代码编辑） | | | | | |")
    fact += ["", "“无写入内容”为导出中没有写入文本的编辑；“无新写入行”为写入内容在编辑前的版本库版本中已存在（或都是短于 15 字符的行）。", ""]
    # 新写入特征行为 0 的文件：给出写入内容原文，便于与第 3.2 节其后提交的 + 行逐字核对
    zero = [p for p in order if not N_of.get(p)]
    if zero:
        fact += ["**新写入特征行为 0 的文件的写入内容**（按内容无法自动核对，请与第 3.2 节其后提交的 + 行对照）：", ""]
        for p in zero[:10]:
            for e in files[p]:
                if not e.get("text"):
                    fact.append(f"- `{Q(p)}`（{e['tool']}，{ts(e['t'])[5:16]}）：无写入内容")
                    continue
                new_lines = [ln for ln in e["text"].splitlines() if ln.strip()]
                old_lines = [ln for ln in (e.get("old") or "").splitlines() if ln.strip()]
                fact.append(f"- `{Q(p)}`（{e['tool']}，{ts(e['t'])[5:16]}）写入 {len(new_lines)} 行" + (f"，替换 {len(old_lines)} 行" if old_lines else "") + "：")
                for ln in new_lines[:10]:
                    fact.append(f"    - `{Q(redact(ln.rstrip())[:160])}`")
                if len(new_lines) > 10:
                    fact.append(f"    - …另 {len(new_lines) - 10} 行")
        fact.append("")
    # 有新写入特征行、但没有任何提交覆盖 ≥ 50% 的文件：列出这些行本身
    uncovered = [p for p in order if N_of.get(p) and not any(N_of[p] & R.added[h].get(p, set()) for h in ep.get("P", ()))]
    if uncovered:
        fact += ["**第 3.1 节的提交都不含的新写入特征行**（原文，每文件至多 10 行；供与第 3.2 节对照）：", ""]
        for p in uncovered[:10]:
            lines = sorted(N_of[p])
            fact.append(f"- `{Q(p)}`（{len(lines)} 行）：")
            fact += [f"    - `{Q(redact(ln)[:160])}`" for ln in lines[:10]]
            if len(lines) > 10:
                fact.append(f"    - …另 {len(lines) - 10} 行")
        fact.append("")

    fact += ["### 2.2 本片段内智能体执行的 git 命令", ""]
    cmds = [(t, c, cid) for t, c, cid in C.bash.get(s, []) if t >= t0 and (t1 is None or t < t1) and GIT_RE.search(c)]
    for t, c, cid in sorted(cmds):
        line = f"- {ts(t)[5:16]} `{Q(redact(c.replace(chr(10), ' ⏎ '))[:CMD_LEN])}`"
        if sw.GIT_COMMIT_RE.search(c):
            m = sw.HASH_RE.search(C.results.get(cid, ""))
            if m:
                h = R.resolve(m.group(1))
                line += f" → 回显 `{m.group(1)[:7]}`" + ("" if h else "（**回显哈希不在公开历史中**）")
            else:
                line += " → 无回显哈希（命令可能失败或输出被截断）"
        fact.append(line)
    if not cmds:
        fact.append("- （无）")
    n_bash = sum(1 for t, c, cid in C.bash.get(s, []) if t >= t0 and (t1 is None or t < t1))
    fact += [f"- 本片段其他终端命令 {n_bash - len(cmds)} 条（未列出）", ""]

    fact += ["### 2.3 本会话其他片段中智能体执行的提交命令", ""]
    others = []
    for j in idxs:
        if j == i:
            continue
        for cid in C.eps[j]["commit_cmds"]:
            m = sw.HASH_RE.search(C.eps[j]["bash_results"].get(cid, ""))
            h = R.resolve(m.group(1)) if m else None
            subj = C.meta(h)[1] if h else ""
            others.append((C.eps[j]["start"], j, m.group(1)[:7] if m else None, h, subj))
    for t, j, h7, h, subj in sorted(others, key=lambda x: (x[0], x[1])):
        hit = "；**含本片段的写入行**" if h and h in ep["P"] else ""
        fact.append(f"- 第 {C.k_of[j]} 条消息（{ts(t)[5:16]}，改代码：{'是' if C.eps[j]['code_edits'] else '否'}）：" +
                    (f"回显 `{h7}`" + (f" “{Q(redact(subj))}”{hit}" if h else "（**回显哈希不在公开历史中**）") if h7 else "无回显哈希"))
    if not others:
        fact.append("- （无）")
    fact.append("")

    # ---- 3 相关提交
    fact += ["## 3 相关提交", "", "### 3.1 含本片段新写入行的提交", "",
             "覆盖 = 该提交在该文件新增行中含本片段新写入特征行的比例。“还含以下片段的内容”按同一口径（覆盖 ≥ 50%）列出。", ""]
    P = sorted(ep.get("P", ()), key=lambda h: R.time[h])
    for h in P:
        an, subj = C.meta(h)
        files_h = [p for p in order if N_of.get(p) and (N_of[p] & R.added[h].get(p, set()))]
        other_eps = sorted(C.sharers[h] - {i}, key=lambda j: C.eps[j]["start"])
        fact += [f"- `{h[:7]}` {ts(R.time[h])}，提交者 {Q(redact(an))}：“{Q(redact(subj))}”",
                 f"  - 提交时间距本片段起点 {lag(t0, R.time[h])}；该提交改动文件 {len(R.files[h])} 个，其中本片段编辑过的 {len([p for p in order if p in R.files[h]])} 个",
                 f"  - 该提交还含以下片段的内容：" + ("；".join(C.ep_label(j, s, Q) for j in other_eps[:MAX_OTHER_EPS]) +
                                                   (f"；另 {len(other_eps) - MAX_OTHER_EPS} 个" if len(other_eps) > MAX_OTHER_EPS else "") if other_eps else "无")]
        for p in files_h:
            N, A = N_of[p], R.added[h].get(p, set())
            hit, miss = sorted(N & A), sorted(N - A)
            fact.append(f"  - `{Q(p)}`：本片段新写入 {len(N)} 行，其中 {len(hit)} 行在该提交中（{len(hit) / len(N):.0%}）")
            for ln in hit[:MAX_LINES_HIT]:
                fact.append(f"    - ✓ `{Q(redact(ln)[:160])}`")
            if len(hit) > MAX_LINES_HIT:
                fact.append(f"    - ✓ …另 {len(hit) - MAX_LINES_HIT} 行")
            for ln in miss[:MAX_LINES_MISS]:
                fact.append(f"    - ✗ `{Q(redact(ln)[:160])}`（不在该提交中）")
            if len(miss) > MAX_LINES_MISS:
                fact.append(f"    - ✗ …另 {len(miss) - MAX_LINES_MISS} 行不在该提交中")
    if not P:
        fact.append("- （无：本片段的新写入特征行没有被任何提交覆盖 ≥ 50%）")
    fact.append("")

    fact += ["### 3.2 其后改过同一文件的其他提交（不含第 3.1 节已列的；每文件至多 3 个）", "",
             "覆盖为该提交含本片段新写入特征行的比例；同一内容先进入分支提交、后随合并或压缩再次进入主线时，主线提交会在这里出现并显示高覆盖。"
             "新写入特征行为 0 的文件改列该提交在此文件上的 + 行，供与第 2.1 节的写入内容原文逐字核对。", ""]
    any_near = False
    for p in order:
        N = N_of.get(p, set())
        later = [(ct, h) for ct, h in R.by_file.get(p, []) if ct >= min(x["t"] for x in files[p]) and h not in P]
        rows = []
        for ct, h in later[:MAX_NEAR]:
            an, subj = C.meta(h)
            if N:
                cov = len(N & R.added[h].get(p, set())) / len(N)
                rows.append(f"  - `{h[:7]}` {ts(ct)}，{Q(redact(an))}：“{Q(redact(subj)[:80])}”，覆盖 {cov:.0%}")
            else:                                        # 无新写入特征行：列出该提交在此文件上的 + 行供逐字核对
                lines, n_all = C.added_lines(h, p, 8)
                rows.append(f"  - `{h[:7]}` {ts(ct)}，{Q(redact(an))}：“{Q(redact(subj)[:80])}”，该提交在此文件新增 {n_all} 行：")
                rows += [f"    - `{Q(redact(ln)[:160])}`" for ln in lines]
                if n_all > len(lines):
                    rows.append(f"    - …另 {n_all - len(lines)} 行")
        if later:
            any_near = True
            fact.append(f"- `{Q(p)}`（本片段新写入特征行 {len(N)} 行；其后共 {len(later)} 个提交改过此文件）")
            fact += rows
        elif p not in R.all_paths:
            any_near = True
            fact.append(f"- `{Q(p)}`：文件从未出现在公开历史中")
        else:
            any_near = True
            fact.append(f"- `{Q(p)}`：本片段之后没有任何提交改过此文件")
    if not any_near:
        fact.append("- （无）")
    fact.append("")

    t_end = t1 or (max([e["t"] for e in ep["edits"]] + [t0]) + 3600)
    during = sorted((ct, h) for h, ct in R.time.items() if t0 <= ct <= t_end)
    fact += [f"### 3.3 片段期间（{ts(t0)} 至 {ts(t_end)}）版本库全部分支上的提交", ""]
    for ct, h in during[:MAX_DURING]:
        an, subj = C.meta(h)
        mark = "；由本会话第 {} 条消息的命令产生".format(C.k_of[C.ref_commits[h]]) if h in C.ref_commits and C.eps[C.ref_commits[h]]["session"] == s else ""
        fact.append(f"- `{h[:7]}` {ts(ct)[5:16]}，{Q(redact(an))}：“{Q(redact(subj)[:100])}”，改动文件 {len(R.files[h])} 个{mark}")
    if len(during) > MAX_DURING:
        fact.append(f"- 另 {len(during) - MAX_DURING} 个")
    if not during:
        fact.append("- （无）")
    fact.append("")

    fact += [f"### 3.4 其他会话在本片段前后 {WINDOW_H} 小时内改过相同代码文件的片段", ""]
    seen, cross = set(), []
    for p in order:
        for t, j in C.file_edits.get(p, []):
            if C.eps[j]["session"] != s and abs(t - t0) <= WINDOW_H * 3600 and j not in seen:
                seen.add(j)
                cross.append((t, j, p))
    for t, j, p in sorted(cross)[:MAX_OTHER_EPS]:
        fact.append(f"- {C.ep_label(j, s, Q)}，改 `{Q(p)}`")
    if len(cross) > MAX_OTHER_EPS:
        fact.append(f"- 另 {len(cross) - MAX_OTHER_EPS} 个")
    if not cross:
        fact.append("- （无）")
    fact.append("")

    fact += ["## 4 待判问题（对照编码手册 v2.6 第 2–5 节与公开语料补充填写录入表）", "",
             "1. 主模式：本片段新写入的代码以什么方式进入版本库？（填一个代码）",
             "2. 次模式：还有哪些方式同时成立？（可多填）",
             "3. 成因线索：是什么导致了这种方式？（可多填）",
             "4. 置信度：1 = 猜测，2 = 较有把握，3 = 证据明确。",
             "5. 证据是否足以判断？不足时写明缺什么。", ""]

    body = "\n".join(fact)
    factual = body
    for q in sorted(set(quoted), key=len, reverse=True):
        if q:
            factual = factual.replace(q, "")
    leaked = [w for w in FORBIDDEN if w in factual]
    assert not leaked, (ep_name, leaked)
    return body


def main():
    args = sys.argv[1:]
    sample_p = Path(args[args.index("--sample") + 1]) if "--sample" in args else SAMPLE
    sample = list(csv.DictReader(open(sample_p, encoding="utf-8")))
    all_eps, conv = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = {}
    written = skipped = 0
    for r in sample:
        repo = r["repo"]
        if repo not in ctx:
            ctx[repo] = RepoCtx(repo, all_eps[repo], conv)
        C = ctx[repo]
        i = next(j for j in C.by_sess[r["session"]] if ep_id(repo, C.eps, j) == r["episode"])
        text = packet(C, i, r["episode"])
        p = OUT / f"{r['episode']}.md"
        if p.exists() and p.read_text(encoding="utf-8") == text:
            continue
        if p.exists() and "--force" not in args:
            skipped += 1
            continue
        p.write_text(text, encoding="utf-8")
        written += 1
    print(f"packets: {len(sample)}; written {written}; differing but not overwritten {skipped}" + ("" if not skipped else " (use --force)"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
