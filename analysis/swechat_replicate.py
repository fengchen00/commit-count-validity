"""在 SWE-chat（Claude Code 子集）上复制内容对齐与 RQ1 / RQ2（S21-T-F4，D27-3）。

输入：data/swechat/{sessions,conversations,commits,checkpoints}.parquet（HF 快照 f66cca95），
      %TEMP%/swechat_probe/<owner>__<name>.git（`git clone --mirror`，非 entire/* 引用为版本库一侧）。
方法：论文 3.2 节切片段、3.4.2 节算法 1 的第 1–3、5、6 步（无检查点差分）、3.4.4 节外部参照与基线、3.6 节映射模式。
      Entire 的 checkpoint→commit 链接（commits 表）只作第三层参照，不进入对齐。
输出：<PAPER_DIR>/results/r21_swechat.md 与 numbers.json 的 swechat 键。

用法：python analysis/swechat_replicate.py [--mirrors DIR] [--repos a/b,c/d]
"""
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR  # noqa: E402

DATA = Path("data/swechat")
MIRRORS = Path(os.environ.get("SWECHAT_MIRRORS", Path(os.environ.get("TEMP", "/tmp")) / "swechat_probe"))
# 按 commits 表唯一 SHA 数取前 5 个可克隆的公开仓库；osabiohq/osabio 与 marcus-sa/brain 为同一仓库，只取前者；
# 第 6 位 oddessentials/ado-git-repo-insights 补位（S21 任务书）。
REPOS = ["osabiohq/osabio", "timelabs-npo/rhea-project", "obsessiondb/rudel",
         "wanshicheng/duckdb-data-agent", "oddessentials/ado-git-repo-insights"]
# 数据集中同一仓库的两个 repo_id（S20 §3a：brain ⊂ osabio 的提交历史；cyyeh 与 wanshicheng 完全相同）；会话多记在别名下。
ALIASES = {"marcus-sa/brain": "osabiohq/osabio", "cyyeh/duckdb-data-agent": "wanshicheng/duckdb-data-agent"}
AGENT = "Claude Code"
EDIT_TOOLS = ("Edit", "MultiEdit", "Write", "NotebookEdit")
THETA, ELL, M_CAP = 0.5, 15, 80                     # 论文 3.4.3 节的参数，原样沿用
CONT_WORDS = {"continue", "go on", "ok", "okay", "yes", "y", "proceed", "next", "继续", "好", "好的", "是"}
NONCODE_EXT = {".md", ".mdc", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".log", ".zip",
               ".gz", ".tar", ".lock", ".csv", ".json", ".yaml", ".yml", ".toml", ".ini", ".env"}
NONCODE_DIRS = ("docs/", ".claude/", ".entire/", "specs/", "plans/")
HASH_RE = re.compile(r"\[[^\]\n]+\s+([0-9a-f]{7,40})\]")
GIT_COMMIT_RE = re.compile(r"\bgit\s+(?:-[-\w=]+\s+)*commit\b")


def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def sig_lines(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) >= ELL:
            out.append(s)
            if len(out) >= M_CAP:
                break
    return out


def is_code(path):
    p = path.lower()
    if any(p.startswith(d) or f"/{d}" in p for d in NONCODE_DIRS):
        return False
    return Path(p).suffix not in NONCODE_EXT


# ------------------------------------------------------------------ 版本库一侧
class Repo:
    """非 entire/* 引用上的全部非合并提交：时间、逐文件新增特征行、改动文件；文件版本按需读取并缓存。"""

    def __init__(self, path, first_parent=False):
        self.path = path
        self.first_parent = first_parent
        if first_parent:
            # D30 对照 V2：只走默认分支的 first-parent 链（含合并提交，其差异即并入主线的内容），不看其他分支
            head = git(path, "symbolic-ref", "--short", "HEAD").strip() or "main"
            refs, log_args = [head], ["--first-parent", "--diff-merges=first-parent"]
        else:
            refs = [r for r in git(path, "for-each-ref", "--format=%(refname)", "refs/heads").split() if "entire/" not in r]
            log_args = ["--no-merges"]
        self.refs = refs
        self.time, self.files, self.added = {}, {}, defaultdict(lambda: defaultdict(set))
        self.all_paths = set()
        sep = b"\x01\x02SWECHAT\x02\x01"
        raw = git(path, "log", *log_args, "-U0", "--format=%x01%x02SWECHAT%x02%x01%H %ct", "--no-renames", *refs, "--", binary=True)
        for chunk in raw.split(sep)[1:]:
            head, _, body = chunk.partition(b"\n")
            h, t = head.decode().split()[:2]
            self.time[h] = int(t)
            self.files[h] = set()
            f = None
            for ln in body.split(b"\n"):
                if ln.startswith(b"diff --git "):
                    f = None
                elif ln.startswith(b"--- a/"):
                    p = ln[6:].decode("utf-8", "replace")
                    self.files[h].add(p)                                   # 删除的文件也算改动
                    self.all_paths.add(p)
                elif ln.startswith(b"+++ b/"):
                    f = ln[6:].decode("utf-8", "replace")
                    self.files[h].add(f)
                    self.all_paths.add(f)
                elif f is not None and ln.startswith(b"+") and not ln.startswith(b"+++"):
                    s = ln[1:].decode("utf-8", "replace").strip()
                    if len(s) >= ELL:
                        self.added[h][f].add(s)
        self.by_file = defaultdict(list)             # path -> [(time, commit)] 升序
        for h, fs in self.files.items():
            for f in fs:
                self.by_file[f].append((self.time[h], h))
        for f in self.by_file:
            self.by_file[f].sort()
        self._ver = {}

    def resolve(self, short):
        for h in self.time:
            if h.startswith(short):
                return h
        return None

    def version_before(self, path, t):
        """path 在 t 之前最后一个提交的版本的特征行集合；从未进入版本库返回 None。"""
        hist = self.by_file.get(path)
        if not hist:
            return None
        prev = None
        for ct, h in hist:
            if ct < t:
                prev = h
            else:
                break
        if prev is None:
            return set()                             # 文件此后才进入版本库：之前不存在
        key = (prev, path)
        if key not in self._ver:
            blob = git(self.path, "show", f"{prev}:{path}", binary=True)
            self._ver[key] = set(sig_lines(blob.decode("utf-8", "replace")))
        return self._ver[key]

    def attribute(self, path, t, N):
        """算法 1 第 3 步：t 之后第一个覆盖 ≥ θ 的提交。返回 (状态, 提交, 覆盖)。"""
        if not N:
            return "no_new", None, 0.0
        later = [(ct, h) for ct, h in self.by_file.get(path, []) if ct >= t]
        for ct, h in later:
            cov = len(N & self.added[h].get(path, set())) / len(N)
            if cov >= THETA:
                return "committed", h, cov
        return ("no_later_commit" if not later else "not_in_commit"), None, 0.0

    def next_commit(self, path, t):
        for ct, h in self.by_file.get(path, []):
            if ct >= t:
                return h
        return None


# ------------------------------------------------------------------ 会话一侧
def load_conversations(repos):
    pf = pq.ParquetFile(DATA / "conversations.parquet")
    cols = ["session_id", "repo_id", "turn_number", "turn_type", "timestamp", "tool_name", "tool_call_id",
            "tool_input_json", "content", "is_continuation", "file_path"]
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=cols).to_pandas()
        parts.append(t[t.repo_id.isin(repos)])
    c = pd.concat(parts, ignore_index=True)
    c["ts"] = c.timestamp.map(lambda x: x.timestamp() if pd.notna(x) else float("nan"))
    return c.sort_values(["session_id", "turn_number"]).reset_index(drop=True)


def rel_path(abs_path, roots, all_paths):
    """把会话中的绝对路径映射为仓库相对路径：取在版本库历史中出现过的最长后缀，返回 (相对路径, 根)。"""
    p = abs_path.replace("\\", "/")
    for root in roots:
        if p.startswith(root + "/"):
            return p[len(root) + 1:], root
    parts = p.split("/")
    for i in range(1, len(parts)):
        suf = "/".join(parts[i:])
        if suf in all_paths:
            return suf, "/".join(parts[:i])
    return None, None


def split_episodes(sess_rows):
    """3.2 节：每条真实开发者消息（有时间戳、非接续）开一个片段；接续消息并入前一片段。"""
    eps, cur = [], None
    for r in sess_rows.itertuples(index=False):
        if r.turn_type == "user_prompt":
            if pd.isna(r.ts):
                continue                             # 误标为 user_prompt 的摘要块（S20 §3a 数据质量项）
            txt = (r.content or "").strip().lower().rstrip(".!。！")
            cont = bool(r.is_continuation) if pd.notna(r.is_continuation) else False
            if cur is not None and (cont or txt in CONT_WORDS):
                continue
            cur = {"session": r.session_id, "start": r.ts, "prompt": (r.content or "")[:200], "rows": []}
            eps.append(cur)
        elif cur is not None:
            cur["rows"].append(r)
    return eps


def parse_edit(name, js):
    """返回 [(path, new_text, old_text)]；解析失败返回 None。"""
    try:
        inp = json.loads(js) if isinstance(js, str) and js else {}
    except Exception:
        return None
    if name == "Edit":
        return [(inp.get("file_path"), inp.get("new_string"), inp.get("old_string") or "")]
    if name == "MultiEdit":
        return [(inp.get("file_path"), e.get("new_string"), e.get("old_string") or "") for e in inp.get("edits") or []]
    if name == "Write":
        return [(inp.get("file_path"), inp.get("content"), "")]
    if name == "NotebookEdit":
        return [(inp.get("notebook_path"), inp.get("new_source"), "")]
    return None


# ------------------------------------------------------------------ 主流程
def replicate_repo(repo_id, conv, entire_links, R=None, write_deinherit=False, keep=None):
    """R：预先构建的版本库一侧（可复用）；write_deinherit：D30 对照 V1，Write 的新写入行再去掉同会话此前其他片段已写入的行。
    keep：给出字典时把片段级中间结果存入其中（D35-6 B6 的分层分析用），不影响返回值。"""
    mirror = MIRRORS / (repo_id.replace("/", "__") + ".git")
    R = R or Repo(mirror)
    rows = conv[conv.repo_id == repo_id]
    sessions = rows.session_id.unique()
    episodes, roots = [], {}
    n_prompt_rows = int((rows.turn_type == "user_prompt").sum())
    for s in sessions:
        sr = rows[rows.session_id == s]
        for ep in split_episodes(sr):
            ep["edits"], ep["commit_cmds"], ep["bash_results"] = [], [], {}
            for r in ep["rows"]:
                if r.turn_type == "tool_use" and r.tool_name in EDIT_TOOLS:
                    parsed = parse_edit(r.tool_name, r.tool_input_json)
                    if parsed is None:
                        ep["edits"].append({"path": None, "t": r.ts, "text": None, "old": "", "tool": r.tool_name})
                        continue
                    for path, new, old in parsed:
                        ep["edits"].append({"path": path, "t": r.ts, "text": new, "old": old, "tool": r.tool_name})
                elif r.turn_type == "tool_use" and r.tool_name == "Bash":
                    try:
                        cmd = json.loads(r.tool_input_json).get("command", "") if isinstance(r.tool_input_json, str) else ""
                    except Exception:
                        cmd = ""
                    if GIT_COMMIT_RE.search(cmd or ""):
                        ep["commit_cmds"].append(r.tool_call_id)
                elif r.turn_type == "tool_result":
                    ep["bash_results"][r.tool_call_id] = r.content or ""
            del ep["rows"]
            episodes.append(ep)
    # 路径映射与代码编辑：先按会话确定工作区根（历史中出现过的最长后缀），再把根下的全部路径取相对路径；
    # 根外的路径（如 ~/.claude/plans/*.md）视为工作区外，不计。
    n_edit_calls = n_unparsed = n_no_text = n_unmapped = 0
    for ep in episodes:
        for e in ep["edits"]:
            if e["path"]:
                rel, root = rel_path(e["path"], [], R.all_paths)
                if root:
                    roots.setdefault(ep["session"], [])
                    if root not in roots[ep["session"]]:
                        roots[ep["session"]].append(root)
    for ep in episodes:
        known = sorted(roots.get(ep["session"], []), key=len, reverse=True)
        for e in ep["edits"]:
            n_edit_calls += 1
            if e["text"] is None and e["path"] is None:
                n_unparsed += 1
                e["rel"], e["code"], e["untracked"] = None, False, False
                continue
            p = (e["path"] or "").replace("\\", "/")
            rel = None
            for rt in known:
                if p.startswith(rt + "/"):
                    rel = p[len(rt) + 1:]
                    break
            if rel is None:
                n_unmapped += 1
            e["rel"] = rel
            e["untracked"] = bool(rel) and rel not in R.all_paths
            e["code"] = bool(rel) and is_code(rel)
            if e["text"] is None or e["text"] == "":
                n_no_text += 1
    # D30 对照 V1：同会话此前其他片段写入过的特征行（Write 整文件写入会把它们再写一遍，若不去掉就会被记为本片段的新写入行）
    earlier = defaultdict(list)                       # session -> [(t, episode_idx, sig_set)]，只含代码编辑
    if write_deinherit:
        for i, ep in enumerate(episodes):
            for e in ep["edits"]:
                if e["code"] and e["text"]:
                    earlier[ep["session"]].append((e["t"], i, set(sig_lines(e["text"]))))
        for s in earlier:
            earlier[s].sort(key=lambda x: x[0])
    n_deinherited = 0
    # 算法 1 第 1–3 步
    status = Counter()
    for i_ep, ep in enumerate(episodes):
        ep["P"], ep["pairs"] = set(), []
        code_edits = [e for e in ep["edits"] if e["code"]]
        ep["code_edits"] = len(code_edits)
        ep["verifiable"] = bool(code_edits) and all(e["text"] is not None for e in code_edits)
        for e in code_edits:
            if e["text"] is None:
                e["status"] = "no_text"
            elif e["untracked"]:
                e["status"] = "untracked"
            else:
                S = set(sig_lines(e["text"]))
                old = set(sig_lines(e["old"])) if e["old"] else set()
                V = R.version_before(e["rel"], e["t"])
                N = S - old - (V or set())
                if write_deinherit and e["tool"] == "Write":
                    inherited = set()
                    for t2, j, S2 in earlier[ep["session"]]:
                        if t2 >= e["t"]:
                            break
                        if j != i_ep:
                            inherited |= S2
                    n_deinherited += len(N & inherited)
                    N = N - inherited
                st, h, cov = R.attribute(e["rel"], e["t"], N)
                e["status"], e["commit"], e["cov"], e["n_new"] = st, h, cov, len(N)
                if h:
                    ep["P"].add(h)
                    ep["pairs"].append((h, e["rel"], len(N & R.added[h][e["rel"]])))
            status[e["status"]] += 1
    # 合并判定：提交被 ≥ 2 个片段的内容进入
    sharers = defaultdict(set)
    for i, ep in enumerate(episodes):
        for h in ep["P"]:
            sharers[h].add(i)
    sess_of = {i: ep["session"] for i, ep in enumerate(episodes)}
    pat = Counter()
    code_eps = [i for i, ep in enumerate(episodes) if ep["code_edits"]]
    for i in code_eps:
        ep = episodes[i]
        P = ep["P"]
        shared = [h for h in P if len(sharers[h]) > 1]
        if not P:
            k = "missing" if ep["verifiable"] else "insufficient"
        elif len(P) == 1 and not shared:
            k = "one_to_one"
        else:
            k = ("split" if len(P) >= 2 else "") + ("+shared" if shared else "")
            k = k.strip("+")
        ep["pattern"] = k
        pat[k] += 1
        if shared:
            others = {sess_of[j] for h in shared for j in sharers[h]} - {ep["session"]}
            pat["shared_cross_session" if others else "shared_same_session"] += 1
    # 外部参照：智能体 git commit 回显的哈希
    ref_pairs, ref_commits, n_cmd, n_cmd_hash, n_hash_found = set(), {}, 0, 0, 0
    for i, ep in enumerate(episodes):
        for cid in ep["commit_cmds"]:
            n_cmd += 1
            m = HASH_RE.search(ep["bash_results"].get(cid, ""))
            if not m:
                continue
            n_cmd_hash += 1
            h = R.resolve(m.group(1))
            if not h:
                continue
            n_hash_found += 1
            ref_commits[h] = i
            ep_files = {e["rel"] for e in ep["edits"] if e["code"] and e["rel"]}
            if ep_files & R.files[h]:
                ref_pairs.add((i, h))
    content_pairs = {(i, h) for i, ep in enumerate(episodes) for h in ep["P"]}
    # 基线：每次代码编辑归到其文件时间不早于编辑的第一个提交
    base_pairs = set()
    for i, ep in enumerate(episodes):
        for e in ep["edits"]:
            if e["code"] and e["rel"] and not e["untracked"]:
                h = R.next_commit(e["rel"], e["t"])
                if h:
                    base_pairs.add((i, h))

    def precision(pairs):
        on_ref = [(i, h) for i, h in pairs if h in ref_commits]
        strict = sum(1 for i, h in on_ref if ref_commits[h] == i)
        loose = sum(1 for i, h in on_ref if ref_commits[h] == i or
                    (sess_of[ref_commits[h]] == sess_of[i] and episodes[i]["start"] <= R.time[h]))
        return {"n": len(on_ref), "strict": strict, "loose": loose}

    # D32 B7 区分检验：只看执行提交的片段本身改过代码的参照提交（排除"单独一条消息让智能体提交"的情形），
    # 若严格一致率在这一子集上明显升高，则低严格一致率主要来自单独提交消息，而不是内容对齐把提交对错了片段。
    exec_code = {h for h, i in ref_commits.items() if i in set(code_eps)}

    def precision_code_exec(pairs):
        on_ref = [(i, h) for i, h in pairs if h in exec_code]
        return {"n": len(on_ref), "strict": sum(1 for i, h in on_ref if ref_commits[h] == i),
                "loose": sum(1 for i, h in on_ref if ref_commits[h] == i or (sess_of[ref_commits[h]] == sess_of[i] and episodes[i]["start"] <= R.time[h])),
                "ref_commits": len(exec_code), "ref_commits_all": len(ref_commits)}

    recall = lambda pairs: {"ref": len(ref_pairs), "hit": len(ref_pairs & pairs)}
    ver_ref = {(i, h) for i, h in ref_pairs if episodes[i]["verifiable"]}
    # 会话层参照：Claude Code 用户常在单独一条消息里让智能体提交，执行片段本身不改代码；
    # 放宽为"该会话中某片段编辑过的代码文件被该提交改动"，看方法是否把提交对到同会话的某个片段。
    sess_files = defaultdict(set)
    for i, ep in enumerate(episodes):
        sess_files[ep["session"]] |= {e["rel"] for e in ep["edits"] if e["code"] and e["rel"]}
    ref_sess = {(sess_of[i], h) for h, i in ref_commits.items() if sess_files[sess_of[i]] & R.files[h]}
    sess_recall = lambda pairs: {"ref": len(ref_sess), "hit": len(ref_sess & {(sess_of[i], h) for i, h in pairs})}
    # 第三层参照：Entire 的 session→commit 链接
    ent = entire_links.get(repo_id, {})
    in_entire = sum(1 for i, h in content_pairs if h in ent.get(sess_of[i], set()))
    ent_pairs = {(i, h) for i in range(len(episodes)) for h in ent.get(sess_of[i], set()) if h in R.time}
    linked = [s for s in sessions if ent.get(s)]
    out = {
        "repo": repo_id, "sessions": len(sessions), "user_prompt_rows": n_prompt_rows, "episodes": len(episodes),
        "code_episodes": len(code_eps), "verifiable_code_episodes": sum(1 for i in code_eps if episodes[i]["verifiable"]),
        "edit_calls": n_edit_calls, "edit_unparsed": n_unparsed, "edit_no_text": n_no_text, "edit_path_unmapped": n_unmapped,
        "code_edit_status": dict(status), "commits_in_mirror": len(R.time), "refs": len(R.refs),
        "variant": {"first_parent": R.first_parent, "write_deinherit": write_deinherit, "write_lines_deinherited": n_deinherited},
        "patterns": dict(pat),
        "reference": {"commit_cmds": n_cmd, "with_hash": n_cmd_hash, "hash_in_mirror": n_hash_found,
                      "ref_commits": len(ref_commits), "ref_pairs_code_overlap": len(ref_pairs)},
        "content": {"pairs": len(content_pairs), "episodes_with_pairs": len({i for i, _ in content_pairs}),
                    "recall": recall(content_pairs), "recall_verifiable": {"ref": len(ver_ref), "hit": len(ver_ref & content_pairs)},
                    "recall_session": sess_recall(content_pairs), "precision": precision(content_pairs),
                    "precision_code_exec": precision_code_exec(content_pairs)},
        "baseline": {"pairs": len(base_pairs), "episodes_with_pairs": len({i for i, _ in base_pairs}),
                     "recall": recall(base_pairs), "recall_session": sess_recall(base_pairs), "precision": precision(base_pairs),
                     "precision_code_exec": precision_code_exec(base_pairs),
                     "only_baseline": len(base_pairs - content_pairs), "only_content": len(content_pairs - base_pairs),
                     "both": len(base_pairs & content_pairs)},
        "entire_layer": {"content_pairs_in_entire_session_links": in_entire, "entire_pairs": len(ent_pairs),
                         "entire_pairs_hit_by_content": len(ent_pairs & content_pairs),
                         "entire_pairs_hit_by_baseline": len(ent_pairs & base_pairs),
                         "sessions_with_links": len(linked),
                         "mean_commits_per_linked_session": sum(len(ent[s]) for s in linked) / max(1, len(linked))},
    }
    if keep is not None:
        keep.update(episodes=episodes, ref_commits=ref_commits, content_pairs=content_pairs, code_eps=code_eps, R=R)
    return out


def entire_links_table(repos):
    """Entire 链接：commits.checkpoint_pk → checkpoints.session_pks（即 session_id 列表）→ 会话；返回 {repo: {session: {sha}}}。"""
    cm = pq.read_table(DATA / "commits.parquet", columns=["commit_sha", "checkpoint_pk", "repo_id"]).to_pandas()
    cp = pq.read_table(DATA / "checkpoints.parquet", columns=["checkpoint_pk", "session_pks", "repo_id"]).to_pandas()
    cm, cp = cm[cm.repo_id.isin(repos) & cm.commit_sha.notna()], cp[cp.repo_id.isin(repos)]
    sess_by_cp = {}
    for r in cp.itertuples(index=False):
        try:
            sess_by_cp[r.checkpoint_pk] = set(json.loads(r.session_pks) or []) if isinstance(r.session_pks, str) else set()
        except Exception:
            sess_by_cp[r.checkpoint_pk] = set()
    links = defaultdict(lambda: defaultdict(set))
    for r in cm.itertuples(index=False):
        for s in sess_by_cp.get(r.checkpoint_pk, ()):
            links[r.repo_id][s].add(r.commit_sha)
    return links


def pct(a, b):
    return f"{a}/{b}（{a / b:.0%}）" if b else "—"


def main():
    args = sys.argv[1:]
    repos = REPOS
    if "--repos" in args:
        repos = args[args.index("--repos") + 1].split(",")
    global MIRRORS
    if "--mirrors" in args:
        MIRRORS = Path(args[args.index("--mirrors") + 1])
    canon = lambda r: ALIASES.get(r, r)
    wanted = [r for r in list(REPOS) + list(ALIASES) if canon(r) in repos]
    ss = pq.read_table(DATA / "sessions.parquet", columns=["session_id", "repo_id", "agent"]).to_pandas()
    keep = set(ss[ss.agent == AGENT].session_id)      # 仓库归属以 conversations 表为准（sessions 表的 repo_id 对部分会话不同）
    conv = load_conversations(wanted)
    conv = conv[conv.session_id.isin(keep)].copy()
    conv["repo_id"] = conv.repo_id.map(canon)
    links_raw = entire_links_table(wanted)
    links = defaultdict(lambda: defaultdict(set))
    for rid, d in links_raw.items():
        for s, shas in d.items():
            links[canon(rid)][s] |= shas
    fast = os.environ.get("SWECHAT_FAST") == "1"      # 只跑基准口径，不跑 D30 的三项对照
    res, res_v1, res_v2 = [], [], []
    for r in repos:
        t0 = time.time()
        mirror = MIRRORS / (r.replace("/", "__") + ".git")
        R = Repo(mirror)
        res.append(replicate_repo(r, conv, links, R=R))
        print(f"[{r}] base {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        if not fast:
            t1 = time.time()
            res_v1.append(replicate_repo(r, conv, links, R=R, write_deinherit=True))
            print(f"[{r}] v1 {time.time() - t1:.0f}s", file=sys.stderr, flush=True)
            t1 = time.time()
            res_v2.append(replicate_repo(r, conv, links, R=Repo(mirror, first_parent=True)))
            print(f"[{r}] v2 {time.time() - t1:.0f}s", file=sys.stderr, flush=True)
        print(f"[{r}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    tot = aggregate(res)
    L = report(res, tot)
    variants = {}
    if not fast:
        miss = lambda r: 1 - r["reference"]["hash_in_mirror"] / max(1, r["reference"]["with_hash"])
        kept = [r for r in res if miss(r) <= HASH_MISS_MAX]
        variants = {
            "base": {"label": "基准口径（全部非合并提交，Write 只去版本库旧行）", "total": dict(tot), "repos": [r["repo"] for r in res]},
            "v1_write_deinherit": {"label": "V1 Write 去继承：Write 的新写入行再去掉同会话此前其他片段写入的行", "total": dict(aggregate(res_v1)),
                                   "repos": [r["repo"] for r in res_v1], "write_lines_deinherited": sum(r["variant"]["write_lines_deinherited"] for r in res_v1),
                                   "per_repo": res_v1},
            "v2_first_parent": {"label": "V2 first-parent：只走默认分支的 first-parent 链（含合并提交）", "total": dict(aggregate(res_v2)),
                                "repos": [r["repo"] for r in res_v2], "per_repo": res_v2},
            "v3_low_missing": {"label": f"V3 剔除回显哈希缺失率 > {HASH_MISS_MAX:.0%} 的仓库", "total": dict(aggregate(kept)),
                               "repos": [r["repo"] for r in kept], "missing_rate": {r["repo"]: round(miss(r), 3) for r in res}},
        }
        L += ["", f"## 6 三项对照（D30；ChatGPT 外审第 ② 条）", "",
              "基准口径的三个可疑之处各做一项对照，只改一处、其余不变：V1 针对 Write 整文件写入继承同会话此前片段内容而被记为共享；"
              "V2 针对所有非合并提交把分支与主线上的重复内容都算作提交；V3 针对智能体回显的提交哈希在公开历史中找不到（历史被改写或压缩）的仓库。",
              f"各仓库的回显哈希缺失率（1 − 哈希在镜像中 / 带哈希）：" + "；".join(f"{k} {v:.0%}" for k, v in variants["v3_low_missing"]["missing_rate"].items()) + "。", "",
              "| 口径 | 仓库数 | 代码修改片段 | 一对一 | 拆分 | 共享 | 拆分且共享 | 缺失 | 召回（片段层 / 会话层） | 严格 / 宽松一致率 | 仅基线 / 仅内容对 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in variants.items():
            t = defaultdict(int, v["total"])
            n = t["code_episodes"]
            L.append(f"| {v['label']} | {len(v['repos'])} | {n} | {pct(t['pat_one_to_one'], n)} | {pct(t['pat_split'], n)} | {pct(t['pat_shared'], n)} | "
                     f"{pct(t['pat_split+shared'], n)} | {pct(t['pat_missing'], n)} | {pct(t['content_hit'], t['content_ref'])} / {pct(t['content_shit'], t['content_sref'])} | "
                     f"{pct(t['content_ps'], t['content_pn'])} / {pct(t['content_pl'], t['content_pn'])} | {t['only_baseline']} / {t['only_content']} |")
        L.append(f"\nV1 从 Write 的新写入行中去掉的继承行共 {variants['v1_write_deinherit']['write_lines_deinherited']} 行。V2 下镜像提交数：" +
                 "；".join(f"{r['repo']} {r['commits_in_mirror']}" for r in res_v2) + "。")
    (RESULTS_DIR / "r21_swechat.md").write_text("\n".join(L), encoding="utf-8")
    nums_p = RESULTS_DIR / "numbers.json"
    num = json.loads(nums_p.read_text(encoding="utf-8")) if nums_p.exists() else {}
    num["swechat"] = {"repos": res, "total": dict(tot), "params": {"theta": THETA, "ell": ELL, "m": M_CAP, "agent": AGENT},
                      "variants": {k: {kk: vv for kk, vv in v.items() if kk != "per_repo"} for k, v in variants.items()},
                      "variants_per_repo": {k: v.get("per_repo", []) for k, v in variants.items() if "per_repo" in v}}
    nums_p.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


HASH_MISS_MAX = 0.25


def aggregate(res):
    tot = defaultdict(int)
    for r in res:
        for k in ("sessions", "episodes", "code_episodes", "verifiable_code_episodes", "edit_calls"):
            tot[k] += r[k]
        for k, v in r["patterns"].items():
            tot["pat_" + k] += v
        for side in ("content", "baseline"):
            tot[f"{side}_hit"] += r[side]["recall"]["hit"]
            tot[f"{side}_ref"] += r[side]["recall"]["ref"]
            tot[f"{side}_shit"] += r[side]["recall_session"]["hit"]
            tot[f"{side}_sref"] += r[side]["recall_session"]["ref"]
            tot[f"{side}_pn"] += r[side]["precision"]["n"]
            tot[f"{side}_ps"] += r[side]["precision"]["strict"]
            tot[f"{side}_pl"] += r[side]["precision"]["loose"]
            tot[f"{side}_pairs"] += r[side]["pairs"]
            pce = r[side].get("precision_code_exec")
            if pce:
                tot[f"{side}_xn"] += pce["n"]
                tot[f"{side}_xs"] += pce["strict"]
                tot[f"{side}_xl"] += pce["loose"]
                if side == "content":
                    tot["ref_commits_exec_code"] += pce["ref_commits"]
                    tot["ref_commits_all"] += pce["ref_commits_all"]
        tot["only_baseline"] += r["baseline"]["only_baseline"]
        tot["only_content"] += r["baseline"]["only_content"]
        tot["cp_in_ent"] += r["entire_layer"]["content_pairs_in_entire_session_links"]
    for k in ("pat_one_to_one", "pat_split", "pat_shared", "pat_split+shared", "pat_missing", "pat_insufficient",
              "pat_shared_cross_session", "pat_shared_same_session"):
        tot.setdefault(k, 0)
    return tot


def report(res, tot):
    mean_links = sum(r["entire_layer"]["mean_commits_per_linked_session"] * r["entire_layer"]["sessions_with_links"] for r in res) / \
        max(1, sum(r["entire_layer"]["sessions_with_links"] for r in res))
    L = ["# SWE-chat（Claude Code 子集）上的复制：内容对齐、外部参照与映射模式（S21-T-F4，D27-3）", "",
         "由 `analysis/swechat_replicate.py` 生成。数据：SWE-chat HF 快照 f66cca95 的 sessions / conversations / commits / checkpoints 表 + 各仓库公开历史的 `git clone --mirror`。"
         "方法沿用论文 3.2 节的片段切分、3.4.2 节算法 1 的第 1–3、5、6 步（θ = 0.5、ℓ = 15、m = 80；SWE-chat 无工作区检查点，故无第 4 步）、"
         "3.4.4 节的外部参照（智能体 `git commit` 回显哈希）与按时间归属基线、3.6 节的映射模式。Entire 的检查点→提交链接只作第三层参照。", "",
         "## 1 规模与覆盖", "",
         "| 仓库 | 会话 | 片段 | 代码修改片段 | 可核验 | 编辑调用 | 无写入内容 | 工作区外 | 镜像提交数 | 代码编辑状态 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['repo']} | {r['sessions']} | {r['episodes']} | {r['code_episodes']} | {r['verifiable_code_episodes']} | {r['edit_calls']} | "
                 f"{r['edit_no_text']} | {r['edit_path_unmapped']} | {r['commits_in_mirror']} | {r['code_edit_status']} |")
    L.append(f"| 合计 | {tot['sessions']} | {tot['episodes']} | {tot['code_episodes']} | {tot['verifiable_code_episodes']} | {tot['edit_calls']} | — | — | — | — |")
    L += ["", "## 2 RQ1：外部参照上的召回与参照一致率", "",
          "| 仓库 | 提交命令 / 带哈希 / 哈希在镜像中 | 参照关系（片段层 / 会话层） | 内容对齐召回（片段层 / 会话层） | 内容对齐严格 / 宽松一致率 | 基线召回（片段层 / 会话层） | 基线严格 / 宽松一致率 | 对数：内容 / 基线 / 仅基线 / 仅内容 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res:
        rf, c, b = r["reference"], r["content"], r["baseline"]
        L.append(f"| {r['repo']} | {rf['commit_cmds']} / {rf['with_hash']} / {rf['hash_in_mirror']} | {rf['ref_pairs_code_overlap']} / {c['recall_session']['ref']} | "
                 f"{pct(c['recall']['hit'], c['recall']['ref'])} / {pct(c['recall_session']['hit'], c['recall_session']['ref'])} | "
                 f"{pct(c['precision']['strict'], c['precision']['n'])} / {pct(c['precision']['loose'], c['precision']['n'])} | "
                 f"{pct(b['recall']['hit'], b['recall']['ref'])} / {pct(b['recall_session']['hit'], b['recall_session']['ref'])} | "
                 f"{pct(b['precision']['strict'], b['precision']['n'])} / {pct(b['precision']['loose'], b['precision']['n'])} | "
                 f"{c['pairs']} / {b['pairs']} / {b['only_baseline']} / {b['only_content']} |")
    L.append(f"| 合计 | — | {tot['content_ref']} / {tot['content_sref']} | {pct(tot['content_hit'], tot['content_ref'])} / {pct(tot['content_shit'], tot['content_sref'])} | "
             f"{pct(tot['content_ps'], tot['content_pn'])} / {pct(tot['content_pl'], tot['content_pn'])} | "
             f"{pct(tot['baseline_hit'], tot['baseline_ref'])} / {pct(tot['baseline_shit'], tot['baseline_sref'])} | "
             f"{pct(tot['baseline_ps'], tot['baseline_pn'])} / {pct(tot['baseline_pl'], tot['baseline_pn'])} | "
             f"{tot['content_pairs']} / {tot['baseline_pairs']} / {tot['only_baseline']} / {tot['only_content']} |")
    if tot.get("ref_commits_all"):
        L += ["", f"区分检验（D32 B7）：{tot['ref_commits_all']} 个参照提交中，执行提交的片段本身改过代码的 {tot['ref_commits_exec_code']} 个"
              f"（其余由不改代码的片段执行，多为单独一条消息让智能体提交）。只在这一子集上，内容对齐的严格 / 宽松一致率为 "
              f"{pct(tot['content_xs'], tot['content_xn'])} / {pct(tot['content_xl'], tot['content_xn'])}，"
              f"基线为 {pct(tot['baseline_xs'], tot['baseline_xn'])} / {pct(tot['baseline_xl'], tot['baseline_xn'])}。"
              "各仓库：" + "；".join(f"{r['repo']} 内容 {pct(r['content']['precision_code_exec']['strict'], r['content']['precision_code_exec']['n'])}"
                                   for r in res if r["content"].get("precision_code_exec")) + "。"]
    L += ["", "## 3 RQ2：映射模式（代码修改片段）", "",
          "| 仓库 | 一对一 | 拆分 | 共享（合并） | 拆分且共享 | 缺失（可核验） | 证据不足 | 共享中跨会话 |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        p, n = r["patterns"], r["code_episodes"]
        L.append(f"| {r['repo']} | {pct(p.get('one_to_one', 0), n)} | {pct(p.get('split', 0), n)} | {pct(p.get('shared', 0), n)} | "
                 f"{pct(p.get('split+shared', 0), n)} | {pct(p.get('missing', 0), n)} | {pct(p.get('insufficient', 0), n)} | "
                 f"{p.get('shared_cross_session', 0)} / {p.get('shared_same_session', 0) + p.get('shared_cross_session', 0)} |")
    n = tot["code_episodes"]
    L.append(f"| 合计 | {pct(tot['pat_one_to_one'], n)} | {pct(tot['pat_split'], n)} | {pct(tot['pat_shared'], n)} | {pct(tot['pat_split+shared'], n)} | "
             f"{pct(tot['pat_missing'], n)} | {pct(tot['pat_insufficient'], n)} | {tot['pat_shared_cross_session']} / {tot['pat_shared_cross_session'] + tot['pat_shared_same_session']} |")
    L += ["", "## 4 第三层参照：Entire 的会话→提交链接", "",
          f"- Entire 通过检查点把提交链接到会话（commits 表），为内容级弱匹配（S20 §2 B4），且一个会话常链接到大量提交（有链接的会话平均 {mean_links:.0f} 个提交），"
          f"粒度远粗于片段—提交对，不能作为对的真值。可比的只有一个方向：内容对齐给出的 {tot['content_pairs']} 对中，{pct(tot['cp_in_ent'], tot['content_pairs'])} 的提交在 Entire 对该会话的链接集合内。", "",
          "## 5 说明与局限", "",
          "- 片段切分：每条有时间戳、非接续的开发者消息开一个片段；无时间戳的 `user_prompt` 行是误标的摘要块，不计。子智能体（Task 工具）的编辑在 SWE-chat 转录中已并入父会话。",
          "- 新写入行：Edit / MultiEdit 去掉 `old_string` 中的行，再去掉该文件在编辑前最后一个提交版本中已有的行（算法 1 第 2 步）；Write 只能去掉版本库中的旧行。",
          "- 路径：先按会话确定工作区根（会话中绝对路径在版本库历史中出现过的最长后缀），根下的路径取相对路径，从未入库的记为未跟踪；根外的路径（如 `~/.claude/plans/*.md`）视为工作区外，不计。",
          "- 无检查点差分：SWE-chat 没有 Cursor 意义上的工作区快照，遗漏只能来自导出覆盖与删除类改动；Claude Code 的 Edit / Write 都带完整写入内容，可核验比例远高于本文案例。",
          "- 提交历史含分支与合并：只用非合并提交、全部非 entire/* 分支；同一内容在多分支上出现时取时间最早者。",
          "- 参照：智能体 `git commit` 命令回显的 `[branch hash]`；开发者手工提交无回显，不入参照。片段层参照要求执行提交的片段本身改过被提交的代码文件；"
          "Claude Code 用户常在单独的消息中让智能体提交，因此另报会话层参照。RQ3 的调节变量在此语料中无对应物，不复制。",
          "- 同一仓库两个 repo_id（osabio / brain，wanshicheng / cyyeh）已合并；会话的仓库归属以 conversations 表为准。",
          "- 全部数字由脚本生成，未经人工核对；样本为 5 个仓库、Claude Code 会话，不代表 SWE-chat 全体。", ""]
    return L


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
