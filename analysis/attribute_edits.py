"""按内容把会话编辑归属到提交，并定位任务书中每个任务在工作区首次写入的编辑。

编辑 e（文件 f，时间下界 t_lo，特征行集合 S）先去掉 f 在 t_lo 之前最后一个提交版本中已有的行，得到新写入行 N。
N 为空的编辑记为 no_new（重写既有内容）。否则归属于：f 的提交中时间 ≥ t_lo 的第一个、其新增行覆盖 N 的
比例 ≥ MATCH 的提交。找不到则按原因记为 after_last_commit（f 此后再无提交）或 not_committed（被覆盖、撤销或未提交）。

任务书编辑：文件在 docs/plans/ 下、特征行中含该任务标题行（如 "### T4 …"、"- [ ] **T4**"）。

输出（data/derived/<系统>/）：
  edit_attribution.csv  每次有内容的编辑：会话、序号、时间、不确定秒数、路径、归属提交、覆盖比例
  plan_task_edits.csv   任务书中任务标题行的每次写入：里程碑、任务、会话、序号、时间
"""
import csv
import json
import os
import re
import subprocess
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import WINDOW_END, excluded  # noqa: E402
from extract_git import MS, PLAN_DIR, SYSTEMS, TASK_LINE, plan_milestone  # noqa: E402
from extract_sessions import SIG_MIN_LEN  # noqa: E402

MATCH = float(os.environ.get("ATTR_MATCH", "0.5"))
OUT_SUFFIX = os.environ.get("ATTR_SUFFIX", "")
SIG_MIN = int(os.environ.get("ATTR_SIG_MIN", SIG_MIN_LEN))
SIG_CAP = int(os.environ.get("ATTR_SIG_CAP", "0"))
# D35-6（B4）去继承口径：一次编辑的特征行若已由更早的、属于其他片段的编辑写过（按时间顺序，不论是否已提交），
# 不计为本次编辑的新写入行。用于检验"整文件重写把别的片段尚未提交的内容算作本片段新写入、从而制造假共享提交"的威胁。
DEINHERIT = os.environ.get("ATTR_DEINHERIT", "0") == "1"


def sig_lines(lines):
    """敏感性口径（D13-3）：提高最短长度或减少每次编辑的条数；默认与抽取时相同。"""
    out = [x for x in lines if TASK_LINE.match(x) or len(x) >= SIG_MIN]
    return out[:SIG_CAP] if SIG_CAP else out


def commit_added_lines(repo):
    out = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotepath=false", "log", "-p", "-U0", "--no-renames",
         "--no-color", "--no-textconv", "--no-ext-diff", "--format=\x1e%H"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    added = defaultdict(set)
    for block in out.split("\x1e")[1:]:
        lines = block.split("\n")
        h = lines[0].strip()
        path = None
        for ln in lines[1:]:
            if ln.startswith("+++ "):
                path = ln[6:] if ln.startswith("+++ b/") else None
            elif path and ln.startswith("+") and not ln.startswith("+++"):
                s = ln[1:].strip()
                if len(s) >= SIG_MIN:
                    added[(h, path.lower())].add(s)
    return added


class Blobs:
    """git cat-file --batch 读取 <commit>:<path> 的内容，返回去空白后的行集合（带缓存）。"""

    def __init__(self, repo):
        self.p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.cache = {}

    def lines(self, commit, path):
        key = (commit, path)
        if key not in self.cache:
            self.p.stdin.write(f"{commit}:{path}\n".encode("utf-8"))
            self.p.stdin.flush()
            header = self.p.stdout.readline().decode("utf-8", "replace").split()
            if len(header) == 3 and header[1] == "blob":
                data = self.p.stdout.read(int(header[2]) + 1)[:-1]
                self.cache[key] = {x.strip() for x in data.decode("utf-8", "replace").splitlines()}
            else:
                self.cache[key] = set()
        return self.cache[key]


class Attributor:
    """D3 规则：attribute(path, t_lo, 特征行集合) → (状态, 提交, 覆盖比例, 新写入行数, 匹配行数)。"""

    def __init__(self, sysname):
        d = Path("data/derived") / sysname
        commits = {c["hash"]: c for c in csv.DictReader(open(d / "commits.csv", encoding="utf-8"))}
        self.by_path = defaultdict(list)
        self.real_path = {}
        for f in csv.DictReader(open(d / "commit_files.csv", encoding="utf-8")):
            self.by_path[f["path"].lower()].append((datetime.fromisoformat(commits[f["hash"]]["time"]), f["hash"]))
            self.real_path[f["path"].lower()] = f["path"]
        for p in self.by_path:
            self.by_path[p].sort()
        self.added = commit_added_lines(SYSTEMS[sysname])
        self.blobs = Blobs(SYSTEMS[sysname])

    def attribute(self, path, t_lo, sig):
        cands = self.by_path.get(path.lower(), [])
        i = bisect_left(cands, (t_lo, ""))
        new = sig - self.blobs.lines(cands[i - 1][1], self.real_path[path.lower()]) if i > 0 else sig
        hit, ratio, state, n_hit = "", 0.0, "", 0
        if not cands:
            state = "untracked"
        elif not new:
            state = "no_new"
        else:
            for ct, h in cands[i:]:
                a = self.added.get((h, path.lower()))
                if not a:
                    continue
                r = len(new & a) / len(new)
                if r >= MATCH:
                    hit, ratio, state, n_hit = h, r, "committed", len(new & a)
                    break
                ratio = max(ratio, r)
            if not hit:
                state = "after_last_commit" if i >= len(cands) else "not_committed"
        return state, hit, ratio, len(new), n_hit

    def first_entry(self, path, t_lo, sig):
        """D29-1 敏感性：新写入行逐行的首次进入提交。返回 {提交: 行数}（不含从未进入任何后续提交的行）。
        与 attribute() 的"取第一个覆盖 ≥ MATCH 的提交"不同，这里不设阈值，用于量化一次编辑的内容分散到多个提交的情形。"""
        cands = self.by_path.get(path.lower(), [])
        i = bisect_left(cands, (t_lo, ""))
        new = sig - self.blobs.lines(cands[i - 1][1], self.real_path[path.lower()]) if i > 0 else sig
        left, out = set(new), defaultdict(int)
        for ct, h in cands[i:]:
            if not left:
                break
            a = self.added.get((h, path.lower()))
            if not a:
                continue
            got = left & a
            if got:
                out[h] = len(got)
                left -= got
        return dict(out)


def run(sysname):
    d = Path("data/derived") / sysname
    att = Attributor(sysname)

    ev = {}
    for e in csv.DictReader(open(d / "events.csv", encoding="utf-8")):
        if e["path"] and e["status"] == "completed" and e["in_workspace"] == "1":
            ev[(e["session"], e["seq"])] = e

    end = WINDOW_END[sysname]
    attr_rows, plan_rows = [], []
    sigs = [json.loads(line) for line in open(d / "edit_signatures.jsonl", encoding="utf-8")]
    ep_of, seen = {}, defaultdict(dict)
    if DEINHERIT:
        ep_of = {(r["session"], r["seq"]): r["episode"] for r in csv.DictReader(open(d / "episode_edits.csv", encoding="utf-8"))
                 if r["episode"] and r["reason"] == ""}
        # 按时间（再按会话、序号）排序，保证"更早"有确定含义
        sigs.sort(key=lambda s: ((ev.get((s["session"], str(s["seq"]))) or {}).get("time") or "", s["session"], int(s["seq"])))
    n_inh_edits = n_inh_lines = n_inh_major = 0
    for s in sigs:
        e = ev.get((s["session"], str(s["seq"])))
        if not e or not e["time"] or excluded(e["session"]):
            continue
        t = datetime.fromisoformat(e["time"])
        t_lo = datetime.fromisoformat(e["t_lo"]) if e["t_lo"] else t
        if t_lo > end:
            continue
        path = e["path"].replace("\\", "/")
        sig = set(sig_lines(s["lines"]))
        if not sig:
            continue
        n_inherited = 0
        if DEINHERIT:
            me = ep_of.get((e["session"], e["seq"]))
            pl = seen[path.lower()]
            inherited = {x for x in sig if x in pl and pl[x] != me}
            for x in sig:
                pl.setdefault(x, me)
            n_inherited = len(inherited)
            if inherited:
                n_inh_edits += 1
                n_inh_lines += n_inherited
                n_inh_major += n_inherited * 2 >= len(sig)
            sig = sig - inherited
            if not sig:
                attr_rows.append({"session": e["session"], "seq": e["seq"], "time": e["time"], "uncert_s": e["uncert_s"],
                                  "path": path, "state": "no_new", "commit": "", "ratio": 0.0, "n_sig": 0, "n_new": 0, "n_hit": 0,
                                  "n_first_commits": 0, "n_entered": 0, "extra_commits": "", "n_inherited": n_inherited})
                continue

        if path.startswith(PLAN_DIR):
            ms = plan_milestone(path)
            for ln in s["lines"]:
                m = TASK_LINE.match(ln)
                if ms and m:
                    plan_rows.append({"milestone": ms, "task": f"T{m.group('n')}", "box": (m.group("box") or "").strip().lower(),
                                      "session": e["session"],
                                      "seq": e["seq"], "time": e["time"], "uncert_s": e["uncert_s"], "path": path})

        state, hit, ratio, n_new, n_hit = att.attribute(path, t_lo, sig)
        fe = att.first_entry(path, t_lo, sig) if state in ("committed", "not_committed", "after_last_commit") else {}
        extra = sorted(((h, n) for h, n in fe.items() if h != hit), key=lambda x: -x[1])
        attr_rows.append({"session": e["session"], "seq": e["seq"], "time": e["time"], "uncert_s": e["uncert_s"],
                          "path": path, "state": state, "commit": hit, "ratio": round(ratio, 3),
                          "n_sig": len(sig), "n_new": n_new, "n_hit": n_hit,
                          # D29-1：逐行首次进入的提交数、进入任一后续提交的行数、主归属之外的提交及其行数
                          "n_first_commits": len(fe), "n_entered": sum(fe.values()),
                          "extra_commits": ";".join(f"{h}:{n}" for h, n in extra),
                          **({"n_inherited": n_inherited} if DEINHERIT else {})})

    if DEINHERIT:
        tot_sig = sum(r["n_sig"] + r.get("n_inherited", 0) for r in attr_rows)
        stats = {"n_edits": len(attr_rows), "n_edits_with_inherited": n_inh_edits, "n_lines_inherited": n_inh_lines,
                 "n_lines_total": tot_sig, "n_edits_inherited_majority": n_inh_major,
                 "n_edits_all_inherited": sum(1 for r in attr_rows if r["n_sig"] == 0 and r.get("n_inherited", 0))}
        (d / f"deinherit_stats{OUT_SUFFIX}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{sysname}{OUT_SUFFIX}: deinherit {stats}")

    outs = [(f"edit_attribution{OUT_SUFFIX}.csv", attr_rows)]
    if not OUT_SUFFIX:
        outs.append(("plan_task_edits.csv", plan_rows))
    for name, rows in outs:
        with open(d / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    from collections import Counter
    code = [r for r in attr_rows if not r["path"].startswith(PLAN_DIR)]
    st = Counter(r["state"] for r in code)
    with_new = sum(v for k, v in st.items() if k in ("committed", "after_last_commit", "not_committed"))
    print(f"{sysname}: code edits={len(code)} states={dict(st)}; committed/with-new-content="
          f"{st['committed'] / with_new:.1%}; plan task-heading writes={len(plan_rows)}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        run(s)
