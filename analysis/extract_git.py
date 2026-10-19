"""从冻结快照的 Git 仓库提取提交、提交文件与任务书中的任务。

输出（data/derived/<系统>/）：
  commits.csv        提交：hash、提交时间（committer，UTC+8）、标题、里程碑、任务列表、类别
  commit_files.csv   每个提交改动的文件
  taskbook_tasks.csv 任务书中每个任务：首次出现的提交与时间、首次打勾的提交与时间
类别：plan（只改 docs/plans）、impl（不改 docs/plans）、mixed（两者都改）、other（无文件）。
"""
import csv
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SNAP = "original-snapshot-EXAMPLE"
SYSTEMS = {
    "SYS_A": Path(f"data/raw/SYS_A_main/{SNAP}/git/clone"),
    "SYS_C": Path(f"data/raw/SYS_B_raw/{SNAP}/git/clone"),
}
PLAN_DIR = "docs/plans/"
MS = r"(?:M\d+(?:\.\d+)?[a-z]?|TECH\d+)"
SUBJ = re.compile(rf"^(?P<ms>{MS})(?:-(?P<rest>[^:：\s]+))?\s*[:：]")
HOTFIX = re.compile(r"^(?:热修|hf|hotfix)\s*\d", re.I)
TASK_LINE = re.compile(r"^\s*(?:#{2,4}\s*|[-*]\s*\[(?P<box>[ xX])\]\s*|\|\s*)\**\s*T(?P<n>\d+)\b")


def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=false", *args], capture_output=True, check=True)
    return r.stdout.decode("utf-8", "replace")


def parse_tasks(rest):
    if not rest:
        return []
    out = []
    for a, b in re.findall(r"T(\d+)\s*[~～]\s*T?(\d+)", rest):
        out.extend(range(int(a), int(b) + 1))
    rest2 = re.sub(r"T\d+\s*[~～]\s*T?\d+", "", rest)
    out.extend(int(x) for x in re.findall(r"T(\d+)", rest2))
    return sorted(set(out))


def classify_subject(s):
    m = SUBJ.match(s)
    if m:
        ms, rest = m.group("ms"), m.group("rest") or ""
        tasks = parse_tasks(rest)
        if tasks:
            return ms, tasks, "task"
        if re.match(r"fix", rest, re.I):
            return ms, [], "fix"
        return ms, [], "milestone"
    if HOTFIX.match(s):
        return "", [], "hotfix"
    return "", [], "unlabeled"


def plan_milestone(path):
    m = re.match(rf"{PLAN_DIR}({MS})[-_. ]", path)
    return m.group(1) if m else ""


def extract(sysname, repo, out):
    out.mkdir(parents=True, exist_ok=True)
    log = git(repo, "log", "--reverse", "--format=\x1e%H\x1f%cI\x1f%aI\x1f%s", "--name-status", "--no-renames")
    commits, files = [], []
    for block in log.split("\x1e")[1:]:
        head, *rest = block.strip("\n").split("\n")
        h, ct, at, subj = head.split("\x1f")
        fl = []
        for line in rest:
            if line.strip():
                st, p = line.split("\t", 1)
                fl.append((st, p))
                files.append({"hash": h, "status": st, "path": p})
        plan = [p for _, p in fl if p.startswith(PLAN_DIR)]
        code = [p for _, p in fl if not p.startswith(PLAN_DIR)]
        kind = "mixed" if plan and code else "plan" if plan else "impl" if code else "other"
        ms, tasks, label = classify_subject(subj)
        commits.append({
            "hash": h, "time": datetime.fromisoformat(ct).isoformat(), "author_time": datetime.fromisoformat(at).isoformat(),
            "subject": subj, "milestone": ms, "tasks": ";".join(f"T{t}" for t in tasks), "label": label,
            "kind": kind, "n_plan_files": len(plan), "n_code_files": len(code),
        })

    tb = {}
    for c in commits:
        touched = sorted({f["path"] for f in files if f["hash"] == c["hash"] and f["path"].startswith(PLAN_DIR) and f["path"].endswith(".md") and f["status"] != "D"})
        for p in touched:
            ms = plan_milestone(p)
            if not ms:
                continue
            try:
                text = git(repo, "show", f"{c['hash']}:{p}")
            except subprocess.CalledProcessError:
                continue
            for line in text.splitlines():
                m = TASK_LINE.match(line)
                if not m:
                    continue
                key = (ms, f"T{m.group('n')}")
                rec = tb.setdefault(key, {"milestone": ms, "task": key[1], "file": p,
                                          "first_commit": c["hash"], "first_time": c["time"],
                                          "tick_commit": "", "tick_time": ""})
                if m.group("box") in ("x", "X") and not rec["tick_commit"]:
                    rec["tick_commit"], rec["tick_time"] = c["hash"], c["time"]

    def dump(name, rows):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("commits.csv", commits)
    dump("commit_files.csv", files)
    dump("taskbook_tasks.csv", sorted(tb.values(), key=lambda r: (r["first_time"], r["milestone"], int(r["task"][1:]))))

    lab = defaultdict(int)
    kinds = defaultdict(int)
    for c in commits:
        lab[c["label"]] += 1
        kinds[c["kind"]] += 1
    ticked = sum(1 for r in tb.values() if r["tick_commit"])
    print(f"{sysname}: commits={len(commits)} labels={dict(lab)} kinds={dict(kinds)}")
    print(f"  taskbook tasks={len(tb)} ticked={ticked} milestones={len({k[0] for k in tb})}")


if __name__ == "__main__":
    for s in sys.argv[1:] or SYSTEMS:
        extract(s, SYSTEMS[s], Path("data/derived") / s)
