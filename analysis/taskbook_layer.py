"""任务书层（S02-T3）：片段与登记任务的对应。

对应规则（按优先级，取第一条成立的；都不成立记 none，不强行对应）：
  request_task       请求中写明"里程碑+任务号"（M40-T3、M40 T3、M40的T1~T3、M40-T1/T5），且该任务在任务书中登记过
  request_milestone  请求中只写了里程碑号（如"继续执行M40"），且该里程碑有任务书
  commit_task        请求中没有里程碑号，但片段按内容进入的提交是登记任务的提交（提交说明带任务号）
  session_milestone  以上都不成立，但同会话此前的请求写过某个有任务书的里程碑号（弱对应，只对应到里程碑）
对应到的任务集合：request_task 取请求中的任务；request_milestone 取片段进入的该里程碑的任务提交所带任务
（没有则只对应到里程碑、无任务级时间）；commit_task 取提交所带任务。

时间：
  plan_written_before  对应任务在工作区首次写入任务标题行（plan_task_edits）早于片段起点：全部 1 / 部分 partial / 都不 0
  plan_committed_before  对应任务首次进入版本库（taskbook_tasks.first_time）早于片段起点：同上
  ticked_git_by_end    对应任务在版本库中打勾（tick_time）不晚于片段最后事件：同上
  ticked_ws_by_end     对应任务在工作区写入"[x]"不晚于片段最后事件：同上
工作区时间只来自带内容的编辑；无内容的任务书编辑不可见，故工作区时间只可能偏晚、不会偏早。

输出 data/derived/<系统>/episode_taskbook.csv。
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_git import MS, parse_tasks  # noqa: E402

REF = re.compile(rf"(?<![A-Za-z0-9])({MS})(?![0-9])(?:\s*(?:-|的|\s)\s*(T\d+(?:\s*[~～/、,，和]\s*T?\d+)*))?")


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def ts(s):
    return datetime.fromisoformat(s) if s else None


def share(flags):
    if not flags:
        return ""
    return "1" if all(flags) else ("0" if not any(flags) else "partial")


def run(sysname):
    d = Path("data/derived") / sysname
    eps = load(d / "episodes.csv")
    tb = {(r["milestone"], r["task"]): r for r in load(d / "taskbook_tasks.csv")}
    tb_ms = {m for m, _ in tb}
    cm = {c["hash"]: c for c in load(d / "commits.csv")}
    P = defaultdict(set)
    for p in load(d / "episode_commit.csv"):
        if p["source"] != "pathtime":
            P[p["episode"]].add(p["commit"])
    ev = {r["episode"]: r["evidence"] for r in load(d / "episode_patterns.csv")}
    ws_first, ws_tick = {}, defaultdict(list)
    for r in load(d / "plan_task_edits.csv"):
        k, t = (r["milestone"], r["task"]), ts(r["time"])
        ws_first[k] = min(ws_first.get(k, t), t)
        if r["box"] == "x":
            ws_tick[k].append(t)

    rows = []
    sess_ms = defaultdict(set)
    for e in sorted(eps, key=lambda e: (e["session"], e["start"], int(e["msg_seq"]))):
        text = e["request_text"]
        prior_ms = set(sess_ms[e["session"]])
        req_tasks, req_ms = set(), set()
        for m, tl in REF.findall(text):
            if m in tb_ms:
                req_ms.add(m)
            for n in parse_tasks(tl):
                if (m, f"T{n}") in tb:
                    req_tasks.add((m, f"T{n}"))
        commit_tasks = set()
        for h in P[e["episode"]]:
            c = cm[h]
            if c["label"] == "task":
                commit_tasks |= {(c["milestone"], f"T{n}") for n in parse_tasks(c["tasks"])} & tb.keys()
        if req_tasks:
            link, tasks = "request_task", req_tasks
        elif req_ms:
            link, tasks = "request_milestone", {k for k in commit_tasks if k[0] in req_ms}
        elif commit_tasks:
            link, tasks = "commit_task", commit_tasks
        elif prior_ms:
            link, tasks = "session_milestone", set()
        else:
            link, tasks = "none", set()
        sess_ms[e["session"]] |= req_ms
        start, end = ts(e["start"]), ts(e["last_event"])
        rows.append({
            "episode": e["episode"], "has_code": int(int(e["n_edits_code"]) > 0), "evidence": ev.get(e["episode"], ""),
            "link": link, "ex_ante": int(bool(req_tasks or req_ms or prior_ms)), "milestones": ";".join(sorted(req_ms or {m for m, _ in tasks} or (prior_ms if link == "session_milestone" else set()))),
            "n_tasks": len(tasks), "tasks": ";".join(f"{m}-{t}" for m, t in sorted(tasks)),
            "plan_written_before": share([k in ws_first and ws_first[k] < start for k in tasks]),
            "plan_committed_before": share([bool(tb[k]["first_time"]) and ts(tb[k]["first_time"]) < start for k in tasks]),
            "ticked_git_by_end": share([bool(tb[k]["tick_time"]) and ts(tb[k]["tick_time"]) <= end for k in tasks]),
            "ticked_ws_by_end": share([any(t <= end for t in ws_tick[k]) for k in tasks]),
        })

    order = {e["episode"]: i for i, e in enumerate(eps)}
    rows.sort(key=lambda r: order[r["episode"]])
    assert len(rows) == len(eps)
    with open(d / "episode_taskbook.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    code = [r for r in rows if r["has_code"]]
    ver = [r for r in code if r["evidence"] == "verified"]
    print(f"{sysname}: code episodes={len(code)} link={dict(Counter(r['link'] for r in code))}; "
          f"unregistered={sum(r['link'] == 'none' for r in code)} ({sum(r['link'] == 'none' for r in code) / len(code):.0%}); "
          f"verified unregistered={sum(r['link'] == 'none' for r in ver)}/{len(ver)}")
    for k in ["plan_written_before", "plan_committed_before", "ticked_git_by_end", "ticked_ws_by_end"]:
        print(f"  {k}: {dict(Counter(r[k] or '-' for r in code if r['n_tasks']))}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        run(s)
