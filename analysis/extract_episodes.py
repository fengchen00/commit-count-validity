"""按 D4 从会话中切分工作片段，并把编辑、终端命令、提问、子代理归入片段。

输入（data/derived/<系统>/）：sessions.csv、events.csv、user_messages.csv
输出：
  episodes.csv       每个片段一行（请求原文只存本地）
  episode_files.csv  片段 × 文件：编辑次数、首末编辑时间
  episode_edits.csv  每次编辑的归属：片段 ID 或未归入原因（unassigned / before_first / orphan_subagent / no_time）
"""
import csv
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import WINDOW_END, excluded  # noqa: E402

EDIT_TOOLS = {"edit_file_v2", "edit_file", "search_replace", "write", "delete_file"}
COMMIT_CMD = re.compile(r"git\s+commit")
DEPLOY_CMD = re.compile(r"deploy|\bscp\b|\bssh\b|rsync|baota", re.I)
CONT_TOKENS = sorted(["继续执行", "继续", "接着", "goon", "快点", "速度快一些", "速度快点", "hi", "hello", "你好",
                      "你还在吗", "还在吗", "你卡了", "卡了", "你是什么模型", "bro"], key=len, reverse=True)
CROSS = re.compile(r"继续|接着|上次|上一个会话|上个会话|刚才")
PUNCT = re.compile(r"[\s\W_]+", re.U)
TASK_REF = re.compile(r"\bM\d+(?:\.\d+)?[a-z]?(?:-T\d+)?\b|docs[/\\]plans|任务书", re.I)
CUSTOMER_CUE = re.compile(r"客户(?:那边|方面|的意见)?(?:说|反馈|要求|提出|提了|希望|想要|觉得|认为|问)|甲方|用户反馈")
CHANGE_CUE = re.compile(r"改成|改为|换成|去掉|删掉|加上|加一个|调整一下")
START_TOL_S = 600
TIE_S = 1.0
NON_CODE_EXT = (".txt", ".tmp", ".log", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".docx", ".xlsx", ".zip", ".bak")


def is_continuation(text):
    s = PUNCT.sub("", text.lower())
    if not s:
        return False
    while s:
        for t in CONT_TOKENS:
            if s.startswith(t):
                s = s[len(t):]
                break
        else:
            return False
    return True


def path_class(path, in_ws):
    if in_ws != "1":
        return "outside"
    if path.startswith("docs/plans/"):
        return "plan"
    low = path.lower()
    if path.startswith("docs/") or low.endswith((".md", ".mdc")):
        return "doc"
    if low.endswith(NON_CODE_EXT):
        return "other"
    return "code"


def ts(s):
    return datetime.fromisoformat(s) if s else None


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def run(sysname):
    d = Path("data/derived") / sysname
    end = WINDOW_END[sysname]
    sessions = {s["session"]: s for s in load(d / "sessions.csv")}
    is_sub = {k for k, v in sessions.items() if v["is_subagent"] == "1"}
    events = load(d / "events.csv")

    parent = {}
    for e in events:
        if e["tool"] == "task_v2" and e["links"]:
            for u in e["links"].split(";"):
                if u in is_sub and u != e["session"]:
                    parent.setdefault(u, e["session"])

    def root(s, depth=0):
        while s in is_sub and depth < 10:
            s, depth = parent.get(s), depth + 1
            if s is None:
                return None
        return s

    in_scope = lambda s: s in sessions and not excluded(s)
    msgs = []
    for m in load(d / "user_messages.csv"):
        if not (in_scope(m["session"]) and m["session"] not in is_sub and m["time"]):
            continue
        lo = ts(m["t_lo"]) or ts(m["time"])
        if lo > end:
            continue
        m["start_uncert_s"] = ((ts(m["t_hi"]) or lo) - lo).total_seconds()
        m["time"] = lo.isoformat()
        msgs.append(m)
    no_time_msgs = [m for m in load(d / "user_messages.csv")
                    if in_scope(m["session"]) and m["session"] not in is_sub and not m["time"]]
    by_sess = defaultdict(list)
    for m in msgs:
        by_sess[m["session"]].append(m)

    episodes = []
    starts = {}
    msg_owner = {}
    n_tied = 0
    for s, ms in by_sess.items():
        ordered = sorted((m for m in ms if m["ordered"] == "1"), key=lambda m: int(m["seq"]))
        for m in sorted((m for m in ms if m["ordered"] != "1"), key=lambda m: ts(m["time"])):
            i = next((k for k, o in enumerate(ordered) if ts(o["time"]) > ts(m["time"])), len(ordered))
            ordered.insert(i, m)
        ms = ordered
        stamps = [ts(m["time"]) for m in ms]
        for i, m in enumerate(ms):
            m["tied"] = any(j != i and abs((stamps[j] - stamps[i]).total_seconds()) < TIE_S for j in range(len(ms)))
        n_tied += sum(m["tied"] for m in ms)
        s_start = ts(sessions[s]["first_event"]) or stamps[0]
        eps = []
        for k, m in enumerate(ms):
            cont = is_continuation(m["text"])
            if cont and eps:
                eps[-1]["merged"].append(m)
            else:
                start = min(s_start, ts(m["time"])) if not eps else ts(m["time"])
                if eps and start < eps[-1]["start"]:
                    start = eps[-1]["start"]
                eps.append({"session": s, "start": start, "msg": m, "merged": [], "group": False,
                            "continuation_start": int(cont and not eps)})
            msg_owner[(s, m["seq"])] = len(eps) - 1
        for i, ep in enumerate(eps):
            ep["first_in_session"] = int(i == 0)
            ep["next_checkpoint"] = eps[i + 1]["msg"].get("checkpoint", "") if i + 1 < len(eps) else ""
        starts[s] = [ep["start"] for ep in eps]
        episodes.append((s, eps))

    ep_list = []
    for s, eps in sorted(episodes, key=lambda x: x[1][0]["start"]):
        for ep in eps:
            ep_list.append(ep)
    ep_list.sort(key=lambda e: (e["start"], e["session"]))
    for n, ep in enumerate(ep_list, 1):
        ep["id"] = f"{sysname}-E{n:04d}"
        ep.update({"n_edits": Counter(), "n_failed_edits": 0, "n_terminal": 0, "n_commit_cmds": 0,
                   "n_deploy_cmds": 0, "n_ask": 0, "subagents": set(), "last": ep["start"], "files": defaultdict(list)})
    ep_index = {(ep["session"], ep["start"], ep["msg"]["seq"]): ep for ep in ep_list}
    by_sess_eps = defaultdict(list)
    for ep in ep_list:
        by_sess_eps[ep["session"]].append(ep)
    for s in by_sess_eps:
        by_sess_eps[s].sort(key=lambda e: (e["start"], int(e["msg"]["seq"])))
        starts[s] = [e["start"] for e in by_sess_eps[s]]

    def locate(s, t):
        i = bisect_right(starts[s], t) - 1
        return i

    edit_rows, ev_rows = [], []
    reasons = Counter()
    for e in events:
        s = e["session"]
        if not in_scope(s):
            continue
        is_edit = e["tool"] in EDIT_TOOLS
        relevant = is_edit or e["tool"].startswith("run_terminal") or e["tool"] == "ask_question"
        if not relevant:
            continue
        host = root(s) if s in is_sub else s
        if not e["time"]:
            if is_edit:
                reasons["no_time"] += 1
                edit_rows.append({"session": s, "seq": e["seq"], "episode": "", "reason": "no_time"})
            continue
        lo, hi = ts(e["t_lo"]) or ts(e["time"]), ts(e["t_hi"]) or ts(e["time"])
        if lo > end:
            continue
        if host is None or host not in starts or not in_scope(host):
            if is_edit:
                reasons["orphan_subagent" if s in is_sub else "no_episode_in_session"] += 1
                edit_rows.append({"session": s, "seq": e["seq"], "episode": "", "reason": "orphan_subagent" if s in is_sub else "no_episode_in_session"})
            continue
        i_lo, i_hi = locate(host, lo), locate(host, min(hi, end))
        if i_hi < 0:
            reason = "before_first"
        elif i_lo != i_hi:
            reason = "unassigned"
        else:
            reason = ""
        if reason:
            if is_edit:
                reasons[reason] += 1
                edit_rows.append({"session": s, "seq": e["seq"], "episode": "", "reason": reason})
            continue
        ep = by_sess_eps[host][i_lo]
        t = ts(e["time"])
        ep["last"] = max(ep["last"], t)
        ev_rows.append({"session": s, "seq": e["seq"], "episode": ep["id"]})
        if s != host:
            ep["subagents"].add(s)
        if is_edit:
            if e["status"] != "completed":
                ep["n_failed_edits"] += 1
                reasons["failed_edit"] += 1
                edit_rows.append({"session": s, "seq": e["seq"], "episode": ep["id"], "reason": "failed"})
                continue
            pc = path_class(e["path"].replace("\\", "/"), e["in_workspace"])
            ep["n_edits"][pc] += 1
            ep["files"][(e["path"].replace("\\", "/"), pc)].append(t)
            reasons["assigned"] += 1
            edit_rows.append({"session": s, "seq": e["seq"], "episode": ep["id"], "reason": ""})
        elif e["tool"].startswith("run_terminal"):
            ep["n_terminal"] += 1
            ep["n_commit_cmds"] += int(bool(COMMIT_CMD.search(e["command"])))
            ep["n_deploy_cmds"] += int(bool(DEPLOY_CMD.search(e["command"])))
        else:
            ep["n_ask"] += 1

    rows, frows = [], []
    for ep in ep_list:
        nxt = [x for x in by_sess_eps[ep["session"]] if x["start"] > ep["start"]]
        text = ep["msg"]["text"]
        rows.append({
            "episode": ep["id"], "system": sysname, "session": ep["session"],
            "start": ep["start"].isoformat(), "last_event": ep["last"].isoformat(),
            "next_start": nxt[0]["start"].isoformat() if nxt else "",
            "msg_seq": ep["msg"]["seq"], "checkpoint": ep["msg"].get("checkpoint", ""),
            "next_checkpoint": ep["next_checkpoint"],
            "request_len": len(text), "n_merged_msgs": len(ep["merged"]),
            "n_edits_code": ep["n_edits"]["code"], "n_edits_plan": ep["n_edits"]["plan"],
            "n_edits_doc": ep["n_edits"]["doc"], "n_edits_outside": ep["n_edits"]["outside"],
            "n_edits_other": ep["n_edits"]["other"], "n_failed_edits": ep["n_failed_edits"], "n_terminal": ep["n_terminal"],
            "n_commit_cmds": ep["n_commit_cmds"], "n_deploy_cmds": ep["n_deploy_cmds"], "n_ask": ep["n_ask"],
            "n_subagents": len(ep["subagents"]), "first_in_session": ep["first_in_session"],
            "start_uncert_s": ep["msg"]["start_uncert_s"],
            "tied_start": int(ep["msg"]["tied"]),
            "start_time_uncertain": int(ep["msg"]["start_uncert_s"] > START_TOL_S),
            "continuation_start": ep["continuation_start"],
            "cross_session_continuation": int(ep["first_in_session"] == 1 and bool(CROSS.search(text[:80]))),
            "f_task_ref": int(bool(TASK_REF.search(text))),
            "f_customer_cue": int(bool(CUSTOMER_CUE.search(text))),
            "f_change_cue": int(bool(CHANGE_CUE.search(text))),
            "f_writes_plan": int(ep["n_edits"]["plan"] > 0),
            "f_edits_code": int(ep["n_edits"]["code"] > 0),
            "f_no_edits": int(sum(ep["n_edits"].values()) == 0),
            "f_commit_cmd": int(ep["n_commit_cmds"] > 0),
            "f_deploy_cmd": int(ep["n_deploy_cmds"] > 0),
            "request_text": text,
        })
        for (p, pc), tl in ep["files"].items():
            frows.append({"episode": ep["id"], "path": p, "class": pc, "n_edits": len(tl),
                          "first": min(tl).isoformat(), "last": max(tl).isoformat()})

    n_in = len(msgs)
    owned = sum(1 + len(ep["merged"]) for ep in ep_list)
    assert owned == n_in, (owned, n_in)
    assert len(msg_owner) == n_in
    assert all(not excluded(r["session"]) for r in rows)
    assert all(ts(r["start"]) <= end for r in rows)
    in_scope_edits = sum(1 for r in edit_rows)
    assert in_scope_edits == sum(reasons.values())
    assert sum(r["n_edits_code"] + r["n_edits_plan"] + r["n_edits_doc"] + r["n_edits_outside"] + r["n_edits_other"] for r in rows) == reasons["assigned"]

    for name, data in [("episodes.csv", rows), ("episode_files.csv", frows), ("episode_edits.csv", edit_rows),
                       ("episode_events.csv", ev_rows)]:
        with open(d / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    linked = sum(1 for s in is_sub if in_scope(s) and root(s))
    print(f"{sysname}: dev messages in window={n_in} (no-time, dropped={len(no_time_msgs)}) episodes={len(rows)} "
          f"merged continuation msgs={sum(r['n_merged_msgs'] for r in rows)}")
    print(f"  subagents linked={linked}/{sum(1 for s in is_sub if in_scope(s))}; edits: {dict(reasons)}")
    print(f"  tied-start msgs={n_tied}; start_time_uncertain episodes={sum(r['start_time_uncertain'] for r in rows)}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        run(s)
