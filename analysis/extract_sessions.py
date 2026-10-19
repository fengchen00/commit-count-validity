"""从 Composer 会话导出中提取事件表与会话表。

导出中同一次工具调用常有多份气泡拷贝（共用 toolCallId）：一份在对话顺序（fullConversationHeadersOnly）中，
其余不在顺序中但带真实时间。本脚本按 (会话, toolCallId) 去重，合并为一条事件：
位置取顺序中的拷贝，时间取各拷贝中最早的有效时间，内容取信息最全的拷贝。

补写时间：Cursor 升级时会给旧消息批量写入同一时刻的 createdAt。同一会话中落在同一秒的气泡
≥ BACKFILL_MIN 条时，该秒内的时间戳一律视为无效。

输出（data/derived/<系统>/）：
  sessions.csv       每个会话一行
  events.csv         每条事件一行（工具调用已去重）：会话、序号、时间（UTC+8）、时间来源、上下界、类型、工具、状态、路径、命令、关联 ID、是否在对话顺序中
  user_messages.csv  开发者消息原文（仅本地）
  edit_signatures.jsonl  编辑写入内容的特征行

时间来源：bubble 为有效 createdAt；interp 为在对话顺序中取前后有效时间的中点（给出上下界）；
session_bounds 为不在顺序中且无时间，取会话起止。
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_git import TASK_LINE  # noqa: E402

TZ = timezone(timedelta(hours=8))
SYSTEMS = {
    "SYS_A": {
        "composer": Path("data/raw/SYS_A_main/sessions/by-workspace/ws-0a13b296-d-sysa/raw/composer"),
        "root": "d:/sysa/",
    },
    "SYS_C": {
        "composer": Path("data/raw/SYS_A_main/sessions/by-workspace/ws-2dfaaae5-d-sysb/raw/composer"),
        "root": "d:/sysb/",
    },
}
RESEARCH = re.compile(r"论文|研究|导出|脱敏|FSE|审稿|证据包|标注|参与者|gate\s*\d|research|export", re.I)
EDIT_TOOLS = {"edit_file_v2", "edit_file", "search_replace", "write", "delete_file"}
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SIG_MIN_LEN = 15
SIG_MAX = 80
BACKFILL_MIN = 50


def parse_params(t):
    p = t.get("params")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except json.JSONDecodeError:
            p = {"_raw": p}
    return p or {}


def signature_lines(tool):
    """编辑写入内容中有辨识度的行（去首尾空白后长度 ≥ SIG_MIN_LEN；任务标题行不受长度与条数限制）。"""
    params = parse_params(tool)
    text = params.get("streamingContent") or params.get("contents") or params.get("new_string") or ""
    lines = text.splitlines()
    ad = tool.get("additionalData") or {}
    if isinstance(ad, str):
        try:
            ad = json.loads(ad)
        except json.JSONDecodeError:
            ad = {}
    pd = ad.get("precomputedDiff") if isinstance(ad, dict) else None
    if pd:
        lines += [x.get("content", "") for x in pd.get("lines", []) if x.get("type") == "added"]
    out, seen = [], set()
    for x in lines:
        s = x.strip()
        if s in seen:
            continue
        if TASK_LINE.match(s) or (len(s) >= SIG_MIN_LEN and len(out) < SIG_MAX):
            seen.add(s)
            out.append(s)
    return out


def iso(ms_or_str):
    if ms_or_str is None or ms_or_str == "":
        return None
    if isinstance(ms_or_str, (int, float)):
        return datetime.fromtimestamp(ms_or_str / 1000, TZ)
    return datetime.fromisoformat(ms_or_str.replace("Z", "+00:00")).astimezone(TZ)


def norm_path(p, root):
    if not p:
        return "", False
    q = p.replace("\\", "/")
    if q.lower().startswith(root):
        return q[len(root):], True
    if re.match(r"^[a-zA-Z]:/", q):
        return q, False
    return q.lstrip("./"), True


def richness(v):
    tf = v.get("toolFormerData") or {}
    return (tf.get("status") == "completed", len(str(tf.get("params") or "")) + len(str(tf.get("additionalData") or "")))


def load_session(cdir):
    head = json.loads((cdir / "header.json").read_text(encoding="utf-8")) if (cdir / "header.json").exists() else {}
    cd = json.loads((cdir / "composerData.json").read_text(encoding="utf-8")) if (cdir / "composerData.json").exists() else {}
    cd = cd.get("_value", cd)
    bubbles = []
    if (cdir / "bubbles.jsonl").exists():
        for line in open(cdir / "bubbles.jsonl", encoding="utf-8"):
            bubbles.append(json.loads(line).get("_value", {}))
    return head, cd, bubbles


def build_sequence(cd, bubbles):
    """去重并排序，返回 [(代表气泡, 是否在对话顺序中, 有效时间, 拷贝数)]。"""
    order = {h["bubbleId"]: i for i, h in enumerate(cd.get("fullConversationHeadersOnly") or [])}
    secs = Counter((v.get("createdAt") or "")[:19] for v in bubbles if v.get("createdAt"))
    bad = {s for s, n in secs.items() if n >= BACKFILL_MIN}

    def valid_time(v):
        c = v.get("createdAt")
        return None if not c or c[:19] in bad else iso(c)

    groups = defaultdict(list)
    singles = []
    for v in bubbles:
        tid = (v.get("toolFormerData") or {}).get("toolCallId")
        if tid:
            groups[tid].append(v)
        else:
            singles.append([v])
    items = []
    for copies in list(groups.values()) + singles:
        pos = min((order[c["bubbleId"]] for c in copies if c.get("bubbleId") in order), default=None)
        times = [t for t in (valid_time(c) for c in copies) if t]
        rep = max(copies, key=richness)
        items.append({"v": rep, "pos": pos, "time": min(times) if times else None, "copies": len(copies)})
    ordered = sorted((x for x in items if x["pos"] is not None), key=lambda x: x["pos"])
    loose = sorted((x for x in items if x["pos"] is None), key=lambda x: (x["time"] is None, x["time"] or datetime.min.replace(tzinfo=TZ)))
    return ordered, loose, len(bad)


def extract(sysname, cfg, out_dir):
    sessions, events, user_rows = [], [], []
    stats = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_f = open(out_dir / "edit_signatures.jsonl", "w", encoding="utf-8")
    for cdir in sorted(cfg["composer"].iterdir()):
        cid = cdir.name
        head, cd, bubbles = load_session(cdir)
        ordered, loose, n_bad = build_sequence(cd, bubbles)
        stats["backfilled_seconds"] += n_bad
        stats["raw_bubbles"] += len(bubbles)

        c0 = iso(head.get("createdAt") or cd.get("createdAt"))
        c1 = iso(head.get("lastUpdatedAt") or cd.get("lastUpdatedAt"))
        valid = [x["time"] for x in ordered + loose if x["time"]]
        s_lo = min([t for t in [c0] + valid if t], default=None)
        s_hi = max([t for t in valid + [c1] if t], default=None)

        known = [(i, x["time"]) for i, x in enumerate(ordered) if x["time"]]
        for i, x in enumerate(ordered):
            if x["time"]:
                x["lo"] = x["hi"] = x["time"]
                x["src"] = "bubble"
                continue
            x["lo"] = max((t for j, t in known if j < i), default=s_lo)
            x["hi"] = min((t for j, t in known if j > i), default=s_hi)
            if x["lo"] and x["hi"] and x["hi"] < x["lo"]:
                x["hi"] = x["lo"]
            x["src"] = "interp"
        for x in loose:
            if x["time"]:
                x["lo"] = x["hi"] = x["time"]
                x["src"] = "bubble"
            else:
                x["lo"], x["hi"], x["src"] = s_lo, s_hi, "session_bounds"

        rows, first_user = [], ""
        for seq, (x, is_ord) in enumerate([(x, 1) for x in ordered] + [(x, 0) for x in loose]):
            v = x["v"]
            tool = v.get("toolFormerData") or {}
            name = tool.get("name") or ""
            params = parse_params(tool) if tool else {}
            path, in_ws = norm_path(params.get("relativeWorkspacePath") or params.get("targetFile") or params.get("path") or "", cfg["root"])
            cmd = (params.get("command") or "") if name.startswith("run_terminal") else ""
            kind = "user" if v.get("type") == 1 else ("tool" if name else "assistant")
            t = x["time"] or (x["lo"] + (x["hi"] - x["lo"]) / 2 if x["lo"] and x["hi"] else None)
            if name in EDIT_TOOLS and tool.get("status") == "completed":
                sig = signature_lines(tool)
                if sig:
                    sig_f.write(json.dumps({"session": cid, "seq": seq, "lines": sig}, ensure_ascii=False) + "\n")
            links = ";".join(sorted(set(UUID.findall(json.dumps(tool, ensure_ascii=False))) - {cid})) if name == "task_v2" else ""
            row = {
                "session": cid, "seq": seq, "time": t, "time_src": x["src"] if t else "",
                "t_lo": x["lo"], "t_hi": x["hi"],
                "uncert_s": round((x["hi"] - x["lo"]).total_seconds(), 1) if x["lo"] and x["hi"] else "",
                "kind": kind, "tool": name, "status": tool.get("status") or "",
                "path": path if name in EDIT_TOOLS or name.startswith("read_file") else "",
                "in_workspace": int(in_ws) if path else "",
                "command": cmd.replace("\n", " ")[:500], "links": links,
                "ordered": is_ord, "copies": x["copies"],
            }
            rows.append(row)
            if kind == "user":
                text = v.get("text") or ""
                if not first_user:
                    first_user = re.sub(r"\s+", " ", text)[:120]
                user_rows.append({"session": cid, "seq": seq, "time": t, "t_lo": x["lo"], "t_hi": x["hi"],
                                  "ordered": is_ord, "checkpoint": v.get("checkpointId") or "", "text": text})
        events.extend(rows)

        times = [r["time"] for r in rows if r["time"]]
        sessions.append({
            "session": cid, "name": cd.get("name") or "",
            "created": c0, "first_event": min(times) if times else None, "last_event": max(times) if times else None,
            "is_subagent": head.get("isSubagent", ""), "n_events": len(rows),
            "n_edits": sum(1 for r in rows if r["tool"] in EDIT_TOOLS), "backfilled_seconds": n_bad,
            "first_user": first_user,
            "research_flag": int(bool(RESEARCH.search(first_user + " " + (cd.get("name") or "")))),
        })

    sig_f.close()
    fmt = lambda x: x.isoformat(timespec="milliseconds") if isinstance(x, datetime) else x
    for fname, data in [("sessions.csv", sessions), ("events.csv", events), ("user_messages.csv", user_rows)]:
        with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            for r in data:
                w.writerow({k: fmt(v) for k, v in r.items()})

    edits = [r for r in events if r["tool"] in EDIT_TOOLS]
    src = Counter(r["time_src"] or "none" for r in edits)
    u = sorted(r["uncert_s"] for r in edits if r["uncert_s"] != "")
    q = lambda p: u[min(len(u) - 1, int(p * len(u)))]
    print(f"{sysname}: sessions={len(sessions)} raw bubbles={stats['raw_bubbles']} events(dedup)={len(events)} "
          f"sessions with backfilled stamps={sum(1 for s in sessions if s['backfilled_seconds'])}")
    print(f"  edits={len(edits)} completed={sum(1 for r in edits if r['status'] == 'completed')} time_src={dict(src)} "
          f"in_workspace={sum(1 for r in edits if r['in_workspace'] == 1)}")
    print(f"  edit time uncertainty (s): p50={q(0.5)} p90={q(0.9)} p99={q(0.99)} >10min={sum(1 for x in u if x > 600)} >1h={sum(1 for x in u if x > 3600)}")
    print(f"  user messages={len(user_rows)} research-flagged sessions={sum(s['research_flag'] for s in sessions)}")


if __name__ == "__main__":
    for s in sys.argv[1:] or SYSTEMS:
        extract(s, SYSTEMS[s], Path("data/derived") / s)
