"""片段级检查点差分（D3b）。

片段在文件 f 上的新增行 = 下一片段起点检查点中 f 的改动行 − 本片段起点检查点中 f 的改动行
（改动行取 files[*].originalModelDiffWrtV0 与 activeInlineDiffs[*].newTextDiffWrtV0 的 modified，去首尾空白、长度 ≥ SIG_MIN_LEN，按集合计）。
差分行按 D3 规则归属到提交（Attributor，起点取片段开始时间）。任一端没有检查点的片段不做差分。
只保留本片段（含其子代理）编辑或删除过的文件（D3c）：其余文件的差分来自并行会话或本会话更早的改动。

输出 data/derived/<系统>/checkpoint_deltas.csv：片段、路径、路径类别、差分行数、状态、提交、覆盖比例、新写入行数、匹配行数。
环境变量 ATTR_SUFFIX 与 attribute_edits.py 相同，用于敏感性分析。
"""
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).parent))
from attribute_edits import SIG_MIN, Attributor  # noqa: E402
from config import excluded  # noqa: E402
from extract_episodes import path_class  # noqa: E402
from extract_sessions import SYSTEMS as SESS  # noqa: E402

WS_ROOT = {"SYS_A": "d:/sysa/", "SYS_C": "d:/sysb/"}
SUFFIX = os.environ.get("ATTR_SUFFIX", "")
CP_INLINE = os.environ.get("CP_INLINE", "1") == "1"


def load_checkpoints(cdir, root):
    """{checkpointId: {相对路径: 改动行集合}}；改动行取 files 与 activeInlineDiffs（D3e）之并。"""
    out = {}
    p = cdir / "checkpoints.jsonl"
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        o = json.loads(line)
        v = o["_value"]
        files = {}
        entries = [(f.get("uri"), f.get("originalModelDiffWrtV0")) for f in v.get("files") or []]
        if CP_INLINE:
            entries += [(f.get("uri"), f.get("newTextDiffWrtV0")) for f in v.get("activeInlineDiffs") or []]
        for uri, diff in entries:
            fp = unquote((uri or {}).get("path", "")).lstrip("/").replace("\\", "/")
            if not fp.lower().startswith(root):
                continue
            lines = files.setdefault(fp[len(root):], set())
            for chunk in diff or []:
                for x in chunk.get("modified") or []:
                    s = x.strip() if isinstance(x, str) else ""
                    if len(s) >= SIG_MIN:
                        lines.add(s)
        out[o["_key"].split(":")[-1]] = files
    return out


def run(sysname):
    d = Path("data/derived") / sysname
    comp = SESS[sysname]["composer"]
    eps = list(csv.DictReader(open(d / "episodes.csv", encoding="utf-8")))
    ev = {(e["session"], e["seq"]): e for e in csv.DictReader(open(d / "events.csv", encoding="utf-8"))
          if e["tool"] in ("edit_file_v2", "delete_file") and e["path"]}
    touched = {}
    for r in csv.DictReader(open(d / "episode_events.csv", encoding="utf-8")):
        e = ev.get((r["session"], r["seq"]))
        if e:
            touched.setdefault(r["episode"], set()).add(e["path"].replace("\\", "/"))
    att = Attributor(sysname)
    cps_by_sess = {}
    rows, avail = [], Counter()
    for e in eps:
        s = e["session"]
        assert not excluded(s)
        if not (e["checkpoint"] and e["next_checkpoint"]):
            avail["no_checkpoint"] += 1
            continue
        if s not in cps_by_sess:
            cps_by_sess[s] = load_checkpoints(comp / s, WS_ROOT[sysname])
        cps = cps_by_sess[s]
        a, b = cps.get(e["checkpoint"]), cps.get(e["next_checkpoint"])
        if a is None or b is None:
            avail["checkpoint_unresolved"] += 1
            continue
        avail["delta"] += 1
        t0 = datetime.fromisoformat(e["start"])
        for path, lines in sorted(b.items()):
            delta = lines - a.get(path, set())
            if not delta:
                continue
            if path not in touched.get(e["episode"], set()):
                avail["dropped_untouched_file"] += 1
                continue
            state, hit, ratio, n_new, n_hit = att.attribute(path, t0, delta)
            rows.append({"episode": e["episode"], "path": path, "class": path_class(path, "1"), "n_delta": len(delta),
                         "state": state, "commit": hit, "ratio": round(ratio, 3), "n_new": n_new, "n_hit": n_hit})

    fields = ["episode", "path", "class", "n_delta", "state", "commit", "ratio", "n_new", "n_hit"]
    with open(d / f"checkpoint_deltas{SUFFIX}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    code = [r for r in rows if r["class"] == "code"]
    print(f"{sysname}{SUFFIX}: episodes={len(eps)} {dict(avail)}; episode-file deltas={len(rows)} "
          f"(code {len(code)}: {dict(Counter(r['state'] for r in code))})")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        run(s)
