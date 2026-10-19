"""片段 → 提交映射（S02-T1）。

对片段内每次成功的代码编辑（path_class == "code"，子代理编辑已并入父片段），取 D3 内容归属的状态：
  committed          进入某个提交（记该提交）
  not_committed      新内容此后未进入该文件的任何提交（被覆盖、撤销或丢弃）
  after_last_commit  该文件此后在窗口内再无提交
  untracked          文件从未进入版本库
  no_new             写入内容在上一版本中已存在（重写既有内容），不计入映射
  no_content         导出中没有写入内容，无法按内容归属

输出（data/derived/<系统>/）：
  episode_commit.csv       片段 × 提交：贡献编辑数、贡献新增行数（匹配行数之和）
  episode_code_states.csv  每个片段的代码编辑按上述状态计数
环境变量 ATTR_SUFFIX 指定读取的归属文件后缀（敏感性分析用），输出文件带同样后缀。
"""
import csv
import json
import os
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_episodes import path_class  # noqa: E402

STATES = ["committed", "not_committed", "after_last_commit", "untracked", "no_new", "no_content"]
SUFFIX = os.environ.get("ATTR_SUFFIX", "")
SIG_VARIANT = bool(os.environ.get("ATTR_SIG_MIN") or os.environ.get("ATTR_SIG_CAP"))
MULTI = os.environ.get("ATTR_MULTI", "0") == "1"           # D29-1：一次编辑可归属多个提交（敏感性口径）
MULTI_MIN = int(os.environ.get("ATTR_MULTI_MIN", "1"))      # 其他提交至少含本次编辑的几条新写入行才计


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def write(p, rows, fields):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def tot_nc(states):
    return sum(c["no_content"] for c in states.values())


def run(sysname):
    d = Path("data/derived") / sysname
    eps = load(d / "episodes.csv")
    ev = {(e["session"], e["seq"]): e for e in load(d / "events.csv") if e["kind"] == "tool"}
    attr = {(a["session"], a["seq"]): a for a in load(d / f"edit_attribution{SUFFIX}.csv")}
    sig = {(o["session"], str(o["seq"])) for o in map(json.loads, open(d / "edit_signatures.jsonl", encoding="utf-8"))}

    commits = {c["hash"]: c for c in load(d / "commits.csv")}
    by_path = defaultdict(list)
    for f in load(d / "commit_files.csv"):
        by_path[f["path"].lower()].append((datetime.fromisoformat(commits[f["hash"]]["time"]), f["hash"]))
    for p in by_path:
        by_path[p].sort()

    def next_commit(path, t):
        c = by_path.get(path.lower(), [])
        i = bisect_left(c, (datetime.fromisoformat(t), ""))
        return c[i][1] if i < len(c) else ""

    cp_rows = load(d / f"checkpoint_deltas{SUFFIX}.csv")
    cp_paths = defaultdict(set)
    for r in cp_rows:
        cp_paths[r["episode"]].add(r["path"])

    states = defaultdict(Counter)
    ucov = defaultdict(Counter)
    ec = defaultdict(lambda: [0, 0])
    pt = Counter()
    pt_states = defaultdict(Counter)
    for r in load(d / "episode_edits.csv"):
        if not r["episode"] or r["reason"]:
            continue
        e = ev[(r["session"], r["seq"])]
        if path_class(e["path"].replace("\\", "/"), e["in_workspace"]) != "code":
            continue
        a = attr.get((r["session"], r["seq"]))
        if a is None:
            assert SIG_VARIANT or (r["session"], r["seq"]) not in sig, ("有内容但无归属", r)
            st = "no_content"
            h = next_commit(e["path"].replace("\\", "/"), e["t_lo"] or e["time"])
            pt_states[r["episode"]]["pt_committed" if h else "pt_none"] += 1
            if h:
                pt[(r["episode"], h)] += 1
        else:
            st = a["state"]
        states[r["episode"]][st] += 1
        if e["path"].replace("\\", "/") not in cp_paths.get(r["episode"], ()):
            ucov[r["episode"]][st] += 1
        if st == "committed":
            k = (r["episode"], a["commit"])
            ec[k][0] += 1
            ec[k][1] += int(a["n_hit"])
        # D29-1 多归属敏感性：一次编辑的新写入行逐行首次进入的其他提交（≥ MULTI_MIN 行）也记为该片段的内容提交
        if MULTI and a is not None and a.get("extra_commits"):
            for item in a["extra_commits"].split(";"):
                h, n = item.rsplit(":", 1)
                if int(n) >= MULTI_MIN:
                    k = (r["episode"], h)
                    ec[k][0] += 1
                    ec[k][1] += int(n)

    for e in eps:
        assert sum(states[e["episode"]].values()) == int(e["n_edits_code"]), e["episode"]
    n_comm = sum(c["committed"] for c in states.values())
    assert MULTI or sum(v[0] for v in ec.values()) == n_comm

    cp = defaultdict(lambda: [0, 0])
    cp_states = defaultdict(Counter)
    cp_eps = set()
    for r in cp_rows:
        cp_eps.add(r["episode"])
        if r["class"] != "code":
            continue
        cp_states[r["episode"]][r["state"]] += 1
        if r["state"] == "committed":
            k = (r["episode"], r["commit"])
            cp[k][0] += 1
            cp[k][1] += int(r["n_hit"])

    rows = []
    for ep, h in sorted(set(ec) | set(cp) | set(pt)):
        a, b = ec.get((ep, h), [0, 0]), cp.get((ep, h), [0, 0])
        src = "both" if a[0] and b[0] else ("edit" if a[0] else ("checkpoint" if b[0] else "pathtime"))
        rows.append({"episode": ep, "commit": h, "commit_time": commits[h]["time"], "commit_label": commits[h]["label"],
                     "source": src, "n_edits": a[0], "n_lines": a[1], "cp_files": b[0], "cp_lines": b[1],
                     "pt_edits": pt.get((ep, h), 0)})
    write(d / f"episode_commit{SUFFIX}.csv", rows,
          ["episode", "commit", "commit_time", "commit_label", "source", "n_edits", "n_lines", "cp_files", "cp_lines", "pt_edits"])
    cp_cols = ["cp_committed", "cp_not_committed", "cp_after_last_commit", "cp_untracked", "cp_no_new"]
    ucov_cols = ["ucov_" + s for s in STATES]
    write(d / f"episode_code_states{SUFFIX}.csv",
          [{"episode": e["episode"], "n_edits_code": e["n_edits_code"], **{s: states[e["episode"]][s] for s in STATES},
            "has_checkpoint_delta": int(e["episode"] in cp_eps),
            **{c: cp_states[e["episode"]][c[3:]] for c in cp_cols},
            **{c: ucov[e["episode"]][c[5:]] for c in ucov_cols},
            **{c: pt_states[e["episode"]][c] for c in ["pt_committed", "pt_none"]}} for e in eps],
          ["episode", "n_edits_code"] + STATES + ["has_checkpoint_delta"] + cp_cols + ucov_cols + ["pt_committed", "pt_none"])
    assert sum(pt.values()) + sum(c["pt_none"] for c in pt_states.values()) == tot_nc(states)

    tot = Counter()
    for c in states.values():
        tot.update(c)
    print(f"{sysname}{SUFFIX}: code edits={sum(tot.values())} states={dict(tot)}")
    print(f"  pairs edit-only={sum(r['source'] == 'edit' for r in rows)} checkpoint-only={sum(r['source'] == 'checkpoint' for r in rows)} "
          f"both={sum(r['source'] == 'both' for r in rows)}; episodes mapped: by edits={len({ep for ep, _ in ec})} "
          f"by content={len({r['episode'] for r in rows if r['source'] != 'pathtime'})} "
          f"pathtime-only pairs={sum(r['source'] == 'pathtime' for r in rows)}; commits reached={len({r['commit'] for r in rows})}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        run(s)
