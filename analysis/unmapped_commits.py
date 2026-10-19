"""不含任何片段内容的提交的刻画（S19-T-E1，口径见 notes/decisions.md D26-3）。

回应独立盲审的共识问题 C2：系统 A 有 436 个提交（36%）没有归属到任何片段的内容，它们进入 RQ3 的日层分母，
却没有被刻画。本脚本按以下维度分类，写入 <PAPER_DIR>/results/r19_unmapped.md 与 numbers.json 的 unmapped 键。

1. 来源：会话中可对上智能体提交命令（有已知执行片段）/ 只有 Cursor 协作尾注 / 二者皆无。
2. 改动：类别（impl / mixed / plan）、代码文件数、增删行数（git numstat）、标题类别。
3. 有已知执行片段者的未归属原因：执行片段无代码编辑；有代码编辑但与提交文件不重叠；
   文件重叠但编辑无写入内容（n_sig = 0）；文件重叠、编辑有内容但被归到其他提交或未入库。
4. 在事前登记分期高 / 低两类日子中的分布，以及把"命令来源"的未归属提交按其执行片段回填后，每片段提交数之比的变化。
5. 仅有尾注、无命令记录的提交（会话覆盖率威胁）：数量、是否改动代码、日期分布。

须在 robustness7.py 之后运行（读 numbers.json 的 robustness6 键取效标）。
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import LABEL, commit_sources, load  # noqa: E402
from extract_git import SYSTEMS, git  # noqa: E402
from robustness import NAME  # noqa: E402
from robustness7 import BOOT, agg, bci, days, load_ctx, split  # noqa: E402
from workitems import wilson  # noqa: E402

NUMS = RESULTS_DIR / "numbers.json"


def numstat(s):
    out = {}
    for blk in git(SYSTEMS[s], "log", "--numstat", "--format=\x1e%H").split("\x1e")[1:]:
        lines = blk.strip().split("\n")
        h, add, dele = lines[0].strip(), 0, 0
        for ln in lines[1:]:
            p = ln.split("\t")
            if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
                add += int(p[0])
                dele += int(p[1])
        out[h] = add + dele
    return out


def pct(k, n):
    lo, hi = wilson(k, n) if n else (0, 0)
    return f"{k}/{n}（{k / n:.0%}，{lo:.0%}–{hi:.0%}）" if n else "0/0"


def run(s, num):
    d = Path("data/derived") / s
    cm, gold, trailer = commit_sources(s)
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    content = defaultdict(set)
    for p in load(d / "episode_commit.csv"):
        if p["source"] != "pathtime":
            content[p["commit"]].add(p["episode"])
    cfiles = defaultdict(set)
    for f in load(d / "commit_files.csv"):
        cfiles[f["hash"]].add(f["path"].lower())
    efiles = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            efiles[f["episode"]].add(f["path"].lower())
    ep_of_edit = {(r["session"], r["seq"]): r["episode"] for r in load(d / "episode_edits.csv")}
    edits = defaultdict(list)
    for r in load(d / "edit_attribution.csv"):
        ep = ep_of_edit.get((r["session"], r["seq"]))
        if ep:
            edits[ep].append(r)
    size = numstat(s)

    unmapped = [h for h in cm if not content[h]]
    n, U = len(cm), len(unmapped)
    src = {h: ("命令" if h in gold else "尾注" if h in trailer else "皆无") for h in unmapped}
    R = {"n_commits": n, "n_unmapped": U, "source": dict(Counter(src.values()))}
    L = [f"## {NAME[s]}", "",
         f"- 提交 {n} 个，不含任何片段内容的 {U} 个（{U / n:.0%}）。来源：" +
         "，".join(f"{k} {v}" for k, v in Counter(src.values()).most_common()) + "。", ""]

    # 2 改动
    def desc(hs, title):
        kind = Counter(cm[h]["kind"] for h in hs)
        lab = Counter(cm[h]["label"] for h in hs)
        code = sum(int(cm[h]["n_code_files"] or 0) > 0 for h in hs)
        sz = sorted(size.get(h, 0) for h in hs)
        med = sz[len(sz) // 2] if sz else 0
        small = sum(v <= 20 for v in sz)
        return {"n": len(hs), "kind": dict(kind), "label": dict(lab), "code": code, "median_lines": med, "le20": small}, (
            f"- {title}（{len(hs)} 个）：改动类别 " + "，".join(f"{k} {v}" for k, v in kind.most_common()) +
            f"；改动了代码文件的 {code} 个；增删行数中位数 {med}，不超过 20 行的 {small} 个；标题类别 " +
            "，".join(f"{LABEL.get(k, k)} {v}" for k, v in lab.most_common()) + "。")

    # D32 B4：三类会话侧证据（提交命令 / 协作尾注 / 片段内容）在全部提交上的交叉表
    cross = Counter((h in gold, h in trailer, bool(content[h])) for h in cm)
    R["evidence_cross"] = {f"cmd={int(a)}|trailer={int(b)}|content={int(c)}": v for (a, b, c), v in cross.items()}
    L += ["### 三类会话侧证据的交叉表（全部提交；D32 B4）", "",
          "| 提交命令可对上 | 协作尾注 | 含片段内容 | 提交数 |", "|---|---|---|---|"]
    for a in (True, False):
        for b in (True, False):
            for c in (True, False):
                L.append(f"| {'是' if a else '否'} | {'是' if b else '否'} | {'是' if c else '否'} | {cross.get((a, b, c), 0)} |")
    L += ["", f"三者皆无的提交 {cross.get((False, False, False), 0)} 个；只有尾注的 {cross.get((False, True, False), 0)} 个；"
          f"命令可对上但不含片段内容的 {cross.get((True, True, False), 0) + cross.get((True, False, False), 0)} 个。", ""]
    # D32 B5：重命名（git -M 检测）与纯删除提交的数量；片段侧的只删除编辑计数在 robustness2 的 del_* 键
    ren, del_only, n_blk = 0, 0, 0
    for blk in git(SYSTEMS[s], "log", "--name-status", "-M", "--format=\x1e%H").split("\x1e")[1:]:
        lines = [ln for ln in blk.strip().split("\n")[1:] if ln.strip()]
        n_blk += 1
        st = [ln.split("\t")[0][:1] for ln in lines]
        ren += any(x == "R" for x in st)
        del_only += bool(st) and all(x == "D" for x in st)
    R["rename_delete"] = {"commits_with_rename": ren, "delete_only_commits": del_only, "n_commits_log": n_blk}
    L += [f"- 重命名与纯删除（D32 B5）：按 `git log -M` 检测，含重命名文件的提交 {ren} 个，只删除文件的提交 {del_only} 个（共 {n_blk} 个提交）。", ""]
    R["all"], line = desc(unmapped, "全部未归属提交")
    L.append(line)
    mapped = [h for h in cm if content[h]]
    R["mapped_ref"], line = desc(mapped, "对照：含片段内容的提交")
    L.append(line)
    L.append("")

    # 3 命令来源者的未归属原因
    reasons = Counter()
    reason_of = {}
    for h in unmapped:
        if h not in gold:
            continue
        ep = gold[h]
        if int(eps[ep]["n_edits_code"] or 0) == 0:
            r = "执行片段无代码编辑"
        elif not (efiles[ep] & cfiles[h]):
            r = "有代码编辑但与提交文件不重叠"
        else:
            rel = [e for e in edits[ep] if e["path"].lower() in cfiles[h]]
            if rel and all(int(e["n_sig"] or 0) == 0 for e in rel):
                r = "文件重叠但编辑无写入内容"
            elif any(e["state"] == "committed" and e["commit"] and e["commit"] != h for e in rel):
                r = "文件重叠、内容被归到其他提交"
            else:
                r = "文件重叠、内容未入库或未命中"
        reasons[r] += 1
        reason_of[h] = r
    R["reason_cmd"] = dict(reasons)
    L += ["### 有已知执行片段的未归属提交：为什么没对上", ""]
    for k, v in reasons.most_common():
        L.append(f"- {k}：{v}")
    L.append("")

    # 5 仅尾注
    tr_only = [h for h in unmapped if src[h] == "尾注"]
    tr_all = [h for h in cm if h in trailer and h not in gold]
    R["trailer_only"] = {"unmapped": len(tr_only), "all": len(tr_all),
                         "all_mapped": len(tr_all) - len(tr_only),
                         "code_unmapped": sum(int(cm[h]["n_code_files"] or 0) > 0 for h in tr_only),
                         "days": len({cm[h]["time"][:10] for h in tr_all})}
    L += ["### 只有协作尾注、没有对上提交命令的提交（会话覆盖率）", "",
          f"- 共 {len(tr_all)} 个，其中 {len(tr_all) - len(tr_only)} 个的内容已归属到某个片段（命令未对上但会话存在），"
          f"{len(tr_only)} 个未归属（改动代码文件的 {R['trailer_only']['code_unmapped']} 个），分布在 {R['trailer_only']['days']} 天。", ""]

    # 4 高 / 低日分布与命令回填
    bl = num["robustness6"]["boundary_labels"]
    nw = bl[f"{s}|final|rates"]["new_works"]
    ctx = load_ctx(s, {True: nw[0], False: nw[1]})
    ds = days(ctx, by="reg")
    H, Lo = split(ds, 2 / 3, 1 / 3)
    hd, ld = {d["d"] for d in H}, {d["d"] for d in Lo}
    day_of = lambda h: cm[h]["time"][:10]
    grp = {"hi": [h for h in cm if day_of(h) in hd], "lo": [h for h in cm if day_of(h) in ld]}
    R["by_group"] = {}
    for g, hs in grp.items():
        um = [h for h in hs if not content[h]]
        R["by_group"][g] = {"commits": len(hs), "unmapped": len(um), "source": dict(Counter(src[h] for h in um)),
                            "reason": dict(Counter(reason_of.get(h, "—") for h in um if h in gold))}
    L += ["### 事前登记分期高 / 低两类日子中的分布", ""]
    for g, lab in (("hi", "高"), ("lo", "低")):
        b = R["by_group"][g]
        L.append(f"- {lab}比例日：提交 {b['commits']}，未归属 {pct(b['unmapped'], b['commits'])}；来源 " +
                 "，".join(f"{k} {v}" for k, v in sorted(b["source"].items())) + "；命令来源者原因 " +
                 "，".join(f"{k} {v}" for k, v in sorted(b["reason"].items(), key=lambda kv: -kv[1])) + "。")

    # 命令回填：把命令来源的未归属提交计给其执行片段所在的日子
    fill = defaultdict(float)
    for h in unmapped:
        if h in gold:
            fill[eps[gold[h]]["start"][:10]] += 1
    rng = random.Random(f"D26|{s}")

    def ratio(attr_key):
        for d in ds:
            d["c_fill"] = d["_share"] + fill.get(d["d"], 0) if attr_key == "fill" else d["_share"]
        a = lambda G: sum(d["c_fill"] for d in G) / sum(d["n"] for d in G)
        r = a(H) / a(Lo)
        bs = []
        for _ in range(BOOT):
            x, y = [rng.choice(H) for _ in H], [rng.choice(Lo) for _ in Lo]
            bs.append(a(x) / a(y) if a(y) else float("nan"))
        return r, bci(bs)

    # 按片段归属的份额（cohort）：重算一次 days 取 c
    ds_c = {d["d"]: d["c"] for d in days(ctx, by="reg", attr="cohort")}
    for d in ds:
        d["_share"] = ds_c.get(d["d"], 0.0)
    r_time = agg(H)["c_ep"] / agg(Lo)["c_ep"]
    r_cohort, ci_cohort = ratio("cohort")
    r_fill, ci_fill = ratio("fill")
    R["ratio"] = {"time": r_time, "cohort": [r_cohort, ci_cohort], "cohort_plus_cmd": [r_fill, ci_fill],
                  "filled_commits": int(sum(fill.values()))}
    L += ["", "### 每片段提交数之比（高对低）在三种提交归属下", "",
          f"- 按提交时间计入当日全部提交：{r_time:.2f}（表 12 口径）。",
          f"- 只计按内容归属到当日片段的提交：{r_cohort:.2f}（{ci_cohort[0]:.2f}–{ci_cohort[1]:.2f}）。",
          f"- 内容归属 + 把命令来源的未归属提交回填给执行片段（回填 {int(sum(fill.values()))} 个）："
          f"{r_fill:.2f}（{ci_fill[0]:.2f}–{ci_fill[1]:.2f}）。", ""]
    return R, L


def main():
    num = json.loads(NUMS.read_text(encoding="utf-8")) if NUMS.exists() else {}
    out = ["# 不含任何片段内容的提交的刻画（S19-T-E1，D26-3）", "",
           "由 `analysis/unmapped_commits.py` 生成。只统计数量；提交标题与命令文本不写出。区间按日重抽 2,000 次。", ""]
    R = {}
    for s in sys.argv[1:] or ["SYS_A", "SYS_C"]:
        R[s], L = run(s, num)
        out += L
    (RESULTS_DIR / "r19_unmapped.md").write_text("\n".join(out), encoding="utf-8")
    num["unmapped"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
