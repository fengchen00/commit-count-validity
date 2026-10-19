"""一次编辑只归属一个提交的漏判量化（D29-1，回应 ChatGPT 审计第 1 条）。

算法 1 第 3 步取第一个覆盖比例 ≥ θ 的提交，一次编辑的新写入行若被分批提交（C1 存一半、C2 存另一半），
只会归到 C1，片段层的"拆分"可能漏判。本脚本：
  1 编辑层：对每次有新写入行的代码编辑，逐行找首次进入的提交（attribute_edits.first_entry），统计
    行分散到 ≥ 2 个提交的编辑数、主归属之外进入其他提交的行数、从未进入任何后续提交的行数；
  2 片段层：以 ATTR_MULTI=1 重跑 map_episodes → patterns（多归属口径：其他提交至少含 MULTI_MIN 条新写入行即计），
    与主口径比较一对一 / 合并 / 拆分的比例。
写 <PAPER_DIR>/results/r22_multi_attrib.md 与 numbers.json 的 multi_attrib 键。须在 attribute_edits、patterns 之后运行。
"""
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR  # noqa: E402
from extract_episodes import path_class  # noqa: E402

SYS = ["SYS_A", "SYS_C"]
NAME = {"SYS_A": "系统 A", "SYS_C": "系统 B"}
MINS = [1, 2, 3]


def py(script, env):
    subprocess.run([sys.executable, f"analysis/{script}"], env={**os.environ, "PYTHONIOENCODING": "utf-8", **env},
                   check=True, stdout=subprocess.DEVNULL)


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def edit_level(s):
    d = Path("data/derived") / s
    rows = [r for r in load(d / "edit_attribution.csv") if path_class(r["path"], "1") == "code" and r["state"] in ("committed", "not_committed", "after_last_commit")]
    out = {"n_edits_new": len(rows)}
    com = [r for r in rows if r["state"] == "committed"]
    out["n_committed"] = len(com)
    spread = [r for r in com if int(r["n_first_commits"]) >= 2]
    out["n_committed_spread2"] = len(spread)
    for m in MINS:
        out[f"n_committed_spread2_min{m}"] = sum(1 for r in com if any(int(x.rsplit(":", 1)[1]) >= m for x in r["extra_commits"].split(";") if x))
    out["lines_new"] = sum(int(r["n_new"]) for r in com)
    out["lines_primary"] = sum(int(r["n_hit"]) for r in com)
    out["lines_extra"] = sum(sum(int(x.rsplit(":", 1)[1]) for x in r["extra_commits"].split(";") if x) for r in com)
    out["lines_never"] = sum(int(r["n_new"]) - int(r["n_entered"]) for r in com)
    # 未归属（not_committed / after_last_commit）的编辑里，其实有行进入了后续提交（只是没有一个提交达到 θ）
    unc = [r for r in rows if r["state"] != "committed"]
    out["n_uncommitted"] = len(unc)
    out["n_uncommitted_some_entered"] = sum(1 for r in unc if int(r["n_entered"]) > 0)
    out["n_uncommitted_spread2"] = sum(1 for r in unc if int(r["n_first_commits"]) >= 2)
    out["uncommitted_lines_entered"] = sum(int(r["n_entered"]) for r in unc)
    out["uncommitted_lines_new"] = sum(int(r["n_new"]) for r in unc)
    return out


def shares(s, suffix, verified_only):
    d = Path("data/derived") / s
    rows = load(d / f"episode_patterns{suffix}.csv")
    rows = [r for r in rows if r["patterns"] != "无代码" and (r["evidence"] == "verified" or not verified_only)]
    c = Counter()
    for r in rows:
        labs = set(r["patterns"].split(";"))
        c["一对一"] += "一对一" in labs
        c["合并"] += bool(labs & {"合并-同会话", "合并-跨会话"})
        c["拆分"] += "拆分" in labs
        c["拆分-逐条任务"] += "拆分-逐条任务" in labs
        c["拆分-非逐条"] += "拆分-非逐条" in labs
    n = len(rows)
    return n, {k: c[k] / n for k in ("一对一", "合并", "拆分", "拆分-逐条任务", "拆分-非逐条")}, {r["episode"]: r for r in rows}


def main():
    R = {"edit": {}, "episode": {}}
    for s in SYS:
        R["edit"][s] = edit_level(s)
    # 片段层多归属口径：归属文件与检查点差分沿用主口径文件，只改 map_episodes 的汇总
    for m in MINS:
        suf = f"_multi{m}"
        for s in SYS:
            d = Path("data/derived") / s
            (d / f"edit_attribution{suf}.csv").write_bytes((d / "edit_attribution.csv").read_bytes())
            (d / f"checkpoint_deltas{suf}.csv").write_bytes((d / "checkpoint_deltas.csv").read_bytes())
        env = {"ATTR_SUFFIX": suf, "ATTR_MULTI": "1", "ATTR_MULTI_MIN": str(m)}
        py("map_episodes.py", env)
        py("patterns.py", env)
    for s in SYS:
        R["episode"][s] = {}
        for vo, tag in ((True, "verified"), (False, "all")):
            n0, base, rows0 = shares(s, "", vo)
            R["episode"][s][tag] = {"n": n0, "main": base}
            for m in MINS:
                n1, alt, rows1 = shares(s, f"_multi{m}", vo)
                assert n1 == n0
                changed = sum(1 for k in rows0 if rows0[k]["patterns"] != rows1[k]["patterns"])
                new_split = sum(1 for k in rows0 if "拆分" in rows1[k]["patterns"].split(";") and "拆分" not in rows0[k]["patterns"].split(";"))
                lost_oto = sum(1 for k in rows0 if "一对一" in rows0[k]["patterns"].split(";") and "一对一" not in rows1[k]["patterns"].split(";"))
                R["episode"][s][tag][f"min{m}"] = {"shares": alt, "changed": changed, "new_split": new_split, "lost_oto": lost_oto}

    L = ["# 一次编辑只归属一个提交的漏判量化（D29-1）", "",
         "由 `analysis/multi_attrib.py` 生成。算法 1 第 3 步取第一个覆盖比例 ≥ θ 的提交；这里对每次编辑的新写入行逐行找首次进入的提交，"
         "量化一次编辑的内容被分批提交、从而漏判片段层拆分的情形。", "", "## 1 编辑层（有新写入行的代码编辑）", "",
         "| 系统 | 有新写入行的编辑 | 已归属 | 其中行分散到 ≥ 2 个提交（其他提交 ≥ 1 / ≥ 2 / ≥ 3 行） | 已归属编辑的新写入行：主归属提交 / 其他提交 / 从未进入 | 未归属编辑 | 其中有行进入后续提交 / 行分散到 ≥ 2 个提交 | 未归属编辑进入后续提交的行 / 全部新写入行 |",
         "|---|---|---|---|---|---|---|---|"]
    for s in SYS:
        e = R["edit"][s]
        L.append(f"| {NAME[s]} | {e['n_edits_new']} | {e['n_committed']} | {e['n_committed_spread2_min1']}（{e['n_committed_spread2_min1'] / e['n_committed']:.1%}） / "
                 f"{e['n_committed_spread2_min2']}（{e['n_committed_spread2_min2'] / e['n_committed']:.1%}） / {e['n_committed_spread2_min3']}（{e['n_committed_spread2_min3'] / e['n_committed']:.1%}） | "
                 f"{e['lines_primary']}（{e['lines_primary'] / e['lines_new']:.1%}） / {e['lines_extra']}（{e['lines_extra'] / e['lines_new']:.1%}） / {e['lines_never']}（{e['lines_never'] / e['lines_new']:.1%}） | "
                 f"{e['n_uncommitted']} | {e['n_uncommitted_some_entered']} / {e['n_uncommitted_spread2']} | {e['uncommitted_lines_entered']} / {e['uncommitted_lines_new']} |")
    L += ["", "## 2 片段层：多归属口径下的模式比例", "",
          "多归属口径：一次编辑除主归属提交外，其新写入行首次进入的其他提交只要含 ≥ k 条该编辑的行，也记为片段的内容提交（k = 1、2、3）；检查点差分沿用主口径。", ""]
    for s in SYS:
        for tag, tn in (("verified", "可核验片段"), ("all", "全部代码修改片段")):
            x = R["episode"][s][tag]
            L += [f"### {NAME[s]}·{tn}（n = {x['n']}）", "", "| 口径 | 一对一 | 合并 | 拆分 | 拆分-逐条任务 | 拆分-非逐条 | 模式有变化的片段 | 新增拆分 | 失去一对一 |", "|---|---|---|---|---|---|---|---|---|",
                  "| 主口径（一次编辑一个提交） | " + " | ".join(f"{x['main'][k]:.1%}" for k in ("一对一", "合并", "拆分", "拆分-逐条任务", "拆分-非逐条")) + " | — | — | — |"]
            for m in MINS:
                y = x[f"min{m}"]
                L.append(f"| 多归属，其他提交 ≥ {m} 行 | " + " | ".join(f"{y['shares'][k]:.1%}" for k in ("一对一", "合并", "拆分", "拆分-逐条任务", "拆分-非逐条")) + f" | {y['changed']} | {y['new_split']} | {y['lost_oto']} |")
            L.append("")
    L += ["## 3 说明", "",
          "- 逐行首次进入：对每条新写入行，在该文件 t_lo 之后的提交中按时间找第一个把它作为新增行的提交；一行只计一次，后续提交中再出现视为移动或重复。",
          "- 主归属提交之外的行可能来自：同一编辑的内容被分批提交（真实拆分）；少数样板行（import、闭合括号等长度 ≥ 15 的行）与其他提交的新增行偶然相同；智能体在后续片段重写并再次提交同一内容。k 越大，后两类越少。",
          "- 未归属编辑中“有行进入后续提交”的，是没有任何单个提交达到 θ 的情形；它们在主口径记为未提交，多归属口径下也不改变（此处只统计，不改状态）。", ""]
    (RESULTS_DIR / "r22_multi_attrib.md").write_text("\n".join(L), encoding="utf-8")
    nums = RESULTS_DIR / "numbers.json"
    num = json.loads(nums.read_text(encoding="utf-8"))
    num["multi_attrib"] = R
    nums.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
