"""S02-T5 敏感性：用不同口径重算候选模式主表（可核验片段），写入 <PAPER_DIR>/results/r1_sensitivity.md。

口径：
  main        内容匹配阈值 0.5，逐次编辑 + 检查点差分（主口径）
  m030 / m070 阈值 0.3 / 0.7（重跑 attribute_edits → checkpoint_deltas → map_episodes → patterns）
  edit_only   只用逐次编辑内容
  cp_only     只用检查点差分
  pathtime    加入路径—时间推定（patterns_pt 列；时间规则的对照）
"只用时间精确的编辑"口径不适用：去重后全部编辑时间来自原始时间戳，不确定度均为 0，结果与主口径相同。
可核验片段集合由编辑是否有内容决定，与阈值无关，各口径分母相同（脚本断言）。
"""
import csv
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from patterns import PATTERNS  # noqa: E402

MAIN = [p for p in PATTERNS if p != "无代码"]
FLAG_PP = 10
VARIANTS = [("main", "阈值 0.5（主口径）"), ("m030", "阈值 0.3"), ("m070", "阈值 0.7"),
            ("edit_only", "只用逐次编辑内容"), ("cp_only", "只用检查点差分"), ("pathtime", "加入路径—时间推定")]


def py(script, env):
    subprocess.run([sys.executable, f"analysis/{script}"], env={**os.environ, "PYTHONIOENCODING": "utf-8", **env},
                   check=True, stdout=subprocess.DEVNULL)


def prepare():
    for thr, suf in [("0.3", "_m030"), ("0.7", "_m070")]:
        env = {"ATTR_MATCH": thr, "ATTR_SUFFIX": suf}
        for s in ["attribute_edits.py", "checkpoint_deltas.py", "map_episodes.py", "patterns.py"]:
            py(s, env)
    py("patterns.py", {"PAIR_SOURCES": "edit", "PATTERN_SUFFIX": "_edit_only"})
    py("patterns.py", {"PAIR_SOURCES": "checkpoint", "PATTERN_SUFFIX": "_cp_only"})
    # D24-8：检查点只读已确认改动（files），不读待确认改动（activeInlineDiffs）的对照；归属文件与主口径相同
    for s in ["SYS_A", "SYS_C"]:
        d = Path("data/derived") / s
        (d / "edit_attribution_noinline.csv").write_bytes((d / "edit_attribution.csv").read_bytes())
    env = {"ATTR_SUFFIX": "_noinline", "CP_INLINE": "0"}
    for s in ["checkpoint_deltas.py", "map_episodes.py", "patterns.py"]:
        py(s, env)


def shares(sysname, variant):
    d = Path("data/derived") / sysname
    f = {"main": "", "pathtime": "", "m030": "_m030", "m070": "_m070", "edit_only": "_edit_only", "cp_only": "_cp_only"}[variant]
    rows = [r for r in csv.DictReader(open(d / f"episode_patterns{f}.csv", encoding="utf-8")) if r["evidence"] == "verified"]
    col = "patterns_pt" if variant == "pathtime" else "patterns"
    c = Counter(p for r in rows for p in r[col].split(";"))
    return len(rows), {p: c[p] / len(rows) for p in MAIN}


def main():
    if "--no-prepare" not in sys.argv:
        prepare()
    L = ["# R1 敏感性：候选模式主表在不同口径下的占比", "",
         "由 `analysis/sensitivity.py` 生成。分母为可核验片段（各口径相同）；单元格为带该标签的片段占比。"
         f"与主口径相差超过 {FLAG_PP} 个百分点的标 **粗体**。"
         "\"只用时间精确的编辑\"口径不适用：去重后全部编辑的时间不确定度为 0，结果与主口径相同。", ""]
    flagged = []
    for s in ["SYS_A", "SYS_C"]:
        res = {v: shares(s, v) for v, _ in VARIANTS}
        ns = {v: n for v, (n, _) in res.items()}
        assert len(set(ns.values())) == 1, ns
        base = res["main"][1]
        L += [f"## {s}（n = {ns['main']}）", "", "| 模式 | " + " | ".join(name for _, name in VARIANTS) + " |",
              "|---|" + "---|" * len(VARIANTS)]
        for p in MAIN:
            if not any(res[v][1][p] for v, _ in VARIANTS):
                continue
            cells = []
            for v, _ in VARIANTS:
                x = res[v][1][p]
                big = abs(x - base[p]) * 100 > FLAG_PP
                cells.append(f"**{x:.0%}**" if big else f"{x:.0%}")
                if big:
                    flagged.append((s, p, v, base[p], x))
            L.append(f"| {p} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 变化超过阈值的模式", ""]
    names = dict(VARIANTS)
    L += [f"- {s} {p}：{names[v]} 下 {b:.0%} → {x:.0%}" for s, p, v, b, x in flagged] or ["- 无"]
    out = (RESULTS_DIR / "r1_sensitivity.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out)
    for f in flagged:
        print(f)


if __name__ == "__main__":
    main()
