"""D35-6（B4）：去继承口径下的映射模式——检验"整文件重写把其他片段尚未提交的内容算作本片段新写入、从而制造假共享提交"的威胁。

口径：attribute_edits.py 的 ATTR_DEINHERIT=1（一次编辑的特征行若已由更早的、属于其他片段的编辑写过，不计为新写入行），
后缀 _deinherit，重跑 attribute_edits → checkpoint_deltas → map_episodes → patterns，与主口径逐片段比较。
写入 <PAPER_DIR>/results/r24_deinherit.md 与 numbers.json 的 deinherit 键。既有文件与键不改。
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
from robustness import MERGE, NAME  # noqa: E402

SUF = "_deinherit"
NUMS = RESULTS_DIR / "numbers.json"
KEYS = (("一对一", ("一对一",)), ("共享提交", MERGE), ("拆分", ("拆分",)), ("缺失", ("缺失",)))


def py(script, env):
    subprocess.run([sys.executable, f"analysis/{script}"], env={**os.environ, "PYTHONIOENCODING": "utf-8", **env},
                   check=True, stdout=subprocess.DEVNULL)


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    if "--no-prepare" not in sys.argv:
        env = {"ATTR_SUFFIX": SUF, "ATTR_DEINHERIT": "1"}
        for s in ["attribute_edits.py", "checkpoint_deltas.py", "map_episodes.py", "patterns.py"]:
            py(s, env)
    R = {}
    L = ["# 去继承口径下的映射模式（D35-6；回应“整文件重写继承未提交内容制造假共享提交”）", "",
         "由 `analysis/deinherit_check.py` 生成。去继承：一次编辑的特征行若已由更早的、属于其他片段的编辑写过（按时间顺序，不论是否已提交），不计为本次编辑的新写入行；"
         "检查点差分本身是相邻检查点之差，不受影响。其余步骤与主口径相同。", ""]
    for s in ("SYS_A", "SYS_C"):
        d = Path("data/derived") / s
        st = json.loads((d / f"deinherit_stats{SUF}.json").read_text(encoding="utf-8"))
        base = {r["episode"]: r for r in load(d / "episode_patterns.csv")}
        var = {r["episode"]: r for r in load(d / f"episode_patterns{SUF}.csv")}
        ea = {(r["session"], r["seq"]): r for r in load(d / "edit_attribution.csv")}
        eb = {(r["session"], r["seq"]): r for r in load(d / f"edit_attribution{SUF}.csv")}
        ch = Counter((ea[k]["state"], eb[k]["state"]) for k in ea if k in eb and (ea[k]["state"], ea[k]["commit"]) != (eb[k]["state"], eb[k]["commit"]))
        pa = {(r["episode"], r["commit"]) for r in load(d / "episode_commit.csv") if r["source"] != "pathtime"}
        pb = {(r["episode"], r["commit"]) for r in load(d / f"episode_commit{SUF}.csv") if r["source"] != "pathtime"}
        out = {"stats": st, "edit_state_changes": sum(ch.values()), "edit_state_change_detail": {f"{a}->{b}": n for (a, b), n in ch.items()},
               "pairs_main": len(pa), "pairs_deinherit": len(pb), "pairs_only_main": len(pa - pb), "pairs_only_deinherit": len(pb - pa)}
        L += [f"## {NAME[s]}", "",
              f"- 有内容的编辑 {st['n_edits']:,} 次、特征行 {st['n_lines_total']:,} 条；含有其他片段先写过的行的编辑 {st['n_edits_with_inherited']:,} 次"
              f"（{st['n_edits_with_inherited'] / st['n_edits']:.1%}），这类行共 {st['n_lines_inherited']:,} 条（占特征行 {st['n_lines_inherited'] / st['n_lines_total']:.1%}）；"
              f"继承行占本次特征行一半以上的编辑 {st['n_edits_inherited_majority']:,} 次，全部为继承行的 {st['n_edits_all_inherited']:,} 次。",
              f"- 去继承后归属状态或归属提交改变的编辑 {sum(ch.values())} 次（" + "、".join(f"{a}→{b} {n}" for (a, b), n in ch.most_common()) + "）；"
              f"片段—提交对（不含路径—时间推定）主口径 {len(pa)} 对、去继承 {len(pb)} 对，只在一方出现的 {len(pa - pb)} / {len(pb - pa)} 对。", "",
              "| 分母 | n | 口径 | " + " | ".join(k for k, _ in KEYS) + " | 共享提交标签改变的片段 |", "|---|---|---|" + "---|" * len(KEYS) + "---|"]
        for dn, pred in (("可核验", lambda r: r["evidence"] == "verified"), ("全部代码修改", lambda r: "无代码" not in r["patterns"])):
            eps = [e for e, r in base.items() if pred(r)]
            n = len(eps)
            res = {}
            for tag, src in (("main", base), ("deinherit", var)):
                lab = {e: set(src[e]["patterns"].split(";")) for e in eps}
                res[tag] = {k: sum(any(x in lab[e] for x in ks) for e in eps) / n for k, ks in KEYS}
            flip = sum((any(x in base[e]["patterns"].split(";") for x in MERGE)) != (any(x in var[e]["patterns"].split(";") for x in MERGE)) for e in eps)
            lost = sum((any(x in base[e]["patterns"].split(";") for x in MERGE)) and not (any(x in var[e]["patterns"].split(";") for x in MERGE)) for e in eps)
            out[dn] = {"n": n, "main": res["main"], "deinherit": res["deinherit"], "merge_flip": flip, "merge_lost": lost}
            for tag, tn in (("main", "主口径"), ("deinherit", "去继承")):
                L.append(f"| {dn} | {n} | {tn} | " + " | ".join(f"{res[tag][k]:.0%}" for k, _ in KEYS) + (f" | {flip}（其中失去标签 {lost}） |" if tag == "deinherit" else " | |"))
        L.append("")
        R[s] = out
    L += ["共享提交标签改变的片段数为两口径下“共享提交”标签不同的片段数；失去标签指主口径为共享提交、去继承后不是。"
          "可核验片段集合由编辑是否有内容决定，两口径相同。", ""]
    (RESULTS_DIR / "r24_deinherit.md").write_text("\n".join(L), encoding="utf-8")
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    num["deinherit"] = R
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
