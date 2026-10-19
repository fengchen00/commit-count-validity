"""人工估计的分层反加权与后验混合说明表（D32 B3；回应模拟审稿甲 M3）。

把 robustness5.human_exclusive 已做的计算逐层写出，不改任何数字：
- 两批样本（原样本 S03、留出样本 S12）× 两个系统，每层的总体片段数 N、样本数 n、权重 N/n、借用两系统合并样本的层；
- 每层对一对一 / 合并 / 拆分三种模式的命中 / 未命中 / 无代码（N0）计数；
- Dirichlet 先验中心（该批全部样本的三类比例，先验总量 1 个样本）；
- 两批各自的反加权点估计、混合权重 w = n1/(n1+n2) 与混合后的中位数、95% 区间（直接读 numbers.json 的 robustness5.human）。
写入 <PAPER_DIR>/results/r23_weighting.md 与 numbers.json 的 weighting_table 键。须在 robustness5.py 之后运行。
"""
import json
import sys
from pathlib import Path

from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from holdout_agreement import holdout_reference  # noqa: E402
from robustness import NAME, load  # noqa: E402
from robustness3 import RES, read_pop  # noqa: E402
from robustness5 import H_ROWS  # noqa: E402
from sim_agreement import final_reference  # noqa: E402

NUMS = RESULTS_DIR / "numbers.json"
SYSTEMS = ["SYS_A", "SYS_C"]
MODES = [("一对一", dict(H_ROWS)["一对一"]), ("合并", dict(H_ROWS)["合并"]), ("拆分", dict(H_ROWS)["拆分"])]


def layer_table(ref, strata, pop, s):
    rows, borrowed = [], []
    es_all = list(ref)
    prior = {}
    for name, pred in MODES:
        k = [sum(ref[e]["main"] != "N0" and pred(ref[e]) for e in es_all), sum(ref[e]["main"] != "N0" and not pred(ref[e]) for e in es_all),
             sum(ref[e]["main"] == "N0" for e in es_all)]
        prior[name] = [v / len(es_all) for v in k]
    for h, ps in pop.items():
        N = ps[s]
        if not N:
            continue
        es = [e for e, st in strata.items() if st == h and e.split("-")[0] == s]
        b = False
        if not es:
            es = [e for e, st in strata.items() if st == h]
            b = True
            borrowed.append(h)
        row = {"stratum": h, "N": N, "n": len(es), "w": N / len(es), "borrowed": b}
        for name, pred in MODES:
            row[name] = [sum(ref[e]["main"] != "N0" and pred(ref[e]) for e in es), sum(ref[e]["main"] != "N0" and not pred(ref[e]) for e in es),
                         sum(ref[e]["main"] == "N0" for e in es)]
        rows.append(row)
    return rows, prior, borrowed


def main():
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    hum = num["robustness5"]["human"]
    orig, hold = final_reference("sim2"), holdout_reference()
    s1 = {r["episode"]: r["stratum"] for r in load("data/coding/sample_S03.csv")}
    s2 = {r["episode"]: r["stratum"] for r in load("data/coding/holdout/sample_S12.csv")}
    p1, p2 = read_pop(RES / "s03_sampling.md"), read_pop(RES / "s12_holdout_sampling.md")
    R = {}
    L = ["# 人工估计的分层反加权与后验混合说明（D32 B3）", "",
         "由 `analysis/weighting_table.py` 生成。把 4.1.4 节人工估计（`robustness5.human_exclusive`）的每一步按层写出；数字与 r13 相同。"
         "层为自动判定的主模式（抽样层），N 为该系统该层的全部片段数，n 为该批样本中该系统落在该层的片段数，权重为 N/n；"
         "该系统某层无样本时借用两系统合并的该层样本（标 *）。每层三元计数为（命中该模式，未命中，无代码 N0）；"
         "Dirichlet 先验中心为该批全部样本的三类比例，先验总量相当于 1 个样本。", ""]
    for s in SYSTEMS:
        n1, n2 = sum(e.startswith(s) for e in s1), sum(e.startswith(s) for e in s2)
        w = n1 / (n1 + n2)
        R[s] = {"n_orig": n1, "n_hold": n2, "w_orig": w}
        L += [f"## {NAME[s]}（原样本 {n1} 个，留出样本 {n2} 个，混合权重 w = {w:.3f}）", ""]
        for tag, ref, strata, pop in (("原样本（S03）", orig, s1, p1), ("留出样本（S12）", hold, s2, p2)):
            rows, prior, borrowed = layer_table(ref, strata, pop, s)
            R[s][tag] = {"rows": rows, "prior": prior, "borrowed": borrowed}
            L += [f"### {tag}", "", "| 层（自动主模式） | N | n | N/n | 一对一（命中, 未命中, N0） | 合并 | 拆分 |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                L.append(f"| {r['stratum']}{'*' if r['borrowed'] else ''} | {r['N']} | {r['n']} | {r['w']:.1f} | {tuple(r['一对一'])} | {tuple(r['合并'])} | {tuple(r['拆分'])} |")
            L.append(f"| 合计 | {sum(r['N'] for r in rows)} | {sum(r['n'] for r in rows if not r['borrowed'])} | | | | |")
            L += ["", "先验中心（命中, 未命中, N0）：" + "；".join(f"{m} ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for m, p in prior.items()) + "。", ""]
        L += ["### 两批点估计与混合后验", "", "| 模式 | 原样本反加权 | 留出样本反加权 | 混合点估计 | 混合后验中位数 | 95% 区间 |", "|---|---|---|---|---|---|"]
        for m, _ in MODES:
            h = hum[f"{s}|{m}"]
            L.append(f"| {m} | {h['orig']:.1%} | {h['hold']:.1%} | {h['point']:.1%} | {h['med']:.1%} | {h['ci'][0]:.0%}–{h['ci'][1]:.0%} |")
        L.append("")
    L += ["反加权点估计 = Σ_h N_h·(命中_h / n_h) / Σ_h N_h·((命中_h + 未命中_h) / n_h)，分母只计有代码的片段；"
          "后验抽样对每层从 Dirichlet(计数 + 先验) 抽三类比例并代入上式，两批各抽 4,000 次后按 w 线性混合。", ""]
    (RESULTS_DIR / "r23_weighting.md").write_text("\n".join(L), encoding="utf-8")
    num["weighting_table"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
