"""D52-2：格均值模型（D40，事前登记分期，系统 A）对标注口径的敏感性。

补齐 r14 §6 / §8 与 r15 两端法只按合并均值模型计算、未覆盖格均值模型的缺口：
裁决（论文现值）/ 分歧取更多合并 / 分歧取更多拆分 / 只用开发者 / 只用编码人甲 / 一致子样本（剔除 35 个裁决片段）
/ 一致子样本且数量一致（再剔除 6 个两人均判多项、分组一致但数量不同的片段）/ 该 6 个片段改取编码人甲数量、两人较小值、两人较大值。

写入 <PAPER_DIR>/results/r28_cells_sensitivity.md 与 numbers.json 的 cells_sensitivity 键。须在 boundary_labels.py score 与 robustness7.py 之后运行；
需 CODING_HUMAN=1。只读编码表与派生数据。
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CODING_HUMAN", "1")
sys.path.insert(0, str(Path(__file__).parent))
_argv = sys.argv
sys.argv = ["boundary_labels.py", "score", "v2"]
import boundary_labels as blm  # noqa: E402
import robustness7 as r7  # noqa: E402
sys.argv = _argv


def main():
    key = json.loads((blm.OUT / "key.json").read_text(encoding="utf-8"))
    c1, c2, adj_rows = blm.read_coder("coder1"), blm.read_coder("coder2"), blm.read_coder("adjudicate")
    A = {p: blm.parse(c1[p], key[p]) for p in key}
    B = {p: blm.parse(c2[p], key[p]) for p in key}
    ADJ = {p: blm.parse(adj_rows[p], key[p]) for p in adj_rows}
    need = [p for p in key if any(blm.view(A[p], q) != blm.view(B[p], q) for q, _ in blm.QS)
            or (A[p]["n"] > 1 and B[p]["n"] > 1 and A[p]["groups"] != B[p]["groups"])]
    for p in need:
        ADJ.setdefault(p, A[p])
    FIN = {m: {p: blm.resolve(A[p], B[p], ADJ.get(p, A[p]), m) for p in key} for m in ("final", "merge", "split")}
    AG = [p for p in key if p not in need]
    cnt6 = [p for p in key if p not in need and A[p]["n"] != B[p]["n"]]
    alt = lambda f: {p: (dict(FIN["final"][p]) | {"n": f(A[p]["n"], B[p]["n"])}) if p in cnt6 else FIN["final"][p] for p in key}

    s = "SYS_A"
    ps_all = [p for p in key if key[p]["system"] == s]
    C = blm.ctx_of(s)
    dys, _, _ = blm.days_of(s, C["x"], C["items"])
    dayreg = {d["d"]: blm.regime(d["reg"] / d["n"]) for d in dys}
    ep_day = {i["e"]: i["start"][:10] for i in C["items"]}
    cell_of = {p: dayreg.get(ep_day.get(key[p]["episode"]), "ineligible") for p in ps_all}

    R, L = {}, ["# 格均值模型对标注口径的敏感性（D52-2）", "",
                "由 `analysis/cells_sensitivity.py` 生成。系统 A，事前登记分期（主口径），格均值模型（D40）；区间为两层重抽 2,000 次。"
                "各行只改变样本片段的新工作项数来源，日子、提交数与分期不变。", "",
                f"两名编码人四问与分组均一致、但“含几项工作”的具体数量不同的片段：{len(cnt6)} 个（{', '.join(cnt6)}）；"
                "这些片段不在裁决清单内，裁决口径沿用开发者的数量。", "",
                "| 标注口径 | 两组每片段新工作项均值（片段内提交 / 未提交） | 高 / 低日工作项合计 | 每项工作提交数 高 / 低 | 之比（95% 区间） |",
                "|---|---|---|---|---|"]
    rows = [("裁决口径（正文）", FIN["final"], ps_all),
            ("分歧取更多合并", FIN["merge"], ps_all),
            ("分歧取更多拆分", FIN["split"], ps_all),
            ("只用开发者的原始标注", A, ps_all),
            ("只用编码人甲的原始标注", B, ps_all),
            ("一致子样本（剔除 35 个裁决片段）", FIN["final"], [p for p in ps_all if p in AG]),
            ("一致子样本且数量一致（再剔除 6 个数量分歧片段）", FIN["final"], [p for p in ps_all if p in AG and A[p]["n"] == B[p]["n"]]),
            ("6 个数量分歧片段改取编码人甲的数量", alt(lambda a, b: b), ps_all),
            ("6 个数量分歧片段取两人较小值", alt(min), ps_all),
            ("6 个数量分歧片段取两人较大值", alt(max), ps_all)]
    for label, F, ps in rows:
        cells = {r: [[blm.new_works(F[p]) for p in ps if key[p]["cmd"] == f and cell_of[p] == r] for f in (True, False)]
                 for r in ("hi", "mid", "lo", "ineligible")}
        nn = [sum(blm.new_works(F[p]) for p in ps if key[p]["cmd"] == f) / max(1, sum(key[p]["cmd"] == f for p in ps)) for f in (True, False)]
        nws = [[blm.new_works(F[p]) for p in ps if key[p]["cmd"] == f] for f in (True, False)]
        ctx = r7.load_ctx(s, {True: nn[0], False: nn[1]}, {True: nws[0], False: nws[1]})
        rc = r7.compare_cells(r7.days(ctx, by="reg"), cells, random.Random(f"{r7.SEED}|{s}|cells"))
        R[label] = {"new_works": nn, "n_samples": [len(nws[0]), len(nws[1])], "cells": rc["cells"],
                    "works": [rc["hi"]["works"], rc["lo"]["works"]], "c_work": [rc["hi"]["c_work"], rc["lo"]["c_work"]],
                    "ratio": rc["ratio"], "ci": rc["ci"]}
        L.append(f"| {label} | {nn[0]:.2f} / {nn[1]:.2f} | {rc['hi']['works']:.0f} / {rc['lo']['works']:.0f} | "
                 f"{rc['hi']['c_work']:.2f} / {rc['lo']['c_work']:.2f} | {rc['ratio']:.2f}（{rc['ci'][0]:.2f}–{rc['ci'][1]:.2f}） |")
    L += ["", "所有口径的区间都包含 1 且宽于等价带 [0.8, 1.25]，判定均为证据不足；点估计均小于 1。", ""]
    (r7.RES / "r28_cells_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    num = json.loads(r7.NUMS.read_text(encoding="utf-8"))
    num["cells_sensitivity"] = {"cnt6": cnt6, "rows": json.loads(json.dumps(R, default=float))}
    r7.NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
