"""S06-T2 客户条目模拟核对汇总（AI 模拟编码，D6、D7）→ <PAPER_DIR>/results/sim_customer_check.md（只含汇总）。

抽样层：原文贴入（pasted）、只有文档级关联（via_doc）、追不到（none）。按层总体数加权推回全部条目。
"""
import csv
import math
from collections import Counter
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

SAMPLE = Path("data/coding/customer_check/sample.csv")
SHEET = Path("data/coding/sim/sim_customer_check.csv")
TRACE = Path("data/derived/customer_trace.csv")
OUT = (RESULTS_DIR / "sim_customer_check.md")


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    trace = list(csv.DictReader(open(TRACE, encoding="utf-8")))
    stratum = lambda r: "pasted" if r["pasted"] else ("via_doc" if r["via_doc"] else "none")
    pop = Counter(stratum(r) for r in trace)
    by_no = {r["item_no"]: stratum(r) for r in trace}
    sample = {r["item"]: by_no[r["item_no"]] for r in csv.DictReader(open(SAMPLE, encoding="utf-8"))}
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    assert {r["item"] for r in rows} == set(sample), "编码表与样本不一致"
    valid = {"pasted": "YDNQ", "via_doc": "YDNQ", "none": "FPNQ"}
    problems = [r["item"] for r in rows if r["verdict"] not in valid[sample[r["item"]]] or r["is_request"] not in "YNQ"]

    L = ["# S06-T2 客户条目核对汇总（AI 模拟编码，不是人工核对）", "",
         "> 核对人由 AI 子代理模拟（D6）。只用于估计字面追踪规则的精确率与漏检量级，不能作为人工核对结果报告。", "",
         "由 `analysis/sim_customer_check.py` 生成；只含汇总。", "",
         f"- 条目总体 {len(trace)}：原文贴入 {pop['pasted']}，只有文档级关联 {pop['via_doc']}，追不到 {pop['none']}。",
         f"- 样本 {len(rows)}；校验问题：{'、'.join(problems) if problems else '无'}。", "",
         "判定：追到的条目 Y = 片段确实处理了该条，D = 片段处理整份文档、该条无法确认，N = 未处理（误追）；"
         "追不到的条目 F = 候选片段确实处理了（漏追），P = 可能处理了，N = 没有候选处理。", "",
         "## 1 各层判定", "", "| 层 | 总体 | 样本 | 判定 | 确实处理的比例（95% 区间） | 其中不是客户问题的段落 |", "|---|---|---|---|---|---|"]
    est, est_hi, nonreq = 0.0, 0.0, 0.0
    for s, hit, hit_hi in [("pasted", "Y", "YD"), ("via_doc", "Y", "YD"), ("none", "F", "FP")]:
        rs = [r for r in rows if sample[r["item"]] == s]
        n = len(rs)
        if not n:
            continue
        k, k_hi = sum(r["verdict"] in hit for r in rs), sum(r["verdict"] in hit_hi for r in rs)
        lo, hi = wilson(k, n)
        nr = sum(r["is_request"] == "N" for r in rs)
        est += pop[s] * k / n
        est_hi += pop[s] * k_hi / n
        nonreq += pop[s] * nr / n
        cnt = "，".join(f"{v} {c}" for v, c in sorted(Counter(r["verdict"] for r in rs).items()))
        L.append(f"| {s} | {pop[s]} | {n} | {cnt} | {k / n:.0%}（{lo:.0%}–{hi:.0%}）；含 {hit_hi[1]} 为 {k_hi / n:.0%} | {nr} |")
    N = len(trace)
    L += ["", "## 2 加权推回全部条目", "",
          f"- 被会话处理过的条目估计 {est:.0f} 条（{est / N:.0%}）；把 D、P 也算上为 {est_hi:.0f} 条（{est_hi / N:.0%}）。"
          f"字面规则（原文贴入）追到 {pop['pasted']} 条（{pop['pasted'] / N:.0%}）。",
          f"- 不是客户问题或要求的段落（标题、链接、说明等）估计 {nonreq:.0f} 条（{nonreq / N:.0%}）。",
          "- 各层样本只有 6–20 条，区间很宽，只看量级。"]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{OUT} problems: {problems or 'none'}")


if __name__ == "__main__":
    main()
