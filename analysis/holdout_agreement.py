"""S12 留出样本独立编码的汇总（D12；AI 模拟编码，不是人工编码）。

输入（本地，不入库）：data/coding/holdout/sample_S12.csv；data/coding/sim/sim4_h1_part*.csv、sim4_h2_part*.csv、
     sim4_h3_adjudicate.csv（裁决）。
用法：python analysis/holdout_agreement.py disputes   生成裁决人用的匿名对照表 data/coding/holdout/adjudication_input.csv
     python analysis/holdout_agreement.py            写 <PAPER_DIR>/results/r9_holdout.md 与 numbers.json 的 holdout 键
                                                     （须在 render_tables 之后运行）
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sim_agreement import CODE, COARSE, NAME, SIM, auto_main, cohen, final_reference, load_sheet  # noqa: E402
import agreement  # noqa: E402

SAMPLE = Path("data/coding/holdout/sample_S12.csv")
ADJ_IN = Path("data/coding/holdout/adjudication_input.csv")
OUT = (RESULTS_DIR / "r9_holdout.md")
NUMS = (RESULTS_DIR / "numbers.json")
HUMAN = os.environ.get("CODING_HUMAN", "0") == "1"  # D29
H1, H2, H3 = ((("holdout_coder1.csv",), ("holdout_coder2.csv",), "holdout_adjudicate.csv") if HUMAN else
              (("sim4_h1_part1.csv", "sim4_h1_part2.csv"), ("sim4_h2_part1.csv", "sim4_h2_part2.csv"), "sim4_h3_adjudicate.csv"))
RULES = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
W_CODES = ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A9", "A10", "U", "X"]
W_DRAWS, SEED = 4000, "S12-T3"


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def auto_labels():
    auto = {}
    for s in ["SYS_A", "SYS_C"]:
        for r in load(f"data/derived/{s}/episode_patterns.csv"):
            labels = r["patterns"].split(";")
            auto[r["episode"]] = {"labels": {CODE[x] for x in labels if x in CODE}, "main": auto_main(labels),
                                  "evidence": r["evidence"]}
    return auto


def coders():
    if not all((SIM / n).exists() for n in H1 + H2):
        return None, None, []
    (h1, p1), (h2, p2) = load_sheet(*H1), load_sheet(*H2)
    return h1, h2, p1 + p2


def disputed(h1, h2):
    return sorted(e for e in h1 if e in h2 and (h1[e]["main"] != h2[e]["main"] or h1[e]["sec"] != h2[e]["sec"]))


def holdout_reference():
    """参照 = 两人主模式与次模式都一致的编码，否则取裁决结果。缺编码表或裁决不全时返回 None。"""
    h1, h2, _ = coders()
    if h1 is None or not (SIM / H3).exists():
        return None
    adj, _ = load_sheet(H3)
    dis = set(disputed(h1, h2))
    if not dis <= set(adj):
        return None
    return {e: (adj[e] if e in dis else h1[e]) for e in h1}


def write_disputes():
    h1, h2, problems = coders()
    assert h1 is not None and not problems, problems
    sample = [r["episode"] for r in load(SAMPLE)]
    assert set(h1) == set(h2) == set(sample), "编码表与样本不一致"
    dis = disputed(h1, h2)
    rng = random.Random("S12-adjudicate")
    raw = {}
    for who, names in (("h1", H1), ("h2", H2)):
        for n in names:
            for r in csv.DictReader(open(SIM / n, encoding="utf-8-sig")):
                raw[(who, r["episode"].strip())] = r
    with open(ADJ_IN, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        cols = ["main_pattern", "secondary_patterns", "causes", "confidence", "evidence_sufficient", "missing_info", "notes"]
        w.writerow(["episode"] + [f"{x}_{c}" for x in ("a", "b") for c in cols])
        for e in dis:
            pair = [raw[("h1", e)], raw[("h2", e)]]
            rng.shuffle(pair)
            w.writerow([e] + [p[c] for p in pair for c in cols])
    print(f"disputed {len(dis)} / {len(h1)}: {','.join(dis)}")


def weighted_share(ref, strata, pop, c, tag):
    """按层反加权的"含该模式的改代码片段 / 改代码片段"；区间为分层 Dirichlet 后验（同 D9）。"""
    def has(e):
        return ref[e]["main"] == c or c in ref[e]["sec"]
    es_all = list(ref)
    k = [sum(ref[e]["main"] != "N0" and has(e) for e in es_all), sum(ref[e]["main"] != "N0" and not has(e) for e in es_all),
         sum(ref[e]["main"] == "N0" for e in es_all)]
    pr = [max(v / len(es_all), 1e-3) for v in k]
    cnt = []
    for h, N in pop.items():
        es = [e for e in ref if strata[e] == h]
        if es:
            cnt.append((N, sum(ref[e]["main"] != "N0" and has(e) for e in es),
                        sum(ref[e]["main"] != "N0" and not has(e) for e in es), sum(ref[e]["main"] == "N0" for e in es), len(es)))
    point = sum(N * a / n for N, a, _, _, n in cnt) / sum(N * (a + b) / n for N, a, b, _, n in cnt)
    rng = random.Random(f"{SEED}|{tag}|{c}")
    draws = []
    for _ in range(W_DRAWS):
        nu, de = 0.0, 0.0
        for N, a, b, z, _ in cnt:
            g = [rng.gammavariate(x + q, 1) for x, q in zip((a, b, z), pr)]
            t = sum(g)
            nu += N * g[0] / t
            de += N * (g[0] + g[1]) / t
        draws.append(nu / de)
    draws.sort()
    return {"p": point, "ci": [draws[int(.025 * W_DRAWS)], draws[int(.975 * W_DRAWS) - 1]]}


def agree(auto, ref, eps):
    a, b = [auto[e]["main"] for e in eps], [ref[e]["main"] for e in eps]
    po, k = cohen(a, b)
    po2, k2 = cohen([COARSE[x] for x in a], [COARSE[x] for x in b])
    return {"n": len(eps), "fine": po, "kappa": k, "coarse": po2, "kappa_coarse": k2}


def confirm(auto, ref, c):
    aa = [e for e in ref if c in auto[e]["labels"]]
    hh = [e for e in ref if ref[e]["main"] == c or c in ref[e]["sec"]]
    return {"k": sum(ref[e]["main"] == c or c in ref[e]["sec"] for e in aa), "n": len(aa),
            "rk": sum(c in auto[e]["labels"] for e in hh), "rn": len(hh)}


def frac(k, n):
    return f"{k}/{n}（{k / n:.0%}）" if n else "—"


def main():
    h1, h2, problems = coders()
    ref = holdout_reference()
    if ref is None:
        print("holdout coding incomplete; skipped")
        return
    sample = {r["episode"]: r for r in load(SAMPLE)}
    assert set(ref) == set(sample), "参照与样本不一致"
    adj, _ = load_sheet(H3)
    auto = auto_labels()
    orig = final_reference("sim2")
    eps = sorted(sample)
    ver = [e for e in eps if auto[e]["evidence"] == "verified"]
    dis = disputed(h1, h2)
    num = {}

    L = ([ "# 留出样本独立编码汇总（真人编码；D12、D29）", "",
          "> 甲（H1，未参与本项目的独立编码人）与开发者（H2）各自盲编 50 个留出片段，"
          "分歧由第二作者（D30-1）依据盲评证据包与两份匿名编码表裁决；两名编码人看不到自动候选，第二作者裁决时看不到答案来源。"] if HUMAN else
         ["# 留出样本独立编码汇总（AI 模拟编码，不是人工编码；D12）", "",
         "> 两名模拟独立编码人（H1 Claude Sonnet 5.5，H2 Gemini 3.8 Flash）各自盲编 50 个留出片段，"
         "分歧由模拟裁决人（H3 GPT 5.6 Sol）依据盲评证据包与两份匿名编码表裁决；三者都看不到自动候选。"]) + [
         "自动候选为 `S12-freeze` 冻结时的结果。", "",
         "由 `analysis/holdout_agreement.py` 生成；只含汇总。", "",
         f"- 样本 {len(eps)}（SYS_A {sum(e.startswith('SYS_A') for e in eps)}，SYS_C {sum(e.startswith('SYS_C') for e in eps)}；"
         f"自动判为可核验 {len(ver)}）；校验问题：{'；'.join(problems) if problems else '无'}。", ""]

    po, k = cohen([h1[e]["main"] for e in eps], [h2[e]["main"] for e in eps])
    po2, k2 = cohen([COARSE[h1[e]["main"]] for e in eps], [COARSE[h2[e]["main"]] for e in eps])
    main_dis = [e for e in dis if h1[e]["main"] != h2[e]["main"]]
    side = Counter("同 H1" if adj[e]["main"] == h1[e]["main"] else "同 H2" if adj[e]["main"] == h2[e]["main"] else "另判"
                   for e in main_dis)
    st_f = agreement.kappa_stats([h1[e]["main"] for e in eps], [h2[e]["main"] for e in eps], "holdout|fine")
    st_c = agreement.kappa_stats([COARSE[h1[e]["main"]] for e in eps], [COARSE[h2[e]["main"]] for e in eps], "holdout|coarse")
    num["coders"] = {"fine": po, "kappa": k, "coarse": po2, "kappa_coarse": k2, "n": len(eps),
                     "disputed": len(dis), "main_disputed": len(main_dis), "adjudication": dict(side),
                     "fine_stats": st_f, "coarse_stats": st_c}
    L += ["## 1 两名编码人之间的一致性", "",
          f"主模式细类一致 {po:.0%}（κ = {k:.2f}），粗类一致 {po2:.0%}（κ = {k2:.2f}），n = {len(eps)}。"
          f"主模式或次模式不同、交裁决的 {len(dis)} 个，其中主模式不同 {len(main_dis)} 个；"
          f"裁决的主模式：{'，'.join(f'{a} {b} 个' for a, b in side.most_common()) or '无'}。", "",
          f"- 细类完整报告（D26-5）：{agreement.fmt(st_f, 'H1', 'H2')}。",
          f"- 粗类完整报告：{agreement.fmt(st_c, 'H1', 'H2')}。", ""]

    L += ["## 2 自动候选与留出参照比对", "", "| 参照 | 范围 | n | 细类一致 | 细类 κ | 粗类一致 | 粗类 κ |", "|---|---|---|---|---|---|---|"]
    rows = [("留出参照", "全部", ref, eps), ("留出参照", "可核验", ref, ver), ("H1 单独", "全部", h1, eps), ("H2 单独", "全部", h2, eps),
            ("原样本参照（50）", "全部", orig, sorted(orig)),
            ("原样本参照（50）", "可核验", orig, [e for e in sorted(orig) if auto[e]["evidence"] == "verified"])]
    extra = []
    for name, scope, r, es in rows:
        a = agree(auto, r, es)
        sf = agreement.kappa_stats([auto[e]["main"] for e in es], [r[e]["main"] for e in es], f"auto|{name}|{scope}|fine")
        sc = agreement.kappa_stats([COARSE[auto[e]["main"]] for e in es], [COARSE[r[e]["main"]] for e in es], f"auto|{name}|{scope}|coarse")
        a["fine_stats"], a["coarse_stats"] = sf, sc
        num.setdefault("auto", {})[f"{name}|{scope}"] = a
        L.append(f"| {name} | {scope} | {a['n']} | {a['fine']:.0%} | {a['kappa']:.2f}（{sf['ci'][0]:.2f}–{sf['ci'][1]:.2f}） | {a['coarse']:.0%} | {a['kappa_coarse']:.2f}（{sc['ci'][0]:.2f}–{sc['ci'][1]:.2f}） |")
        if name in ("留出参照", "原样本参照（50）") and scope == "全部":
            extra += [f"- {name}·{scope} 细类：{agreement.fmt(sf, '自动', '参照')}。", f"- {name}·{scope} 粗类：{agreement.fmt(sc, '自动', '参照')}。"]
    L += ["", "括号内为 κ 的 95% 自助区间（按片段重抽 2,000 次，D26-5）。", ""] + extra

    L += ["", "## 3 逐模式确认率", "",
          "确认率 = 自动标出该模式的片段中参照也标了（主或次）的比例；召回 = 参照标了的片段中自动也标了的比例。", "",
          "| 代码 | 模式 | 留出：确认 | 留出：召回 | 原样本：确认 | 合并 100 个：确认 |", "|---|---|---|---|---|---|"]
    for c in RULES:
        h, o = confirm(auto, ref, c), confirm(auto, orig, c)
        num.setdefault("confirmation", {})[c] = {"holdout": h, "original": o}
        if h["n"] or h["rn"] or o["n"]:
            L.append(f"| {c} | {NAME[c]} | {frac(h['k'], h['n'])} | {frac(h['rk'], h['rn'])} | {frac(o['k'], o['n'])} | "
                     f"{frac(h['k'] + o['k'], h['n'] + o['n'])} |")

    cols = ["进入版本库", "未进入", "无代码或无新增", "证据不足", "其他"]
    cm = Counter((COARSE[auto[e]["main"]], COARSE[ref[e]["main"]]) for e in eps)
    L += ["", "粗类混淆矩阵（行 = 自动，列 = 留出参照）：", "", "| 自动 \\ 参照 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for rn in cols:
        if any(cm[(rn, c)] for c in cols):
            L.append(f"| {rn} | " + " | ".join(str(cm[(rn, c)]) for c in cols) + " |")

    pop = {}
    for line in (RESULTS_DIR / "s12_holdout_sampling.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\S+) \| (\d+)（", line)
        if m:
            pop[m.group(1)] = int(m.group(2))
    strata = {e: r["stratum"] for e, r in sample.items()}
    orig_w = json.loads(NUMS.read_text(encoding="utf-8")).get("weighted", {}).get("合计", {})
    L += ["", "## 4 按层反加权的模式比例（留出参照，推回留出总体的改代码片段）", "",
          f"留出总体 {sum(pop.values())} 个片段；可多标签；括号内为分层 Dirichlet 后验 95% 区间（同 D9）。原估计为 50 个原样本推回全部片段（两个系统合计）。", "",
          "| 模式 | 留出估计 | 原估计 |", "|---|---|---|"]
    for c in W_CODES:
        w = weighted_share(ref, strata, pop, c, "holdout")
        num.setdefault("weighted", {})[c] = w
        o = orig_w.get(c)
        L.append(f"| {NAME[c]} | {w['p']:.0%}（{w['ci'][0]:.0%}–{w['ci'][1]:.0%}） | "
                 + (f"{o['p']:.0%}（{o['ci'][0]:.0%}–{o['ci'][1]:.0%}） |" if o else "— |"))

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    allnum = json.loads(NUMS.read_text(encoding="utf-8"))
    allnum["holdout"] = num
    NUMS.write_text(json.dumps(allnum, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    if sys.argv[1:] == ["disputes"]:
        write_disputes()
    else:
        main()
