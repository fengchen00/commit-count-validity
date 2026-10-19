"""D6 模拟编码预演的汇总：一致性、自动候选与模拟编码比对、按层加权的模式比例。

输入（本地，不入库）：data/coding/sim/sim_author_part1.csv、sim_author_part2.csv、sim_wang.csv、sim_zhang.csv
输出：<PAPER_DIR>/results/sim_coding.md（只含汇总，不列逐条结果；D6 要求转人工前只能看汇总）
全部数字是 AI 模拟编码，不是人工编码。
"""
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sample_episodes import stratum  # noqa: E402
import agreement  # noqa: E402

HUMAN = os.environ.get("CODING_HUMAN", "0") == "1"  # D29：真人编码数据在 data/coding/human/
SIM = Path("data/coding/human") if HUMAN else Path("data/coding/sim")
OUT = (RESULTS_DIR / "sim_coding.md")
MAIN = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9", "N0", "N1", "U", "X"]
SECONDARY = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"}
CAUSES = [f"C{i}" for i in range(1, 12)] + ["CX"]
NAME = {"A1": "一对一", "A2": "合并-同会话", "A3": "合并-跨会话", "A4": "拆分-逐条任务", "A5": "拆分-非逐条",
        "A6": "随他项提交", "A7": "顺序压平", "A8": "部分未提交", "A9": "缺失", "A10": "含未跟踪文件",
        "N0": "无代码", "N1": "无新增内容", "U": "证据不足", "X": "其他"}
CODE = {v: k for k, v in NAME.items()}
AUTO_PRIORITY = ["A9", "A6", "A3", "A2", "A7", "A5", "A4", "A1"]
BOOT = 2000
SEED_BOOT = "S10-boot"
RECHECK = {"sim2": ("sim3_recheck.csv", "sim3_recheck_part2.csv")}
COARSE = {**{c: "进入版本库" for c in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]},
          "A9": "未进入", "N0": "无代码或无新增", "N1": "无代码或无新增", "U": "证据不足", "X": "其他"}


def split(s):
    return [x.strip() for x in (s or "").replace("；", ";").replace(",", ";").split(";") if x.strip()]


def load_sheet(*names):
    rows, problems = {}, []
    for n in names:
        p = SIM / n
        if not p.exists():
            problems.append(f"{n} 不存在")
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            ep = r["episode"].strip()
            m = r["main_pattern"].strip().upper()
            if m not in MAIN:
                problems.append(f"{n}: 主模式代码无效 {m!r}")
            bad = [x for x in split(r["secondary_patterns"]) if x.upper() not in SECONDARY]
            bad += [x for x in split(r["causes"]) if x.upper() not in CAUSES]
            if bad:
                problems.append(f"{n}: 次模式或成因代码无效 {bad}")
            if ep in rows:
                problems.append(f"{n}: 重复片段")
            rows[ep] = {"main": m, "sec": {x.upper() for x in split(r["secondary_patterns"])},
                        "causes": {x.upper() for x in split(r["causes"])}, "conf": r["confidence"].strip(),
                        "suff": r["evidence_sufficient"].strip().upper(), "x": m == "X"}
    return rows, problems


def cohen(a, b):
    n = len(a)
    if not n:
        return float("nan"), float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(table):
    cats = sorted({c for row in table for c in row})
    n, k = len(table), len(table[0])
    pj = {c: sum(row.count(c) for row in table) / (n * k) for c in cats}
    pi = [(sum(row.count(c) ** 2 for c in cats) - k) / (k * (k - 1)) for row in table]
    pbar, pe = sum(pi) / n, sum(v * v for v in pj.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else 1.0


def auto_main(labels):
    codes = {CODE[x] for x in labels if x in CODE}
    for c in ["N0", "N1", "U"]:
        if c in codes:
            return c
    for c in AUTO_PRIORITY:
        if c in codes:
            return c
    return "X"


def final_reference(P="sim2"):
    """人工参照（审定结果；D3e 后证据包变化的片段改用复核结果），与 main() 第 7 节同口径。无审定表时返回 None。
    CODING_HUMAN=1 时（D29）参照为作者的真人编码 human_author_part1/2.csv，不经审定与复核。"""
    if HUMAN:
        ref, _ = load_sheet("human_author_part1.csv", "human_author_part2.csv")
        return ref or None
    if not (SIM / f"{P}_review.csv").exists():
        return None
    review, _ = load_sheet(f"{P}_review.csv")
    rc_names = [n for n in RECHECK.get(P, ()) if (SIM / n).exists()]
    if not rc_names:
        return review
    recheck, _ = load_sheet(*rc_names)
    old_dir, new_dir = Path("data/coding/packets_blind"), Path("data/coding/recheck_d3e/packets_blind")

    def evidence(p):
        return [x for x in p.read_text(encoding="utf-8").splitlines() if not x.startswith("本片段有检查点差分")]
    moved = {e for e in review if (new_dir / f"{e}.md").exists()
             and evidence(old_dir / f"{e}.md") != evidence(new_dir / f"{e}.md")}
    return {e: (recheck[e] if e in moved and e in recheck else r) for e, r in review.items()}


def main():
    P = sys.argv[1] if len(sys.argv) > 1 else ("human" if HUMAN else "sim")
    out = OUT.with_name(f"{P}_coding.md")
    author, pa = load_sheet(f"{P}_author_part1.csv", f"{P}_author_part2.csv")
    wang, pw = load_sheet(f"{P}_wang.csv")
    zhang, pz = load_sheet(f"{P}_zhang.csv") if (SIM / f"{P}_zhang.csv").exists() else ({}, [])  # D29：第三表可缺
    review, decision = {}, {}
    if (SIM / f"{P}_review.csv").exists():
        review, pr = load_sheet(f"{P}_review.csv")
        pa += pr
        decision = {r["episode"]: r["decision"].strip() for r in csv.DictReader(open(SIM / f"{P}_review.csv", encoding="utf-8-sig"))}
    sample = {r["episode"]: r for r in csv.DictReader(open("data/coding/sample_S03.csv", encoding="utf-8"))}
    auto = {}
    for s in ["SYS_A", "SYS_C"]:
        for r in csv.DictReader(open(f"data/derived/{s}/episode_patterns.csv", encoding="utf-8")):
            labels = r["patterns"].split(";")
            auto[r["episode"]] = {"labels": {CODE[x] for x in labels if x in CODE}, "main": auto_main(labels),
                                  "evidence": r["evidence"], "stratum": stratum(labels)}
    pop = {}
    for line in (RESULTS_DIR / "s03_sampling.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\S+) \| (\d+)（", line)
        if m:
            pop[m.group(1)] = int(m.group(2))
    moved = sum(auto[e]["stratum"] != r["stratum"] for e, r in sample.items())
    blind = sorted(e for e, r in sample.items() if r["blind_candidate"] == "1")
    problems = pa + pw + pz
    problems += [f"作者缺 {len(set(sample) - set(author))} 个样本"] if set(sample) - set(author) else []
    problems += [f"王缺 {len(set(blind) - set(wang))} 个盲评样本"] if set(blind) - set(wang) else []
    problems += [f"张缺 {len(set(blind) - set(zhang))} 个盲评样本"] if zhang and set(blind) - set(zhang) else []

    WHO = ("作者", "甲", "乙") if HUMAN else ("模拟作者", "模拟王", "模拟张")
    L = (["# 原样本 50 的人工编码汇总（真人编码，D29）", "",
          "> 作者编码 50 个原样本；甲、乙（未参与本项目）各盲编其中 20 个。三人只看证据包与编码手册，互不可见。", ""] if HUMAN else
         ["# 模拟编码预演汇总（AI 模拟编码，不是人工编码）", "",
         "> 本文件全部数字来自 AI 子代理模拟的三名编码人（D6）。它们只看盲评证据包与编码手册，互不可见。",
         "> 用途：在投入人工前估计分类框架是否可编、自动候选是否可信。", ""]) + [
         "由 `analysis/sim_agreement.py` 生成；只含汇总。", "",
         f"- 样本 {len(sample)}（盲评 {len(blind)}）；已收：{WHO[0]} {len(author)}、{WHO[1]} {len(wang)}、{WHO[2]} {len(zhang)}。",
         f"- 校验问题：{'；'.join(problems) if problems else '无'}。",
         ("- 证据包为 D3c 修正后重建，手册 v2.6（冻结版）；参照为作者编码，不经审定。" if HUMAN else
          f"- 模拟编码用的是 D3c 修正前生成的证据包；修正后 50 个样本中有 {moved} 个的自动层发生变化。" if P == "sim" else
          f"- 第二轮：证据包按 D3c 重建，手册 v2.1；不同模型分别模拟不同编码人（见 D6 第二轮）。"
          "审查人可看全部证据包与三份编码表，不属于盲评。")
         + "第 3 节与当前（修正后）的自动候选比对；第 4 节的权重用抽样时的层（`s03_sampling.md`）。", ""]

    L += ["## 1 主模式分布", "", f"| 代码 | 模式 | {WHO[0]}（50） | {WHO[1]}（20） | {WHO[2]}（20） | 自动候选（50） |", "|---|---|---|---|---|---|"]
    ca, cw, cz = Counter(r["main"] for r in author.values()), Counter(r["main"] for r in wang.values()), Counter(r["main"] for r in zhang.values())
    cauto = Counter(auto[e]["main"] for e in sample)
    for c in MAIN:
        if ca[c] or cw[c] or cz[c] or cauto[c]:
            L.append(f"| {c} | {NAME[c]} | {ca[c]} | {cw[c]} | {cz[c]} | {cauto[c]} |")
    conf = Counter(r["conf"] for r in author.values())
    L += ["", f"{WHO[0]}置信度：1 = {conf['1']}，2 = {conf['2']}，3 = {conf['3']}；"
          f"证据不充分（N）{sum(r['suff'] == 'N' for r in author.values())} 个；填 X {ca['X']} 个。", ""]

    L += ["## 2 编码人之间的一致性（盲评 20 个）", "", "| 比较 | 细类 一致率 | 细类 κ | 粗类 一致率 | 粗类 κ | n |", "|---|---|---|---|---|---|"]
    pairs = [(f"{WHO[1]} vs {WHO[2]}", wang, zhang), (f"{WHO[0]} vs {WHO[1]}", author, wang), (f"{WHO[0]} vs {WHO[2]}", author, zhang)]
    pairs = [t for t in pairs if t[1] and t[2]]
    for name, x, y in pairs:
        eps = [e for e in blind if e in x and e in y]
        a, b = [x[e]["main"] for e in eps], [y[e]["main"] for e in eps]
        po, k = cohen(a, b)
        po2, k2 = cohen([COARSE[v] for v in a], [COARSE[v] for v in b])
        L.append(f"| {name} | {po:.0%} | {k:.2f} | {po2:.0%} | {k2:.2f} | {len(eps)} |")
    # D26-5：κ 的完整报告（区间、P_o、P_e、边缘分布），供 4.1.4 节引用
    L += ["", "一致性的完整报告（自助区间按片段重抽 2,000 次；粗类 κ 低于细类时多因粗类边缘分布偏斜，见边缘分布）：", ""]
    for name, x, y in pairs:
        eps = [e for e in blind if e in x and e in y]
        a, b = [x[e]["main"] for e in eps], [y[e]["main"] for e in eps]
        na, nb = name.split(" vs ")
        L.append(f"- {name} 细类：{agreement.fmt(agreement.kappa_stats(a, b, name + '|fine'), na, nb)}。")
        L.append(f"- {name} 粗类：{agreement.fmt(agreement.kappa_stats([COARSE[v] for v in a], [COARSE[v] for v in b], name + '|coarse'), na, nb)}。")
    three = [e for e in blind if e in author and e in wang and e in zhang]
    if three:
        fk = fleiss([[author[e]["main"], wang[e]["main"], zhang[e]["main"]] for e in three])
        fk2 = fleiss([[COARSE[author[e]["main"]], COARSE[wang[e]["main"]], COARSE[zhang[e]["main"]]] for e in three])
        L += ["", f"三人 Fleiss κ：细类 {fk:.2f}，粗类 {fk2:.2f}（n = {len(three)}）。"]
    L += ["", "粗类：进入版本库（A1–A7）/ 未进入（A9）/ 无代码或无新增（N0、N1）/ 证据不足（U）/ 其他（X）。", "",
          f"次模式与成因（{WHO[1]} vs {WHO[2]}，逐代码二值 κ，只列至少一人用过的代码）：", "",
          f"| 代码 | {WHO[1]}用 | {WHO[2]}用 | 一致率 | κ |", "|---|---|---|---|---|"]
    eps = [e for e in blind if e in wang and e in zhang] if zhang else []
    for c in ["A8", "A10"] + CAUSES:
        a = [c in (wang[e]["sec"] | wang[e]["causes"]) for e in eps]
        b = [c in (zhang[e]["sec"] | zhang[e]["causes"]) for e in eps]
        if any(a) or any(b):
            po, k = cohen(a, b)
            L.append(f"| {c} | {sum(a)} | {sum(b)} | {po:.0%} | {k:.2f} |")

    def ref_sections(ref, who, n3, n4, n5):
        nonlocal L
        L += ["", f"## {n3} 自动候选与{who}比对（50 个）", ""]
        eps = [e for e in sample if e in ref]
        po, k = cohen([auto[e]["main"] for e in eps], [ref[e]["main"] for e in eps])
        po2, k2 = cohen([COARSE[auto[e]["main"]] for e in eps], [COARSE[ref[e]["main"]] for e in eps])
        vf = [e for e in eps if auto[e]["evidence"] == "verified"]
        po3, k3 = cohen([auto[e]["main"] for e in vf], [ref[e]["main"] for e in vf])
        L += [f"主模式：细类一致 {po:.0%}（κ = {k:.2f}），粗类一致 {po2:.0%}（κ = {k2:.2f}）；"
              f"只看 verified 片段（{len(vf)} 个）细类一致 {po3:.0%}（κ = {k3:.2f}）。", "",
              f"逐模式：自动标出该模式的片段中，{who}也标了（主或次）的比例；以及{who}标了的片段中，自动也标了的比例。", "",
              f"| 代码 | 模式 | 自动标出 | 其中{who}确认 | {who}标出 | 其中自动也标 |", "|---|---|---|---|---|---|"]
        for c in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "N0", "N1", "U"]:
            aa = [e for e in eps if c in auto[e]["labels"]]
            hh = [e for e in eps if c == ref[e]["main"] or c in ref[e]["sec"]]
            if aa or hh:
                ok1 = sum(c == ref[e]["main"] or c in ref[e]["sec"] for e in aa)
                ok2 = sum(c in auto[e]["labels"] for e in hh)
                L.append(f"| {c} | {NAME[c]} | {len(aa)} | {ok1}（{ok1 / len(aa):.0%}） | {len(hh)} | "
                         f"{ok2}（{ok2 / len(hh):.0%}） |" if aa and hh else
                         f"| {c} | {NAME[c]} | {len(aa)} | {ok1} | {len(hh)} | {ok2} |")
        cm = Counter((COARSE[auto[e]["main"]], COARSE[ref[e]["main"]]) for e in eps)
        cols = ["进入版本库", "未进入", "无代码或无新增", "证据不足", "其他"]
        L += ["", f"粗类混淆矩阵（行 = 自动，列 = {who}）：", "", f"| 自动 \\ {who} | " + " | ".join(cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
        for rname in cols:
            if any(cm[(rname, c)] for c in cols):
                L.append(f"| {rname} | " + " | ".join(str(cm[(rname, c)]) for c in cols) + " |")

        L += ["", f"## {n4} 按层加权的模式比例（{who}，推回 701 个片段）", "",
              f"权重 = 层总体数 / 层样本数。样本小，只看量级。区间为层内有放回重抽 {BOOT} 次的 2.5% 与 97.5% 分位数。", "",
              "| 代码 | 模式 | 加权占全部片段 | 加权占改代码片段 | 改代码片段 95% 区间 |", "|---|---|---|---|---|"]
        ns = Counter(sample[e]["stratum"] for e in eps)
        w = {e: pop[sample[e]["stratum"]] / ns[sample[e]["stratum"]] for e in eps}
        tot = sum(w.values())

        def has(e, c):
            return ref[e]["main"] == c or c in ref[e]["sec"]

        def code_share(es, c):
            code = [e for e in es if ref[e]["main"] != "N0"]
            den = sum(pop[sample[e]["stratum"]] / ns[sample[e]["stratum"]] for e in code)
            return sum(pop[sample[e]["stratum"]] / ns[sample[e]["stratum"]] for e in code if has(e, c)) / den if den else 0.0

        by_st = defaultdict(list)
        for e in eps:
            by_st[sample[e]["stratum"]].append(e)
        rng = random.Random(SEED_BOOT)
        boots = [[x for es in by_st.values() for x in rng.choices(es, k=len(es))] for _ in range(BOOT)]
        for c in MAIN + ["A8", "A10"]:
            s1 = sum(w[e] for e in eps if has(e, c))
            if s1:
                bs = sorted(code_share(b, c) for b in boots)
                L.append(f"| {c} | {NAME[c]} | {s1 / tot:.0%} | {code_share(eps, c):.0%} | "
                         f"{bs[int(.025 * BOOT)]:.0%}–{bs[int(.975 * BOOT) - 1]:.0%} |")

        L += ["", f"## {n5} 成因分布（{who}，未加权）", "", "| 成因 | 片段数 |", "|---|---|"]
        cc = Counter(c for r in ref.values() for c in r["causes"])
        for c in CAUSES:
            if cc[c]:
                L.append(f"| {c} | {cc[c]} |")

    ref_sections(author, WHO[0], 3, 4, 5)
    if review:
        changed = sum(review[e]["main"] != author[e]["main"] for e in review if e in author)
        L += ["", "## 6 模拟作者审查（审定结果）", "",
              f"审查 {len(review)} 个；主模式改判 {changed} 个；审查结论：{dict(Counter(decision.values()))}。"
              "以下各节以审定结果为参照。", ""]
        ref_sections(review, "审定结果", "6.1", "6.2", "6.3")
    rc_names = [n for n in RECHECK.get(P, ()) if (SIM / n).exists()]
    if review and rc_names:
        recheck, prc = load_sheet(*rc_names)
        rc_name = "、".join(rc_names)
        problems += prc
        old_dir, new_dir = Path("data/coding/packets_blind"), Path("data/coding/recheck_d3e/packets_blind")
        def evidence(p):
            return [x for x in p.read_text(encoding="utf-8").splitlines() if not x.startswith("本片段有检查点差分")]
        moved_pk = {e for e in sample if (new_dir / f"{e}.md").exists()
                    and evidence(old_dir / f"{e}.md") != evidence(new_dir / f"{e}.md")}
        redo = sorted(set(recheck) & moved_pk)
        ctrl = sorted(set(recheck) - moved_pk)
        final = {e: (recheck[e] if e in moved_pk and e in recheck else r) for e, r in review.items()}
        stable = sum(recheck[e]["main"] == review[e]["main"] for e in ctrl)
        L += ["", "## 7 D3e 复核后的参照", "",
              f"检查点补读（D3e）后，50 个样本中 {len(moved_pk)} 个的盲评证据包内容变化，其中 {len(redo)} 个已由复核人按新证据包盲编"
              f"（`{rc_name}`），参照改用复核结果"
              + ("" if len(redo) == len(moved_pk) else f"；另 {len(moved_pk) - len(redo)} 个未复核，仍用审定结果") + "。"
              f"对照片段 {len(ctrl)} 个（证据包未变），复核主模式与审定结果一致 {stable} 个。", ""]
        ref_sections(final, "复核后参照", "7.1", "7.2", "7.3")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out, "problems:", problems or "none")


if __name__ == "__main__":
    main()
