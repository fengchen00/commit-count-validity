"""S03-T1 分层抽样。

分层：每个片段按 PRIORITY 取其候选模式标签中排最前的一个作为层（多标签片段只进一层）。
配额：目标 TOTAL 个；层权重 = 层总体数 × 加权系数（S02-T5 登记的不稳定模式 ×2）；
     配额 = max(MIN_PER, 按权重比例取整)，不超过层总体数；超出 TOTAL 时从配额最大的层逐个减回。
层内：按片段 ID 排序后用种子 SEED 简单随机抽取。
试编 10 个：从每个非空层轮流各取 1 个（层顺序按 PRIORITY），种子 SEED。
盲评候选 20 个：从正式组（其余 40 个）中用种子 BLIND_SEED 简单随机抽取。

输出：data/coding/sample_S03.csv（片段 ID、系统、层、组别、是否盲评候选；不含内容）；
     <PAPER_DIR>/results/s03_sampling.md（各层总体数与抽样数）。
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

SEED, BLIND_SEED = "S03-20260929", "S05-20260929"
TOTAL, MIN_PER, N_PILOT, N_BLIND = 50, 3, 10, 20
PRIORITY = ["缺失", "顺序压平", "随他项提交", "合并-跨会话", "合并-同会话", "拆分-非逐条", "拆分-逐条任务",
            "部分未提交", "含未跟踪文件", "一对一", "无新增内容", "证据不足", "无代码"]
UNSTABLE = {"缺失", "随他项提交", "合并-跨会话", "拆分-非逐条", "拆分-逐条任务"}


def stratum(labels):
    return next(p for p in PRIORITY if p in labels)


def allocate(sizes):
    w = {s: n * (2 if s in UNSTABLE else 1) for s, n in sizes.items()}
    tw = sum(w.values())
    q = {s: min(sizes[s], max(MIN_PER, round(TOTAL * w[s] / tw))) for s in sizes}
    while sum(q.values()) > TOTAL:
        s = max((s for s in q if q[s] > MIN_PER), key=lambda s: q[s])
        q[s] -= 1
    return q


def main():
    pop = []
    for sysname in ["SYS_A", "SYS_C"]:
        for r in csv.DictReader(open(f"data/derived/{sysname}/episode_patterns.csv", encoding="utf-8")):
            pop.append({"episode": r["episode"], "system": sysname, "evidence": r["evidence"],
                        "stratum": stratum(r["patterns"].split(";"))})
    by = defaultdict(list)
    for p in pop:
        by[p["stratum"]].append(p)
    sizes = {s: len(by[s]) for s in PRIORITY if by[s]}
    quota = allocate(sizes)

    rng = random.Random(SEED)
    sample = {}
    for s in PRIORITY:
        if s in quota:
            members = sorted(by[s], key=lambda p: p["episode"])
            sample[s] = rng.sample(members, quota[s])

    pilot = []
    queues = {s: list(v) for s, v in sample.items()}
    rng_p = random.Random(SEED)
    for q in queues.values():
        rng_p.shuffle(q)
    while len(pilot) < N_PILOT:
        for s in PRIORITY:
            if s in queues and queues[s] and len(pilot) < N_PILOT:
                pilot.append(queues[s].pop()["episode"])
    formal = sorted(p["episode"] for v in sample.values() for p in v if p["episode"] not in pilot)
    blind = set(random.Random(BLIND_SEED).sample(formal, N_BLIND))

    rows = []
    for s in PRIORITY:
        for p in sorted(sample.get(s, []), key=lambda p: p["episode"]):
            rows.append({"episode": p["episode"], "system": p["system"], "stratum": s, "evidence": p["evidence"],
                         "group": "pilot" if p["episode"] in pilot else "formal", "blind_candidate": int(p["episode"] in blind)})
    assert len(rows) == TOTAL and len({r["episode"] for r in rows}) == TOTAL
    assert sum(r["group"] == "pilot" for r in rows) == N_PILOT
    assert sum(r["blind_candidate"] for r in rows) == N_BLIND and all(r["group"] == "formal" for r in rows if r["blind_candidate"])

    out = Path("data/coding/sample_S03.csv")
    if out.exists() and "--force" not in sys.argv:
        old = list(csv.DictReader(open(out, encoding="utf-8")))
        same = [dict(r, blind_candidate=str(r["blind_candidate"])) for r in rows] == old
        print(f"{out} exists (identical to rerun: {same}); not overwritten. Use --force before coding starts only.")
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sysc = Counter((r["stratum"], r["system"]) for r in rows)
    popc = Counter((p["stratum"], p["system"]) for p in pop)
    L = ["# S03 分层抽样", "",
         f"由 `analysis/sample_episodes.py` 生成（种子 `{SEED}`，盲评候选种子 `{BLIND_SEED}`）。"
         f"总体为两个系统全部工作片段（{len(pop)} 个）；每个片段按优先顺序取一个层；"
         f"S02-T5 登记的不稳定模式（{'、'.join(p for p in PRIORITY if p in UNSTABLE)}）权重加倍；每层至少 {MIN_PER} 个。", "",
         "| 层 | 总体（SYS_A / SYS_C） | 抽样（SYS_A / SYS_C） | 试编 | 盲评候选 |", "|---|---|---|---|---|"]
    for s in PRIORITY:
        if s not in sizes:
            continue
        rs = [r for r in rows if r["stratum"] == s]
        L.append(f"| {s} | {sizes[s]}（{popc[(s, 'SYS_A')]} / {popc[(s, 'SYS_C')]}） | {len(rs)}（{sysc[(s, 'SYS_A')]} / {sysc[(s, 'SYS_C')]}） | "
                 f"{sum(r['group'] == 'pilot' for r in rs)} | {sum(r['blind_candidate'] for r in rs)} |")
    L += [f"| 合计 | {len(pop)} | {len(rows)} | {N_PILOT} | {N_BLIND} |", "",
          f"层总体不足 {MIN_PER} 个时全部抽取。样本中 SYS_C 占 {sum(r['system'] == 'SYS_C' for r in rows)}/{len(rows)}，"
          f"总体中占 {sum(p['system'] == 'SYS_C' for p in pop)}/{len(pop)}。", "",
          "说明：层是抽样用的，不是结论。人工编码时看的是完整证据，可以给出与自动候选不同的模式。"
          "估计总体比例时按层反加权（权重 = 层总体数 / 层抽样数）。"]
    (RESULTS_DIR / "s03_sampling.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out, dict(Counter(r["stratum"] for r in rows)))


if __name__ == "__main__":
    main()
