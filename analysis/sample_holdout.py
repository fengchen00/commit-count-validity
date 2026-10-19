"""S12-T1 留出样本（D12）：从未进入 S03 样本的片段中，按与 S03 相同的分层与配额再抽 50 个。

输出：data/coding/holdout/sample_S12.csv（片段 ID、系统、层、证据层；不含内容）；
     <PAPER_DIR>/results/s12_holdout_sampling.md（各层总体数与抽样数）。
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sample_episodes import MIN_PER, PRIORITY, TOTAL, UNSTABLE, allocate, stratum  # noqa: E402

SEED = "S12-holdout"
OUT = Path("data/coding/holdout/sample_S12.csv")


def main():
    used = {r["episode"] for r in csv.DictReader(open("data/coding/sample_S03.csv", encoding="utf-8"))}
    pop = []
    for sysname in ["SYS_A", "SYS_C"]:
        for r in csv.DictReader(open(f"data/derived/{sysname}/episode_patterns.csv", encoding="utf-8")):
            if r["episode"] not in used:
                pop.append({"episode": r["episode"], "system": sysname, "evidence": r["evidence"],
                            "stratum": stratum(r["patterns"].split(";"))})
    by = defaultdict(list)
    for p in pop:
        by[p["stratum"]].append(p)
    sizes = {s: len(by[s]) for s in PRIORITY if by[s]}
    quota = allocate(sizes)
    rng = random.Random(SEED)
    rows = []
    for s in PRIORITY:
        if s in quota:
            for p in sorted(rng.sample(sorted(by[s], key=lambda p: p["episode"]), quota[s]), key=lambda p: p["episode"]):
                rows.append({"episode": p["episode"], "system": p["system"], "stratum": s, "evidence": p["evidence"]})
    assert len(rows) == TOTAL and not ({r["episode"] for r in rows} & used)

    if OUT.exists() and "--force" not in sys.argv:
        same = rows == list(csv.DictReader(open(OUT, encoding="utf-8")))
        print(f"{OUT} exists (identical to rerun: {same}); not overwritten.")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sysc = Counter((r["stratum"], r["system"]) for r in rows)
    popc = Counter((p["stratum"], p["system"]) for p in pop)
    L = ["# S12 留出样本抽样", "",
         f"由 `analysis/sample_holdout.py` 生成（种子 `{SEED}`）。总体为两个系统中未进入 S03 样本的片段（{len(pop)} 个）；"
         f"分层与配额同 S03：不稳定模式（{'、'.join(p for p in PRIORITY if p in UNSTABLE)}）权重加倍，每层至少 {MIN_PER} 个。", "",
         "| 层 | 总体（SYS_A / SYS_C） | 抽样（SYS_A / SYS_C） |", "|---|---|---|"]
    for s in PRIORITY:
        if s in sizes:
            n = sum(r["stratum"] == s for r in rows)
            L.append(f"| {s} | {sizes[s]}（{popc[(s, 'SYS_A')]} / {popc[(s, 'SYS_C')]}） | {n}（{sysc[(s, 'SYS_A')]} / {sysc[(s, 'SYS_C')]}） |")
    L += [f"| 合计 | {len(pop)} | {len(rows)} |", "",
          "估计总体比例时按层反加权；总体取全部片段的层规模（留出总体 + S03 样本），与 S03 的推算口径一致。"]
    (RESULTS_DIR / "s12_holdout_sampling.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(OUT, dict(Counter(r["stratum"] for r in rows)))


if __name__ == "__main__":
    main()
