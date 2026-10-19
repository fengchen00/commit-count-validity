"""S06-T2 客户条目核对证据包：可追到与追不到的条目各抽 20 条。

可追到的条目：列出追到的片段（最多 3 个，原文贴入优先）及其请求原文，核对该片段是否确实处理了这一条。
追不到的条目：列出文档日期后 30 天内请求与条目字符 3-gram 重合最多的 5 个片段，核对是否有片段处理了这一条而被漏掉。
输出 data/coding/customer_check/<条目号>.md（含商业内容，不进版本库）与 sample.csv。
"""
import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).parent))
from trace_customer import DOCS, MIN_ITEM, doc_date, load, norm  # noqa: E402

SEED, N_EACH, N_CAND, WINDOW_DAYS, REQ_LEN = "S06-T2-20260929", 20, 5, 30, 800
OUT = Path("data/coding/customer_check")


def tri(s):
    return {s[i:i + 3] for i in range(len(s) - 2)}


def main():
    items = []
    for p in sorted(DOCS.glob("*.docx")):
        for para in docx.Document(p).paragraphs:
            t = norm(para.text)
            if len(t) >= MIN_ITEM:
                items.append({"text": para.text.strip(), "norm": t, "date": doc_date(p.name)})
    trace = load(Path("data/derived/customer_trace.csv"))
    assert len(trace) == len(items)
    eps = {}
    for s in ["SYS_A", "SYS_C"]:
        for e in load(Path("data/derived") / s / "episodes.csv"):
            e["tri"] = tri(norm(e["request_text"]))
            eps[e["episode"]] = e

    linked = [r for r in trace if r["pasted"] or r["via_doc"]]
    unlinked = [r for r in trace if not (r["pasted"] or r["via_doc"])]
    rng = random.Random(SEED)
    lp = [r for r in linked if r["pasted"]]
    ld = [r for r in linked if not r["pasted"]]
    k = round(N_EACH * len(lp) / len(linked))
    sample = [("linked", r) for r in rng.sample(lp, k) + rng.sample(ld, N_EACH - k)]
    sample += [("unlinked", r) for r in rng.sample(unlinked, N_EACH)]
    rng.shuffle(sample)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (group, r) in enumerate(sample, 1):
        it = items[int(r["item_no"])]
        cid = f"CI{i:02d}"
        L = [f"# {cid}", "", f"- 条目日期：{it['date'] or '无'}（文档 {int(r['doc_no']):02d}）", "", "## 条目原文", "", it["text"], ""]
        if group == "linked":
            pasted = [x for x in r["pasted"].split(";") if x]
            via = [x for x in r["via_doc"].split(";") if x and x not in pasted]
            shown = [(x, "原文贴入请求") for x in pasted] + [(x, "agent 读取该文档或请求提到文档名") for x in via]
            L += ["## 自动追到的片段", "", f"共 {len(shown)} 个，列出前 3 个。", ""]
            for x, how in shown[:3]:
                e = eps[x]
                L += [f"### 片段 {x}（{how}）", "", f"- 开始：{e['start']}", "", "请求原文：", "", "> " + e["request_text"][:REQ_LEN].replace("\n", "\n> "), ""]
        else:
            t0 = datetime.combine(it["date"], datetime.min.time()) if it["date"] else None
            g = tri(it["norm"])
            cand = []
            for e in eps.values():
                st = datetime.fromisoformat(e["start"]).replace(tzinfo=None)
                if t0 and not (t0 <= st <= t0 + timedelta(days=WINDOW_DAYS)):
                    continue
                ov = len(g & e["tri"]) / len(g) if g else 0
                if ov > 0:
                    cand.append((ov, e))
            cand.sort(key=lambda c: -c[0])
            L += ["## 候选片段", "", f"文档日期后 {WINDOW_DAYS} 天内，请求与条目字面最接近的 {N_CAND} 个片段（字符 3-gram 重合比例）。", ""]
            if not cand:
                L += ["（没有任何字面重合的片段）", ""]
            for ov, e in cand[:N_CAND]:
                L += [f"### 片段 {e['episode']}（重合 {ov:.0%}）", "", f"- 开始：{e['start']}", "", "请求原文：", "",
                      "> " + e["request_text"][:REQ_LEN].replace("\n", "\n> "), ""]
        (OUT / f"{cid}.md").write_text("\n".join(L), encoding="utf-8")
        rows.append({"item": cid, "group": group, "item_no": r["item_no"], "pasted": int(bool(r["pasted"]))})
    with open(OUT / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "group", "item_no", "pasted"])
        w.writeheader()
        w.writerows(rows)
    print(f"linked={len(linked)} unlinked={len(unlinked)} sample={len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
