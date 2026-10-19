"""S06 客户问题清单追踪：问题条目 → 工作片段 → 提交。写入 <PAPER_DIR>/results/r6_customer_trace.md（只含汇总）。

条目：需求文档中每个非空段落（去空白后 ≥ MIN_ITEM 字）视为一条客户问题或要求。
条目 → 片段（D7：只用原文贴入）：
  pasted   条目的字符 N-gram（N = NG）有 ≥ PASTE_COV 出现在片段请求原文中（开发者把原文贴进了对话）
  doc_read 片段中 agent 读取了该文档文件，或请求中出现该文档的文件名（去扩展名）；只记文档级关联，单独报告
  只在文档日期（文件名中的 8 位日期，没有则不限）当天及之后的片段中查找。
片段 → 提交：只用内容证据的片段—提交对（episode_commit.csv，source ≠ pathtime）。
结论只到"可追踪"：条目被某片段处理过的证据，不判断问题是否已解决。
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR  # noqa: E402

import docx

sys.path.insert(0, str(Path(__file__).parent))

DOCS = Path("data/raw/SYS_A_main/customer-docs/需求文档")
MIN_ITEM, NG, PASTE_COV = 8, 6, 0.5
NORM = re.compile(r"[\s\W_]+", re.U)


def norm(s):
    return NORM.sub("", s)


def grams(s):
    return {s[i:i + NG] for i in range(len(s) - NG + 1)}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def doc_date(name):
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name.replace("202260916", "20260916"))
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date() if m else None


def main():
    items = []
    for p in sorted(DOCS.glob("*.docx")):
        for para in docx.Document(p).paragraphs:
            t = norm(para.text)
            if len(t) >= MIN_ITEM:
                items.append({"doc": p.stem, "date": doc_date(p.name), "text": t, "grams": grams(t)})
    docs = sorted({i["doc"] for i in items} | {p.stem for p in DOCS.glob("*.docx")})

    eps, pairs, reads = [], defaultdict(set), defaultdict(set)
    plans_written, ep_ms = defaultdict(set), {}
    for s in ["SYS_A", "SYS_C"]:
        d = Path("data/derived") / s
        ev = {(e["session"], e["seq"]): e for e in load(d / "events.csv") if e["kind"] == "tool" and e["path"]}
        for r in load(d / "episode_events.csv"):
            e = ev.get((r["session"], r["seq"]))
            if not e:
                continue
            path = e["path"].replace("\\", "/")
            if path.lower().endswith(".docx"):
                reads[r["episode"]].add(Path(path).stem)
            m = re.match(r"M\d+", Path(path).name)
            if e["tool"] == "edit_file_v2" and "docs/plans/" in path and m:
                plans_written[r["episode"]].add((s, m.group()))
        for r in load(d / "episode_taskbook.csv"):
            ep_ms[r["episode"]] = {(s, x) for x in r["milestones"].split(";") if x}
        for e in load(d / "episodes.csv"):
            e["sys"] = s
            e["norm"] = norm(e["request_text"])
            e["grams"] = grams(e["norm"])
            eps.append(e)
        for p in load(d / "episode_commit.csv"):
            if p["source"] != "pathtime":
                pairs[p["episode"]].add((s, p["commit"], p["commit_time"]))

    doc_eps = defaultdict(set)
    for e in eps:
        for dname in docs:
            if dname in reads[e["episode"]] or norm(dname) in e["norm"]:
                doc_eps[dname].add(e["episode"])
    ep_by_id = {e["episode"]: e for e in eps}

    rows = []
    for it in items:
        cand = [e for e in eps if it["date"] is None or datetime.fromisoformat(e["start"]).date() >= it["date"]]
        pasted = {e["episode"] for e in cand if it["grams"] and len(it["grams"] & e["grams"]) / len(it["grams"]) >= PASTE_COV}
        via_doc = {x for x in doc_eps[it["doc"]] if it["date"] is None or datetime.fromisoformat(ep_by_id[x]["start"]).date() >= it["date"]}
        linked = pasted
        commits = set().union(*(pairs[x] for x in linked)) if linked else set()
        ms = set().union(*(plans_written[x] for x in linked)) if linked else set()
        t0 = min((ep_by_id[x]["start"] for x in linked), default="")
        impl = {e["episode"] for e in eps if e["start"] >= t0 and ep_ms.get(e["episode"], set()) & ms} - linked
        commits_plan = set().union(*(pairs[x] for x in impl)) - commits if impl else set()
        rows.append({"doc": it["doc"], "date": it["date"], "pasted": pasted, "via_doc": via_doc, "linked": linked,
                     "commits": commits, "plan_ms": ms, "impl": impl, "commits_plan": commits_plan})

    n = len(rows)
    lp = sum(bool(r["pasted"]) for r in rows)
    ld = sum(bool(r["via_doc"]) and not r["pasted"] for r in rows)
    docs_assoc = {r["doc"] for r in rows if r["via_doc"]}
    ll = sum(bool(r["linked"]) for r in rows)
    lc = sum(bool(r["commits"]) for r in rows)
    lpm = sum(bool(r["plan_ms"]) for r in rows)
    lcp = sum(bool(r["commits"] or r["commits_plan"]) for r in rows)
    lcp_only = sum(bool(r["commits_plan"]) and not r["commits"] for r in rows)
    handling = [x for r in rows for x in r["linked"]]
    per_ep = Counter(handling)
    sys_of = Counter(ep_by_id[x]["sys"] for x in per_ep)

    lag_ep, lag_c = [], []
    for r in rows:
        if r["date"] and r["linked"]:
            first = min(datetime.fromisoformat(ep_by_id[x]["start"]) for x in r["linked"])
            lag_ep.append((first.date() - r["date"]).days)
            if r["commits"]:
                fc = min(datetime.fromisoformat(t) for _, _, t in r["commits"])
                lag_c.append((fc - first).total_seconds() / 3600)

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")

    L = ["# S06 客户问题清单追踪：条目 → 工作片段 → 提交", "",
         "由 `analysis/trace_customer.py` 生成；规则见脚本文件头。只到“可追踪”，不判断问题是否已解决。只含汇总。", "",
         f"- 需求文档 {len(docs)} 份，抽出条目 {n} 条（段落，去空白后 ≥ {MIN_ITEM} 字）。",
         f"- 能追到至少一个工作片段的条目 {ll}（{ll / n:.0%}），即原文被贴进请求的条目（D7）。",
         f"- 文档级关联：{len(docs_assoc)} 份文档被 agent 读取或在请求中被提到；其中另有 {ld} 条（{ld / n:.0%}）条目只有文档级关联，不计入条目可追踪。",
         f"- 处理片段本身的内容能追到提交的条目 {lc}（{lc / n:.0%}；占可追到片段的 {lc / ll:.0%}）。" if ll else "- 无可追踪条目。",
         f"- 处理片段写了任务书的条目 {lpm}；再经“该任务书里程碑 → 之后执行该里程碑的片段 → 提交”追到提交的另有 {lcp_only} 条"
         f"（里程碑粒度，一个里程碑可能含多条客户条目，偏宽）。两条路合计追到提交 {lcp}（{lcp / n:.0%}；占可追到片段的 {lcp / max(ll, 1):.0%}）。",
         f"- 处理过客户条目的片段 {len(per_ep)} 个（SYS_A {sys_of['SYS_A']}、SYS_C {sys_of['SYS_C']}）；"
         f"每个片段处理的条目数：中位数 {q(list(per_ep.values()), .5)}，P90 {q(list(per_ep.values()), .9)}，最大 {max(per_ep.values()) if per_ep else 0}。",
         f"- 文档日期到首个处理片段的天数（有日期的条目 {len(lag_ep)} 条）：中位数 {q(lag_ep, .5)}，P90 {q(lag_ep, .9)}。",
         f"- 首个处理片段到首个内容提交的小时数（{len(lag_c)} 条）：中位数 {q(lag_c, .5):.1f}，P90 {q(lag_c, .9):.1f}。", "",
         "## 按文档", "", "| 文档 | 日期 | 条目 | 可追到片段 | 原文贴入 | 直接追到提交 | 含经任务书 |", "|---|---|---|---|---|---|---|"]
    for i, dname in enumerate(docs, 1):
        rs = [r for r in rows if r["doc"] == dname]
        if not rs:
            continue
        L.append(f"| 文档 {i:02d} | {rs[0]['date'] or '—'} | {len(rs)} | {sum(bool(r['linked']) for r in rs)} | "
                 f"{sum(bool(r['pasted']) for r in rs)} | {sum(bool(r['commits']) for r in rs)} | "
                 f"{sum(bool(r['commits'] or r['commits_plan']) for r in rs)} |")
    L += ["", "文档以编号代替文件名（文件名含客户信息）。"]
    (RESULTS_DIR / "r6_customer_trace.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open("data/derived/customer_trace.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_no", "item_no", "date", "pasted", "via_doc", "n_commits"])
        for k, r in enumerate(rows):
            w.writerow([docs.index(r["doc"]) + 1, k, r["date"] or "", ";".join(sorted(r["pasted"])),
                        ";".join(sorted(r["via_doc"])), len(r["commits"])])
    print(f"items={n} linked={ll} pasted={lp} via_doc_only={ld} to_commit={lc} episodes={len(per_ep)}")


if __name__ == "__main__":
    main()
