"""正文表格（S11-T3，D8）。写入 <PAPER_DIR>/results/tables.md 与 numbers.json，只含汇总。

表 1  资料概况与检查点差分的作用（D24）：观察窗口、提交来源、会话、文件编辑（去重前后）、开发者消息、片段、
      检查点前后可映射片段数、只读已确认改动时的可核验片段与无差分文件比例。
表 5  可核验与全部代码修改片段两个分母；区间按会话整体有放回重抽；附带提交、部分未提交、缺失三行按人工确认率折算。
表 6  不同工作单位下进入版本库的单位的主要模式；区间按会话重抽。
表 7  系统 A 可核验片段中各模式的规模与事前登记情况。
（表号为 v4.0 正文表号，S19-T-C2 起；函数名 table3/table4/table5 与 numbers.json 键沿用旧号。）
加权估计  人工参照按层反加权推回改代码片段，分系统报告（D9）。
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR, WINDOW_END, excluded  # noqa: E402
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import commit_sources  # noqa: E402
from extract_episodes import EDIT_TOOLS  # noqa: E402
from patterns import prepare  # noqa: E402
from holdout_agreement import holdout_reference  # noqa: E402
from sim_agreement import CODE, NAME, final_reference  # noqa: E402
from workitems import coded_decisions, group, units, wilson  # noqa: E402

BOOT = 2000
SEED = "S11-T3"
SYSTEMS = ["SYS_A", "SYS_C"]
SYS_NAME = {"SYS_A": "系统 A", "SYS_C": "系统 C"}
OUT = RESULTS_DIR
REF = "sim2"

ROWS3 = [("一对一", ["一对一"]), ("合并（同会话或跨会话）", ["合并-同会话", "合并-跨会话"]),
         ("　同会话合并", ["合并-同会话"]), ("　跨会话合并", ["合并-跨会话"]), ("拆分", ["拆分"]),
         ("　逐条任务拆分", ["拆分-逐条任务"]), ("　非逐条拆分", ["拆分-非逐条"]), ("附带提交", ["随他项提交"]),
         ("部分未提交", ["部分未提交"]), ("含未跟踪文件", ["含未跟踪文件"]), ("缺失", ["缺失"]), ("证据不足", ["证据不足"])]
DISCOUNT = {"附带提交": "A6", "部分未提交": "A8", "缺失": "A9"}
RULES4 = [("episode", "片段"), ("strict", "规则·严格"), ("loose", "规则·宽松"), ("coded", "编码工作项"), ("session", "整会话")]
ROWS4 = [("一对一", ["一对一"]), ("合并", ["合并-同会话", "合并-跨会话"]), ("拆分", ["拆分"])]
ROWS5 = [("一对一", "一对一"), ("同会话合并", "合并-同会话"), ("跨会话合并", "合并-跨会话"), ("逐条任务拆分", "拆分-逐条任务"),
         ("非逐条拆分", "拆分-非逐条"), ("附带提交", "随他项提交"), ("部分未提交", "部分未提交"),
         ("含未跟踪文件", "含未跟踪文件"), ("缺失", "缺失")]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pc(v):
    return f"{v:.0%}"


class Boot:
    """按会话整体有放回重抽；同一组单位的各行共用同一批重抽样本。"""

    def __init__(self, items, session_of, tag):
        by_s = defaultdict(list)
        for it in items:
            by_s[session_of(it)].append(it)
        sess = sorted(by_s)
        rng = random.Random(f"{SEED}|{tag}")
        self.items = items
        self.samples = [[it for k in rng.choices(sess, k=len(sess)) for it in by_s[k]] for _ in range(BOOT)]

    def stat(self, pred):
        n = len(self.items)
        k = sum(pred(it) for it in self.items)
        vs = sorted(sum(pred(it) for it in s) / len(s) for s in self.samples)
        lo, hi = vs[int(.025 * BOOT)], vs[int(.975 * BOOT) - 1]
        wlo, whi = wilson(k, n)
        return {"k": k, "n": n, "p": k / n if n else 0.0, "ci": [lo, hi], "wilson": [wlo, whi]}


def cell(s):
    return f"{pc(s['p'])}（{pc(s['ci'][0])}–{pc(s['ci'][1])}）"


def has_any(labels, pats):
    return any(p in labels for p in pats)


def confirmation():
    """人工参照下各自动规则的确认率（两个系统合并）：自动标出的样本中参照也标了（主或次）的比例。
    留出样本编码完成后，原样本与留出样本合并计算（D12）。"""
    ref = final_reference(REF)
    if not ref:
        return {}
    ref = {**ref, **(holdout_reference() or {})}
    auto = {}
    for s in SYSTEMS:
        for r in load(Path("data/derived") / s / "episode_patterns.csv"):
            auto[r["episode"]] = {CODE[x] for x in r["patterns"].split(";") if x in CODE}
    out = {}
    for c in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]:
        aa = [e for e in ref if c in auto[e]]
        ok = sum(ref[e]["main"] == c or c in ref[e]["sec"] for e in aa)
        out[c] = {"k": ok, "n": len(aa)}
    return out


def episode_units(s):
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    tb = {r["episode"]: r for r in load(d / "episode_taskbook.csv")}
    rows = []
    for r in load(d / "episode_patterns.csv"):
        if not r["evidence"]:
            continue
        e = eps[r["episode"]]
        rows.append({"episode": r["episode"], "session": e["session"], "evidence": r["evidence"],
                     "labels": set(r["patterns"].split(";")), "n_code": int(e["n_edits_code"]),
                     "ex_ante": int(tb[r["episode"]]["ex_ante"])})
    return rows


def table3(num, conf):
    cols = []
    for s in SYSTEMS:
        code = episode_units(s)
        ver = [u for u in code if u["evidence"] == "verified"]
        cols.append((s, "verified", ver, Boot(ver, lambda u: u["session"], f"t3|{s}|verified")))
        cols.append((s, "code", code, Boot(code, lambda u: u["session"], f"t3|{s}|code")))
    head = [f"{SYS_NAME[s]}·{'可核验' if den == 'verified' else '全部代码修改'}（n = {len(us)}）" for s, den, us, _ in cols]
    L = ["## 表 6 代码修改片段的模式分布", "",
         "可多标签；括号内为按会话整体有放回重抽 2,000 次的 95% 区间。", "",
         "| 模式 | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for name, pats in ROWS3:
        cells = []
        for s, den, us, b in cols:
            st = b.stat(lambda u: has_any(u["labels"], pats))
            num.setdefault(s, {}).setdefault(den, {"n": len(us)})[name.strip("　")] = st
            cells.append("—" if name == "证据不足" and den == "verified" else cell(st))
        L.append(f"| {name}{'†' if name in DISCOUNT else ''} | " + " | ".join(cells) + " |")
    L += ["", "† 自动规则的人工确认率低，该行只作上限读。"]
    if conf:
        L += ["", "### 附表：低确认率三行按人工确认率折算（可核验片段）", "",
              "折算 = 可核验比例 × 该自动规则的人工确认率（两个系统合并；留出样本编码完成后含原样本与留出样本，D12）。", "",
              "| 模式 | 人工确认率 | " + " | ".join(f"{SYS_NAME[s]}：原始 → 折算" for s in SYSTEMS) + " |",
              "|---|---|" + "---|" * len(SYSTEMS)]
        for name, c in DISCOUNT.items():
            r = conf[c]["k"] / conf[c]["n"]
            cells = []
            for s in SYSTEMS:
                p = num[s]["verified"][name]["p"]
                num[s]["verified"][name + "·折算"] = p * r
                cells.append(f"{pc(p)} → 约 {pc(p * r)}")
            L.append(f"| {name} | {conf[c]['k']}/{conf[c]['n']} | " + " | ".join(cells) + " |")
    L += ["", "对照：Wilson 95% 区间（不考虑同会话相关）", "",
          "| 模式 | " + " | ".join(f"{SYS_NAME[s]}·{'可核验' if den == 'verified' else '全部代码修改'}" for s, den, _, _ in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for name, _ in ROWS3:
        key = name.strip("　")
        L.append(f"| {name} | " + " | ".join(
            f"{pc(num[s][den][key]['wilson'][0])}–{pc(num[s][den][key]['wilson'][1])}" for s, den, _, _ in cols) + " |")
    return L + [""]


def table4(num):
    coded = coded_decisions()
    rules = [(r, n) for r, n in RULES4 if coded or r != "coded"]
    stats = {}
    for s in SYSTEMS:
        x = prepare(s)
        code_files = defaultdict(set)
        for f in load(x["d"] / "episode_files.csv"):
            if f["class"] == "code":
                code_files[f["episode"]].add(f["path"])
        for r, _ in rules:
            us, eps_of = units(x, group(x["eps"], code_files, r, coded))
            mapped = [dict(u, uid=k) for k, u in us.items() if u["n_code"] and u["commits"]]
            b = Boot(mapped, lambda u: u["uid"].split("|")[0], f"t4|{s}|{r}")
            multi = sum(len(v) >= 2 for v in eps_of.values())
            stats[(s, r)] = {"n": len(mapped), "multi_commits": [multi, len(eps_of)],
                             **{name: b.stat(lambda u: has_any(u["labels"], pats)) for name, pats in ROWS4}}
    num["table4"] = {f"{s}|{r}": v for (s, r), v in stats.items()}
    head = []
    for s in SYSTEMS:
        head += [f"{SYS_NAME[s]}：n"] + [f"{n}" for n, _ in ROWS4]
    L = ["## 表 7 不同工作单位下进入版本库的单位的主要模式", "",
         "进入版本库的单位 = 任意证据层、至少一个内容提交；可多标签；括号内为按会话重抽的 95% 区间。", "",
         "| 单位 | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for r, rname in rules:
        cells = []
        for s in SYSTEMS:
            st = stats[(s, r)]
            cells += [str(st["n"])] + [cell(st[n]) for n, _ in ROWS4]
        L.append(f"| {rname} | " + " | ".join(cells) + " |")
    L += ["", "含多个单位的提交：" + "；".join(
        f"{SYS_NAME[s]} {rname} {stats[(s, r)]['multi_commits'][0]}/{stats[(s, r)]['multi_commits'][1]}"
        for s in SYSTEMS for r, rname in rules) + "。"]
    return L + [""]


def table5(num):
    ver = [u for u in episode_units("SYS_A") if u["evidence"] == "verified"]
    L = ["## 补充表 系统 A 可核验片段中各模式的特征（正文未列）", "", "| 模式 | 片段数 | 代码编辑中位数 | 事前未登记 |", "|---|---|---|---|"]
    t5 = {}
    for name, pat in ROWS5 + [("全部可核验片段", None)]:
        us = [u for u in ver if pat is None or pat in u["labels"]]
        med = median(u["n_code"] for u in us)
        unreg = sum(not u["ex_ante"] for u in us) / len(us)
        t5[name] = {"n": len(us), "median_code_edits": med, "unregistered": unreg}
        L.append(f"| {name} | {len(us)} | {med:g} | {pc(unreg)} |")
    num["table5"] = t5
    return L + [""]


W_CODES = ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A9", "A10", "U", "X"]
W_DRAWS = 4000
W_PRIOR = 1.0


def weighted(num):
    """加权估计（D9）：按层反加权推回改代码片段；分系统；区间为分层 Dirichlet 后验（先验以全样本比例为中心）。"""
    ref = final_reference(REF)
    if not ref:
        return []
    sample = {r["episode"]: r["stratum"] for r in load("data/coding/sample_S03.csv")}
    pop = {}
    for line in (OUT / "s03_sampling.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\S+) \| (\d+)（(\d+) / (\d+)）", line)
        if m:
            pop[m.group(1)] = {"SYS_A": int(m.group(3)), "SYS_C": int(m.group(4))}
    assert set(sample.values()) <= set(pop) and set(ref) == set(sample)

    def has(e, c):
        return ref[e]["main"] == c or c in ref[e]["sec"]

    def prior(c):
        es = list(sample)
        k = [sum(ref[e]["main"] != "N0" and has(e, c) for e in es), sum(ref[e]["main"] != "N0" and not has(e, c) for e in es),
             sum(ref[e]["main"] == "N0" for e in es)]
        return [max(W_PRIOR * v / len(es), 1e-3) for v in k]

    scopes = [("SYS_A", ["SYS_A"]), ("SYS_C", ["SYS_C"]), ("合计", SYSTEMS)]
    res, borrowed = {}, {}
    for scope, syss in scopes:
        strata = []
        for h, ps in pop.items():
            N = sum(ps[s] for s in syss)
            if not N:
                continue
            es = [e for e, st in sample.items() if st == h and e.split("-")[0] in syss]
            if not es:
                es = [e for e, st in sample.items() if st == h]
                borrowed[scope] = borrowed.get(scope, 0) + N
            strata.append((N, es))
        total = sum(N for N, _ in strata)
        res[scope] = {}
        for c in W_CODES:
            cnt = [(N, sum(ref[e]["main"] != "N0" and has(e, c) for e in es),
                    sum(ref[e]["main"] != "N0" and not has(e, c) for e in es),
                    sum(ref[e]["main"] == "N0" for e in es), len(es)) for N, es in strata]
            den = sum(N * (a + b) / n for N, a, b, _, n in cnt)
            point = sum(N * a / n for N, a, _, _, n in cnt) / den
            rng = random.Random(f"{SEED}|weighted|{scope}|{c}")
            pr = prior(c)
            draws = []
            for _ in range(W_DRAWS):
                num_, den_ = 0.0, 0.0
                for N, a, b, z, _ in cnt:
                    g = [rng.gammavariate(k + q, 1) for k, q in zip((a, b, z), pr)]
                    t = sum(g)
                    num_ += N * g[0] / t
                    den_ += N * (g[0] + g[1]) / t
                draws.append(num_ / den_)
            draws.sort()
            res[scope][c] = {"p": point, "ci": [draws[int(.025 * W_DRAWS)], draws[int(.975 * W_DRAWS) - 1]]}
        res[scope]["borrowed_share"] = borrowed.get(scope, 0) / total
        res[scope]["n_sample"] = sum(len(es) for N, es in strata if all(e.split("-")[0] in syss for e in es))
    num["weighted"] = res
    L = ["## 加权估计（人工参照，按层反加权推回改代码片段；D9）", "",
         "可多标签（主模式与次模式合计）；括号内为分层 Dirichlet 后验的 95% 区间"
         "（每层先验均值取全部样本的三类比例，先验总量相当于 1 个样本）。", "",
         "| 模式 | " + " | ".join(f"{SYS_NAME.get(s, s)}（样本 {res[s]['n_sample']}）" for s, _ in scopes) + " |",
         "|---|" + "---|" * len(scopes)]
    for c in W_CODES:
        L.append(f"| {NAME[c]} | " + " | ".join(cell(res[s][c]) for s, _ in scopes) + " |")
    L += ["", "该系统没有样本、借用两个系统合并样本的层，占该系统片段的比例："
          + "；".join(f"{SYS_NAME.get(s, s)} {pc(res[s]['borrowed_share'])}" for s, _ in scopes[:2]) + "。"]
    return L + [""]


def _rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def table1_counts(s):
    """资料概况计数（D24）。"""
    d = Path("data/derived") / s
    end = WINDOW_END[s].isoformat()
    cm, gold, trailer = commit_sources(s)
    days = sorted(datetime.fromisoformat(c["time"]).date() for c in cm.values())
    sess = _rows(d / "sessions.csv")
    eps = _rows(d / "episodes.csv")
    ev = [e for e in _rows(d / "events.csv") if e["kind"] == "tool" and e["tool"] in EDIT_TOOLS]
    main_sess = {r["session"] for r in sess if r["is_subagent"] != "1"}
    msgs = [m for m in _rows(d / "user_messages.csv") if m["session"] in main_sess and not excluded(m["session"])
            and (m["time"] <= end if m["time"] else m["ordered"] == "1")]
    out = {"window_days": (days[-1] - days[0]).days + 1, "commits": len(cm), "commits_agent_cmd": len(gold),
           "commits_trailer": len(trailer), "commits_agent_any": len(set(gold) | trailer),
           "sessions_main": len({e["session"] for e in eps}), "sessions_subagent": sum(r["is_subagent"] == "1" for r in sess),
           "edits_dedup": len(ev), "edits_raw": sum(int(e["copies"] or 1) for e in ev),
           "user_msgs_window": len(msgs), "episodes": len(eps), "code_episodes": sum(int(e["n_edits_code"]) > 0 for e in eps)}
    for k, suf in [("mapped_full", ""), ("mapped_edit_only", "_edit_only"), ("mapped_noinline", "_noinline")]:
        out[k] = sum(int(r["n_commits"]) > 0 for r in _rows(d / f"episode_patterns{suf}.csv"))
    for k, suf in [("verified_full", ""), ("verified_noinline", "_noinline")]:
        out[k] = sum(r["evidence"] == "verified" for r in _rows(d / f"episode_patterns{suf}.csv"))
    # 两端都有检查点的代码修改片段所编辑的代码文件中，没有检查点差分的比例（主口径 / 只读已确认改动）
    path_of = {(e["session"], e["seq"]): e["path"].replace("\\", "/") for e in _rows(d / "events.csv")
               if e["tool"] in ("edit_file_v2", "delete_file") and e["path"]}
    touched = defaultdict(set)
    for r in _rows(d / "episode_events.csv"):
        if (r["session"], r["seq"]) in path_of:
            touched[r["episode"]].add(path_of[(r["session"], r["seq"])])
    code_paths = {(f["episode"], f["path"].replace("\\", "/")) for f in _rows(d / "episode_files.csv") if f["class"] == "code"}
    both = {e["episode"] for e in eps if e["checkpoint"] and e["next_checkpoint"] and int(e["n_edits_code"]) > 0}
    for k, suf in [("no_delta_share_full", ""), ("no_delta_share_noinline", "_noinline")]:
        have = {(r["episode"], r["path"]) for r in _rows(d / f"checkpoint_deltas{suf}.csv")}
        files = {(ep, p) for ep in both for p in touched[ep] if (ep, p) in code_paths}
        out[k] = round(1 - len(files & have) / len(files), 3) if files else None
        out["cp_code_files"] = len(files)
    return out


def table1(num):
    t = {s: table1_counts(s) for s in SYSTEMS}
    num["table1"] = t
    rows = [("观察窗口（天）", "window_days"), ("提交", "commits"), ("　其中智能体执行或署名协作", "commits_agent_any"),
            ("　　可对上智能体提交命令", "commits_agent_cmd"), ("　　带 Cursor 协作尾注", "commits_trailer"),
            ("主会话（有工作片段）", "sessions_main"), ("子智能体会话", "sessions_subagent"),
            ("文件编辑（去重后，含失败调用）", "edits_dedup"), ("　去重前记录", "edits_raw"),
            ("观察窗口内开发者消息", "user_msgs_window"), ("工作片段", "episodes"), ("　其中代码修改片段", "code_episodes"),
            ("按内容映射到提交的片段：只用逐次编辑", "mapped_edit_only"), ("　加入检查点差分（主口径）", "mapped_full"),
            ("　检查点只读已确认改动", "mapped_noinline"),
            ("可核验片段：主口径", "verified_full"), ("　检查点只读已确认改动", "verified_noinline"),
            ("两端有检查点的代码修改片段所编辑的代码文件", "cp_code_files"),
            ("　其中无差分的比例：主口径", "no_delta_share_full"), ("　其中无差分的比例：只读已确认改动", "no_delta_share_noinline")]

    def fmt(v):
        return "—" if v is None else (f"{v:.0%}" if isinstance(v, float) else f"{v:,}")
    L = ["## 表 1 资料概况与检查点差分的作用（D24）", "",
         "文件编辑 = `events.csv` 中属于 `EDIT_TOOLS` 的工具调用（去重后）；去重前记录为其 `copies` 之和。"
         "开发者消息 = 主会话、非排除会话、观察窗口内。\"只读已确认改动\"为 `CP_INLINE=0` 的对照（D3e）。", "",
         "| 项目 | " + " | ".join(SYS_NAME[s] for s in SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    for name, k in rows:
        L.append(f"| {name} | " + " | ".join(fmt(t[s][k]) for s in SYSTEMS) + " |")
    return L + [""]


if __name__ == "__main__":
    num = {}
    conf = confirmation()
    num["confirmation"] = conf
    L = ["# 正文表格（S11-T3）", "",
         "由 `analysis/render_tables.py` 生成（口径见 `notes/decisions.md` D8、D24）；正文表格以本文件为准，不手填。"
         f"人工确认率取自 `{REF}` 编码参照。", ""]
    L += table1(num)
    L += table3(num, conf)
    L += table4(num)
    L += table5(num)
    L += weighted(num)
    OUT.joinpath("tables.md").write_text("\n".join(L), encoding="utf-8")
    OUT.joinpath("numbers.json").write_text(json.dumps(num, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print("\n".join(L))
