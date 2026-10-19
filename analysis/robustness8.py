"""回应第七轮审稿人 2 与不利情景审稿、不依赖边界标注的补充分析（S18-T3，口径见 notes/decisions.md D20）。

1 工作单位敏感性带：系统比较与事前登记分期比较在片段、规则聚合、整会话、代码编辑、改动代码文件、请求中列举的事项数与工作项下的结果
2 不设切点：合格日事前登记比例与每片段、每百次代码编辑提交数的 Spearman 相关；逐日删除后主口径比值的范围；日提交数与日片段数、日代码编辑数的相关
3 外部参照的补充：未找回关系的去向；按会话重抽的召回率区间
写入 <PAPER_DIR>/results/r16_robustness8.md 与 numbers.json 的 robustness8 键（须在 robustness7.py 之后运行）。
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import command_commits  # noqa: E402
from robustness import NAME  # noqa: E402
from robustness7 import BOOT, NUMS, RES, agg, bci, compare, days, groups, load_ctx, spearman, split  # noqa: E402
from robustness7 import SEED as SEED7  # noqa: E402

SEED = "S18b"
ITEM = re.compile(r"(?m)^\s*(?:\d{1,2}|[一二三四五六七八九十])\s*[、.．)）:：]")
UNITS = [("episodes", "代码修改片段"), ("strict", "规则·严格聚合"), ("loose", "规则·宽松聚合"), ("session", "整会话"),
         ("items", "请求中列举的事项"), ("files", "改动代码文件"), ("edits", "代码编辑"), ("coded", "编码工作项"),
         ("works", "开发者视角工作项")]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def unit_counts(s, ctx):
    """每个代码修改片段在各单位下的计数：聚合单位只在其第一个片段上计 1，片段级单位直接计数。"""
    d = Path("data/derived") / s
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    wi = {r["episode"]: r for r in load(d / "workitems.csv")}
    files = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            files[f["episode"]].add(f["path"])
    items = sorted(ctx["items"], key=lambda i: i["start"])
    seen = defaultdict(set)
    for i in items:
        e = i["e"]
        i["u"] = {"episodes": 1, "edits": i["n"], "files": len(files[e]), "works": i["w_hat"],
                  "items": max(1, len(ITEM.findall(eps[e]["request_text"])))}
        for r in ("strict", "loose", "coded", "session"):
            g = wi[e][f"wi_{r}"]
            i["u"][r] = int(g not in seen[r])
            seen[r].add(g)


def by_day(ctx, key, min_eps=3):
    """合格日（按 robustness7.days 的口径）上某一单位的日计数，挂到 days() 的输出上。"""
    ds = days(ctx, by="reg", min_eps=min_eps)
    tot = defaultdict(float)
    for i in ctx["items"]:
        tot[i["start"][:10]] += i["u"][key]
    for d in ds:
        d["k"] = tot[d["d"]]
    return ds


def band(CTX, rng):
    R, L = {}, ["## 1 工作单位敏感性带（表 14 的来源；D30-4 比率分解 R_C = R_W^(u) × D^(u)）", "",
                "| 工作单位 u | 系统 A 总数 | 系统 C 总数 | R_W^(u)：系统 A 事前登记高对低的日均工作量之比 | D^(u)：每单位提交数之比（高对低） | 系统比较：A 对 C 的日均量之比 | 以提交计的系统比值 ÷ 以该单位计 |",
                "|---|---|---|---|---|---|---|"]
    act = {s: sorted({i["start"][:10] for i in CTX[s]["items"]}) for s in CTX}
    day_c = {s: sum(CTX[s]["cday"].values()) / len(act[s]) for s in CTX}
    sys_c = day_c["SYS_A"] / day_c["SYS_C"]
    R["sys_commits"] = sys_c
    rows = []
    for key, name in UNITS:
        tot = {s: sum(i["u"][key] for i in CTX[s]["items"]) for s in CTX}
        r_sys = (tot["SYS_A"] / len(act["SYS_A"])) / (tot["SYS_C"] / len(act["SYS_C"]))
        ds = by_day(CTX["SYS_A"], key)
        r = compare(ds, rng=random.Random(f"{SEED7}|SYS_A"))
        g = groups(ds, rng)
        R[key] = {"tot": tot, "sys": r_sys, "sys_over": sys_c / r_sys, "per_unit": r["ratio"]["c_work"], "per_unit_ci": r["ci"]["c_work"],
                  "daily_unit": g["daily"]["k"], "daily_unit_ci": g["daily_ci"]["k"], "daily_c": g["daily"]["c"], "daily_c_ci": g["daily_ci"]["c"]}
        rows.append(f"| {name} | {tot['SYS_A']:,.0f} | {tot['SYS_C']:,.0f} | "
                    f"{g['daily']['k']:.2f}（{g['daily_ci']['k'][0]:.2f}–{g['daily_ci']['k'][1]:.2f}） | "
                    f"{r['ratio']['c_work']:.2f}（{r['ci']['c_work'][0]:.2f}–{r['ci']['c_work'][1]:.2f}） | {r_sys:.2f} | {sys_c / r_sys:.2f} |")
    rc, rc_ci = R["episodes"]["daily_c"], R["episodes"]["daily_c_ci"]
    R["r_c"], R["r_c_ci"] = rc, rc_ci
    L.append(f"| 提交（对照） | {sum(CTX['SYS_A']['cday'].values())} | {sum(CTX['SYS_C']['cday'].values())} | R_C = {rc:.2f}（{rc_ci[0]:.2f}–{rc_ci[1]:.2f}） | 1 | {sys_c:.2f} | 1.00 |")
    L += rows
    vals = [R[k]["sys"] for k, _ in UNITS]
    per = [R[k]["per_unit"] for k, _ in UNITS]
    dly = [R[k]["daily_unit"] for k, _ in UNITS]
    R["range_sys"], R["range_per_unit"], R["range_daily_unit"] = [min(vals), max(vals)], [min(per), max(per)], [min(dly), max(dly)]
    L += ["", f"事前登记分期：以提交计的日均量之比 R_C = {rc:.2f}（{rc_ci[0]:.2f}–{rc_ci[1]:.2f}）；九种单位下 R_W^(u) 为 {min(dly):.2f}–{max(dly):.2f}，D^(u) 为 {min(per):.2f}–{max(per):.2f}，"
          f"每行满足 R_C = R_W^(u) × D^(u)（点估计；区间各自按日重抽）。以提交计的系统比较为 {sys_c:.2f}；以九种工作单位计为 {min(vals):.2f}–{max(vals):.2f}。"
          "聚合单位与改动文件数按片段开始的日子计；请求中列举的事项数为请求中以编号开头的行数（至少 1）。区间按日重抽 2,000 次；两种标注单位的区间不含效标样本的不确定性。", ""]
    return R, L


def continuous(ctx, rng):
    ds = days(ctx, by="reg")
    x = [d["g"] for d in ds]
    ys = {"c_ep": [d["c"] / d["n"] for d in ds], "c_100w": [100 * d["c"] / d["w"] for d in ds]}
    out = {}
    for m, y in ys.items():
        bs = []
        for _ in range(BOOT):
            idx = [rng.randrange(len(ds)) for _ in ds]
            bs.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
        out[m] = {"rho": spearman(x, y), "ci": bci(bs)}
    lodo = defaultdict(list)
    for j in range(len(ds)):
        H, Lo = split(ds[:j] + ds[j + 1:], 2 / 3, 1 / 3)
        a, b = agg(H), agg(Lo)
        for m in ("c_ep", "c_100w", "c_work"):
            lodo[m].append(a[m] / b[m])
    out["lodo"] = {m: [min(v), max(v)] for m, v in lodo.items()}
    out["daily_rho"] = {"n": spearman([d["c"] for d in ds], [d["n"] for d in ds]), "w": spearman([d["c"] for d in ds], [d["w"] for d in ds])}
    out["n_days"] = len(ds)
    return out


def validation(s):
    d = Path("data/derived") / s
    cm = {c["hash"]: c for c in load(d / "commits.csv")}
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    gold = command_commits(d, cm)
    content, eps_of = defaultdict(set), defaultdict(set)
    for p in load(d / "episode_commit.csv"):
        if p["source"] != "pathtime":
            content[p["episode"]].add(p["commit"])
            eps_of[p["commit"]].add(p["episode"])
    files_of_c, code_files = defaultdict(set), defaultdict(set)
    for f in load(d / "commit_files.csv"):
        files_of_c[f["hash"]].add(f["path"].lower())
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"].lower())
    rel = [(h, ep) for h, ep in gold.items() if code_files[ep] & files_of_c[h]]
    miss = [(h, ep) for h, ep in rel if h not in content[ep]]
    fate = {"ep_no_content": sum(not content[ep] for h, ep in miss),
            "ep_other_commits": sum(bool(content[ep]) for h, ep in miss),
            "c_to_same_session": sum(any(eps[o]["session"] == eps[ep]["session"] for o in eps_of[h]) for h, ep in miss),
            "c_to_other_session_only": sum(bool(eps_of[h]) and not any(eps[o]["session"] == eps[ep]["session"] for o in eps_of[h]) for h, ep in miss),
            "c_unattributed": sum(not eps_of[h] for h, ep in miss)}
    by_s = defaultdict(list)
    for h, ep in rel:
        by_s[eps[ep]["session"]].append(h in content[ep])
    ss = list(by_s.values())
    rng = random.Random(f"{SEED}|val|{s}")
    bs = []
    for _ in range(BOOT):
        pick = [rng.choice(ss) for _ in ss]
        bs.append(sum(sum(g) for g in pick) / sum(len(g) for g in pick))
    return {"n_rel": len(rel), "n_miss": len(miss), "recall": 1 - len(miss) / len(rel), "recall_ci_session": bci(bs),
            "n_sessions": len(ss), "fate": fate}


DELTAS = (1.1, 1.25, 1.5)
MAIN_DELTA = 1.25


def classify(lo, hi, delta, rule):
    """D35-2：主判据（顺序）与对称判据。lo/hi 为 D^(u) 的 95% 区间端点。"""
    if lo >= 1 / delta and hi <= delta:
        return "稳定"
    if rule == "main":
        return "不稳定" if (lo > 1 or hi < 1) else "证据不足"
    return "不稳定" if (lo >= delta or hi <= 1 / delta) else "证据不足"


def criteria(R):
    """D35-2（B1/B7）：九种单位在三个 δ 与两种判据下的分类，以及区间宽度相对等价界宽度的精度量。"""
    import math
    L = ["## 4 稳定性判据的敏感性（D35-2；主判据为顺序判定：界内 → 稳定，否则不含 1 → 不稳定，其余证据不足；对称判据：不稳定须区间整体落在界外）", "",
         "| 工作单位 u | D^(u)（95% 区间） | 主判据 δ=1.1 | δ=1.25 | δ=1.5 | 对称判据 δ=1.1 | δ=1.25 | δ=1.5 | 区间对数宽度 ÷ 界宽（δ=1.25） | 达到界宽所需天数倍数 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for key, name in UNITS:
        lo, hi = R[key]["per_unit_ci"]
        pe = R[key]["per_unit"]
        width = math.log(hi / lo)
        band_w = 2 * math.log(MAIN_DELTA)
        ratio = width / band_w
        row = {"ci": [lo, hi], "main": {str(d): classify(lo, hi, d, "main") for d in DELTAS},
               "sym": {str(d): classify(lo, hi, d, "sym") for d in DELTAS}, "log_width": width, "width_over_band": ratio,
               "days_multiplier": ratio ** 2}
        out[key] = row
        L.append(f"| {name} | {pe:.2f}（{lo:.2f}–{hi:.2f}） | " + " | ".join(row["main"][str(d)] for d in DELTAS) + " | "
                 + " | ".join(row["sym"][str(d)] for d in DELTAS) + f" | {ratio:.1f} | ≈{ratio ** 2:.0f} |")
    n_main = {str(d): sum(v["main"][str(d)] == "不稳定" for v in out.values()) for d in DELTAS}
    n_sym = {str(d): sum(v["sym"][str(d)] == "不稳定" for v in out.values()) for d in DELTAS}
    changed = [name for key, name in UNITS if out[key]["main"][str(MAIN_DELTA)] != out[key]["sym"][str(MAIN_DELTA)]]
    out["n_unstable_main"], out["n_unstable_sym"], out["changed_at_main_delta"] = n_main, n_sym, changed
    L += ["", f"- 主判据下判为不稳定的单位数：δ=1.1 为 {n_main['1.1']}，δ=1.25 为 {n_main['1.25']}，δ=1.5 为 {n_main['1.5']}；"
          f"对称判据下为 {n_sym['1.1']} / {n_sym['1.25']} / {n_sym['1.5']}。没有一种单位在任一 δ 下判为稳定。",
          f"- δ=1.25 时两种判据分类不同的单位：{'、'.join(changed) if changed else '无'}（主判据不稳定、对称判据证据不足）。",
          "- 精度：区间对数宽度 ln(上端/下端) 除以等价界对数宽度 2·ln δ；若标准误随合格日数按平方根缩小，使区间宽度缩到界宽所需的天数约为现有 16 天乘以该比值的平方。此为粗估，只说明量级。", ""]
    return out, L


def content_numerator(CTX, R, rng):
    """D35-6（B2）：分子只计按内容归属到某片段的提交（按提交时间归日），重算 R_C 与九种单位的 D^(u)。"""
    from robustness import base
    x, _, _ = base("SYS_A")
    attributed = {c for eps in x["content"].values() for c in eps}
    cday = defaultdict(int)
    for h, c in x["cm"].items():
        if h in attributed:
            cday[c["time"][:10]] += 1
    ctx = dict(CTX["SYS_A"])
    ctx["cday"] = cday
    L = ["## 5 分子只计按内容归属到片段的提交（D35-6；系统 A 事前登记分期，按提交时间归日）", "",
         "| 工作单位 u | D^(u)：全部提交为分子 | D^(u)：仅内容归属提交为分子 | 主判据分类（δ=1.25，仅内容归属） |", "|---|---|---|---|"]
    out = {"n_attributed": len(attributed), "n_commits": len(x["cm"])}
    ds0 = by_day(ctx, "episodes")
    g = groups(ds0, random.Random(f"{SEED}|content|rc"))
    out["r_c"], out["r_c_ci"] = g["daily"]["c"], g["daily_ci"]["c"]
    for key, name in UNITS:
        ds = by_day(ctx, key)
        r = compare(ds, rng=random.Random(f"{SEED}|content|{key}"))
        lo, hi = r["ci"]["c_work"]
        out[key] = {"per_unit": r["ratio"]["c_work"], "per_unit_ci": [lo, hi], "main_class": classify(lo, hi, MAIN_DELTA, "main")}
        a = R[key]
        L.append(f"| {name} | {a['per_unit']:.2f}（{a['per_unit_ci'][0]:.2f}–{a['per_unit_ci'][1]:.2f}） | "
                 f"{r['ratio']['c_work']:.2f}（{lo:.2f}–{hi:.2f}） | {out[key]['main_class']} |")
    per = [out[k]["per_unit"] for k, _ in UNITS]
    out["range_per_unit"] = [min(per), max(per)]
    L += ["", f"- 系统 A {len(x['cm'])} 个提交中按内容归属到至少一个片段的 {len(attributed)} 个；只以这些提交为分子时 R_C = {out['r_c']:.2f}（{out['r_c_ci'][0]:.2f}–{out['r_c_ci'][1]:.2f}），"
          f"九种单位的 D^(u) 为 {min(per):.2f}–{max(per):.2f}。区间按日重抽 2,000 次；分期、日集与分母与表 12 相同。", ""]
    return out, L


def main():
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    bl = num["robustness6"]["boundary_labels"]
    CTX = {}
    for s in ("SYS_A", "SYS_C"):
        nw = bl[f"{s}|final|rates"]["new_works"]
        CTX[s] = load_ctx(s, {True: nw[0], False: nw[1]})
        unit_counts(s, CTX[s])
    rng = random.Random(SEED)
    R, L = band(CTX, rng)
    # D40：开发者视角工作项一行另附格均值模型（提交状态 × 事前登记日组的样本均值外推，robustness7 两层区间）
    c7 = num.get("robustness7", {}).get("SYS_A|事前登记比例分期（主口径）|cells")
    if c7:
        lo, hi = c7["ci"]
        R["works"].update({"per_unit_cells": c7["ratio"], "per_unit_cells_ci": [lo, hi], "cells_class": classify(lo, hi, MAIN_DELTA, "main"),
                           "tot_cells": {"hi": c7["hi"]["works"], "lo": c7["lo"]["works"]}, "tot_pooled": {"hi": c7["hi"]["works_pooled"], "lo": c7["lo"]["works_pooled"]}})
        L += [f"开发者视角工作项一行按合并均值模型外推（高 / 低日工作项 {c7['hi']['works_pooled']:.0f} / {c7['lo']['works_pooled']:.0f}）；"
              f"按格均值模型外推（{c7['hi']['works']:.0f} / {c7['lo']['works']:.0f}）D^(u) = {c7['ratio']:.2f}（两层重抽 {lo:.2f}–{hi:.2f}），"
              f"主判据分类为{R['works']['cells_class']}；两种模型方向相反，该单位的判定取决于折算模型（D40，r15 事前登记主口径的折算模型行）。", ""]
    L = ["# 第七轮审稿补充分析（二）：工作单位敏感性带、不设切点的关联与外部参照（S18-T3，D20）", "",
         "由 `analysis/robustness8.py` 生成。", ""] + L
    L += ["## 2 不设切点的关联与逐日删除（系统 A 合格日）", ""]
    c = continuous(CTX["SYS_A"], random.Random(f"{SEED}|cont"))
    R["continuous"] = c
    L += [f"- 当日事前登记比例与每片段提交数的 Spearman ρ = {c['c_ep']['rho']:.2f}（{c['c_ep']['ci'][0]:.2f}–{c['c_ep']['ci'][1]:.2f}），"
          f"与每百次代码编辑提交数 ρ = {c['c_100w']['rho']:.2f}（{c['c_100w']['ci'][0]:.2f}–{c['c_100w']['ci'][1]:.2f}）；{c['n_days']} 天，区间按日重抽。",
          f"- 逐日删除一个合格日后，事前登记分期的比值范围：每片段 {c['lodo']['c_ep'][0]:.2f}–{c['lodo']['c_ep'][1]:.2f}，"
          f"每百次代码编辑 {c['lodo']['c_100w'][0]:.2f}–{c['lodo']['c_100w'][1]:.2f}，每项工作 {c['lodo']['c_work'][0]:.2f}–{c['lodo']['c_work'][1]:.2f}。",
          f"- 日提交数与日片段数的 Spearman ρ = {c['daily_rho']['n']:.2f}，与日代码编辑数 ρ = {c['daily_rho']['w']:.2f}。", ""]
    L += ["## 3 外部参照的补充", ""]
    for s in ("SYS_A", "SYS_C"):
        v = validation(s)
        R[f"{s}|validation"] = v
        f = v["fate"]
        L += [f"- {NAME[s]}：召回 {v['n_rel'] - v['n_miss']}/{v['n_rel']}（{v['recall']:.0%}），按会话重抽（{v['n_sessions']} 个会话）的 95% 区间 "
              f"{v['recall_ci_session'][0]:.0%}–{v['recall_ci_session'][1]:.0%}。未找回的 {v['n_miss']} 组中，片段没有任何按内容归属的提交 {f['ep_no_content']} 组，"
              f"片段的内容归到了其他提交 {f['ep_other_commits']} 组；该提交的内容归到同会话其他片段 {f['c_to_same_session']} 组、只归到其他会话 {f['c_to_other_session_only']} 组、"
              f"未归属任何片段 {f['c_unattributed']} 组。"]
    L.append("")
    crit, Lc = criteria(R)
    R["criteria"] = crit
    L += Lc
    cn, Ln = content_numerator(CTX, R, rng)
    R["content_numerator"] = cn
    L += Ln
    (RES / "r16_robustness8.md").write_text("\n".join(L), encoding="utf-8")
    num["robustness8"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
