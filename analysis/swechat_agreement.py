"""S27 / D38（B9）：公开语料人工核验的汇总——两名编码人（甲、作者）对 SWE-chat 样本片段的编码与自动判定的对照。

输入（本地，不入库）：data/coding/swechat/frame.csv（抽样框与分层变量）、sample_S27.csv、
     human/coder_甲.csv、human/coder_作者.csv（经 tools/coding/swechat_handoff.py validate 落盘的规范表）、可选 human/adjudicate.csv。
口径（tools/repro/analysis_plan.md S2b，抽样前冻结）：
  - Cohen κ：主模式全集；粗分五类（一对一 A1 / 合并 A2 A3 / 拆分 A4 A5 / 缺失 A9 / 其他）。
  - 确认率：自动一对一 ↔ A1；共享 ↔ A2/A3；拆分 ↔ A4/A5；缺失 ↔ A9；人工 U 不计入分母、另报比例。每格、每种自动模式，
    两名编码人各自 + 两人一致子集（有裁决时另列裁决口径），Wilson 95%。
  - 两组（保留 / 排除）的人工一对一、合并、缺失比例：按层（格 × 自动模式）反加权到两层总体，分层方差近似（含有限总体校正）。
  - 自动判定对人工合并与人工缺失的召回：同样反加权。
用法：python analysis/swechat_agreement.py            写 <PAPER_DIR>/results/r27_swechat_human.md 与 numbers.json 的 swechat_human 键
     python analysis/swechat_agreement.py disputes   写 human/disputes_06.txt 与匿名对照表（供裁决，可选）
环境变量 SWECHAT_CODING_DIR 可改输入目录（测试用）。
"""
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR  # noqa: E402

DIR = Path(os.environ.get("SWECHAT_CODING_DIR", "data/coding/swechat"))
RES = Path(os.environ.get("SWECHAT_RESULTS_DIR", RESULTS_DIR))   # 测试时可改，避免写入论文结果目录
HUMAN = DIR / "human"
CODERS = [("甲", "coder_甲.csv"), ("作者", "coder_作者.csv")]
ADJ = HUMAN / "adjudicate.csv"
GROUPS, LAYERS, MODES = ["保留", "排除"], ["单独提交指令", "片段内提交"], ["一对一", "共享", "拆分", "缺失"]
EXPECT = {"一对一": {"A1"}, "共享": {"A2", "A3"}, "拆分": {"A4", "A5"}, "缺失": {"A9"}, "证据不足": {"U"}}
COARSE = {"A1": "一对一", "A2": "合并", "A3": "合并", "A4": "拆分", "A5": "拆分", "A9": "缺失"}
HUMAN_CLASSES = {"一对一": {"A1"}, "合并": {"A2", "A3"}, "缺失": {"A9"}}
AUTO_FOR = {"合并": "共享", "缺失": "缺失"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def pct(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n}（{k / n:.0%}；{lo:.0%}–{hi:.0%}）"


def coarse(m):
    return COARSE.get(m, "其他")


def read_coders(sample_ids):
    out = {}
    for name, fn in CODERS:
        p = HUMAN / fn
        if not p.exists():
            return None
        rows = {r["episode"].strip(): r for r in load(p)}
        assert set(rows) == set(sample_ids), f"{fn} 的片段集合与样本不一致"
        out[name] = {e: {"main": r["main_pattern"].strip().upper(), "sec": {x.strip().upper() for x in r["secondary_patterns"].replace("；", ";").split(";") if x.strip()},
                         "conf": r["confidence"].strip(), "suff": r["evidence_sufficient"].strip().upper()} for e, r in rows.items()}
    return out


def strat_estimate(frame_g, sample_rows, indicator):
    """分层反加权：frame_g 为该组在两层上的全部片段，sample_rows 为该组样本（带 stratum 与 main）。
    层 = (layer, auto_mode)。返回 (估计, 标准误)。"""
    N = Counter(r["stratum"] for r in frame_g)
    Ntot = sum(N.values())
    by = defaultdict(list)
    for r in sample_rows:
        by[r["stratum"]].append(indicator(r))
    est = var = 0.0
    for h, Nh in N.items():
        xs = by.get(h, [])
        if not xs:
            continue                                     # 未抽到的层（总体为 0 或名额为 0）按缺失处理，不外推
        nh, ph = len(xs), sum(xs) / len(xs)
        W = Nh / Ntot
        est += W * ph
        pv = (sum(xs) + 1) / (nh + 2)                   # 方差项用平滑比例（Agresti–Coull），避免层内全 0 / 全 1 时区间宽度为零
        var += W * W * max(0.0, 1 - nh / Nh) * pv * (1 - pv) / nh
    return est, math.sqrt(var)


def fmt_est(e, se):
    return f"{e:.0%}（{max(0, e - 1.96 * se):.0%}–{min(1, e + 1.96 * se):.0%}）"


def zero_feature_flags():
    """片段 id → 该片段是否没有任何新写入特征行（所有代码编辑的 n_new 之和为 0）。
    从 swechat_sample 的片段级缓存读；缓存不全时返回 {}（对应小节跳过）。"""
    try:
        import pickle
        from swechat_sample import SHORT, ep_id
    except Exception:
        return {}
    out = {}
    for repo in SHORT:
        p = DIR / "cache" / f"{repo.replace('/', '__')}.pkl"
        if not p.exists():
            return {}
        kp = pickle.loads(p.read_bytes())
        eps = kp["episodes"]
        for i in kp["code_eps"]:
            out[ep_id(repo, eps, i)] = sum(x.get("n_new", 0) for x in eps[i]["edits"] if x.get("code")) == 0
    return out


def main():
    frame = load(DIR / "frame.csv")
    for r in frame:
        r["stratum"] = (r["layer"], "缺失" if r["auto_mode"] == "证据不足" else r["auto_mode"])
    sample = load(DIR / "sample_S27.csv")
    ids = [r["episode"] for r in sample]
    key = {r["episode"]: r for r in frame}
    coders = read_coders(ids)
    if coders is None:
        print("编码表尚未落盘（human/coder_甲.csv、coder_作者.csv）；不输出。")
        return
    names = [n for n, _ in CODERS]
    A, B = coders[names[0]], coders[names[1]]
    adj = {r["episode"].strip(): r["main_pattern"].strip().upper() for r in load(ADJ)} if ADJ.exists() else {}
    if sys.argv[1:] == ["disputes"]:
        write_disputes(ids, A, B)
        return
    # 一致性
    a_main, b_main = [A[e]["main"] for e in ids], [B[e]["main"] for e in ids]
    po, k_full = cohen(a_main, b_main)
    po_c, k_coarse = cohen([coarse(x) for x in a_main], [coarse(x) for x in b_main])
    # 参照口径：两人一致（主模式相同）子集；有裁决时为一致 ∪ 裁决
    agree = [e for e in ids if A[e]["main"] == B[e]["main"]]
    ref = {e: A[e]["main"] for e in agree}
    ref_adj = dict(ref, **{e: adj[e] for e in ids if e not in ref and e in adj}) if adj else None
    views = [(names[0], {e: A[e]["main"] for e in ids}), (names[1], {e: B[e]["main"] for e in ids}), ("两人一致子集", ref)]
    if ref_adj:
        views.append(("一致 + 裁决", ref_adj))

    def confirm_table(rows_sel):
        """rows_sel: [(episode, auto_mode)] → 各视角的 (n, nU, k)。"""
        out = {}
        for vn, labels in views:
            n = nU = k = 0
            for e, am in rows_sel:
                if e not in labels:
                    continue
                m = labels[e]
                n += 1
                if m == "U":
                    nU += 1
                elif m in EXPECT[am]:
                    k += 1
            out[vn] = (n, nU, k)
        return out

    L = ["# S27 公开语料人工核验：两名编码人与自动判定的对照（D38）", "",
         f"由 `analysis/swechat_agreement.py` 生成。样本 {len(ids)} 个片段（抽样见 `s27_swechat_sampling.md`），编码人 {names[0]} 与 {names[1]} 各编全部；"
         "编码人看不到自动模式、组与触发层。口径见 `tools/repro/analysis_plan.md` S2b。", "",
         "## 1 编码人之间的一致性", "",
         f"- 主模式全集：观察一致率 {po:.0%}，Cohen κ = {k_full:.2f}；粗分五类（一对一 / 合并 / 拆分 / 缺失 / 其他）：一致率 {po_c:.0%}，κ = {k_coarse:.2f}。",
         f"- 两人主模式一致的片段 {len(agree)} 个（{len(agree) / len(ids):.0%}）；" + (f"裁决覆盖分歧 {len([e for e in ids if e not in ref and e in adj])} 个。" if adj else "未裁决，分歧全部保留。"),
         f"- 人工 U（证据不足）：{names[0]} {sum(1 for e in ids if A[e]['main'] == 'U')} 个，{names[1]} {sum(1 for e in ids if B[e]['main'] == 'U')} 个。", ""]
    # 主模式交叉表
    cats = sorted({*a_main, *b_main}, key=lambda x: (x[0] != "A", x))
    L += ["主模式交叉表（行 = " + names[0] + "，列 = " + names[1] + "）：", "", "| | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    cross = Counter(zip(a_main, b_main))
    for ca in cats:
        L.append(f"| {ca} | " + " | ".join(str(cross[(ca, cb)] or "") for cb in cats) + " |")
    L.append("")

    # 确认率
    num = {"n": len(ids), "kappa_main": k_full, "kappa_coarse": k_coarse, "agree_n": len(agree), "po_main": po, "cells": {}, "modes": {}, "groups": {}}
    L += ["## 2 自动主模式的人工确认率", "",
          "确认 = 人工主模式落在自动模式对应的集合内（一对一 ↔ A1；共享 ↔ A2/A3；拆分 ↔ A4/A5；缺失 ↔ A9）；人工 U 不计入分母。括号内为 Wilson 95% 区间。", ""]
    hdr = "| 组 | 层 | 自动模式 | n | " + " | ".join(f"{vn}" for vn, _ in views) + " |"
    L += [hdr, "|---|---|---|---|" + "---|" * len(views)]
    for g in GROUPS:
        for ly in LAYERS:
            for m in MODES + ["合计"]:
                sel = [(e, key[e]["auto_mode"]) for e in ids if key[e]["group"] == g and key[e]["layer"] == ly and (m == "合计" or key[e]["stratum"][1] == m)]
                if not sel:
                    continue
                ct = confirm_table(sel)
                num["cells"][f"{g}|{ly}|{m}"] = {vn: {"n": n, "U": nU, "confirmed": k} for vn, (n, nU, k) in ct.items()}
                L.append(f"| {g} | {ly} | {m} | {len(sel)} | " + " | ".join(pct(k, n - nU) + (f"，U {nU}" if nU else "") for vn, (n, nU, k) in ct.items()) + " |")
    L += ["", "按自动模式合计（四格并）：", "", "| 自动模式 | n | " + " | ".join(vn for vn, _ in views) + " |", "|---|---|" + "---|" * len(views)]
    for m in MODES:
        sel = [(e, key[e]["auto_mode"]) for e in ids if key[e]["stratum"][1] == m]
        ct = confirm_table(sel)
        num["modes"][m] = {vn: {"n": n, "U": nU, "confirmed": k} for vn, (n, nU, k) in ct.items()}
        L.append(f"| {m} | {len(sel)} | " + " | ".join(pct(k, n - nU) + (f"，U {nU}" if nU else "") for vn, (n, nU, k) in ct.items()) + " |")
    L += ["", "按组合计：", "", "| 组 | n | " + " | ".join(vn for vn, _ in views) + " |", "|---|---|" + "---|" * len(views)]
    for g in GROUPS:
        sel = [(e, key[e]["auto_mode"]) for e in ids if key[e]["group"] == g]
        ct = confirm_table(sel)
        num["groups"][g] = {"confirm": {vn: {"n": n, "U": nU, "confirmed": k} for vn, (n, nU, k) in ct.items()}}
        L.append(f"| {g} | {len(sel)} | " + " | ".join(pct(k, n - nU) + (f"，U {nU}" if nU else "") for vn, (n, nU, k) in ct.items()) + " |")

    # 事后补充口径（不在冻结计划内，单独标注）：(a) 人工主模式或次模式落在对应集合内即算确认；(b) 自动"拆分且共享"对应 A2–A5。
    # 动机：手册 v2.6 规定主模式取大部分新写入行所在的提交，内容分进两个提交而大头在一个干净提交时人工填 A1 + 次模式 A5，
    # 而自动"拆分"只看 |P| ≥ 2——两者定义不同，主模式口径会系统性低估拆分的确认率。
    def confirm_table_post(rows_sel):
        out = {}
        for vn, labels in views:
            n = nU = k = 0
            for e, am in rows_sel:
                if e not in labels:
                    continue
                m = labels[e]
                sec = (A if vn == names[0] else B)[e]["sec"] if vn in names else (A[e]["sec"] | B[e]["sec"])
                exp = EXPECT[am] | ({"A2", "A3"} if key[e]["auto_pattern"] == "split+shared" else set())
                n += 1
                if m == "U":
                    nU += 1
                elif m in exp or (sec & exp):
                    k += 1
            out[vn] = (n, nU, k)
        return out

    L += ["", "**事后补充口径**（不在冻结计划内）：人工主模式**或次模式**落在对应集合内即算确认，自动\"拆分且共享\"对应 A2–A5。"
          "手册 v2.6 规定主模式取大部分新写入行所在的提交，内容分进两个提交而大头在一个干净提交时人工填 A1 + 次模式 A5，自动\"拆分\"只看 |P| ≥ 2，两者定义不同。", "",
          "| 自动模式 | n | " + " | ".join(vn for vn, _ in views) + " |", "|---|---|" + "---|" * len(views)]
    num["modes_post"] = {}
    for m in MODES:
        sel = [(e, key[e]["auto_mode"]) for e in ids if key[e]["stratum"][1] == m]
        ct = confirm_table_post(sel)
        num["modes_post"][m] = {vn: {"n": n, "U": nU, "confirmed": k} for vn, (n, nU, k) in ct.items()}
        L.append(f"| {m} | {len(sel)} | " + " | ".join(pct(k, n - nU) + (f"，U {nU}" if nU else "") for vn, (n, nU, k) in ct.items()) + " |")

    # 自动"缺失"按有无新写入特征行拆分（从片段级缓存读 n_new；缓存缺失时跳过）
    zero = zero_feature_flags()
    if zero:
        miss_all = [e for e in key if key[e]["auto_mode"] == "缺失"]
        z_all = [e for e in miss_all if zero.get(e)]
        miss_s = [e for e in ids if key[e]["auto_mode"] == "缺失"]
        L += ["", f"**自动\"缺失\"按有无新写入特征行拆分**（事后）：抽样框 {len(key)} 个片段中自动缺失 {len(miss_all)} 个（{len(miss_all) / len(key):.0%}），"
              f"其中没有任何 ≥ 15 字符新写入行的 {len(z_all)} 个（占全部片段 {len(z_all) / len(key):.0%}、占缺失 {len(z_all) / len(miss_all):.0%}）——"
              "短行、纯删除、重排都落在这里，自动判定对它们无法按内容核对而一律记为缺失。样本中：", "",
              "| 自动缺失子类 | n | " + " | ".join(f"{vn} 主模式分布" for vn, _ in views[:2]) + " |", "|---|---|---|---|"]
        num["missing_zero_feature"] = {"frame_missing": len(miss_all), "frame_zero": len(z_all), "sample": {}}
        for label, sel in (("无新写入特征行", [e for e in miss_s if zero.get(e)]), ("有新写入特征行", [e for e in miss_s if not zero.get(e)])):
            dist = [Counter(A[e]["main"] for e in sel), Counter(B[e]["main"] for e in sel)]
            num["missing_zero_feature"]["sample"][label] = {names[0]: dict(dist[0]), names[1]: dict(dist[1])}
            L.append(f"| {label} | {len(sel)} | " + " | ".join("；".join(f"{k} {v}" for k, v in sorted(d.items())) or "—" for d in dist) + " |")
    L.append("")

    # 两组的人工比例（反加权）
    L += ["## 3 两组的人工模式比例（反加权到两层总体）", "",
          "层 = 格 × 自动模式；权重为层总体数 / 组总体数（只含单独提交指令与片段内提交两层）；区间为分层抽样方差近似（含有限总体校正）的 ±1.96 SE。"
          "人工 U 计入分母（它既不是一对一也不是合并）。自动判定一行为同一总体上的自动比例（r25 §3 为全部四层，此处只含两层，数字略有不同）。", "",
          "| 组 | 两层总体 | 口径 | 一对一 | 合并（A2/A3） | 缺失（A9） | U |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        fg = [r for r in frame if r["group"] == g and r["layer"] in LAYERS]
        auto = {"一对一": sum(1 for r in fg if r["auto_mode"] == "一对一") / len(fg), "合并": sum(1 for r in fg if r["auto_mode"] == "共享") / len(fg),
                "缺失": sum(1 for r in fg if r["auto_mode"] in ("缺失", "证据不足")) / len(fg)}
        L.append(f"| {g} | {len(fg)} | 自动判定 | {auto['一对一']:.0%} | {auto['合并']:.0%} | {auto['缺失']:.0%} | — |")
        num["groups"][g]["N_two_layers"] = len(fg)
        num["groups"][g]["auto"] = auto
        for vn, labels in views:
            srows = [{"stratum": key[e]["stratum"], "main": labels[e]} for e in ids if key[e]["group"] == g and e in labels]
            if not srows:
                continue
            ests = {c: strat_estimate(fg, srows, lambda r, S=S: 1.0 if r["main"] in S else 0.0) for c, S in list(HUMAN_CLASSES.items()) + [("U", {"U"})]}
            num["groups"][g][vn] = {c: {"est": e, "se": se} for c, (e, se) in ests.items()}
            L.append(f"| {g} | | {vn}（n = {len(srows)}） | " + " | ".join(fmt_est(*ests[c]) for c in ("一对一", "合并", "缺失", "U")) + " |")
    L.append("")

    # 自动召回
    L += ["## 4 自动判定对人工合并与人工缺失的召回（反加权）", "",
          "| 组 | 口径 | 人工合并中自动为共享 | 人工缺失中自动为缺失 |", "|---|---|---|---|"]
    for g in GROUPS:
        fg = [r for r in frame if r["group"] == g and r["layer"] in LAYERS]
        for vn, labels in views:
            cells = []
            for hc in ("合并", "缺失"):
                srows = [{"stratum": key[e]["stratum"], "main": labels[e], "auto": key[e]["stratum"][1]} for e in ids if key[e]["group"] == g and e in labels]
                num_e, _ = strat_estimate(fg, srows, lambda r: 1.0 if r["main"] in HUMAN_CLASSES[hc] and r["auto"] == AUTO_FOR[hc] else 0.0)
                den_e, _ = strat_estimate(fg, srows, lambda r: 1.0 if r["main"] in HUMAN_CLASSES[hc] else 0.0)
                raw_k = sum(1 for r in srows if r["main"] in HUMAN_CLASSES[hc] and r["auto"] == AUTO_FOR[hc])
                raw_n = sum(1 for r in srows if r["main"] in HUMAN_CLASSES[hc])
                cells.append(f"{num_e / den_e:.0%}（未加权 {raw_k}/{raw_n}）" if den_e else "—")
                num["groups"][g].setdefault("recall", {}).setdefault(vn, {})[hc] = {"weighted": (num_e / den_e if den_e else None), "k": raw_k, "n": raw_n}
            L.append(f"| {g} | {vn} | " + " | ".join(cells) + " |")
    L += ["", "## 5 说明", "",
          "- 样本按格与自动模式分层并对拆分、缺失超额抽样，未加权的比例不代表总体；第 3、4 节已反加权，第 2 节的确认率按层报告、不外推。",
          "- 两人一致子集不是随机子集（分歧集中在难判的片段），其确认率偏高；两名编码人各自的确认率是无选择的估计。",
          "- 人工编码不用于修改对齐规则或自动判定；这些结果只回答自动判定在这一语料上的可信程度。", ""]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "r27_swechat_human.md").write_text("\n".join(L), encoding="utf-8")
    nums_p = RES / "numbers.json"
    nums = json.loads(nums_p.read_text(encoding="utf-8")) if nums_p.exists() else {}
    nums["swechat_human"] = json.loads(json.dumps(num, default=float))
    nums_p.write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


def write_disputes(ids, A, B):
    """分歧清单与匿名对照表（a / b 顺序逐条随机），供可选的裁决。"""
    dis = [e for e in ids if A[e]["main"] != B[e]["main"] or A[e]["sec"] != B[e]["sec"]]
    HUMAN.mkdir(parents=True, exist_ok=True)
    (HUMAN / "disputes_06.txt").write_text("\n".join(dis) + "\n", encoding="utf-8")
    raw = {}
    for (name, fn) in CODERS:
        for r in load(HUMAN / fn):
            raw[(name, r["episode"].strip())] = r
    rng = random.Random("S27-adjudicate")
    cols = ["main_pattern", "secondary_patterns", "causes", "confidence", "evidence_sufficient", "missing_info", "notes"]
    with open(HUMAN / "06_两份编码对照（匿名，a-b 顺序每行随机）.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + [f"{x}_{c}" for x in ("a", "b") for c in cols])
        for e in dis:
            pair = [raw[(CODERS[0][0], e)], raw[(CODERS[1][0], e)]]
            rng.shuffle(pair)
            w.writerow([e] + [p.get(c, "") for p in pair for c in cols])
    print(f"分歧 {len(dis)}/{len(ids)} 个，已写 {HUMAN / 'disputes_06.txt'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
