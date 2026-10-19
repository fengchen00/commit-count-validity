"""论文配图（只用汇总数据）。写入 <PAPER_DIR>/figures/。

文件名沿用历史编号，与论文图号的对应为：fig3_patterns = 图 2 可核验代码修改片段的模式分布；
fig4_tipping = 图 3 边界扰动；fig6_daily = 图 5 每日提交与工作量；fig7_ratio = 图 6 比率分解
（图 1 研究框架、图 4 案例时间线为手绘）。
旧图 fig2_modes（D42 删去，信息由表 3 给出）、fig4_granularity / fig6_multiplicity / fig7_lag（D35-7）已停用；
fig_modes() 保留但不再调用。
"""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR, FIGURES_DIR  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import matplotlib.patheffects  # noqa: E402
from workitems import RULES, coded_decisions, group, prepare, units, wilson  # noqa: E402
import figtext  # noqa: E402,F401  图内文字体例（半角标点、无中西文空格）与正文一致

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

OUT = FIGURES_DIR
SYS = {"SYS_A": "系统 A", "SYS_C": "系统 B"}
COL = {"SYS_A": "#3b6ea5", "SYS_C": "#d08a3c"}


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pairs(s):
    return [p for p in load(f"data/derived/{s}/episode_commit.csv") if p["source"] != "pathtime"]


def box(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.17), 0.84, 0.34, boxstyle="round,pad=0.02",
                                fc=color, ec="#444", lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=8)


def arrow(ax, a, b, dashed=False):
    ax.add_patch(FancyArrowPatch((a[0] + 0.44, a[1]), (b[0] - 0.44, b[1]), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color="#333", ls="--" if dashed else "-"))


def fig_modes():
    E, C, O = "#dbe7f3", "#f6e3c8", "#e8e8e8"
    panels = [
        ("一对一", [("片段", 1)], [("提交", 1)], [(0, 0)], []),
        ("共享提交", [("片段 1", 1.4), ("片段 2", 0.6)], [("提交", 1)], [(0, 0), (1, 0)], []),
        ("拆分", [("片段", 1)], [("提交 1", 1.4), ("提交 2", 0.6)], [(0, 0), (0, 1)], []),
        ("附带提交", [("片段", 1)], [("提交\n（任务 X）", 1)], [(0, 0)], ["标题指向另一任务"]),
        ("部分未提交 / 缺失", [("片段", 1)], [("提交", 1.4), ("（未入库）", 0.6)], [(0, 0), (0, 1, "--")],
         ["全部内容未入库即为缺失"]),
    ]
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 6, wspace=0.5, hspace=0.25, left=0.02, right=0.98, top=0.98, bottom=0.02)
    cells = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    for cell, (name, left, right, links, note) in zip(cells, panels):
        ax = fig.add_subplot(cell)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 2.3)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes,
                                    fc="none", ec="#999", lw=0.7, clip_on=False))
        ax.text(1, 2.08, name, ha="center", va="center", fontsize=10)
        for t, y in left:
            box(ax, 0, y, t, E)
        for t, y in right:
            box(ax, 2, y, t, O if t.startswith("（") else C)
        for i, j, *ls in links:
            arrow(ax, (0, left[i][1]), (2, right[j][1]), dashed=bool(ls))
        for t in note:
            ax.text(1, 0.12, t, ha="center", va="bottom", fontsize=7, color="#555")
    figtext.save(fig, OUT / "fig2_modes.png")
    plt.close(fig)


def fig_multiplicity():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    bins = ["1", "2", "3", "4", "5–9", "≥10"]

    def bucket(n):
        return str(n) if n <= 4 else ("5–9" if n < 10 else "≥10")

    for ax, view in zip(axes, ["commit", "episode"]):
        w = 0.38
        for k, s in enumerate(SYS):
            e2c, c2e = defaultdict(set), defaultdict(set)
            for p in pairs(s):
                e2c[p["episode"]].add(p["commit"])
                c2e[p["commit"]].add(p["episode"])
            groups = c2e if view == "commit" else e2c
            cnt = Counter(bucket(len(v)) for v in groups.values())
            ys = [cnt[b] / len(groups) for b in bins]
            xs = [i + (k - 0.5) * w for i in range(len(bins))]
            ax.bar(xs, ys, w, color=COL[s], label=f"{SYS[s]}（n = {len(groups)}）")
        ax.set_xticks(range(len(bins)), bins)
        ax.set_xlabel("每个提交包含的片段数" if view == "commit" else "每个片段进入的提交数")
        ax.set_title("（a）提交视角" if view == "commit" else "（b）片段视角", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("占比")
    fig.tight_layout()
    figtext.save(fig, OUT / "fig6_multiplicity.png")
    plt.close(fig)


def fig_patterns():
    """与表 3 可核验列同源：读取 render_tables.py 写出的 numbers.json（区间按会话重抽，D8）。"""
    rows = ["一对一", "合并（同会话或跨会话）", "　同会话合并", "　跨会话合并", "拆分", "　逐条任务拆分", "　非逐条拆分",
            "含未跟踪文件", "缺失"]
    num = json.loads((RESULTS_DIR / "numbers.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 2.5))  # v5.2:压矮(3.1→2.5)以放进表6所在页的余白
    h = 0.38
    for k, s in enumerate(SYS):
        st = [num[s]["verified"][r.strip("　")] for r in rows]
        n = num[s]["verified"]["n"]
        vals = [x["p"] for x in st]
        ci = [tuple(x["ci"]) for x in st]
        ys = [i + (k - 0.5) * h for i in range(len(rows))]
        ax.barh(ys, vals, h, color=COL[s], label=f"{SYS[s]}（n = {n}）",
                xerr=[[v - lo for v, (lo, _) in zip(vals, ci)], [hi - v for v, (_, hi) in zip(vals, ci)]],
                error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#555"})
        for y, v in zip(ys, vals):  # S33-A8：数值标签紧贴柱端（不是误差线端），白边描字保证压在误差线上仍可读
            ax.text(v + 0.008, y, f"{v:.0%}", va="center", fontsize=7, zorder=5,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")])
    disp = {"合并（同会话或跨会话）": "共享提交（同会话或跨会话）", "　同会话合并": "　同会话共享", "　跨会话合并": "　跨会话共享"}  # D32-5：显示名改"共享提交"，numbers.json 键不变
    ax.set_yticks(range(len(rows)), [disp.get(r, r) for r in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("占可核验代码修改片段的比例（可多标签；误差线为按会话重抽的 95% 区间）")
    ax.set_xlim(0, 0.95)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figtext.save(fig, OUT / "fig3_patterns.png")
    plt.close(fig)


def fig_granularity():
    x = {s: prepare(s) for s in SYS}
    rules = [r for r in RULES if r not in ("coded_hi",)]
    coded = coded_decisions()
    rules = [r for r in rules if coded or r != "coded"]
    names = {"episode": "片段", "strict": "严格规则聚合", "loose": "宽松规则聚合", "coded": "编码工作项", "session": "整会话"}
    pats = [("一对一", "一对一", "o-"), ("共享提交（并集）", "合并", "s--"), ("拆分", "拆分", "^:")]
    rob = json.loads((RESULTS_DIR / "numbers.json").read_text(encoding="utf-8"))["robustness"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, s in zip(axes, SYS):
        cf = defaultdict(set)
        for f in load(f"data/derived/{s}/episode_files.csv"):
            if f["class"] == "code":
                cf[f["episode"]].add(f["path"])
        ns = [rob[s]["refine_task_all_n"]]
        vals = {p: [rob[s][f"refine_task_all_{p[:2]}"]] for _, p, _ in pats}
        for r in rules:
            us, _ = units(x[s], group(x[s]["eps"], cf, r, coded))
            mapped = [u for u in us.values() if u["n_code"] and u["commits"]]
            ns.append(len(mapped))
            for _, p, _ in pats:
                vals[p].append(sum(p in u["labels"] for u in mapped) / len(mapped))
        for label, p, st in pats:
            ax.plot(range(1, len(rules) + 1), vals[p][1:], st, color=COL[s], label=label, lw=1.2, ms=4)
            ax.plot([0], vals[p][:1], st[0], color=COL[s], ms=4, mfc="white")
        ax.axvline(0.5, color="0.6", lw=0.6, ls=":")
        ax.set_xticks(range(len(rules) + 1), [f"{n_}\n(n = {n})" for n_, n in zip(["任务条目"] + [names[r] for r in rules], ns)], fontsize=6.5)
        ax.set_title(SYS[s], fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("占进入版本库单位的比例")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    figtext.save(fig, OUT / "fig4_granularity.png")
    plt.close(fig)


def fig_lag():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for s in SYS:
        eps = {e["episode"]: e for e in load(f"data/derived/{s}/episodes.csv")}
        lags = sorted((datetime.fromisoformat(p["commit_time"]) - datetime.fromisoformat(eps[p["episode"]]["last_event"]))
                      .total_seconds() / 3600 for p in pairs(s))
        n = len(lags)
        ax.step(lags, [(i + 1) / n for i in range(n)], where="post", color=COL[s], label=f"{SYS[s]}，片段—提交对（{n}）")
        last = {}
        for p in pairs(s):
            last[p["episode"]] = max(last.get(p["episode"], p["commit_time"]), p["commit_time"])
        el = sorted((datetime.fromisoformat(t) - datetime.fromisoformat(eps[e]["last_event"])).total_seconds() / 3600
                    for e, t in last.items())
        m = len(el)
        ax.step(el, [(i + 1) / m for i in range(m)], where="post", color=COL[s], ls="--",
                label=f"{SYS[s]}，片段末次提交（{m}）")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-16, 24 * 7)
    ticks = [-12, -1, 0, 1, 24, 24 * 7]
    ax.set_xticks(ticks, ["−12 小时", "−1 小时", "0", "1 小时", "1 天", "1 周"])
    for x in (1, 24):
        ax.axvline(x, color="#aaa", lw=0.6, ls=":")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("提交时间 − 片段最后事件时间（负值表示片段结束前已提交）")
    ax.set_ylabel("累积占比")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figtext.save(fig, OUT / "fig7_lag.png")
    plt.close(fig)


def fig_tipping():
    """边界扰动下的主要比例（系统 A，D17-4）；若有边界标注（D17-5），以竖线标出 ρ̂ 与 σ̂。"""
    R = json.loads((RESULTS_DIR / "numbers.json").read_text(encoding="utf-8"))["robustness6"]
    B = R["boundary_SYS_A"]
    bl = R.get("boundary_labels", {}).get("SYS_A|final|rates")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, kind, xl, marker in ((axes[0], "O", "过度切分 ρ（相邻边界被移除的概率）", "rho"), (axes[1], "U", "切分不足 σ（片段按文件拆成两份的概率）", "sigma")):
        xs = [lv for lv, _ in B[kind]]
        for key, lab, st in (("oto", "一对一", "o-"), ("merge", "共享提交", "s--"), ("split", "拆分", "^:")):
            med = [r[key]["med"] for _, r in B[kind]]
            lo = [r[key]["ci"][0] for _, r in B[kind]]
            hi = [r[key]["ci"][1] for _, r in B[kind]]
            ax.plot(xs, med, st, color="#3b6ea5", lw=1.9, ms=5, label=lab)
            ax.fill_between(xs, lo, hi, color="#3b6ea5", alpha=0.12, lw=0)
        ax.axhline(0.5, color="#999", lw=0.6, ls=":")
        if bl and bl.get(marker) == bl.get(marker):
            ax.axvline(bl[marker], color="#c0392b", lw=0.9, ls="-.")
            ax.text(bl[marker], 0.97, " 边界标注估计", color="#c0392b", fontsize=6.5, va="top")
        ax.set_xlabel(xl, fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("（a）", fontsize=9)
    axes[1].set_title("（b）", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[2]
    ms = [m for m, _ in B["U1"]]
    for key, lab, st in (("ratio_share", "平分口径", "o-"), ("ratio_count", "各计 1 口径", "s--")):
        ax.plot(ms, [r[key]["med"] for _, r in B["U1"]], st, color="#d08a3c", lw=1.9, ms=5, label=lab)
    ax.axhline(2, color="#999", lw=0.6, ls=":")
    works = R.get("boundary_labels", {}).get("SYS_A|final|rates", {}).get("new_works")
    if works and works[1]:
        ax.axvline(works[0] / works[1], color="#c0392b", lw=0.9, ls="-.")
        ax.text(works[0] / works[1], 8.8, " 边界标注估计", color="#c0392b", fontsize=6.5, va="top")
    ax.set_xlabel("片段内提交的片段按文件分为 q 份", fontsize=7.5)
    ax.set_ylabel("每单位分得提交数之比", fontsize=7.5)
    ax.set_ylim(0, 9)
    ax.set_title("（c）", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figtext.save(fig, OUT / "fig4_tipping.png")
    plt.close(fig)


def fig_daily():
    """每日提交数与代码编辑数、片段数（D17-2、D17-3）；颜色为当日事前登记片段的比例（D18-1）。"""
    from robustness import base  # noqa: E402
    from robustness6 import days_of  # noqa: E402
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)  # v5.1:压矮以与图4同页
    sc = None
    for s, mk in (("SYS_A", "o"), ("SYS_C", "^")):
        x, items, _ = base(s)
        days, _, _ = days_of(s, x, items)
        for ax, key, xl in ((axes[0], "w", "当日代码编辑数"), (axes[1], "n", "当日代码修改片段数")):
            sc = ax.scatter([d[key] for d in days], [d["c"] for d in days], c=[d["reg"] / d["n"] for d in days], cmap="coolwarm_r", vmin=0, vmax=1,
                            marker=mk, s=36, edgecolors="#222", linewidths=0.7, label=SYS[s])
            ax.set_xlabel(xl, fontsize=11)
    axes[0].set_ylabel("当日提交数", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    from matplotlib.lines import Line2D  # noqa: E402
    axes[0].legend(handles=[Line2D([], [], marker=m, ls="", mfc="#ccc", mec="#222", mew=0.7, ms=7, label=SYS[s]) for s, m in (("SYS_A", "o"), ("SYS_C", "^"))],
                   frameon=False, fontsize=9.5, loc="upper left")
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=9.5)
    cb.set_label("当日事前登记比例", fontsize=9.5)
    figtext.save(fig, OUT / "fig6_daily.png", bbox_inches="tight")
    plt.close(fig)


def fig_ratio():
    """比率分解（D30-4）：R_C = R_W^(u) × D^(u)。每种工作单位一行，画 R_W^(u)（日均工作量之比）与 D^(u)（每单位提交数之比）
    及其按日重抽区间；1.0 为基线，R_C 为以提交计的比值。数据来自 numbers.json 的 robustness8 键（r16 §1）。"""
    from robustness8 import UNITS  # noqa: E402
    R = json.loads((RESULTS_DIR / "numbers.json").read_text(encoding="utf-8"))["robustness8"]
    rc = R["episodes"]["daily_c"]
    # D40：按正文 2.5 节的三族排列显示——编码工作项属"请求的聚合"族，置于规则·宽松与整会话之间；只改显示顺序，不改 robustness8 的计算顺序
    order = ["episodes", "strict", "loose", "coded", "session", "items", "files", "edits", "works"]
    units = sorted(UNITS, key=lambda u: order.index(u[0]))
    # D40/D42：与表 11 同口径——开发者视角工作项只画格均值主模型：D^(u) 为 per_unit_cells（两层重抽区间），
    # R_W^(u) 由恒等式 R_C / D^(u) 得到、只画点估计（表 11 的 2.61）；合并均值模型为诊断对照，不入图。
    disp = {"strict": "严格规则聚合", "loose": "宽松规则聚合", "items": "请求中列举的事项(格式代理)",
            "works": "开发者视角工作项(格均值主模型)"}  # S33-B：显示名去间隔号，robustness8.UNITS 与 numbers.json 键不变
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ys = list(range(len(units)))[::-1]
    for y, (key, name) in zip(ys, units):
        r = R[key]
        if "per_unit_cells" in r:
            d, d_ci = r["per_unit_cells"], r["per_unit_cells_ci"]
            rw, rw_ci = rc / d, None
            ax.axhspan(y - 0.45, y + 0.45, color="#e9eef3", zorder=0)
        else:
            d, d_ci, rw, rw_ci = r["per_unit"], r["per_unit_ci"], r["daily_unit"], r["daily_unit_ci"]
        for dy, val, ci, mk, col, lab in ((0.16, rw, rw_ci, "o", "#3b6ea5", "$R_W^{(u)}$ 日均单位计数之比"),
                                          (-0.16, d, d_ci, "s", "#d08a3c", "$D^{(u)}$ 每单位提交数之比")):
            if ci is not None:
                ax.plot(ci, [y + dy, y + dy], color=col, lw=1.1)
            ax.plot([val], [y + dy], marker=mk, ms=5, mfc=col, mec="#333", mew=0.4, ls="", label=lab if y == ys[0] else None)
    ax.axvline(1.0, color="#555", lw=0.8, ls="--")
    ax.axvline(rc, color="#777", lw=0.8, ls=":", label=f"$R_C$ = {rc:.2f}")
    from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402
    ax.set_xscale("log")
    ax.set_xlim(0.25, 8)
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.6, len(units) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([disp.get(k, n) for k, n in units])
    ax.set_xlabel("系统 A 事前登记比例高、低两类日子之比（对数坐标；横线为 95% 重抽区间，开发者视角工作项的 $R_W^{(u)}$ 只报点估计）")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.14), ncol=3)
    figtext.save(fig, OUT / "fig7_ratio.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_patterns()
    fig_tipping()
    fig_daily()
    fig_ratio()
    for stale in ("fig2_modes.png", "fig4_granularity.png", "fig6_multiplicity.png", "fig7_lag.png"):
        p = OUT / stale
        if p.exists():
            p.unlink()
    print("figures ->", OUT)
