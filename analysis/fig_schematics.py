"""示意图（不依赖数据）：图 1 研究框架、图 4 案例的片段视角与提交视角。

用法：python analysis/fig_schematics.py [输出目录]   默认 doc/v4.9/figures
取代此前手工渲染的 Mermaid 源（doc/v4.0/figures/fig1_method.mmd、fig5_case.mmd）。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figtext  # noqa: E402,F401  图内文字体例与正文一致

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "doc" / "v4.9" / "figures"
FILL = {"data": "#f2f2f2", "step": "#e8eef7", "eval": "#fff4e0", "out": "#eaf4ea", "plain": "#ffffff"}
EDGE = "#404040"


def box(ax, x, y, w, h, text, kind="step", fs=8.5, bold_first=False, lw=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=FILL[kind], ec=EDGE, lw=lw))
    if bold_first and "\n" in text:
        first, rest = text.split("\n", 1)
        ax.text(x, y + h * 0.18, first, ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(x, y - h * 0.2, rest, ha="center", va="center", fontsize=fs - 1, color="#333")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.35)


def arrow(ax, a, b, dashed=False, color=EDGE, rad=0.0, shrink=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=shrink, shrinkB=shrink,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.axis("off")

    # ---------- (a) ----------
    ax.text(1, 101, "(a) 会话—提交内容对齐与验证", fontsize=10.5, fontweight="bold", va="center")
    y1, y2, y3 = 93, 81, 69
    bw, bh = 27, 8
    box(ax, 16, y1, bw, bh, "会话导出\n消息、工具调用、工作区检查点", "data")
    box(ax, 50, y1, bw, bh, "Git 仓库\n逐提交新增行、文件版本", "data")
    box(ax, 84, y1, bw, bh, "智能体提交命令\n独立于对齐规则的参照", "data")
    box(ax, 16, y2, bw, bh, "工作片段切分\n按请求边界;去重、时间戳修复", "step")
    box(ax, 50, y2, bw, bh, "算法 1:按内容归属\n逐次编辑 + 检查点差分", "step")
    box(ax, 50, y3, bw, bh, "片段—提交对 P\n证据分层:可核验 / 部分 / 推定", "out")
    box(ax, 84, y3, bw, bh, "评估\n召回率、参照一致率、对级盲评\n对照:按时间归属基线", "eval", fs=8)
    arrow(ax, (16, y1 - bh / 2), (16, y2 + bh / 2))
    arrow(ax, (50, y1 - bh / 2), (50, y2 + bh / 2))
    arrow(ax, (16 + bw / 2, y2), (50 - bw / 2, y2))
    arrow(ax, (50, y2 - bh / 2), (50, y3 + bh / 2))
    arrow(ax, (50 + bw / 2, y3), (84 - bw / 2, y3))
    arrow(ax, (84, y1 - bh / 2), (84, y3 + bh / 2), dashed=True, color="#777")
    ax.text(85.5, (y1 + y3) / 2 + 1.5, "不参与对齐\n只用于评估", fontsize=7.5, color="#666", va="center")

    # divider
    ax.plot([0, 100], [62, 62], color="#bbb", lw=0.8)

    # ---------- (b) ----------
    ax.text(1, 59.5, "(b) 提交计数度量效度的检验流程 S1–S8", fontsize=10.5, fontweight="bold", va="center")
    ys1, ys2, ys3, ys4 = 51, 38, 25, 10
    # inputs to (b)
    arrow(ax, (50, y3 - bh / 2), (50, ys1 + 3.5))
    box(ax, 48, ys1, 38, 7, "S1 映射模式判定\n一对一、共享提交、拆分、缺失等;可多标签", "step", bold_first=True)
    box(ax, 85, ys1, 26, 7, "日提交数(Git)、片段与编辑计数\n请求文本的分期变量", "data", fs=7.6)

    sw, sh = 22, 8.5
    xs = [13, 37, 61, 85]
    box(ax, xs[0], ys2, sw, sh, "S2 人工编码校正\n原样本 + 留出样本\n两名编码人 + 开发者", "step", bold_first=True)
    box(ax, xs[1], ys2, sw, sh, "S3 边界扰动\n过度 / 不足切分\n预设临界点", "step", bold_first=True)
    box(ax, xs[2], ys2, sw, sh, "S4 开发者视角边界标注\n按标注边界重算", "step", bold_first=True)
    box(ax, xs[3], ys2, sw, sh, "S5 按事前登记分期\n每单位提交数之比\n置换检验、计数模型", "step", bold_first=True)
    # S1 -> S2..S4（母线 + 竖直箭头）
    ax.plot([xs[0], xs[2]], [ys1 - 5.5, ys1 - 5.5], color=EDGE, lw=0.9)
    ax.plot([48, 48], [ys1 - 3.5, ys1 - 5.5], color=EDGE, lw=0.9)
    for x in xs[:3]:
        arrow(ax, (x, ys1 - 5.5), (x, ys2 + sh / 2))
    # inputs -> S5
    arrow(ax, (85, ys1 - 3.5), (85, ys2 + sh / 2))
    box(ax, 61, ys3, sw, sh, "S6 九种工作单位\n比率分解 $R_C = R_W^{(u)} \\times D^{(u)}$\n按单位族判定", "step", bold_first=True)
    box(ax, 85, ys3, sw, sh, "S7 仅用版本库的代理\n提交信息带任务编号比例", "step", bold_first=True)
    arrow(ax, (85, ys2 - sh / 2), (85, ys3 + sh / 2))
    # S5 -> S6
    arrow(ax, (77, ys2 - sh / 2), (69, ys3 + sh / 2))
    arrow(ax, (58, ys2 - sh / 2), (58, ys3 + sh / 2))
    ax.text(57, (ys2 + ys3) / 2, "工作项单位", fontsize=7, color="#555", va="center", ha="right")
    box(ax, 50, ys4, 60, 7, "S8 证据性质分类\n可由机制解释 / 描述性量级 / 经验发现 / 辅助分析 / 探索性对照", "out", bold_first=True)
    for x in (xs[0], xs[1]):
        arrow(ax, (x, ys2 - sh / 2), (x, ys4 + 3.5))
    arrow(ax, (61, ys3 - sh / 2), (61, ys4 + 3.5))
    arrow(ax, (85, ys3 - sh / 2), (80, ys4 + 3.5))
    ax.text(1, 3.2, "注:工作片段的边界同时是 S3、S4 的检验对象;S5 的主口径不经过内容对齐.", fontsize=7.5, color="#555")
    fig.tight_layout(pad=0.3)
    figtext.save(fig, OUT / "fig1_method.png")
    plt.close(fig)


def fig4():
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    # 泳道底色与标签
    lane_e, lane_c = (35, 50), (10, 24)
    for (y0, y1), label in ((lane_e, "片段视角\n(会话)"), (lane_c, "提交视角\n(Git)")):
        ax.add_patch(Rectangle((11, y0), 89, y1 - y0, fc="#f6f6f6", ec="none", zorder=0))
        ax.text(5.5, (y0 + y1) / 2, label, fontsize=8.5, va="center", ha="center", color="#333")

    # 时间轴（底部）
    ty = 4.5
    ax.annotate("", xy=(99, ty), xytext=(11, ty), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=EDGE))
    ax.text(98.5, ty + 1.2, "时间", fontsize=7.5, ha="right", va="bottom", color="#333")
    x_e1, x_e2, x_dep, x_c1, x_c2 = 20, 41, 58, 78, 94
    ticks = [(x_e1, "片段 1 开始"), (x_e2, "片段 2 开始"), (x_dep, "部署上线"), (x_c1, "提交 1"), (x_c2, "提交 2")]
    for x, t in ticks:
        ax.plot([x, x], [ty - 1.0, ty + 1.0], color=EDGE, lw=0.9)
        ax.text(x, ty - 1.8, t, fontsize=7.3, ha="center", va="top", color="#333")
    # 25 分钟标注
    ax.annotate("", xy=(x_c1, ty + 2.8), xytext=(x_dep, ty + 2.8), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#777"))
    ax.text((x_dep + x_c1) / 2, ty + 3.4, "25 分钟", fontsize=7.3, ha="center", va="bottom", color="#555")
    # 部署的竖向参考线（改动先上线、后入库）
    ax.plot([x_dep, x_dep], [ty + 1, lane_e[1] + 0.5], color="#999", lw=0.8, ls=(0, (2, 2)))
    ax.text(x_dep + 0.8, lane_e[1] + 0.6, "改动先直接从工作区部署,后补入版本库", fontsize=6.8, color="#666", va="bottom", ha="left")

    # 片段框（左边缘对齐开始时刻）
    bh = 11
    ye = sum(lane_e) / 2
    box(ax, x_e1 + 9, ye, 18, bh, "片段 1\n请求:列表页筛选\n改为两行", "step", fs=7.5)
    box(ax, x_e2 + 10.5, ye, 21, bh, "片段 2\n请求:返工,标签文字对齐\n智能体另自行调整页面 B 样式", "step", fs=7.5)
    for x in (x_e1, x_e2):
        ax.plot([x, x], [ty + 1, ye - bh / 2], color="#bbb", lw=0.7, ls=(0, (1.5, 2)), zorder=0)
    # 提交框
    yc = sum(lane_c) / 2
    box(ax, x_c1, yc, 19, bh, "提交 1(无标签热修)\n筛选与标签对齐", "out", fs=7.5)
    box(ax, x_c2, yc, 11, bh, "提交 2\n(无标签热修)\n页面 B 样式", "out", fs=7)
    for x in (x_c1, x_c2):
        ax.plot([x, x], [ty + 1, yc - bh / 2], color="#bbb", lw=0.7, ls=(0, (1.5, 2)), zorder=0)

    # 内容去向
    def flow(a, b, color, text, dashed=False):
        arrow(ax, a, b, color=color, lw=1.1, dashed=dashed)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my, text, fontsize=7.2, color=color, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.6))
    flow((x_e1 + 12, ye - bh / 2), (x_c1 - 6, yc + bh / 2), "#1f4e79", "主体改动")
    flow((x_e2 + 8, ye - bh / 2), (x_c1 + 1, yc + bh / 2), "#1f4e79", "主体改动")
    flow((x_e2 + 17, ye - bh / 2), (x_c2 - 1, yc + bh / 2), "#b05a00", "额外改动", dashed=True)
    fig.tight_layout(pad=0.3)
    figtext.save(fig, OUT / "fig5_case.png")
    plt.close(fig)


def fig1_simple():
    """图 1 简图（v5.3 正文用）：两条线——内容对齐→对应结构,直接计数→比率稳定性;完整 S1–S8 版进附录."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    xl, xr = 27, 73
    bw, bh = 40, 12
    ys = [90, 66, 42, 18]
    rows = [
        (("会话记录\n开发者请求、智能体编辑、工作区检查点", "data"), ("Git 记录\n提交、逐提交新增行与文件版本", "data")),
        (("会话—提交内容对齐\n按新写入内容归属;检查点差分补足", "step"), ("直接计数\n日提交数;片段、编辑等工作单位计数", "step")),
        (("验证\n智能体提交命令参照、对级盲评、人工编码", "eval"), ("多单位比较\n事前登记比例分期;九种工作单位", "eval")),
        (("对应结构(RQ2)\n一对一、共享提交、拆分及其边界依赖", "out"), ("比率稳定性(RQ3)\n$D^{(u)}$ 的判定:稳定 / 不稳定 / 证据不足", "out")),
    ]
    for y, (left, right) in zip(ys, rows):
        box(ax, xl, y, bw, bh, left[0], left[1], fs=8.2, bold_first=True)
        box(ax, xr, y, bw, bh, right[0], right[1], fs=8.2, bold_first=True)
    for i in range(3):
        arrow(ax, (xl, ys[i] - bh / 2), (xl, ys[i + 1] + bh / 2))
        arrow(ax, (xr, ys[i] - bh / 2), (xr, ys[i + 1] + bh / 2))
    # 两种记录都进入内容对齐;Git 记录也进入直接计数
    arrow(ax, (xr - bw / 2, ys[0] - bh / 2 + 1), (xl + 6, ys[1] + bh / 2), rad=0.0, color="#777")
    arrow(ax, (xl + bw / 2, ys[0] - bh / 2 + 1), (xr - 6, ys[1] + bh / 2), rad=0.0, color="#777")
    ax.text(50, ys[1] - bh / 2 - 5.5, "RQ1:对齐方法的召回率与局限", fontsize=7.8, ha="center", color="#444")
    ax.text(50, 3, "比率稳定性的主口径使用直接计数,不经过内容对齐;检验流程 S1–S8 见附录图 A1.",
            fontsize=7.5, ha="center", color="#555")
    fig.tight_layout(pad=0.3)
    figtext.save(fig, OUT / "fig1_overview.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--simple" in sys.argv:
        fig1_simple()
    else:
        fig1()
        fig4()
    print("figures ->", OUT)
