# 复现包：编码智能体辅助开发中提交计数的度量效度

本包配合论文《编码智能体辅助开发中提交计数的度量效度》。包含分析脚本、编码手册、编码表模板、论文中的汇总结果，以及一份与真实导出格式相同的合成示例数据生成器。

会话原文、源代码与客户文档属于案例企业的商业数据，不能公开。工业案例的正文表 1、表 5、表 6、附录各表与图 2–5、附录图 D1 不能由第三方从原始数据重算。包内给出这些表、图的汇总结果，供核对。

许可：CC BY 4.0。

## 目录

| 路径 | 内容 |
|---|---|
| `analysis/` | 分析脚本。`run_all.ps1` 为全部检验的入口 |
| `run_core.ps1` | 核心流程：对齐、模式判定、召回与参照评估。不需要人工编码表 |
| `tools/make_synthetic.py` | 生成合成示例数据到 `data/raw/` |
| `manuals/` | 编码手册、边界编码手册、研究设计、分析计划 |
| `templates/` | 空的编码表模板 |
| `results_paper/` | 论文所报告的汇总结果（`r*.md`、`tables.md`、`numbers.json`） |
| `results_paper/figures/` | 脚本输出的图。文件名沿用历史编号，与正文图号的对应见下表 |
| `data/coding/swechat/` | 公开语料的抽样框、80 片段样本、置换与两名编码人的编码表 |
| `data/raw/`、`data/derived/` | 空。合成数据或自有数据放这里 |
| `data/swechat/` | 空。SWE-chat 原始数据须自行申请后放这里 |
| `swechat_mirror_manifest.json` | 五个公开仓库镜像的克隆时间、HEAD 与引用 |
| `doc/v4.0/results/`、`doc/v4.0/figures/` | 空。重新运行时脚本把新结果写到这里 |

## 用合成数据检查流程

在仓库根目录执行：

```powershell
python tools/make_synthetic.py
.\run_core.ps1
```

合成场景切分后为 22 个片段（系统 A 16 个、系统 B 6 个），用来检查实现是否自洽，不能复算论文中的表和图。脚本里系统 B 的键是 `SYS_C`，输出时写成「系统 B」。

重新运行时，明细写到 `data/derived/`，汇总写到 `doc/v4.0/results/`。包里已经算好的论文结果在 `results_paper/`，不要和重跑结果混读。

## 公开语料

SWE-chat v1 修订 `f66cca95`（ODC-BY，需先接受条款）。四个 parquet 放在 `data/swechat/`。五个公开仓库按 `swechat_mirror_manifest.json` 自行克隆。二者缺失时，`run_all.ps1` 跳过公开语料步骤，并保留 `results_paper/` 里的结果表。

随包提供的只有抽样框、样本、置换和编码表。盲评证据包不包含在本包中。

## 与正文的对应

结果文件名保持生成时的历史编号。下表按现行正文的表号、图号对应。

| 正文 | 脚本 | 结果 |
|---|---|---|
| 表 1 资料概况 | `render_tables.py` | `results_paper/tables.md` 表 1 |
| 表 2 工作单位 | 文字表 | 正文 2.2 节；规则见 `manuals/research_design.md` |
| 表 3 映射模式 | 文字表 | `manuals/coding_manual.md` |
| 表 4 人工验证设计 | 文字表 | `manuals/coding_manual.md`、`manuals/boundary_manual.md` |
| 表 5 模式分布 | `render_tables.py`、`patterns.py` | `tables.md` 表 6；`r13_robustness5.md` |
| 表 6 比率分解 | `robustness7.py`、`robustness8.py` | `r15_robustness7.md`、`r16_robustness8.md` |
| 表 7 公开语料 | `swechat_replicate.py`、`swechat_strata.py`、`swechat_agreement.py` | `r21_swechat.md`、`r25_swechat_strata.md`、`r27_swechat_human.md` |
| 表 8 证据性质 | 文字表 | 正文 4.1 节 |
| 附录表 A1 检验流程 | 文字表 | `manuals/analysis_plan.md` |
| 附录表 C1 参照评估 | `agent_commits.py`、`align_eval.py` | `r18_align_eval.md` |
| 附录表 C2 对级盲评 | `pair_validation.py` | `r20_pair_validation.md` |
| 附录表 C3 边界标注 | `boundary_labels.py` | `r14_boundary.md` |
| 附录表 D1 工作单位下的模式 | `workitems.py`、`render_tables.py` | `tables.md` 表 7 |
| 附录表 D2 片段内提交 | `robustness5.py` | `r13_robustness5.md` |
| 附录表 E1 分期敏感性 | `robustness7.py`、`robustness9.py` | `r15_robustness7.md`、`r17_robustness9.md` |
| 附录表 F1 公开语料人工核验 | `swechat_agreement.py` | `r27_swechat_human.md` |
| 图 1 研究框架 | `fig_schematics.py` | `figures/fig1_overview.png` |
| 图 2 模式分布 | `figures.py` `fig_patterns` | `figures/fig3_patterns.png` |
| 图 3 案例 | `fig_schematics.py` | `figures/fig5_case.png` |
| 图 4 每日提交与活动 | `figures.py` `fig_daily` | `figures/fig6_daily.png` |
| 图 5 比率分解 | `figures.py` `fig_ratio` | `figures/fig7_ratio.png` |
| 附录图 A1 研究框架（完整版） | `fig_schematics.py` | `figures/fig1_method.png` |
| 附录图 D1 边界扰动 | `figures.py` `fig_tipping` | `figures/fig4_tipping.png` |

图 2、图 4、图 5、附录图 D1 与投稿稿件中的嵌入图一致。图 1、图 3、附录图 A1 的脚本输出是上表这些文件；稿件排版时改过幅面，投稿图包里的 PNG 与这三个脚本输出文件不是同一张。

算法 1 在正文 2.3 节。公开语料对照在正文 3.4 节，移植时的九处决定在附录 F，实现对应 `swechat_replicate.py`。

## 运行范围

`run_core.ps1` 只跑对齐、模式判定、召回与参照评估，使用合成数据或自有导出即可。

`analysis/run_all.ps1` 是更长的入口，不是“从原始工业数据重建论文”的开关。不设环境变量时，编码评分读 `data/coding/sim/`（模拟编码，只供预演）。工业案例的真人编码须先备好未公开的编码表，再在 PowerShell 里执行：

```powershell
$env:CODING_HUMAN = "1"
.\analysis\run_all.ps1
```

该入口不调用 `analysis/cells_sensitivity.py`。这个脚本对应附录 E.6 的格计数敏感性，须在 `CODING_HUMAN=1` 且边界标注、`robustness7.py` 已经跑过之后单独执行，并依赖未公开的工业人工编码表。没有客户 Word 文档时，`trace_customer.py` 会因条目数为 0 而失败；这不影响论文中已经给出的汇总结果。

公开语料的原始 parquet 与五个仓库镜像不在包内。SWE-chat 须先在其发布页接受访问条款；缺失时相关步骤跳过，以 `results_paper/` 里的表为准。

## 环境

Windows PowerShell 5.1 或 7，Python 3.11 及以上，Git 2.40 及以上。核心流程（`run_core.ps1`、合成数据、作图）依赖 numpy、matplotlib。公开语料脚本另需 pandas、pyarrow。读取客户 Word 文档的步骤另需 python-docx。在仓库根目录运行。核心流程未在 Linux 上作为投稿检查的正式环境。
