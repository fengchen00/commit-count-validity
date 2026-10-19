# 复现包核心流程：会话—提交内容对齐与模式判定（论文 2.3 节算法 1 与附录 A 的 S1–S4）。
# 只依赖 data/raw 下的 Git 仓库与会话导出，不需要人工编码表；合成示例数据可直接跑通。
# 完整流程（含人工编码一致性、留出样本、边界扰动等）见 analysis/run_all.ps1 与 README。
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

$steps = @(
    "extract_git",        # 提交、提交文件、任务书任务（S1 数据）
    "extract_sessions",   # 会话事件、开发者消息、编辑特征行
    "extract_episodes",   # 工作片段切分（D4）
    "summarize_episodes", # 覆盖与质量报告
    "attribute_edits",    # 算法 1：按内容把编辑归属到提交
    "checkpoint_deltas",  # 检查点差分（D3b）
    "map_episodes",       # 片段 × 提交映射
    "patterns",           # 映射模式（一一对应 / 拆分 / 合并 …）
    "taskbook_layer",     # 任务书层：事前登记
    "summarize_mapping",  # R1 汇总表
    "sensitivity",        # 阈值与口径敏感性
    "align",              # 任务时钟
    "agent_commits",      # 智能体提交命令：召回率参照（S2）
    "align_eval"          # 召回、精确率、按时间归属基线
)
foreach ($s in $steps) {
    Write-Output "== $s"
    python "analysis/$s.py"
    if ($LASTEXITCODE -ne 0) { throw "step failed: $s (exit $LASTEXITCODE)" }
}
Write-Output "done: 明细在 data/derived/<系统>/，汇总在 doc/v4.0/results/"
