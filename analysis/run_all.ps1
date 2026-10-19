$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
# D29：$env:CODING_HUMAN = "1" 时各编码评分脚本读 data/coding/human/（真人编码），否则读 data/coding/sim/（AI 模拟，预演用）
# 本清单不包含 cells_sensitivity.py。该脚本对应附录 E.6，须在 CODING_HUMAN=1 且 robustness7.py 之后单独运行，并依赖未公开的工业人工编码表。没有客户文档时 trace_customer.py 会失败。
$human = ($env:CODING_HUMAN -eq "1")
$codingDir = if ($human) { "data/coding/human" } else { "data/coding/sim" }

$steps = @(
    "extract_git", "extract_sessions", "extract_episodes", "summarize_episodes",
    "attribute_edits", "checkpoint_deltas", "map_episodes", "patterns", "taskbook_layer",
    "summarize_mapping", "sensitivity", "rq3_bias", "trace_customer", "sim_customer_check", "sim_agreement", "align", "summarize",
    "workitems", "agent_commits", "rq2_factors", "render_tables", "multi_attrib", "align_eval", "holdout_agreement", "robustness", "robustness2", "robustness3", "robustness4", "robustness5", "weighting_table", "robustness6", "robustness7", "unmapped_commits", "robustness8", "robustness9", "figures"
)
$i = 0
foreach ($s in $steps) {
    $i++
    Write-Output ("[{0}/{1}] {2}  {3:HH:mm:ss}" -f $i, $steps.Count, $s, (Get-Date))
    python "analysis/$s.py"
    if ($LASTEXITCODE -ne 0) { throw "step failed: $s (exit $LASTEXITCODE)" }
    if ($s -eq "sim_agreement" -and -not $human) {
        python "analysis/sim_agreement.py" sim2
        if ($LASTEXITCODE -ne 0) { throw "step failed: sim_agreement sim2 (exit $LASTEXITCODE)" }
    }
    if ($s -eq "robustness6") {
        foreach ($v in "v1", "v2") {
            $dir = if ($v -eq "v1") { "boundary" } else { "boundary_v2" }
            $probe = if ($human) { "$codingDir/$dir/coder2_part1.csv" } else { "$codingDir/$dir/coder2_part4.csv" }
            if (Test-Path $probe) {
                python "analysis/boundary_labels.py" score $v
                if ($LASTEXITCODE -ne 0) { throw "step failed: boundary_labels score $v (exit $LASTEXITCODE)" }
            }
        }
    }
    if ($s -eq "workitems") {
        $adjDir = if ($human) { "$codingDir/workitem_pairs" } else { "data/coding/workitem_pairs" }
        $modes = if ($human) { @("score_full") } else { @("score", "score_full") }
        foreach ($m in ($modes | Where-Object { Test-Path "$adjDir/adjudicate_all.csv" })) {
            python "analysis/workitem_pairs.py" $m
            if ($LASTEXITCODE -ne 0) { throw "step failed: workitem_pairs $m (exit $LASTEXITCODE)" }
        }
    }
    if ($s -eq "align_eval") {
        # 片段—提交对级盲评（D26-1）：有编码表时评分，之后附上未找回对的可见性刻画；PV_SIM=1 时读模拟目录
        $pvdir = if ($env:PV_SIM -eq "1") { "data/coding/sim/pair_validation" } else { "data/coding/pair_validation" }
        if (Test-Path "$pvdir/coder1.csv") {
            python "analysis/pair_validation.py" score
            if ($LASTEXITCODE -ne 0) { throw "step failed: pair_validation score (exit $LASTEXITCODE)" }
        }
        python "analysis/pair_validation.py" characterize
        if ($LASTEXITCODE -ne 0) { throw "step failed: pair_validation characterize (exit $LASTEXITCODE)" }
    }
    if ($s -eq "robustness9") {
        # SWE-chat 外部复制（D27-3）：需要 data/swechat/*.parquet 与 $env:TEMP/swechat_probe 下的镜像（或 SWECHAT_MIRRORS）
        $mir = if ($env:SWECHAT_MIRRORS) { $env:SWECHAT_MIRRORS } else { Join-Path $env:TEMP "swechat_probe" }
        if ((Test-Path "data/swechat/conversations.parquet") -and (Test-Path "$mir/obsessiondb__rudel.git")) {
            python "analysis/swechat_replicate.py"
            if ($LASTEXITCODE -ne 0) { throw "step failed: swechat_replicate (exit $LASTEXITCODE)" }
            # D35-6 B5/B6 + D36-5：逐仓库对照、按提交触发方式分层、按无歧义归属标准切分（r25；约 12 分钟）
            python "analysis/swechat_strata.py"
            if ($LASTEXITCODE -ne 0) { throw "step failed: swechat_strata (exit $LASTEXITCODE)" }
            # D38（S27 / B9）：公开语料人工核验的抽样（已有样本则不覆盖）；约 12 分钟
            python "analysis/swechat_sample.py"
            if ($LASTEXITCODE -ne 0) { throw "step failed: swechat_sample (exit $LASTEXITCODE)" }
        } else {
            Write-Output "skip swechat_replicate: data/swechat or mirrors missing (results/r21_swechat.md kept as is)"
        }
        # D38：两名编码人与自动判定的对照（r27）。只需 data/coding/swechat/ 下的抽样框、样本与编码表（随包提供）；编码表缺失时只打印提示；
        # "自动缺失按有无特征行拆分"一节需要片段级缓存，缓存缺失时跳过该节
        if (Test-Path "data/coding/swechat/frame.csv") {
            python "analysis/swechat_agreement.py"
            if ($LASTEXITCODE -ne 0) { throw "step failed: swechat_agreement (exit $LASTEXITCODE)" }
        }
        # D35-6 B4：去继承口径（r24）
        python "analysis/deinherit_check.py"
        if ($LASTEXITCODE -ne 0) { throw "step failed: deinherit_check (exit $LASTEXITCODE)" }
    }
}
Write-Output "done: results written under config.PAPER_DIR (see analysis/config.py)"
