"""片段—提交对级盲评（S19-T-E3，口径见 notes/decisions.md D26-1）。

独立盲审共识问题 C1：表 5 对"基线独有对"的核对以内容归属为准，是用被评估方法裁判对手。本脚本从四类候选对中混抽，
生成不含来源类别、不含自动候选的证据包，由独立编码人回答"该片段的内容是否进入该提交"，再按类别汇总人工判定。

  python analysis/pair_validation.py sample            抽样，写 <OUT>/key.csv（含类别，编码人不可见）与 <OUT>/manifest.csv
  python analysis/pair_validation.py packets           生成 <OUT>/packets/<对号>.md 与录入模板 <OUT>/coder_template.csv
  python analysis/pair_validation.py score [coder.csv] 汇总，写 <PAPER_DIR>/results/r20_pair_validation.md 与 numbers.json 的 pair_validation 键

四类（系统 A，种子固定）：
  both          内容对齐与按时间归属都给出的对（抽 QUOTA 对）
  content_only  只有内容对齐给出的对（全取）
  baseline_only 只有按时间归属给出的对（表 5 中"基线独有对"）
  missed_ref    参照关系（片段执行提交命令产生该提交，且提交改动了片段编辑过的代码文件）中内容对齐未找回的对
后两类按"可核对"分层（D29-3 严格层）：片段在重叠代码文件上的**全部**编辑都有写入内容、或检查点在这些文件上有新增行，为可核对；
可核对层全取（普查，不抽样），其余（含部分编辑无内容的对）为不可核对层，抽 QUOTA_UNCHK 对核实 U 率。
（D26-1 的旧口径为"任一编辑有写入内容"并抽 25 对；模拟第四轮显示该层 76% 只能答 U，与手册 v2.5 的 U 条不配套，故改。）
答案：Y 是（片段的内容进入了该提交）/ P 部分 / N 否 / U 无法判断。
环境变量 PV_SIM=1 时目录为 data/coding/sim/pair_validation（AI 模拟），否则为 data/coding/pair_validation（真人）。
"""
import csv
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from agent_commits import commit_sources, load  # noqa: E402
from align_eval import baseline_pairs  # noqa: E402
from build_packets import hm, quote, redact  # noqa: E402
from checkpoint_deltas import WS_ROOT, load_checkpoints  # noqa: E402
from extract_git import SYSTEMS  # noqa: E402
from extract_sessions import SYSTEMS as SESS  # noqa: E402
from workitems import wilson  # noqa: E402

SYS = "SYS_A"
QUOTA = 25
QUOTA_UNCHK = 5  # 不可核对层每类抽取数
SEED = "D26-pair-validation"
SIM = os.environ.get("PV_SIM", "0") == "1"
OUT = Path("data/coding") / ("sim/pair_validation" if SIM else "pair_validation")
NUMS = RESULTS_DIR / "numbers.json"
MAX_LINES_EDIT, MAX_LINES_CP, MAX_LINES_DIFF, MAX_FILES = 60, 60, 120, 8  # 差异按 ★ 文件逐个截断（模拟第一轮后调整）
MAX_STAR_EDITS = 30  # 显示写入内容的 ★ 编辑上限；超出者明确标"（有写入内容，包中未显示）"（模拟第三轮后调整）
CATS = ["both", "content_only", "baseline_only", "missed_ref"]
CAT_ZH = {"both": "两法一致", "content_only": "仅内容对齐", "baseline_only": "仅时间基线", "missed_ref": "参照中内容对齐未找回"}


def universe():
    d = Path("data/derived") / SYS
    commits, gold, _ = commit_sources(SYS)
    eps = {e["episode"]: e for e in load(d / "episodes.csv")}
    files_of_c = defaultdict(set)
    for f in load(d / "commit_files.csv"):
        files_of_c[f["hash"]].add(f["path"].lower())
    code_files = defaultdict(set)
    for f in load(d / "episode_files.csv"):
        if f["class"] == "code":
            code_files[f["episode"]].add(f["path"].lower())
    content = defaultdict(set)
    for p in load(d / "episode_commit.csv"):
        if p["source"] != "pathtime":
            content[p["episode"]].add(p["commit"])
    base = baseline_pairs(d, eps, commits)
    cp = {(e, c) for e, cs in content.items() for c in cs}
    bp = {(e, c) for e, cs in base.items() for c in cs}
    ref = {(ep, c) for c, ep in gold.items() if code_files[ep] & files_of_c[c]}
    pools = {"both": sorted(cp & bp), "content_only": sorted(cp - bp), "baseline_only": sorted(bp - cp),
             "missed_ref": sorted(p for p in ref if p not in cp)}
    return pools, eps, commits, files_of_c, code_files


def visibility(d, eps, files_of_c, code_files, pairs):
    """对每个 (episode, commit) 判断可核对性：any_sig = 片段在重叠代码文件上任一编辑有写入内容；all_sig = 全部重叠编辑都有写入内容（且至少一次）；
    has_cp = 检查点在这些文件上有新增行。返回 {(episode, commit): (any_sig, has_cp, strict)}，strict = all_sig or has_cp（D29-3）。"""
    sig = edit_lines(d)
    ev = {(e["session"], e["seq"]): e for e in load(d / "events.csv") if e["kind"] == "tool"}
    ep_events = defaultdict(list)
    for r in load(d / "episode_events.csv"):
        ep_events[r["episode"]].append(ev[(r["session"], r["seq"])])
    comp = SESS[SYS]["composer"]
    cps_cache = {}
    out = {}
    for ep, ch in pairs:
        e = eps[ep]
        cpaths = files_of_c[ch]
        edits = [x for x in ep_events[ep] if x["tool"] == "edit_file_v2" and x["status"] == "completed" and x["path"]]
        ov = [x for x in edits if x["path"].replace("\\", "/").lower() in cpaths and x["path"].replace("\\", "/").lower() in code_files[ep]]
        has_sig = any(sig.get((x["session"], int(x["seq"]))) for x in ov)
        all_sig = bool(ov) and all(sig.get((x["session"], int(x["seq"]))) for x in ov)
        has_cp = False
        if e["checkpoint"] and e["next_checkpoint"]:
            ss = e["session"]
            if ss not in cps_cache:
                cps_cache[ss] = load_checkpoints(comp / ss, WS_ROOT[SYS])
            a, b = cps_cache[ss].get(e["checkpoint"]), cps_cache[ss].get(e["next_checkpoint"])
            if a is not None and b is not None:
                has_cp = any((b[p] - a.get(p, set())) for p in b if p.lower() in cpaths and p.lower() in code_files[ep])
        out[(ep, ch)] = (has_sig, has_cp, all_sig or has_cp)
    return out


def sample():
    pools, eps, commits, files_of_c, code_files = universe()
    d = Path("data/derived") / SYS
    rng = random.Random(SEED)
    rows = []
    strat = {cat: visibility(d, eps, files_of_c, code_files, pools[cat]) for cat in ("baseline_only", "missed_ref")}
    for cat in CATS:
        pool = pools[cat]
        if cat in strat:
            chk = [p for p in pool if strat[cat][p][2]]
            unchk = [p for p in pool if not strat[cat][p][2]]
            for stratum, sub, q in (("checkable", chk, len(chk)), ("uncheckable", unchk, QUOTA_UNCHK)):
                pick = sorted(sub) if q >= len(sub) else rng.sample(sub, q)
                rows += [{"episode": e, "commit": c, "category": cat, "stratum": stratum, "pool_size": len(pool), "stratum_size": len(sub)} for e, c in pick]
        else:
            pick = rng.sample(pool, min(QUOTA, len(pool)))
            rows += [{"episode": e, "commit": c, "category": cat, "stratum": "all", "pool_size": len(pool), "stratum_size": len(pool)} for e, c in pick]
    # 同一对可同属两类（如既是基线独有对又是参照中未找回的对）：只编一次，类别以 ";" 连接，评分时两类都计
    merged = {}
    for r in rows:
        k = (r["episode"], r["commit"])
        if k in merged:
            for f in ("category", "stratum", "pool_size", "stratum_size"):
                merged[k][f] = f"{merged[k][f]};{r[f]}"
        else:
            merged[k] = dict(r)
    dups = len(rows) - len(merged)
    rows = list(merged.values())
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["pair"] = f"P{i:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "episode", "commit", "category", "stratum", "pool_size", "stratum_size"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "episode", "commit"])
        w.writeheader()
        w.writerows([{k: r[k] for k in ("pair", "episode", "commit")} for r in rows])
    print(f"{len(rows)} pairs（跨类重复合并 {dups} 对）→ {OUT / 'key.csv'}; pools " + ", ".join(f"{c} {len(pools[c])}" for c in CATS)
          + "; 严格可核对层（全取） " + ", ".join(f"{c} {sum(1 for p in pools[c] if strat[c][p][2])}/{len(pools[c])}" for c in strat)
          + "; 旧口径任一编辑可见 " + ", ".join(f"{c} {sum(1 for p in pools[c] if strat[c][p][0] or strat[c][p][1])}/{len(pools[c])}" for c in strat))


def edit_lines(d):
    sig = {}
    for line in open(d / "edit_signatures.jsonl", encoding="utf-8"):
        o = json.loads(line)
        sig[(o["session"], o["seq"])] = o["lines"]
    return sig


def clip(lines, n):
    lines = list(lines)
    return lines[:n] + ([f"…（另 {len(lines) - n} 行略）"] if len(lines) > n else [])


def git_diff(h, paths):
    args = ["git", "-C", str(SYSTEMS[SYS]), "-c", "core.quotepath=false", "show", "--format=", "--unified=1", h, "--"] + paths
    r = subprocess.run(args, capture_output=True)
    return r.stdout.decode("utf-8", "replace").splitlines()


def packets():
    d = Path("data/derived") / SYS
    key = load(OUT / "manifest.csv")
    pools, eps, commits, files_of_c, code_files = universe()
    sig = edit_lines(d)
    ev = {(e["session"], e["seq"]): e for e in load(d / "events.csv") if e["kind"] == "tool"}
    ep_events = defaultdict(list)
    for r in load(d / "episode_events.csv"):
        ep_events[r["episode"]].append(ev[(r["session"], r["seq"])])
    cfiles_status = defaultdict(list)
    for f in load(d / "commit_files.csv"):
        cfiles_status[f["hash"]].append((f["status"], f["path"]))
    comp = SESS[SYS]["composer"]
    cps_cache = {}
    pdir = OUT / "packets"
    pdir.mkdir(parents=True, exist_ok=True)
    for r in key:
        e, c, pid = eps[r["episode"]], commits[r["commit"]], r["pair"]
        L = [f"# 对 {pid}", "",
             "问题：**这个片段在下面这个提交所改文件上写入的内容，是否进入了该提交？** 只看内容：片段写入的代码行（含子智能体的编辑、检查点差分）是否出现在该提交的改动中。",
             "答案：Y 是（片段在这些文件上的写入内容进入了该提交）/ P 部分（一部分进入，其余进入了别的提交或未入库）/ N 否（该提交对这些文件的改动不是这个片段写的，或片段的写入未进入该提交）/ U 无法判断（片段在这些文件上的编辑没有可比的内容）。",
             "判断依据见编码手册第 7 节。要点：检查点差分中的行视同片段的写入内容；写入行以上下文行（无 +/− 前缀）出现说明已在更早提交中，不算进入；写入内容与检查点都没有、或差异被截断而看不到可比行时答 U，不凭请求与差异\"吻合\"给 Y。不要依据时间先后推断。片段所改文件中与该提交重叠的以 ★ 标出。", "",
             "## 片段", "",
             f"- 会话 {e['session'][:8]}，开始 {hm(e['start'])}，最后动作 {hm(e['last_event'])}；代码编辑 {e['n_edits_code']} 次。", "",
             "开发者请求：", "", quote(e["request_text"][:1500]), ""]
        # 编辑内容
        edits = [x for x in ep_events[r["episode"]] if x["tool"] == "edit_file_v2" and x["status"] == "completed" and x["path"]]
        code_edits = sorted([x for x in edits if x["path"].replace("\\", "/").lower() in code_files[r["episode"]]], key=lambda x: x["time"])
        cpaths = files_of_c[r["commit"]]
        star = lambda p: "★ " if p.lower() in cpaths else ""
        L += [f"### 片段中的代码编辑（{len(code_edits)} 次，按时间；★ = 该提交也改了这个文件）", ""]
        shown = 0
        for x in sorted(code_edits, key=lambda x: (not star(x["path"].replace(chr(92), "/")), x["time"])):
            lines = sig.get((x["session"], int(x["seq"])), [])
            is_star = bool(star(x["path"].replace(chr(92), "/")))
            tag = "（导出中无写入内容）" if not lines else ("" if (is_star and shown < MAX_STAR_EDITS) else "（有写入内容，包中未显示）")
            L.append(f"- {star(x['path'].replace(chr(92), '/'))}{hm(x['time'])} `{x['path'].replace(chr(92), '/')}`" + tag)
            if lines and is_star and shown < MAX_STAR_EDITS:
                L += ["", "```"] + [redact(ln) for ln in clip(lines, MAX_LINES_EDIT)] + ["```", ""]
                shown += 1
        # 检查点差分
        has_cp_section = False
        if e["checkpoint"] and e["next_checkpoint"]:
            s = e["session"]
            if s not in cps_cache:
                cps_cache[s] = load_checkpoints(comp / s, WS_ROOT[SYS])
            a, b = cps_cache[s].get(e["checkpoint"]), cps_cache[s].get(e["next_checkpoint"])
            if a is not None and b is not None:
                touched = {x["path"].replace("\\", "/") for x in edits}
                rows = [(p, sorted(b[p] - a.get(p, set()))) for p in sorted(b) if p in touched and b[p] - a.get(p, set())]
                rows = [(p, ls) for p, ls in rows if p.lower() in code_files[r["episode"]]]
                if rows:
                    has_cp_section = True
                    rows.sort(key=lambda t: t[0].lower() not in cpaths)
                    L += [f"### 检查点差分（片段起点到下一片段起点之间，这些文件新出现的行；{len(rows)} 个文件；这些行视同片段的写入内容，★ 文件排在前）", ""]
                    for p, ls in rows[:MAX_FILES]:
                        L += [f"- {star(p)}`{p}`（{len(ls)} 行）", "", "```"] + [redact(ln) for ln in clip(ls, MAX_LINES_CP)] + ["```", ""]
        if not has_cp_section:
            L += ["### 检查点差分", "", "本片段没有可用的检查点差分（两端检查点缺失或所改代码文件上没有新出现的行）。", ""]
        # 提交
        L += ["## 提交", "",
              f"- 提交 {c['hash'][:8]}，时间 {hm(c['time'])}，标题：{redact(c['subject'])}",
              f"- 改动文件 {len(cfiles_status[c['hash']])} 个：", ""]
        for st, p in cfiles_status[c["hash"]][:40]:
            L.append(f"  - {st} `{p}`")
        if len(cfiles_status[c["hash"]]) > 40:
            L.append(f"  - …（另 {len(cfiles_status[c['hash']]) - 40} 个）")
        overlap = [p for _, p in cfiles_status[c["hash"]] if p.lower() in code_files[r["episode"]]]
        L += ["", f"### 提交在片段所改代码文件上的差异（{len(overlap)} 个文件，逐文件列出，每个至多 {MAX_LINES_DIFF} 行；无重叠文件时为空，此时一般答 N 或 U）", ""]
        for fp in overlap[:MAX_FILES]:
            diff = git_diff(c["hash"], [fp])
            L += [f"#### `{fp}`（差异 {len(diff)} 行" + (f"，只显示前 {MAX_LINES_DIFF} 行" if len(diff) > MAX_LINES_DIFF else "") + "）", "", "```diff"] + [redact(ln) for ln in clip(diff, MAX_LINES_DIFF)] + ["```", ""]
        if len(overlap) > MAX_FILES:
            L.append(f"（另 {len(overlap) - MAX_FILES} 个重叠文件的差异未显示）")
        L += ["## 作答", "", f"在录入表中 `{pid}` 一行填写：answer（Y / P / N / U）、confidence（1 低 / 2 中 / 3 高）、note（可空，写你据以判断的行或文件）。", ""]
        (pdir / f"{pid}.md").write_text("\n".join(L), encoding="utf-8")
    with open(OUT / "coder_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "answer", "confidence", "note"])
        for r in key:
            w.writerow([r["pair"], "", "", ""])
    print(f"{len(key)} packets → {pdir}; template → {OUT / 'coder_template.csv'}")


def characterize():
    """全部参照中未找回的对（missed_ref）的可见性刻画（D26-1 / 6 节）：
    片段在重叠代码文件上的编辑是否有写入内容、检查点是否有新增行、提交在这些文件上的差异是否只有删除。"""
    d = Path("data/derived") / SYS
    pools, eps, commits, files_of_c, code_files = universe()
    vis = visibility(d, eps, files_of_c, code_files, pools["missed_ref"])
    vis_b = visibility(d, eps, files_of_c, code_files, pools["baseline_only"])
    n = Counter()
    for ep, ch in pools["missed_ref"]:
        has_sig, has_cp, _ = vis[(ep, ch)]
        overlap = [p for p in files_of_c[ch] if p in code_files[ep]]
        diff = git_diff(ch, sorted(overlap)[:MAX_FILES])
        adds = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
        dels = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
        n["total"] += 1
        n["no_visible_content"] += not (has_sig or has_cp)
        n["deletion_only_commit"] += (dels > 0 and adds == 0)
        n["no_visible_and_deletion_only"] += (not (has_sig or has_cp)) and (dels > 0 and adds == 0)
        n["has_sig"] += has_sig
        n["has_cp_only"] += (has_cp and not has_sig)
    # 内容对齐给出的对所依赖的匹配行数（编辑特征行 + 检查点行）
    ec = [r for r in load(d / "episode_commit.csv") if r["source"] != "pathtime"]
    strength = lambda r: int(r["n_lines"]) + int(r["cp_lines"])
    weak = sum(1 for r in ec if strength(r) <= 2)
    co = {(e, c) for e, c in pools["content_only"]}
    weak_co = sum(1 for r in ec if (r["episode"], r["commit"]) in co and strength(r) <= 2)
    b_chk = sum(1 for v in vis_b.values() if v[0] or v[1])
    b_strict = sum(1 for v in vis_b.values() if v[2])
    R = dict(n, pairs_total=len(ec), pairs_le2=weak, content_only_total=len(co), content_only_le2=weak_co,
             baseline_only_total=len(vis_b), baseline_only_checkable=b_chk, baseline_only_strict=b_strict,
             missed_ref_strict=sum(1 for v in vis.values() if v[2]))
    lines = ["", f"## 参照中未找回的 {n['total']} 对的可见性（`characterize`）", "",
             f"- 内容对齐给出的 {len(ec)} 对中，只依赖不超过 2 行匹配（编辑特征行 + 检查点行）的 {weak} 对（{weak / len(ec):.0%}）；仅内容对齐给出的 {len(co)} 对中为 {weak_co} 对。",
             f"- 仅时间基线给出的 {len(vis_b)} 对中，任一重叠编辑有写入内容或检查点新增行的 {b_chk} 对；全部重叠编辑可见或有检查点差分（D29-3 严格层）的 {b_strict} 对。", ""]
    lines += [
             f"- 片段在重叠代码文件上的编辑既无写入内容、检查点也无新增行（对齐不可见）：{n['no_visible_content']} 对；其中提交在这些文件上的差异只有删除行：{n['no_visible_and_deletion_only']} 对。",
             f"- 提交在重叠文件上只有删除行（不论片段侧）：{n['deletion_only_commit']} 对。",
             f"- 片段侧有写入内容：{n['has_sig']} 对；只有检查点新增行：{n['has_cp_only']} 对。", ""]
    print("\n".join(lines))
    out = RESULTS_DIR / "r20_pair_validation.md"
    txt = out.read_text(encoding="utf-8") if out.exists() else "# 片段—提交对级盲评（D26-1）\n"
    if "## 参照中未找回的" in txt:
        txt = txt[:txt.index("\n## 参照中未找回的")]
    out.write_text(txt.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")
    nums = json.loads(NUMS.read_text(encoding="utf-8")) if NUMS.exists() else {}
    nums.setdefault("pair_validation", {})["missed_ref_visibility"] = R
    NUMS.write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")


def score_rows(key, ans):
    """按类别（后两类再按层）汇总 Y/P/N/U 与"进入"比例。"""
    out = []
    def parts(r):  # 一对可属多类：[(category, stratum, stratum_size), ...]
        cs, ss, zs = r["category"].split(";"), r.get("stratum", "all").split(";"), str(r.get("stratum_size", r.get("pool_size", 0))).split(";")
        return list(zip(cs, ss, zs))
    for cat in CATS:
        rows = [(r, st, z) for r in key for c, st, z in parts(r) if c == cat]
        strata = sorted({st for _, st, _ in rows}, key=lambda x: ("all", "checkable", "uncheckable").index(x))
        for st in strata:
            sub = [(r, z) for r, s2, z in rows if s2 == st]
            ps = [r["pair"] for r, _ in sub]
            cnt = Counter(ans[p]["answer"].strip().upper() for p in ps)
            k, nn = cnt["Y"] + cnt["P"], len(ps) - cnt["U"]
            out.append({"category": cat, "stratum": st, "n": len(ps), "Y": cnt["Y"], "P": cnt["P"], "N": cnt["N"], "U": cnt["U"],
                        "in": k, "checkable": nn, "in_rate": (k / nn) if nn else None, "ci": list(wilson(k, nn)) if nn else None,
                        "stratum_size": int(sub[0][1]) if sub else 0})
    return out


def score():
    coder = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "coder1.csv"
    key = {r["pair"]: r for r in load(OUT / "key.csv")}
    ans = {r["pair"]: r for r in load(coder)}
    miss = [p for p in key if p not in ans or ans[p]["answer"].strip().upper() not in ("Y", "P", "N", "U")]
    assert not miss, f"missing or invalid answers: {miss[:5]}"
    R, L = {}, [f"# 片段—提交对级盲评（S19-T-E3，D26-1{'；AI 模拟编码' if SIM else ''}）", "",
                f"由 `analysis/pair_validation.py score` 生成。编码人只见片段的写入内容、检查点差分与提交差异，不知对的来源类别，不见自动候选。"
                f"录入表 `{coder.name}`。Y 是 / P 部分 / N 否 / U 无法判断。", "",
                "| 类别 | 层（层内总数） | n | Y | P | N | U | 人工判\"进入\"（Y + P）比例 | 含义 |", "|---|---|---|---|---|---|---|---|---|"]
    meaning = {"both": "两法都对的比例", "content_only": "内容对齐独有对中真实的比例（内容对齐的精确率证据）",
               "baseline_only": "基线独有对中真实的比例（Y + P 高则基线未必错；N 高则表 5 的判定成立）",
               "missed_ref": "参照中未找回的对里片段内容确实进入了该提交的比例（N 表示参照本身为假阳性）"}
    ST_ZH = {"all": "全部", "checkable": "可核对", "uncheckable": "不可核对"}
    R["rows"] = score_rows(list(key.values()), ans)
    for r in R["rows"]:
        cat, st = r["category"], r["stratum"]
        rate = f"{r['in']}/{r['checkable']}（{r['in_rate']:.0%}，{r['ci'][0]:.0%}–{r['ci'][1]:.0%}）" if r["checkable"] else "—"
        L.append(f"| {CAT_ZH[cat]} | {ST_ZH[st]}（{r['stratum_size']}） | {r['n']} | {r['Y']} | {r['P']} | {r['N']} | {r['U']} | {rate} | {meaning[cat]} |")
    for cat in CATS:  # 兼容旧键：按类别合计
        ps = [p for p in key if cat in key[p]["category"].split(";")]
        cnt = Counter(ans[p]["answer"].strip().upper() for p in ps)
        judged = len(ps) - cnt["U"]
        yp = cnt["Y"] + cnt["P"]
        R[cat] = {"n": len(ps), "counts": dict(cnt), "in_rate": yp / judged if judged else None,
                  "in_ci": list(wilson(yp, judged)) if judged else None, "judged": judged,
                  "pool_size": int(dict(zip(key[ps[0]]["category"].split(";"), key[ps[0]]["pool_size"].split(";")))[cat]) if ps else 0}
    conf = Counter(ans[p]["confidence"].strip() for p in key)
    R["confidence"] = dict(conf)
    L += ["", f"置信度分布：1 = {conf.get('1', 0)}，2 = {conf.get('2', 0)}，3 = {conf.get('3', 0)}。", "",
          "解读：内容对齐给出的对（前两类）中人工判\"进入\"的比例是内容对齐精确率的独立估计；基线独有对中人工判\"否\"的比例检验表 5 的判定；"
          "参照中未找回的对里人工判\"否\"的比例估计参照关系自身的假阳性。区间为 Wilson 95%，分母不含\"无法判断\"。"
          "仅时间基线与参照中未找回两类按可核对分层（层内总数见括号）：可核对层为全部重叠编辑可见或有检查点差分的对，全取；不可核对层抽样，只用来核实 U 率。", ""]
    (RESULTS_DIR / "r20_pair_validation.md").write_text("\n".join(L), encoding="utf-8")
    num = json.loads(NUMS.read_text(encoding="utf-8")) if NUMS.exists() else {}
    num["pair_validation"] = json.loads(json.dumps(R, default=float))
    NUMS.write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"sample": sample, "packets": packets, "characterize": characterize, "score": score}[sys.argv[1]]()
