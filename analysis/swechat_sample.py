"""S27 / D38（B9）：SWE-chat 五仓库代码修改片段的人工核验抽样。

抽样框 = `swechat_replicate.py` 基准口径的全部代码修改片段（与 numbers.json 的 swechat 键逐仓库核对），
每个片段带三项分层变量：组（r25 §3 会话层：保留 = 严格 + 宽松 / 排除 = 有歧义 + 无提交）、触发层（r25 §2）、自动模式。
格子 = 组 × {单独提交指令, 片段内提交}，每格 20 个；格内拆分、缺失各至少 4 个，其余按一对一 / 共享比例；同一仓库一格内至多 60%。
判据与配额见 tools/repro/analysis_plan.md S2b（抽样前冻结）。种子 S27-swechat；每格每层的置换整体保存，样本嵌套。

输出（不入库，data/coding/ 在 .gitignore）：
  data/coding/swechat/frame.csv       抽样框（含分层变量与自动模式——这是 key，编码人不得查看）
  data/coding/swechat/sample_S27.csv  样本（episode、repo、session、idx、order；分层变量在 frame.csv 中）
  data/coding/swechat/perm.json       每格每层的置换（嵌套扩缩样本用）
输出（入库）：<PAPER_DIR>/results/s27_swechat_sampling.md  各格各层总体数与抽样数

用法：python analysis/swechat_sample.py [--frame-only] [--n 20] [--force]
"""
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR  # noqa: E402
import swechat_replicate as sw  # noqa: E402

SEED = "S27-swechat"
OUT_DIR = Path("data/coding/swechat")
FRAME, SAMPLE, PERM = OUT_DIR / "frame.csv", OUT_DIR / "sample_S27.csv", OUT_DIR / "perm.json"
NUMS = RESULTS_DIR / "numbers.json"
LAYERS = ["单独提交指令", "片段内提交"]              # 进入样本的两层
ALL_LAYERS = LAYERS + ["混合", "无可对上的智能体提交"]
GROUPS = ["保留", "排除"]
MODES = ["一对一", "共享", "拆分", "缺失"]             # 证据不足并入缺失名额
PER_CELL, MIN_MINOR, REPO_CAP = 20, 4, 0.6
REPO_CACHE_MAX_LINES = 3_000_000
SHORT = {"osabiohq/osabio": "osa", "timelabs-npo/rhea-project": "rhea", "obsessiondb/rudel": "rudel",
         "wanshicheng/duckdb-data-agent": "duck", "oddessentials/ado-git-repo-insights": "ado"}


def mode_of(pattern):
    return {"one_to_one": "一对一", "shared": "共享", "split": "拆分", "split+shared": "拆分", "missing": "缺失",
            "insufficient": "证据不足"}[pattern]


def load_all(repos=None, use_cache=True):
    """复算五仓库的片段级中间结果（与 numbers.json 的基准结果断言一致）。返回 {repo: keep 字典}、conv。
    片段级结果（不含 Repo 对象）缓存到 data/coding/swechat/cache/<repo>.pkl；Repo 对象每次重建。"""
    import pickle
    repos = repos or sw.REPOS
    canon = lambda r: sw.ALIASES.get(r, r)
    wanted = [r for r in list(sw.REPOS) + list(sw.ALIASES) if canon(r) in repos]
    ss = pq.read_table(sw.DATA / "sessions.parquet", columns=["session_id", "repo_id", "agent"]).to_pandas()
    keep_s = set(ss[ss.agent == sw.AGENT].session_id)
    conv = sw.load_conversations(wanted)
    conv = conv[conv.session_id.isin(keep_s)].copy()
    conv["repo_id"] = conv.repo_id.map(canon)
    links_raw = sw.entire_links_table(wanted)
    links = defaultdict(lambda: defaultdict(set))
    for rid, d in links_raw.items():
        for s, shas in d.items():
            links[canon(rid)][s] |= shas
    num = json.loads(NUMS.read_text(encoding="utf-8"))
    base = {r["repo"]: r for r in num["swechat"]["repos"]}
    out = {}
    t0 = time.time()
    cache_dir = OUT_DIR / "cache"
    for r in repos:
        cp = cache_dir / (r.replace("/", "__") + ".pkl")
        rp = cache_dir / (r.replace("/", "__") + ".repo.pkl")
        if use_cache and rp.exists():
            R = _load_repo(pickle.loads(rp.read_bytes()))
        else:
            R = sw.Repo(sw.MIRRORS / (r.replace("/", "__") + ".git"))
            # 镜像须完整可读：提交数与 numbers.json 记录的一致（Temp 清理曾删掉空的 refs/ 目录，使 git 读不到镜像而静默返回空历史）
            assert len(R.time) == base[r]["commits_in_mirror"], (r, len(R.time), base[r]["commits_in_mirror"])
            cache_dir.mkdir(parents=True, exist_ok=True)
            n_lines = sum(len(v) for d in R.added.values() for v in d.values())
            if n_lines <= REPO_CACHE_MAX_LINES:            # rhea-project 的新增行集合有数千万行，缓存会达数 GB，不缓存
                rp.write_bytes(pickle.dumps(_dump_repo(R)))
        if use_cache and cp.exists():
            kp = pickle.loads(cp.read_bytes())
            kp["R"] = R
        else:
            kp = {}
            res = sw.replicate_repo(r, conv, links, R=R, keep=kp)
            assert res["patterns"] == base[r]["patterns"] and res["code_episodes"] == base[r]["code_episodes"], r
            cache_dir.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(pickle.dumps({k: v for k, v in kp.items() if k != "R"}))
        out[r] = kp
        print(f"[{r}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return out, conv


def _dump_repo(R):
    """Repo 对象含 lambda 型 defaultdict，转成普通 dict 后再 pickle（版本库一侧对大仓库的 git log 解析要数分钟）。"""
    d = dict(R.__dict__)
    d["added"] = {h: dict(v) for h, v in R.added.items()}
    d["by_file"] = dict(R.by_file)
    d["_ver"] = {}
    return d


def _load_repo(d):
    R = sw.Repo.__new__(sw.Repo)
    R.__dict__.update(d)
    R.added = defaultdict(lambda: defaultdict(set), {h: defaultdict(set, v) for h, v in d["added"].items()})
    R.by_file = defaultdict(list, d["by_file"])
    return R


def ep_id(repo, episodes, i):
    """片段 ID：仓库缩写-会话 ID 末 8 位-会话内序号（含不改代码的片段，1 起）。"""
    ep = episodes[i]
    k = sum(1 for j in range(i + 1) if episodes[j]["session"] == ep["session"])
    return f"SW-{SHORT[repo]}-{ep['session'][-8:]}-{k:02d}"


def session_vars(episodes, ref_commits, content_pairs):
    """每会话的组（r25 §3）与触发层（r25 §2），口径与 swechat_strata.py 相同。"""
    c_of = defaultdict(set)
    for i, h in content_pairs:
        c_of[i].add(h)
    sess_of_commit = defaultdict(set)
    for i, hs in c_of.items():
        for h in hs:
            sess_of_commit[h].add(episodes[i]["session"])
    exec_by = defaultdict(lambda: {"code": 0, "nocode": 0})
    for h, i in ref_commits.items():
        exec_by[episodes[i]["session"]]["code" if episodes[i]["code_edits"] else "nocode"] += 1
    S = {}
    for i, ep in enumerate(episodes):
        if not ep["code_edits"]:
            continue
        d = S.setdefault(ep["session"], {"C": set(), "shared_other": False})
        for h in c_of.get(i, set()):
            d["C"].add(h)
            if sess_of_commit[h] - {ep["session"]}:
                d["shared_other"] = True
    out = {}
    for s, d in S.items():
        if not d["C"]:
            g4 = "无提交"
        elif not d["shared_other"] and len(d["C"]) == 1:
            g4 = "无歧义（严格）"
        elif not d["shared_other"]:
            g4 = "无歧义（宽松）"
        else:
            g4 = "有歧义"
        e = exec_by[s]
        layer = ("片段内提交" if e["code"] and not e["nocode"] else "单独提交指令" if e["nocode"] and not e["code"]
                 else "混合" if e["code"] and e["nocode"] else "无可对上的智能体提交")
        out[s] = {"group4": g4, "group": "保留" if g4.startswith("无歧义") else "排除", "layer": layer}
    return out


def build_frame(all_eps):
    rows = []
    for repo, kp in all_eps.items():
        episodes = kp["episodes"]
        sv = session_vars(episodes, kp["ref_commits"], kp["content_pairs"])
        n_in_sess = Counter(ep["session"] for ep in episodes)
        for i in kp["code_eps"]:
            ep = episodes[i]
            v = sv[ep["session"]]
            rows.append({"episode": ep_id(repo, episodes, i), "repo": repo, "session": ep["session"],
                         "idx": int(ep_id(repo, episodes, i).rsplit("-", 1)[1]), "n_in_session": n_in_sess[ep["session"]],
                         "group": v["group"], "group4": v["group4"], "layer": v["layer"], "auto_pattern": ep["pattern"],
                         "auto_mode": mode_of(ep["pattern"]), "code_edits": ep["code_edits"], "n_commits": len(ep["P"])})
    assert len(rows) == len({r["episode"] for r in rows}), "片段 ID 不唯一"
    return rows


def allocate(cell_rows, per_cell):
    """格内按自动模式的名额：拆分、缺失（含证据不足）各至少 MIN_MINOR，其余按一对一 / 共享比例四舍五入。"""
    by = defaultdict(list)
    for r in cell_rows:
        by["缺失" if r["auto_mode"] == "证据不足" else r["auto_mode"]].append(r)
    n_cell = min(per_cell, len(cell_rows))
    quota = {}
    for m in ("拆分", "缺失"):
        quota[m] = min(MIN_MINOR, len(by[m]))
    rest = n_cell - sum(quota.values())
    major = {m: len(by[m]) for m in ("一对一", "共享")}
    tot = sum(major.values())
    if tot:
        q1 = round(rest * major["一对一"] / tot)
        quota["一对一"] = min(q1, major["一对一"])
        quota["共享"] = min(rest - quota["一对一"], major["共享"])
        # 某层不足时把余额给另一层，再给拆分 / 缺失
        short = rest - quota["一对一"] - quota["共享"]
        for m in ("一对一", "共享", "拆分", "缺失"):
            add = min(short, len(by[m]) - quota[m])
            quota[m] += add
            short -= add
    return by, quota


def draw(frame, per_cell, rng_seed=SEED):
    rng = random.Random(rng_seed)
    perm, chosen, report = {}, [], []
    for g in GROUPS:
        for ly in LAYERS:
            cell = [r for r in frame if r["group"] == g and r["layer"] == ly]
            by, quota = allocate(cell, per_cell)
            picked = []
            for m in MODES:
                pool = sorted(by[m], key=lambda r: r["episode"])
                rng.shuffle(pool)
                perm[f"{g}|{ly}|{m}"] = [r["episode"] for r in pool]
                picked += [(m, r) for r in pool[:quota[m]]]
            # 仓库上限：一格内同一仓库至多 REPO_CAP；超出的名额在同层内让给其他仓库（按置换顺序取下一个）
            cap = int(REPO_CAP * len(picked))
            for _ in range(len(picked)):
                cnt = Counter(r["repo"] for _, r in picked)
                over = [rp for rp, c in cnt.items() if c > cap]
                if not over:
                    break
                rp = over[0]
                # 换掉该仓库在本格置换中最靠后的一个
                victim = max((k for k, (m, r) in enumerate(picked) if r["repo"] == rp),
                             key=lambda k: perm[f"{g}|{ly}|{picked[k][0]}"].index(picked[k][1]["episode"]))
                m = picked[victim][0]
                taken = {r["episode"] for _, r in picked}
                repl = next((r for r in sorted(by[m], key=lambda r: perm[f"{g}|{ly}|{m}"].index(r["episode"]))
                             if r["episode"] not in taken and r["repo"] != rp and cnt[r["repo"]] < cap), None)
                if repl is None:
                    break                                # 让无可让，保留
                picked[victim] = (m, repl)
            for m, r in picked:
                chosen.append(r)
            report.append({"group": g, "layer": ly, "N": len(cell), "n": len(picked),
                           "N_mode": {m: len(by[m]) for m in MODES}, "n_mode": {m: sum(1 for mm, _ in picked if mm == m) for m in MODES},
                           "repos": dict(Counter(r["repo"] for _, r in picked))})
    order = list(range(len(chosen)))
    rng.shuffle(order)
    for k, r in zip(order, chosen):
        r["order"] = k + 1
    chosen.sort(key=lambda r: r["order"])
    return chosen, perm, report


def write_report(frame, report, sample, per_cell):
    N = Counter((r["group"], r["layer"]) for r in frame)
    L = [f"# S27 公开语料人工核验的抽样（D38）", "",
         f"由 `analysis/swechat_sample.py` 生成（种子 `{SEED}`，每格 {per_cell} 个）。抽样框为 SWE-chat 五仓库基准口径的全部 {len(frame)} 个代码修改片段；"
         "格子为组（保留 / 排除，r25 §3）× 触发层（单独提交指令 / 片段内提交，r25 §2）；混合层与无可对上层不进入样本。"
         f"格内拆分、缺失各至少 {MIN_MINOR} 个，其余按一对一 / 共享比例；同一仓库一格内至多 {REPO_CAP:.0%}。判据见 `tools/repro/analysis_plan.md` S2b。", "",
         "## 1 抽样框：组 × 触发层的片段数", "", "| 组 \\ 层 | " + " | ".join(ALL_LAYERS) + " | 合计 |", "|---|" + "---|" * (len(ALL_LAYERS) + 1)]
    for g in GROUPS:
        L.append(f"| {g} | " + " | ".join(str(N[(g, ly)]) for ly in ALL_LAYERS) + f" | {sum(N[(g, ly)] for ly in ALL_LAYERS)} |")
    L += ["", "## 2 各格的总体与样本（按自动模式）", "",
          "| 组 | 层 | 总体 | 样本 | " + " | ".join(f"{m}（总体 / 样本）" for m in MODES) + " | 样本中各仓库 |", "|---|---|---|---|" + "---|" * (len(MODES) + 1)]
    for c in report:
        L.append(f"| {c['group']} | {c['layer']} | {c['N']} | {c['n']} | " +
                 " | ".join(f"{c['N_mode'][m]} / {c['n_mode'][m]}" for m in MODES) + " | " +
                 "；".join(f"{SHORT[k]} {v}" for k, v in sorted(c["repos"].items())) + " |")
    L += ["", f"样本合计 {len(sample)} 个片段，发给编码人的顺序为整体随机置换。证据不足层并入缺失名额（抽样框中证据不足 "
          f"{sum(1 for r in frame if r['auto_mode'] == '证据不足')} 个）。", ""]
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    per_cell = int(args[args.index("--n") + 1]) if "--n" in args else PER_CELL
    all_eps, _ = load_all()
    frame = build_frame(all_eps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["episode", "repo", "session", "idx", "n_in_session", "group", "group4", "layer", "auto_pattern", "auto_mode", "code_edits", "n_commits"]
    with open(FRAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(frame, key=lambda r: r["episode"]))
    if "--frame-only" in args:
        N = Counter((r["group"], r["layer"], r["auto_mode"]) for r in frame)
        for g in GROUPS:
            for ly in ALL_LAYERS:
                print(g, ly, {m: N[(g, ly, m)] for m in MODES + ["证据不足"]})
        return
    sample, perm, report = draw(frame, per_cell)
    if SAMPLE.exists() and "--force" not in args:
        old = [r["episode"] for r in csv.DictReader(open(SAMPLE, encoding="utf-8"))]
        print(f"{SAMPLE} exists (identical to rerun: {old == [r['episode'] for r in sample]}); not overwritten.")
        return
    with open(SAMPLE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["order", "episode", "repo", "session", "idx"])
        w.writeheader()
        w.writerows([{k: r[k] for k in ("order", "episode", "repo", "session", "idx")} for r in sample])
    PERM.write_text(json.dumps({"seed": SEED, "per_cell": per_cell, "perm": perm}, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / "s27_swechat_sampling.md").write_text(write_report(frame, report, sample, per_cell), encoding="utf-8")
    print(write_report(frame, report, sample, per_cell))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
