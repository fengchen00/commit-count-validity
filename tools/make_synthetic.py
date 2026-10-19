# -*- coding: utf-8 -*-
"""生成与真实导出格式相同的合成示例数据（不含任何商业内容），供复现包在没有真实数据时跑通核心流程。

生成物（相对当前目录，与 analysis/ 各脚本的默认路径一致）：
  data/raw/SYS_A_main/original-snapshot-EXAMPLE/git/clone      合成 Git 仓库（系统 A）
  data/raw/SYS_B_raw/original-snapshot-EXAMPLE/git/clone     合成 Git 仓库（系统 B，脚本内键为 SYS_C）
  data/raw/SYS_A_main/sessions/by-workspace/ws-0a13b296-d-sysa/raw/composer/<会话>/
  data/raw/SYS_A_main/sessions/by-workspace/ws-2dfaaae5-d-sysb/raw/composer/<会话>/
      header.json / composerData.json / bubbles.jsonl / checkpoints.jsonl（Cursor 会话导出的字段子集）

场景由下方 SCENARIO 表写死：每个片段一条开发者请求、若干次编辑、可选的智能体提交命令；
提交由脚本按时间写入 Git 仓库，因此一一对应、拆分、合并、未提交等模式都有样例。
用法：python tools/make_synthetic.py   （在复现包根目录运行；已存在的 data/raw 会被覆盖）
"""
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))
SNAP = "original-snapshot-EXAMPLE"
SYSTEMS = {
    "SYS_A": {
        "git": Path(f"data/raw/SYS_A_main/{SNAP}/git/clone"),
        "composer": Path("data/raw/SYS_A_main/sessions/by-workspace/ws-0a13b296-d-sysa/raw/composer"),
        "root": "d:/sysa/",
        "day0": datetime(2026, 9, 1, 9, 0, tzinfo=TZ),
    },
    "SYS_C": {
        "git": Path(f"data/raw/SYS_B_raw/{SNAP}/git/clone"),
        "composer": Path("data/raw/SYS_A_main/sessions/by-workspace/ws-2dfaaae5-d-sysb/raw/composer"),
        "root": "d:/sysb/",
        "day0": datetime(2026, 9, 18, 9, 0, tzinfo=TZ),
    },
}


def L(name, n, k):
    """生成有辨识度的代码行（长度 ≥ 15，且全局唯一）。"""
    return [f"def {name}_{k}_{i}(request, ctx):  # synthetic line {k}-{i}" for i in range(n)]


# (session, day, hour, request, edits=[(path, n_lines)], commit=subject|None, agent_commit=bool)
# 说明：commit 由“开发者”在片段结束后提交；agent_commit=True 表示片段内智能体执行 git commit 命令。
SCENARIO = {
    "SYS_A": [
        # M1：任务书先登记（事前登记），片段一一对应
        ("s1", 0, 9.0, "按 docs/plans/M1-用户中心.md 建里程碑 M1 任务书，T1 登录 T2 注册 T3 找回密码", [("docs/plans/M1-用户中心.md", 0)], "M1: 任务书", False),
        ("s1", 0, 9.5, "M1-T1 实现登录页", [("src/auth/login.py", 6), ("src/auth/forms.py", 4)], "M1-T1: 登录页", False),
        ("s1", 0, 10.5, "M1-T2 实现注册页", [("src/auth/register.py", 7)], "M1-T2: 注册页", True),
        ("s1", 0, 11.2, "继续", [("src/auth/register.py", 3)], None, False),
        ("s1", 0, 14.0, "M1-T3 找回密码，顺手把注册页的校验也改一下", [("src/auth/reset.py", 6), ("src/auth/register.py", 2)], "M1-T3: 找回密码与注册校验", True),
        # 拆分：一个片段进入两个提交
        ("s2", 1, 9.0, "M2-T1 订单列表页，先做接口再做页面", [("src/orders/api.py", 8), ("src/orders/page.py", 6)], "M2-T1: 订单接口", False),
        ("s2", 1, 9.6, "接着", [("src/orders/page.py", 4)], "M2-T1: 订单页面", True),
        # 合并：两个片段进入一个提交
        ("s2", 1, 11.0, "客户说订单页要能按状态筛选", [("src/orders/page.py", 5)], None, False),
        ("s2", 1, 11.5, "再加一个导出按钮", [("src/orders/export.py", 5), ("src/orders/page.py", 2)], "热修1 订单页筛选与导出", True),
        # 未提交：被覆盖的编辑
        ("s3", 2, 9.0, "试一下用缓存优化订单接口", [("src/orders/cache.py", 6)], None, False),
        ("s3", 2, 9.8, "缓存方案先不要了，改回去，把接口日志加上", [("src/orders/api.py", 3)], "M2: fix 接口日志", False),
        # 无代码片段
        ("s3", 2, 10.5, "解释一下现在订单模块的结构", [], None, False),
        # M3：无事前登记，日内多提交
        ("s4", 3, 9.0, "做一个仪表盘页面，先出个能看的", [("src/dash/page.py", 7)], "仪表盘初版", True),
        ("s4", 3, 9.4, "图表换成折线图", [("src/dash/page.py", 3)], "仪表盘折线图", True),
        ("s4", 3, 9.7, "标题字号大一点", [("src/dash/page.py", 2)], "仪表盘样式", True),
        ("s4", 3, 10.1, "把仪表盘任务补记到任务书 M3", [("docs/plans/M3-仪表盘.md", 0)], "M3: 补任务书", False),
        ("s5", 4, 9.0, "M3-T1 仪表盘加权限控制", [("src/dash/perm.py", 6), ("src/dash/page.py", 2)], "M3-T1: 仪表盘权限", True),
        ("s5", 4, 10.0, "部署到测试环境", [], None, False),
    ],
    "SYS_C": [
        ("c1", 0, 9.0, "按 docs/plans/M1-站点.md 建 M1 任务书，T1 首页 T2 栏目页", [("docs/plans/M1-站点.md", 0)], "M1: 任务书", False),
        ("c1", 0, 9.5, "M1-T1 首页", [("src/site/home.py", 6)], "M1-T1: 首页", True),
        ("c1", 0, 10.5, "M1-T2 栏目页", [("src/site/column.py", 6)], "M1-T2: 栏目页", True),
        ("c2", 1, 9.0, "客户说首页横幅要能轮播", [("src/site/home.py", 4)], None, False),
        ("c2", 1, 9.5, "栏目页加分页", [("src/site/column.py", 3)], "热修1 首页轮播与栏目分页", True),
        ("c2", 1, 10.5, "总结一下站点目前的页面", [], None, False),
    ],
}

TASKBOOK = {
    "docs/plans/M1-用户中心.md": "# M1 用户中心\n\n- [ ] **T1** 登录页\n- [ ] **T2** 注册页\n- [ ] **T3** 找回密码\n",
    "docs/plans/M3-仪表盘.md": "# M3 仪表盘\n\n- [x] **T0** 仪表盘初版（补记）\n- [ ] **T1** 权限控制\n",
    "docs/plans/M1-站点.md": "# M1 站点\n\n- [ ] **T1** 首页\n- [ ] **T2** 栏目页\n",
}


def utc(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def git(repo, *args, env=None):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def build(sysname, cfg):
    repo, comp, root = cfg["git"], cfg["composer"], cfg["root"]
    for p in (repo, comp):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "dev")
    git(repo, "config", "user.email", "dev@example.com")

    files = {}          # 工作区当前内容（行列表）
    sessions = {}       # session_key -> dict(cid, bubbles, order, created, last, cp_lines)
    counter = 0

    def bubble(sess, t, **kw):
        b = {"bubbleId": str(uuid.uuid4()), "createdAt": utc(t)}
        b.update(kw)
        sess["bubbles"].append(b)
        sess["order"].append(b["bubbleId"])
        sess["last"] = t
        return b

    def checkpoint(sess):
        """把会话内到目前为止写入的行记为检查点（files[*].originalModelDiffWrtV0.modified）。"""
        cid = str(uuid.uuid4())
        entries = []
        for path, lines in sess["cp_lines"].items():
            entries.append({"uri": {"path": "/" + root + path}, "originalModelDiffWrtV0": [{"modified": list(lines)}]})
        sess["checkpoints"].append({"_key": f"checkpointId:{cid}", "_value": {"files": entries, "activeInlineDiffs": []}})
        return cid

    def commit(subject, t, paths):
        for p in paths:
            fp = repo / p
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text("\n".join(files[p]) + "\n", encoding="utf-8")
            git(repo, "add", p)
        env = dict(os.environ, GIT_AUTHOR_DATE=t.isoformat(), GIT_COMMITTER_DATE=t.isoformat())
        git(repo, "commit", "-q", "-m", subject, "--allow-empty", env=env)

    for skey, day, hour, text, edits, subject, agent_commit in SCENARIO[sysname]:
        t = cfg["day0"] + timedelta(days=day, hours=hour - 9)
        if skey not in sessions:
            sessions[skey] = {"cid": str(uuid.uuid4()), "bubbles": [], "order": [], "created": t, "last": t,
                              "cp_lines": {}, "checkpoints": [], "name": text[:20]}
        sess = sessions[skey]
        cp = checkpoint(sess)
        bubble(sess, t, type=1, text=text, checkpointId=cp)
        t += timedelta(seconds=20)
        bubble(sess, t, type=2, text="好的，开始处理。")
        touched = []
        for path, n in edits:
            counter += 1
            t += timedelta(seconds=40)
            if path in TASKBOOK:
                new_lines = TASKBOOK[path].splitlines()
                files[path] = list(new_lines)
                content = TASKBOOK[path]
            else:
                new_lines = L(Path(path).stem, n, counter)
                files.setdefault(path, []).extend(new_lines)
                content = "\n".join(new_lines) + "\n"
            sess["cp_lines"].setdefault(path, []).extend(x for x in new_lines if len(x.strip()) >= 15)
            touched.append(path)
            bubble(sess, t, type=2, text="", toolFormerData={
                "toolCallId": str(uuid.uuid4()), "name": "edit_file_v2", "status": "completed",
                "params": json.dumps({"relativeWorkspacePath": path, "streamingContent": content}, ensure_ascii=False),
                "additionalData": "{}",
            })
        if subject and agent_commit:
            t += timedelta(seconds=30)
            bubble(sess, t, type=2, text="", toolFormerData={
                "toolCallId": str(uuid.uuid4()), "name": "run_terminal_cmd_v2", "status": "completed",
                "params": json.dumps({"command": f'git add -A; git commit -m "{subject}"'}, ensure_ascii=False),
                "additionalData": "{}",
            })
            commit(subject, t + timedelta(seconds=5), touched)
        elif subject:
            commit(subject, t + timedelta(minutes=8), touched)
        t += timedelta(seconds=15)
        bubble(sess, t, type=2, text="已完成。")

    for sess in sessions.values():
        d = comp / sess["cid"]
        d.mkdir(parents=True)
        (d / "header.json").write_text(json.dumps({"composerId": sess["cid"], "createdAt": utc(sess["created"]),
                                                   "lastUpdatedAt": utc(sess["last"]), "isSubagent": False}, ensure_ascii=False), encoding="utf-8")
        (d / "composerData.json").write_text(json.dumps({"_value": {
            "composerId": sess["cid"], "name": sess["name"], "createdAt": utc(sess["created"]), "lastUpdatedAt": utc(sess["last"]),
            "fullConversationHeadersOnly": [{"bubbleId": b} for b in sess["order"]]}}, ensure_ascii=False), encoding="utf-8")
        with open(d / "bubbles.jsonl", "w", encoding="utf-8") as f:
            for b in sess["bubbles"]:
                f.write(json.dumps({"_key": f"bubbleId:{sess['cid']}:{b['bubbleId']}", "_value": b}, ensure_ascii=False) + "\n")
        with open(d / "checkpoints.jsonl", "w", encoding="utf-8") as f:
            for c in sess["checkpoints"]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_commits = subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{sysname}: sessions={len(sessions)} episodes={len(SCENARIO[sysname])} commits={n_commits} -> {repo}, {comp}")


if __name__ == "__main__":
    for s in sys.argv[1:] or SYSTEMS:
        build(s, SYSTEMS[s])
    # 供 trace_customer.py 读取的需求文档目录（合成数据无客户文档，留空目录）
    Path("data/raw/SYS_A_main/customer-docs/需求文档").mkdir(parents=True, exist_ok=True)
    Path("data/coding").mkdir(parents=True, exist_ok=True)
    Path("data/derived").mkdir(parents=True, exist_ok=True)
    print("done")
