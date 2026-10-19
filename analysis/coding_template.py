"""生成编码表模板 data/coding/coding_sheet_template.csv 与试编用编码表 data/coding/pilot_sheet.csv。
字段与 <PAPER_DIR>/coding_manual.md 第 5 节一一对应。"""
import csv
import re
from pathlib import Path
from config import PAPER_DIR  # noqa: E402

FIELDS = ["episode", "coder", "main_pattern", "secondary_patterns", "causes", "confidence",
          "evidence_sufficient", "missing_info", "notes"]


def main():
    manual = (PAPER_DIR / "coding_manual.md").read_text(encoding="utf-8")
    sec5 = manual.split("## 5 编码表字段")[1].split("## 6")[0]
    assert re.findall(r"^\| `(\w+)` \|", sec5, re.M) == FIELDS
    codes = set(re.findall(r"^#### (A\d+|N\d|U|X) ", manual, re.M))
    assert codes == {f"A{i}" for i in range(1, 11)} | {"N0", "N1", "U", "X"}, codes

    Path("data/coding").mkdir(parents=True, exist_ok=True)
    with open("data/coding/coding_sheet_template.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(FIELDS)
    sample = list(csv.DictReader(open("data/coding/sample_S03.csv", encoding="utf-8")))
    if Path("data/coding/pilot_sheet.csv").exists():
        print("pilot_sheet.csv exists, not overwritten")
        return
    with open("data/coding/pilot_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted((r for r in sample if r["group"] == "pilot"), key=lambda r: r["episode"]):
            w.writerow({"episode": r["episode"], "coder": "author"})
    print("template + pilot sheet written")


if __name__ == "__main__":
    main()
