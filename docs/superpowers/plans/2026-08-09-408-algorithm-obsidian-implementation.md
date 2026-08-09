# 408 Algorithm Obsidian Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Create a native Obsidian tracker under 数据结构/ containing 47 LeetCode exercises and the 2017—2026 ten 408 exam questions, ordered by mother-problem module and ready for progress recording.

**Architecture:** Treat the existing Excel workbook as the canonical import source. A temporary Python generator reads its 刷题记录 and 母题索引 sheets, writes one Markdown note per problem plus a Base, dashboard, and index; a separate validator checks counts, years, properties, order, YAML, formulas, links, and rendering before commit.

**Tech Stack:** Obsidian 1.13.4, Obsidian Bases, Obsidian Flavored Markdown, bundled Python 3, openpyxl, PyYAML, Obsidian CLI.

---

## File map

Source, read only:

- /Users/heyuhang/Documents/Codex/2026-08-09/chatgpt-conversation-6a7803c0-39f0-83e8-997e/outputs/408算法题单刷题记录表_母题到真题.xlsx

Temporary build files:

- /tmp/408-obsidian-build/generate_tracker.py
- /tmp/408-obsidian-build/validate_tracker.py

Created vault files:

- 数据结构/408算法刷题总览.md
- 数据结构/408算法刷题记录.base
- 数据结构/408算法母题索引.md
- 数据结构/题目/*.md, exactly 57 files

The temporary scripts are not committed.

### Task 1: Create a failing validator

**Files:**

- Create: /tmp/408-obsidian-build/validate_tracker.py
- Test: /tmp/408-obsidian-build/validate_tracker.py

- [ ] **Step 1: Create the temporary directory**

Run:

~~~bash
mkdir -p /tmp/408-obsidian-build
~~~

Expected: the directory exists.

- [ ] **Step 2: Write the validator**

Create /tmp/408-obsidian-build/validate_tracker.py with the following complete implementation:

~~~python
from pathlib import Path
import re
import sys
import yaml

VAULT = Path("/Users/heyuhang/Documents/Obsidian Vault")
ROOT = VAULT / "数据结构"
PROBLEMS = ROOT / "题目"
BASE = ROOT / "408算法刷题记录.base"
DASHBOARD = ROOT / "408算法刷题总览.md"
INDEX = ROOT / "408算法母题索引.md"

REQUIRED = {
    "title", "tags", "order", "mother_id", "mother", "chapter",
    "source_type", "problem_id", "priority", "time_limit",
    "handwriting", "goal", "status", "first_date", "first_minutes",
    "first_result", "second_date", "second_result", "paper_date",
    "paper_score", "error_type", "next_review", "review_count",
    "mastery", "weakness", "note",
}


def read_frontmatter(path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, f"missing frontmatter: {path.name}"
    return yaml.safe_load(match.group(1)), text


def main():
    assert ROOT.is_dir(), "数据结构 directory missing"
    assert PROBLEMS.is_dir(), "题目 directory missing"
    files = sorted(PROBLEMS.glob("*.md"))
    assert len(files) == 57, f"expected 57 notes, got {len(files)}"

    rows = []
    for path in files:
        data, text = read_frontmatter(path)
        missing = REQUIRED - set(data)
        assert not missing, f"{path.name} missing {sorted(missing)}"
        assert "408/数据结构/刷题" in data["tags"], f"bad tag: {path.name}"
        assert "## 思路记录" in text and "## 错因复盘" in text, f"bad body: {path.name}"
        rows.append(data)

    orders = sorted(row["order"] for row in rows)
    assert orders == list(range(1, 58)), "order must be continuous from 1 to 57"
    assert len(set(orders)) == 57, "duplicate order"

    lc_rows = [row for row in rows if row["source_type"] == "LeetCode"]
    exam_rows = [row for row in rows if row["source_type"] == "408真题"]
    assert len(lc_rows) == 47, f"expected 47 LeetCode notes, got {len(lc_rows)}"
    assert len(exam_rows) == 10, f"expected 10 exam notes, got {len(exam_rows)}"
    years = sorted(str(row["problem_id"]) for row in exam_rows)
    assert years == [str(year) for year in range(2017, 2027)], f"bad years: {years}"

    by_module = {}
    for row in sorted(rows, key=lambda item: item["order"]):
        by_module.setdefault(row["mother_id"], []).append(row["source_type"])
    for module_id, sources in by_module.items():
        seen_exam = False
        for source in sources:
            if source == "408真题":
                seen_exam = True
            assert not (seen_exam and source == "LeetCode"), f"LeetCode after exam in {module_id}"

    assert BASE.is_file() and DASHBOARD.is_file() and INDEX.is_file(), "root files missing"
    base_data = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    assert set(base_data) >= {"filters", "formulas", "properties", "views"}, "invalid Base root"
    formula_names = set(base_data["formulas"])
    assert formula_names == {"completion", "status_icon", "days_to_review", "overdue"}
    assert len(base_data["views"]) == 5, "expected five views"
    for view in base_data["views"]:
        for prop in view.get("order", []):
            if prop.startswith("formula."):
                assert prop.removeprefix("formula.") in formula_names, f"undefined formula: {prop}"

    dashboard = DASHBOARD.read_text(encoding="utf-8")
    for view in ["全部题目", "408 真题", "待复习"]:
        embed = f"![[408算法刷题记录.base#{view}]]"
        assert embed in dashboard, f"missing embed: {view}"
    assert "[[408算法母题索引]]" in dashboard, "missing index link"

    print("PASS: 57 notes, 47 LeetCode, 10 exams, years 2017-2026, valid YAML and ordering")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise
~~~

- [ ] **Step 3: Prove the validator fails before implementation**

Run:

~~~bash
/Users/heyuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /tmp/408-obsidian-build/validate_tracker.py
~~~

Expected: FAIL with 数据结构 directory missing.

### Task 2: Generate the Obsidian tracker from Excel

**Files:**

- Create: /tmp/408-obsidian-build/generate_tracker.py
- Create: 数据结构/408算法刷题总览.md
- Create: 数据结构/408算法刷题记录.base
- Create: 数据结构/408算法母题索引.md
- Create: 数据结构/题目/*.md

- [ ] **Step 1: Write the complete generator**

Create /tmp/408-obsidian-build/generate_tracker.py with this implementation:

~~~~python
from pathlib import Path
import re
import yaml
from openpyxl import load_workbook

SOURCE = Path("/Users/heyuhang/Documents/Codex/2026-08-09/chatgpt-conversation-6a7803c0-39f0-83e8-997e/outputs/408算法题单刷题记录表_母题到真题.xlsx")
VAULT = Path("/Users/heyuhang/Documents/Obsidian Vault")
ROOT = VAULT / "数据结构"
PROBLEMS = ROOT / "题目"


def load_rows():
    workbook = load_workbook(SOURCE, data_only=True)
    sheet = workbook["刷题记录"]
    rows = []
    for row in range(5, 62):
        values = [sheet.cell(row=row, column=column).value for column in range(1, 12)]
        assert all(value is not None for value in values), f"empty source cell in row {row}"
        rows.append(values)
    return rows


def load_modules():
    workbook = load_workbook(SOURCE, data_only=True)
    sheet = workbook["母题索引"]
    modules = []
    for row in range(5, 17):
        values = [sheet.cell(row=row, column=column).value for column in range(1, 8)]
        assert all(value is not None for value in values), f"empty module cell in row {row}"
        modules.append(values)
    return modules


def safe_filename(text):
    return re.sub(r'[\\/:*?"<>|]', "-", str(text)).strip()


def note_name(order, source_type, problem_id, title):
    identity = f"{problem_id}真题" if source_type == "408真题" else str(problem_id).replace(" ", "")
    return f"{int(order):03d}_{safe_filename(identity)}_{safe_filename(title)}.md"


def render_note(row):
    order, mother_id, mother, chapter, source_type, problem_id, title, priority, time_limit, handwriting, goal = row
    properties = {
        "title": title,
        "tags": ["408/数据结构/刷题"],
        "order": int(order),
        "mother_id": mother_id,
        "mother": mother,
        "chapter": chapter,
        "source_type": source_type,
        "problem_id": problem_id,
        "priority": priority,
        "time_limit": int(time_limit),
        "handwriting": handwriting,
        "goal": goal,
        "status": "未开始",
        "first_date": None,
        "first_minutes": None,
        "first_result": "未完成",
        "second_date": None,
        "second_result": "未完成",
        "paper_date": None,
        "paper_score": None,
        "error_type": "无",
        "next_review": None,
        "review_count": 0,
        "mastery": None,
        "weakness": "",
        "note": "",
    }
    frontmatter = yaml.safe_dump(properties, allow_unicode=True, sort_keys=False).strip()
    exam_notice = ""
    if source_type == "408真题":
        exam_notice = (
            "> [!warning] 真题作答要求\n"
            "> 本页题名是训练摘要。正式手写时必须对照真题原卷，严格使用题目给定的数据结构、函数接口和复杂度要求。\n\n"
        )
    body = f"""---
{frontmatter}
---

# {title}

> [!info] 训练要求
> - 母题：{mother_id} {mother}
> - 来源：{source_type} · {problem_id}
> - 优先级：{priority}
> - 建议限时：{int(time_limit)} 分钟
> - 手写要求：{handwriting}
> - 核心目标：{goal}

{exam_notice}## 思路记录

写清所用数据结构、关键观察、循环不变量或递归函数含义。

## 代码或伪代码

~~~cpp

~~~

## 复杂度分析

- 时间复杂度：
- 空间复杂度：

## 错因复盘

- 错误发生在哪里：
- 正确处理方式：
- 下次识别信号：

## 再次手写

- [ ] 首刷后 1～3 天独立重做
- [ ] 第 7 天纸笔完整书写
- [ ] 第 14 天更换数据结构接口迁移
- [ ] 第 21 天限时模拟

返回：[[408算法刷题总览]]
"""
    return note_name(order, source_type, problem_id, title), body


BASE_YAML = """filters:
  and:
    - 'file.inFolder("数据结构/题目")'
    - 'file.hasTag("408/数据结构/刷题")'
formulas:
  completion: 'if(status == "可纸笔", 1, if(status == "已二刷", 0.75, if(status == "已AC", 0.5, if(status == "学习中", 0.25, 0))))'
  status_icon: 'if(status == "可纸笔", "✅", if(status == "已二刷", "🔁", if(status == "已AC", "🟡", if(status == "学习中", "🟠", "⚪"))))'
  days_to_review: 'if(next_review, (date(next_review) - today()).days, "")'
  overdue: 'if(next_review && status != "可纸笔" && date(next_review) <= today(), "⚠️ 到期", "")'
properties:
  file.name: {displayName: "题目笔记"}
  order: {displayName: "顺序"}
  mother_id: {displayName: "母题"}
  mother: {displayName: "母题名称"}
  source_type: {displayName: "来源"}
  problem_id: {displayName: "题号/年份"}
  priority: {displayName: "优先级"}
  time_limit: {displayName: "限时(分钟)"}
  handwriting: {displayName: "手写"}
  goal: {displayName: "核心训练目标"}
  status: {displayName: "状态"}
  paper_date: {displayName: "纸笔日期"}
  paper_score: {displayName: "纸笔得分"}
  error_type: {displayName: "错误类型"}
  next_review: {displayName: "下次复习"}
  review_count: {displayName: "复习次数"}
  mastery: {displayName: "掌握评分"}
  weakness: {displayName: "薄弱点"}
  formula.completion: {displayName: "完成率"}
  formula.status_icon: {displayName: ""}
  formula.days_to_review: {displayName: "距复习天数"}
  formula.overdue: {displayName: "提醒"}
views:
  - type: table
    name: "全部题目"
    groupBy: {property: mother_id, direction: ASC}
    order: [order, formula.status_icon, file.name, source_type, problem_id, priority, time_limit, handwriting, status, formula.completion, next_review, formula.overdue]
    summaries: {formula.completion: Average, review_count: Sum}
  - type: table
    name: "母题路线"
    groupBy: {property: mother_id, direction: ASC}
    order: [order, file.name, mother, source_type, goal, status, formula.completion]
  - type: table
    name: "408 真题"
    filters:
      and:
        - 'source_type == "408真题"'
    order: [order, file.name, problem_id, mother, time_limit, status, paper_date, paper_score, error_type, next_review]
    summaries: {paper_score: Average, review_count: Sum}
  - type: table
    name: "待复习"
    filters:
      and:
        - 'next_review != null'
        - 'status != "可纸笔"'
    order: [formula.overdue, next_review, formula.days_to_review, file.name, status, weakness]
  - type: table
    name: "可纸笔"
    filters:
      and:
        - 'status == "可纸笔"'
    order: [order, file.name, mother, priority, paper_score, mastery]
"""


DASHBOARD = """---
title: 408 算法刷题总览
tags:
  - 408/数据结构
aliases:
  - 408 数据结构算法题单
---

# 408 算法刷题总览

> [!abstract] 使用方式
> 按“母题 → LeetCode 训练题 → 对应真题”完成。只需在题目笔记顶部属性中更新状态、日期、耗时、得分和薄弱点，下面的 Bases 视图会自动同步。

母题说明：[[408算法母题索引]]

## 状态标准

| 状态 | 完成率 | 判定标准 |
|---|---:|---|
| 未开始 | 0% | 尚未独立分析 |
| 学习中 | 25% | 正在理解思路或模板 |
| 已AC | 50% | 已在 LeetCode 通过，或真题完成一次 |
| 已二刷 | 75% | 不看旧代码可以独立重做 |
| 可纸笔 | 100% | 可限时写出思想、代码和复杂度 |

## 复习节奏

1. 第 0 天：首刷并记录耗时。
2. 第 1～3 天：不看旧代码独立二刷。
3. 第 7 天：纸笔完整书写。
4. 第 14 天：更换存储结构或接口迁移。
5. 第 21 天：按照真题标准限时模拟。

## 全部题目

![[408算法刷题记录.base#全部题目]]

## 近十年 408 真题

![[408算法刷题记录.base#408 真题]]

## 待复习

![[408算法刷题记录.base#待复习]]
"""


def write_index(modules):
    lines = [
        "---",
        "title: 408 算法母题索引",
        "tags:",
        "  - 408/数据结构",
        "---",
        "",
        "# 408 算法母题索引",
        "",
        "返回：[[408算法刷题总览]]",
        "",
        "| 编号 | 母题模块 | 核心思想 | LeetCode 训练题 | 对应真题 | 过关要求 | 推荐顺序 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in modules:
        clean = [str(value).replace("|", "｜").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(clean) + " |")
    (ROOT / "408算法母题索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = load_rows()
    modules = load_modules()
    assert len(rows) == 57
    assert len([row for row in rows if row[4] == "LeetCode"]) == 47
    assert len([row for row in rows if row[4] == "408真题"]) == 10
    ROOT.mkdir(parents=True, exist_ok=True)
    PROBLEMS.mkdir(parents=True, exist_ok=True)
    assert not list(PROBLEMS.glob("*.md")), "题目 directory must be empty before generation"
    for row in rows:
        filename, content = render_note(row)
        (PROBLEMS / filename).write_text(content, encoding="utf-8")
    (ROOT / "408算法刷题记录.base").write_text(BASE_YAML, encoding="utf-8")
    (ROOT / "408算法刷题总览.md").write_text(DASHBOARD, encoding="utf-8")
    write_index(modules)
    print("CREATED: 57 problem notes and 3 root tracker files")


if __name__ == "__main__":
    main()
~~~~

- [ ] **Step 2: Run the generator**

Run:

~~~bash
/Users/heyuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /tmp/408-obsidian-build/generate_tracker.py
~~~

Expected: CREATED: 57 problem notes and 3 root tracker files.

- [ ] **Step 3: Commit the generated content only after validation**

Do not stage any file until Task 3 and Task 4 pass.

### Task 3: Validate generated Markdown and Base data

**Files:**

- Test: /tmp/408-obsidian-build/validate_tracker.py
- Inspect: 数据结构/408算法刷题总览.md
- Inspect: 数据结构/408算法刷题记录.base
- Inspect: 数据结构/408算法母题索引.md
- Inspect: 数据结构/题目/001_LC27_移除元素.md
- Inspect: 数据结构/题目/005_2018真题_最小未出现正整数.md
- Inspect: 数据结构/题目/057_LC164_最大间距.md

- [ ] **Step 1: Run the validator**

Run:

~~~bash
/Users/heyuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /tmp/408-obsidian-build/validate_tracker.py
~~~

Expected:

~~~text
PASS: 57 notes, 47 LeetCode, 10 exams, years 2017-2026, valid YAML and ordering
~~~

- [ ] **Step 2: Inspect representative files**

Run:

~~~bash
sed -n '1,220p' 数据结构/408算法刷题总览.md
sed -n '1,280p' 数据结构/408算法刷题记录.base
sed -n '1,220p' 数据结构/408算法母题索引.md
sed -n '1,180p' 数据结构/题目/001_LC27_移除元素.md
sed -n '1,180p' 数据结构/题目/005_2018真题_最小未出现正整数.md
sed -n '1,180p' 数据结构/题目/057_LC164_最大间距.md
~~~

Expected: readable Chinese, valid YAML, all required sections, the exam warning in the 2018 note, and correct return links.

- [ ] **Step 3: Scan for placeholders and check embeds**

Run:

~~~bash
rg -n "T""BD|T""ODO|待""定" 数据结构
rg -n "408算法刷题记录\.base#(全部题目|408 真题|待复习)" 数据结构/408算法刷题总览.md
~~~

Expected: no placeholder matches and exactly three embed matches.

### Task 4: Test in Obsidian

**Files:**

- Open: 数据结构/408算法刷题总览.md
- Open: 数据结构/408算法刷题记录.base

- [ ] **Step 1: Open the dashboard and Base**

Run:

~~~bash
obsidian open path="数据结构/408算法刷题总览.md"
obsidian open path="数据结构/408算法刷题记录.base"
~~~

Expected: both files open in the active vault.

- [ ] **Step 2: Check Obsidian errors**

Run:

~~~bash
obsidian dev:errors
obsidian dev:console level=error
~~~

Expected: no YAML or formula error attributable to the new tracker.

- [ ] **Step 3: Capture and inspect a screenshot**

Run:

~~~bash
obsidian dev:screenshot path="/tmp/408-obsidian-build/408-tracker.png"
~~~

Expected: five Base views are available, the current view contains problem rows, and headers are readable without clipping.

### Task 5: Final verification and commit

**Files:**

- Add: 数据结构/408算法刷题总览.md
- Add: 数据结构/408算法刷题记录.base
- Add: 数据结构/408算法母题索引.md
- Add: 数据结构/题目/*.md
- Add: docs/superpowers/plans/2026-08-09-408-algorithm-obsidian-implementation.md

- [ ] **Step 1: Run fresh verification**

Run:

~~~bash
/Users/heyuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /tmp/408-obsidian-build/validate_tracker.py
git diff --check -- 数据结构 docs/superpowers/plans/2026-08-09-408-algorithm-obsidian-implementation.md
~~~

Expected: validator prints PASS and git diff --check prints nothing.

- [ ] **Step 2: Confirm only intended files are in scope**

Run:

~~~bash
git status --short -- 数据结构 docs/superpowers/plans/2026-08-09-408-algorithm-obsidian-implementation.md
~~~

Expected: only the new 数据结构 tree and this plan are listed; unrelated existing changes remain untouched.

- [ ] **Step 3: Commit**

Run:

~~~bash
git add 数据结构 docs/superpowers/plans/2026-08-09-408-algorithm-obsidian-implementation.md
git commit -m "feat: 添加408算法Obsidian刷题记录"
~~~

Expected: one commit containing the Base, dashboard, index, 57 problem notes, and implementation plan.
