# 408 Data Structure Apex Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a visually coherent, evidence-driven 408 data-structure training system whose Apex home page selects the next useful work, whose Obsidian notes preserve real training history, and whose 12 pattern pages plus 10 exam questions support closed-book handwritten practice.

**Architecture:** Problem-note frontmatter remains the only source of training truth. A shared state engine performs atomic, evidence-gated updates; small DataviewJS views render controls, statistics, queues, navigation, and tables; Apex reads the same frontmatter through its built-in DQL engine. Project-scoped CSS supplies the Matcha visual language without changing notes outside `11_数据结构`.

**Tech Stack:** Obsidian 1.13.x CLI, Dataview/DataviewJS, Apex Dashboard, Obsidian Flavored Markdown, JavaScript, Node.js `node:test`, Ruby/Psych for mechanical YAML migration, scoped CSS.

---

## Scope and execution order

The dashboard, state controls, training pages, and content are coupled through one frontmatter contract, so they remain one ordered plan rather than separate plans. The implementation is split into independently committable tasks: contract tests → data migration → state engine → page normalization → dashboards → Apex → styling → mother-pattern content → exam content → documentation and visual QA.

Run every shell command from the Git root `/Users/heyuhang/Documents/Obsidian Vault` unless a step explicitly states otherwise.

Do not modify any path outside `11_数据结构` except these Vault-level files:

- `.obsidian/community-plugins.json` and `.obsidian/plugins/apex-dashboard/**` for the approved Apex installation;
- `.obsidian/appearance.json` and `.obsidian/snippets/408-training.css` for the approved project-scoped snippet.

The pre-existing dirty path `08_DL基础及论文进阶/.claude/worktrees/wizardly-gould-b23ea3` is out of scope and must remain untouched.

## File map

| Responsibility | Exact path |
|---|---|
| Apex home data file | `11_数据结构/408作战台.md` |
| Deep analytics page | `11_数据结构/总览.md` |
| Exam sprint page | `11_数据结构/真题/真题冲刺.md` |
| Error review page | `11_数据结构/错题回炉.md` |
| Evidence calibration page | `11_数据结构/训练数据校准.md` |
| Pure transition rules | `11_数据结构/系统/共享/state-engine.js` |
| Per-problem evidence UI | `11_数据结构/系统/状态控件/view.js`, `view.css` |
| Previous/next navigation | `11_数据结构/系统/题目导航/view.js`, `view.css` |
| Overview KPI cards | `11_数据结构/系统/统计卡/view.js`, `view.css` |
| Today/review queues | `11_数据结构/系统/复习队列/view.js`, `view.css` |
| Filterable problem table | `11_数据结构/系统/题库表/view.js`, `view.css` |
| Mother-template evidence UI | `11_数据结构/系统/母题复写/view.js`, `view.css` |
| Mechanical migration | `11_数据结构/系统/脚本/migrate-frontmatter.rb` |
| Mechanical page normalization | `11_数据结构/系统/脚本/normalize-problem-pages.rb` |
| Vault acceptance checks | `11_数据结构/系统/脚本/validate-training-vault.rb` |
| State-rule tests | `11_数据结构/系统/测试/state-engine.test.mjs` |
| Future note templates | `11_数据结构/系统/模板/题目模板.md`, `母题模板.md` |
| Project visual system | `.obsidian/snippets/408-training.css` |

## Canonical data contract

Problem status values are exactly `未开始`, `学习中`, `已AC`, `已二刷`, `可纸笔`. Evidence status values are exactly `有效`, `待校准`. Result values are:

- `首刷结果`: `未完成`, `独立通过`, `订正通过`, `失败`;
- `二刷结果`: `未完成`, `独立通过`, `失败`.

Paper scores use `思想得分` 0–2, `正确性得分` 0–2, `代码得分` 0–4, and `复杂度得分` 0–2. `纸笔得分` is derived from those four fields. Empty dates and scores remain the empty YAML string; the migration must not invent training history.

Review intervals are fixed: independent first pass +3 days, corrected first pass +1 day, second pass +7 days, paper pass +14 days, migration/retention pass +30 days, and any failed attempt +1 day.

## Implementation tasks

### Task 1: Add a phase-aware acceptance harness

**Files:**
- Create: `11_数据结构/系统/脚本/validate-training-vault.rb`

- [ ] **Step 1: Create the validator with explicit data, structure, and content checks**

Use this complete shape; keep every assertion and error label so later tasks receive actionable failures:

```ruby
#!/usr/bin/env ruby
require "yaml"
require "date"
require "pathname"

ROOT = Pathname.new(__dir__).join("../..").cleanpath
PHASES = %w[data structure content all].freeze
phase = (ARGV[0] || "all").sub(/^--phase=/, "")
abort "phase must be one of #{PHASES.join(', ')}" unless PHASES.include?(phase)

def parse_note(path)
  raw = path.read
  match = raw.match(/\A---\s*\n(.*?)\n---\s*\n/m)
  raise "#{path}: missing YAML frontmatter" unless match
  fm = YAML.safe_load(match[1], permitted_classes: [Date], aliases: false) || {}
  [fm, raw[match.end(0)..] || ""]
end

errors = []
check = ->(condition, message) { errors << message unless condition }
problems = ROOT.join("刷题记录").glob("*.md").sort
patterns = ROOT.join("母题模板").glob("*.md").sort
check.call(problems.length == 57, "DATA count/problems expected=57 actual=#{problems.length}")
check.call(patterns.length == 12, "DATA count/patterns expected=12 actual=#{patterns.length}")

problem_required = %w[题号 母题 母题模块 章节 来源 引用 标题 优先级 限时 手写 训练目标 状态
  首刷日期 首刷耗时 首刷结果 二刷日期 二刷结果 纸笔日期 纸笔得分 错误类型 下次复习
  复习次数 掌握评分 薄弱点 备注 证据状态 最后训练 思想得分 正确性得分 代码得分 复杂度得分]
statuses = %w[未开始 学习中 已AC 已二刷 可纸笔]
evidence_states = %w[有效 待校准]
exam_count = 0
pending_count = 0

problems.each do |path|
  fm, body = parse_note(path)
  problem_required.each { |key| check.call(fm.key?(key), "DATA #{path.basename}: missing #{key}") }
  check.call(statuses.include?(fm["状态"]), "DATA #{path.basename}: invalid 状态=#{fm['状态'].inspect}")
  check.call(evidence_states.include?(fm["证据状态"]), "DATA #{path.basename}: invalid 证据状态=#{fm['证据状态'].inspect}")
  check.call(Array(fm["cssclasses"]).include?("ds-408-problem"), "DATA #{path.basename}: missing ds-408-problem")
  exam_count += 1 if fm["来源"] == "408真题"
  pending_count += 1 if fm["证据状态"] == "待校准"

  parts = %w[思想得分 正确性得分 代码得分 复杂度得分].map { |key| fm[key] }
  unless parts.all? { |v| v.nil? || v == "" }
    nums = parts.map { |v| Float(v) rescue nil }
    check.call(nums.none?(&:nil?), "DATA #{path.basename}: paper score components must all be numeric")
    if nums.none?(&:nil?)
      check.call(nums[0].between?(0, 2) && nums[1].between?(0, 2) && nums[2].between?(0, 4) && nums[3].between?(0, 2),
                 "DATA #{path.basename}: paper score component out of range")
      total = Float(fm["纸笔得分"]) rescue nil
      check.call(total == nums.sum, "DATA #{path.basename}: 纸笔得分 must equal component sum")
    end
  end

  next unless %w[structure content all].include?(phase)
  check.call(body.include?('dv.view("11_数据结构/系统/状态控件"'), "STRUCTURE #{path.basename}: missing shared state view")
  check.call(body.include?('dv.view("11_数据结构/系统/题目导航"'), "STRUCTURE #{path.basename}: missing shared navigation view")
  check.call(!body.include?("const statuses = ['未开始'"), "STRUCTURE #{path.basename}: duplicated legacy state script")
  check.call(!body.include?("当前状态：**未开始**"), "STRUCTURE #{path.basename}: static status conflict")

  next unless %w[content all].include?(phase)
  %w[独立作答 训练控制台 错因复盘].each do |heading|
    check.call(body.include?(heading), "CONTENT #{path.basename}: missing #{heading}")
  end
  if fm["来源"] == "408真题"
    %w[算法思想 正确性说明 C/C++代码 复杂度分析 评分点].each do |label|
      compact = body.gsub(/\s+/, "")
      check.call(compact.include?(label), "CONTENT #{path.basename}: missing #{label}")
    end
    check.call(body.include?("[!success]-"), "CONTENT #{path.basename}: exam answer must be collapsed")
  else
    check.call(body.include?("我的实现与复盘"), "CONTENT #{path.basename}: missing personal implementation area")
  end
end

check.call(exam_count == 10, "DATA count/exams expected=10 actual=#{exam_count}")
check.call(pending_count.between?(0, 15), "DATA count/pending-evidence must remain within 0..15 actual=#{pending_count}")

patterns.each do |path|
  fm, body = parse_note(path)
  %w[模板状态 上次复写 下次复写 复写得分].each { |key| check.call(fm.key?(key), "DATA #{path.basename}: missing #{key}") }
  check.call(Array(fm["cssclasses"]).include?("ds-408-pattern"), "DATA #{path.basename}: missing ds-408-pattern")
  next unless %w[content all].include?(phase)
  %w[识别信号 核心不变量 C/C++纸笔模板 正确性说明 复杂度 常见边界与失分点 可迁移变化 模板闭卷复写].each do |heading|
    check.call(body.include?(heading), "CONTENT #{path.basename}: missing #{heading}")
  end
  check.call(!body.include?("在此手写核心模板代码"), "CONTENT #{path.basename}: legacy empty pattern prompt")
end

if %w[structure content all].include?(phase)
  required_pages = %w[408作战台.md 总览.md 错题回炉.md 训练数据校准.md 真题/真题冲刺.md]
  required_pages.each { |rel| check.call(ROOT.join(rel).file?, "STRUCTURE missing #{rel}") }
  overview = ROOT.join("总览.md").read
  %w[统计卡 复习队列 题库表].each { |view| check.call(overview.include?("系统/#{view}"), "STRUCTURE 总览 missing #{view} view") }
  apex = ROOT.join("408作战台.md").read
  check.call(apex.include?("dashboard: true"), "STRUCTURE Apex dashboard flag missing")
  check.call(apex.include?("type: dataview"), "STRUCTURE Apex DQL sections missing")
end

if errors.empty?
  puts "PASS phase=#{phase} problems=#{problems.length} exams=#{exam_count} patterns=#{patterns.length} pending=#{pending_count}"
else
  warn errors.join("\n")
  warn "FAIL phase=#{phase} errors=#{errors.length}"
  exit 1
end
```

- [ ] **Step 2: Run the data phase and confirm it fails for the missing new contract**

Run:

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=data
```

Expected: exit 1 with missing `证据状态`, `最后训练`, scoring fields, pattern evidence fields, and CSS-class messages. This is the red test for Task 2.

- [ ] **Step 3: Commit the failing acceptance harness**

```bash
git add 11_数据结构/系统/脚本/validate-training-vault.rb
git commit -m "test: define 408 vault acceptance contract"
```

### Task 2: Migrate frontmatter without inventing history

**Files:**
- Create: `11_数据结构/系统/脚本/migrate-frontmatter.rb`
- Modify: `11_数据结构/刷题记录/*.md`
- Modify: `11_数据结构/母题模板/*.md`

- [ ] **Step 1: Write the idempotent migration script**

The script must split frontmatter from body, retain every existing non-empty value, append only missing fields, and write a note only when its serialized content changed. Use these core rules in the implementation:

```ruby
PROBLEM_DEFAULTS = {
  "最后训练" => "", "思想得分" => "", "正确性得分" => "",
  "代码得分" => "", "复杂度得分" => ""
}.freeze

def present?(value)
  !value.nil? && value != ""
end

def evidenced?(fm)
  case fm["状态"]
  when "未开始", "学习中" then true
  when "已AC" then present?(fm["首刷日期"]) && %w[独立通过 订正通过].include?(fm["首刷结果"])
  when "已二刷" then present?(fm["二刷日期"]) && fm["二刷结果"] == "独立通过"
  when "可纸笔" then present?(fm["纸笔日期"]) && (Float(fm["纸笔得分"]) rescue -1) >= 8
  else false
  end
end
```

For every problem:

```ruby
fm["cssclasses"] = (Array(fm["cssclasses"]) + ["ds-408-problem"]).uniq
PROBLEM_DEFAULTS.each { |key, value| fm[key] = value unless fm.key?(key) }
fm["证据状态"] = evidenced?(fm) ? "有效" : "待校准" unless fm.key?("证据状态")
```

For every mother-pattern page:

```ruby
fm["cssclasses"] = (Array(fm["cssclasses"]) + ["ds-408-pattern"]).uniq
fm["模板状态"] = "未开始" unless fm.key?("模板状态")
fm["上次复写"] = "" unless fm.key?("上次复写")
fm["下次复写"] = "" unless fm.key?("下次复写")
fm["复写得分"] = "" unless fm.key?("复写得分")
```

Use `YAML.safe_load(..., permitted_classes: [Date], aliases: false)` and `YAML.dump` for serialization. Preserve the original body byte-for-byte after the closing frontmatter delimiter.

- [ ] **Step 2: Run the migration twice to prove idempotence**

```bash
ruby 11_数据结构/系统/脚本/migrate-frontmatter.rb
git diff --stat -- 11_数据结构/刷题记录 11_数据结构/母题模板
before=$(git diff -- 11_数据结构/刷题记录 11_数据结构/母题模板 | shasum -a 256)
ruby 11_数据结构/系统/脚本/migrate-frontmatter.rb
after=$(git diff -- 11_数据结构/刷题记录 11_数据结构/母题模板 | shasum -a 256)
test "$before" = "$after"
```

Expected after the first run: 57 problem notes and 12 pattern notes receive only contract fields/classes. Expected from the second script run: `changed=0`; the existing Git diff remains unchanged rather than growing.

- [ ] **Step 3: Run the data acceptance phase**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=data
```

Expected: `PASS phase=data problems=57 exams=10 patterns=12 pending=15`.

- [ ] **Step 4: Verify old progress was preserved exactly**

```bash
ruby -ryaml -e '
files=Dir["11_数据结构/刷题记录/*.md"];
counts=Hash.new(0);
files.each{|f| y=YAML.safe_load(File.read(f)[/\A---\n(.*?)\n---/m,1]); counts[y["状态"]]+=1};
abort counts.inspect unless counts=={"已AC"=>11,"可纸笔"=>4,"未开始"=>42};
puts counts.inspect
'
```

Expected: `{"已AC"=>11, "可纸笔"=>4, "未开始"=>42}` in any hash display order.

- [ ] **Step 5: Commit the migration separately**

```bash
git add 11_数据结构/系统/脚本/migrate-frontmatter.rb 11_数据结构/刷题记录 11_数据结构/母题模板
git commit -m "feat: migrate 408 training evidence fields"
```

### Task 3: Implement and test the atomic state engine

**Files:**
- Create: `11_数据结构/系统/共享/state-engine.js`
- Create: `11_数据结构/系统/测试/state-engine.test.mjs`

- [ ] **Step 1: Write failing tests for every scoring and scheduling rule**

Tests must load the CommonJS-compatible engine with `createRequire` and cover these exact cases:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const engine = require("../共享/state-engine.js");
const base = {
  状态: "未开始", 证据状态: "有效", 首刷日期: "", 首刷耗时: "", 首刷结果: "未完成",
  二刷日期: "", 二刷结果: "未完成", 纸笔日期: "", 纸笔得分: "",
  思想得分: "", 正确性得分: "", 代码得分: "", 复杂度得分: "",
  最后训练: "", 下次复习: "", 复习次数: 0
};

test("independent first pass schedules three days", () => {
  const patch = engine.buildAction(base, { kind: "transition", target: "已AC", evidence: { 首刷结果: "独立通过", 首刷耗时: 18 } }, "2026-08-18");
  assert.equal(patch.首刷日期, "2026-08-18");
  assert.equal(patch.下次复习, "2026-08-21");
  assert.equal(patch.状态, "已AC");
});

test("corrected first pass schedules one day", () => {
  const patch = engine.buildAction(base, { kind: "transition", target: "已AC", evidence: { 首刷结果: "订正通过" } }, "2026-08-18");
  assert.equal(patch.下次复习, "2026-08-19");
});

test("second pass schedules seven days and keeps first date", () => {
  const current = { ...base, 状态: "已AC", 首刷日期: "2026-08-01", 首刷结果: "独立通过" };
  const patch = engine.buildAction(current, { kind: "transition", target: "已二刷", evidence: { 二刷结果: "独立通过" } }, "2026-08-18");
  assert.equal(patch.首刷日期, undefined);
  assert.equal(patch.二刷日期, "2026-08-18");
  assert.equal(patch.下次复习, "2026-08-25");
});

test("paper score below eight cannot enter 可纸笔", () => {
  const current = { ...base, 状态: "已二刷", 二刷日期: "2026-08-01", 二刷结果: "独立通过" };
  assert.throws(() => engine.buildAction(current, {
    kind: "transition", target: "可纸笔",
    evidence: { 思想得分: 2, 正确性得分: 1, 代码得分: 3, 复杂度得分: 1 }
  }, "2026-08-18"), /纸笔得分必须不低于 8/);
});

test("paper score is derived and schedules fourteen days", () => {
  const current = { ...base, 状态: "已二刷", 二刷日期: "2026-08-01", 二刷结果: "独立通过" };
  const patch = engine.buildAction(current, {
    kind: "transition", target: "可纸笔",
    evidence: { 思想得分: 2, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 }
  }, "2026-08-18");
  assert.equal(patch.纸笔得分, 8);
  assert.equal(patch.纸笔日期, "2026-08-18");
  assert.equal(patch.下次复习, "2026-09-01");
});

test("backward transition retains historical evidence", () => {
  const current = { ...base, 状态: "可纸笔", 首刷日期: "2026-07-01", 二刷日期: "2026-07-04", 纸笔日期: "2026-07-11", 纸笔得分: 9 };
  const patch = engine.buildAction(current, { kind: "transition", target: "学习中" }, "2026-08-18");
  assert.equal(patch.状态, "学习中");
  assert.equal(patch.首刷日期, undefined);
  assert.equal(patch.纸笔得分, undefined);
});

test("failed action schedules tomorrow without advancing", () => {
  const patch = engine.buildAction(base, { kind: "failure", stage: "首刷" }, "2026-08-18");
  assert.equal(patch.状态, undefined);
  assert.equal(patch.首刷结果, "失败");
  assert.equal(patch.下次复习, "2026-08-19");
});

test("retention pass schedules thirty days", () => {
  const patch = engine.buildAction({ ...base, 状态: "可纸笔" }, { kind: "retention", passed: true }, "2026-08-18");
  assert.equal(patch.下次复习, "2026-09-17");
});

test("same-stage closed-book validation clears old pending evidence", () => {
  const current = { ...base, 状态: "已AC", 证据状态: "待校准" };
  const patch = engine.buildAction(current, { kind: "transition", target: "已AC", evidence: { 首刷结果: "独立通过" } }, "2026-08-18");
  assert.equal(patch.首刷日期, "2026-08-18");
  assert.equal(patch.证据状态, "有效");
  assert.equal(patch.状态, "已AC");
});
```

- [ ] **Step 2: Run the tests and confirm the module-not-found failure**

```bash
node --test 11_数据结构/系统/测试/state-engine.test.mjs
```

Expected: FAIL because `系统/共享/state-engine.js` does not exist.

- [ ] **Step 3: Implement the engine as a pure patch builder**

The module must expose `STATUSES`, `buildAction`, `paperTotal`, and `evidenceSatisfied`. It must never mutate the supplied frontmatter. Calculate and validate the entire patch before `processFrontMatter` receives it.

```javascript
const STATUSES = ["未开始", "学习中", "已AC", "已二刷", "可纸笔"];
const LEVEL = Object.fromEntries(STATUSES.map((name, index) => [name, index]));
const present = value => value !== undefined && value !== null && value !== "";
const passedFirst = value => ["独立通过", "订正通过"].includes(value);

function addDays(iso, days) {
  const date = new Date(`${iso}T00:00:00`);
  date.setDate(date.getDate() + days);
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function paperTotal(source) {
  const keys = ["思想得分", "正确性得分", "代码得分", "复杂度得分"];
  const limits = [2, 2, 4, 2];
  if (keys.some(key => !present(source[key]))) throw new Error("四项纸笔评分必须全部填写");
  const values = keys.map(key => Number(source[key]));
  if (values.some(Number.isNaN)) throw new Error("四项纸笔评分必须全部填写");
  values.forEach((value, index) => {
    if (value < 0 || value > limits[index]) throw new Error(`${keys[index]}超出范围`);
  });
  return values.reduce((sum, value) => sum + value, 0);
}

function evidenceSatisfied(source, status = source.状态) {
  if (["未开始", "学习中"].includes(status)) return true;
  if (status === "已AC") return present(source.首刷日期) && passedFirst(source.首刷结果);
  if (status === "已二刷") return present(source.二刷日期) && source.二刷结果 === "独立通过";
  if (status === "可纸笔") return present(source.纸笔日期) && Number(source.纸笔得分) >= 8;
  return false;
}

function setIfEmpty(patch, draft, key, value) {
  if (!present(draft[key])) { patch[key] = value; draft[key] = value; }
}

function buildAction(current, action, today) {
  const draft = { ...current };
  const patch = {};
  const evidence = action.evidence || {};
  for (const [key, value] of Object.entries(evidence)) {
    if (value !== "" && value !== undefined) { patch[key] = value; draft[key] = value; }
  }

  if (["transition", "failure", "retention"].includes(action.kind)) {
    patch.最后训练 = today;
    patch.复习次数 = Number(current.复习次数 || 0) + 1;
  }

  if (action.kind === "failure") {
    if (action.stage === "首刷") patch.首刷结果 = "失败";
    if (action.stage === "二刷") patch.二刷结果 = "失败";
    patch.下次复习 = addDays(today, 1);
    return patch;
  }

  if (action.kind === "retention") {
    if (current.状态 !== "可纸笔") throw new Error("只有可纸笔题目可记录迁移或保温训练");
    patch.下次复习 = addDays(today, action.passed ? 30 : 1);
    return patch;
  }

  if (action.kind !== "transition" || !STATUSES.includes(action.target)) throw new Error("未知训练动作");
  const target = action.target;
  const movingForward = LEVEL[target] > LEVEL[current.状态];
  const calibratingCurrent = target === current.状态 && current.证据状态 === "待校准";
  const needsEvidence = movingForward || calibratingCurrent;

  if (target === "学习中" && needsEvidence) setIfEmpty(patch, draft, "首刷日期", today);
  if (target === "已AC" && needsEvidence) {
    if (!passedFirst(draft.首刷结果)) throw new Error("进入已AC前必须记录独立通过或订正通过");
    setIfEmpty(patch, draft, "首刷日期", today);
    patch.下次复习 = addDays(today, draft.首刷结果 === "独立通过" ? 3 : 1);
  }
  if (target === "已二刷" && needsEvidence) {
    if (draft.二刷结果 !== "独立通过") throw new Error("进入已二刷前必须闭卷独立通过");
    setIfEmpty(patch, draft, "二刷日期", today);
    patch.下次复习 = addDays(today, 7);
  }
  if (target === "可纸笔" && needsEvidence) {
    const total = paperTotal(draft);
    if (total < 8) throw new Error("纸笔得分必须不低于 8");
    patch.纸笔得分 = total;
    draft.纸笔得分 = total;
    setIfEmpty(patch, draft, "纸笔日期", today);
    patch.下次复习 = addDays(today, 14);
  }

  patch.状态 = target;
  draft.状态 = target;
  if (current.证据状态 === "待校准" && evidenceSatisfied(draft, target)) patch.证据状态 = "有效";
  return patch;
}

const api = { STATUSES, buildAction, paperTotal, evidenceSatisfied };
if (typeof module !== "undefined" && module.exports) module.exports = api;
```

- [ ] **Step 4: Run tests and require all eight to pass**

```bash
node --test 11_数据结构/系统/测试/state-engine.test.mjs
```

Expected: `tests 9`, `pass 9`, `fail 0`.

- [ ] **Step 5: Commit engine and tests**

```bash
git add 11_数据结构/系统/共享/state-engine.js 11_数据结构/系统/测试/state-engine.test.mjs
git commit -m "feat: add evidence-gated training state engine"
```

### Task 4: Build shared problem controls and navigation

**Files:**
- Create: `11_数据结构/系统/状态控件/view.js`
- Create: `11_数据结构/系统/状态控件/view.css`
- Create: `11_数据结构/系统/题目导航/view.js`
- Create: `11_数据结构/系统/题目导航/view.css`

- [ ] **Step 1: Add a DataviewJS control that uses the tested engine**

The view must load the engine once from Vault text and create it in a function scope:

```javascript
const engineSource = await dv.io.load("11_数据结构/系统/共享/state-engine.js");
const engine = Function(`${engineSource}\nreturn api;`)();
const page = input?.path ? dv.page(input.path) : dv.current();
const file = dv.app.vault.getAbstractFileByPath(page.file.path);
const root = dv.container.createDiv({ cls: `ds408-control ${input?.compact ? "is-compact" : ""}` });
```

Render, in this order:

1. current-state badge and `证据待补` warning badge;
2. five state buttons;
3. `首刷结果`, `首刷耗时`, `二刷结果`, four paper-score numeric inputs;
4. `记录首刷失败`, `记录二刷失败`, `迁移通过`, `迁移失败` actions;
5. read-only evidence summary for dates, total, review count, and next review.

Each action must perform exactly one `processFrontMatter` call:

```javascript
async function applyAction(action) {
  try {
    await dv.app.fileManager.processFrontMatter(file, fm => {
      const patch = engine.buildAction({ ...fm }, action, window.moment().format("YYYY-MM-DD"));
      Object.assign(fm, patch);
    });
    new Notice("训练证据已保存");
  } catch (error) {
    new Notice(`未保存：${error.message}`);
  }
}
```

State-button clicks must include the visible form as `evidence`, so a completed form and a transition are one atomic click. A thrown validation error occurs before `Object.assign`; this is the rollback guarantee.

- [ ] **Step 2: Add compact, accessible control styles**

`view.css` must use only selectors under `.ds408-control`. Include visible `:focus-visible`, `aria-live="polite"` for result notices, 44 px minimum touch height on narrow screens, and no fixed text colors that fail dark mode.

- [ ] **Step 3: Add previous, mother, overview, and next navigation**

`系统/题目导航/view.js` reads all records sorted by `题号`, finds the current index, and renders links in this exact order:

```javascript
const records = dv.pages('"11_数据结构/刷题记录"').where(p => p.题号).sort(p => p.题号, "asc").array();
const current = input?.path ? dv.page(input.path) : dv.current();
const index = records.findIndex(p => p.file.path === current.file.path);
const links = [
  index > 0 ? records[index - 1].file.link : null,
  dv.fileLink(`11_数据结构/母题模板/${current.母题}_${current.母题模块}`),
  dv.fileLink("11_数据结构/总览"),
  index + 1 < records.length ? records[index + 1].file.link : null
];
```

Label the links `← 上一题`, `母题 Mxx`, `训练总览`, `下一题 →`; omit missing ends instead of creating dead links.

- [ ] **Step 4: Smoke-test views in one note before mass replacement**

Temporarily invoke both views at the bottom of `刷题记录/001_M01_LC27_移除元素.md`, open that note, and run:

```bash
obsidian dev:errors
obsidian dev:dom selector=".ds408-control" total
obsidian dev:dom selector=".ds408-problem-nav" total
```

Expected: no new project error, control count `1`, navigation count `1`. Revert only the temporary two invocations before Task 5; keep the view files.

- [ ] **Step 5: Commit shared views**

```bash
git add 11_数据结构/系统/状态控件 11_数据结构/系统/题目导航
git commit -m "feat: add shared 408 training controls"
```

### Task 5: Normalize all 57 problem pages and add future templates

**Files:**
- Create: `11_数据结构/系统/脚本/normalize-problem-pages.rb`
- Create: `11_数据结构/系统/模板/题目模板.md`
- Create: `11_数据结构/系统/模板/母题模板.md`
- Modify: `11_数据结构/刷题记录/*.md`

- [ ] **Step 1: Write the page-normalization script with preservation guards**

For each problem, capture and preserve:

- everything before `## ⚡ 状态快速切换` (title card, question, hint);
- the exact text under `## 🎯 训练目标`;
- the exact text under `## 📝 解题笔记` (the 2018 answer must survive unchanged).

Set `training_goal` to the captured goal. Build `answer_or_personal_section` as `## 📝 我的实现与复盘` plus the captured notes for LeetCode; for a 408 page, build `## 🧠 满分参考答案` plus a collapsed `[!success]-` callout whose quoted body is the captured notes. Tasks 12–14 replace the 408 callout bodies with the specified full-score answers.

Rebuild the remaining body with this deterministic order:

````markdown
## 🎯 训练目标

#{training_goal.rstrip}

## ✍️ 独立作答

> [!question] 闭卷作答区
> 先写输入、输出、约束和朴素方法，再写关键不变量、边界与复杂度。完成后再展开参考内容。

- **算法思想**：
- **关键不变量或递归含义**：
- **边界条件**：
- **时间复杂度及依据**：
- **空间复杂度及依据**：

## 🧭 训练控制台

```dataviewjs
await dv.view("11_数据结构/系统/状态控件", { mode: "problem" });
```

#{answer_or_personal_section.rstrip}

## 🔁 错因复盘

- **错误类型**：以顶部属性为准
- **薄弱点**：以顶部属性为准
- **下一步**：根据控制台中的下次复习日期闭卷重做

## 🧭 导航

```dataviewjs
await dv.view("11_数据结构/系统/题目导航");
```
````

The script must abort if a file lacks any capture boundary or if it finds content after `## 📚 错题复盘` that it cannot classify. It must print `normalized=57` and be idempotent.

- [ ] **Step 2: Run the normalizer twice and inspect representative diffs**

```bash
ruby 11_数据结构/系统/脚本/normalize-problem-pages.rb
ruby 11_数据结构/系统/脚本/normalize-problem-pages.rb
git diff -- 11_数据结构/刷题记录/001_M01_LC27_移除元素.md
git diff -- 11_数据结构/刷题记录/005_M01_2018_最小未出现正整数.md
git diff -- 11_数据结构/刷题记录/041_M09_2024_判断有向图是否存在唯一拓扑序列.md
```

Expected: second run reports `normalized=0`; 2018 answer text remains present; legacy inline scripts and static status sentences are gone.

- [ ] **Step 3: Create exact future templates**

The problem template must include every canonical property, `cssclasses: [ds-408-problem]`, the normalized section order, both shared view calls, empty personal evidence, and no fabricated date/result. The mother template must include `cssclasses: [ds-408-pattern]`, four mother evidence fields, the eight required teaching sections, training/exam Dataview queries parameterized by the chosen mother number, and the mother rewrite view.

- [ ] **Step 4: Run structure checks that can pass at this stage**

```bash
rg -l "const statuses = \['未开始'" 11_数据结构/刷题记录 | wc -l
rg -l '系统/状态控件' 11_数据结构/刷题记录 | wc -l
rg -l '系统/题目导航' 11_数据结构/刷题记录 | wc -l
```

Expected: `0`, `57`, `57`.

- [ ] **Step 5: Commit normalized pages and templates**

```bash
git add 11_数据结构/系统/脚本/normalize-problem-pages.rb 11_数据结构/系统/模板 11_数据结构/刷题记录
git commit -m "refactor: standardize 408 problem training pages"
```

### Task 6: Replace the monolithic overview with focused views and entry pages

**Files:**
- Create: `11_数据结构/系统/统计卡/view.js`, `view.css`
- Create: `11_数据结构/系统/复习队列/view.js`, `view.css`
- Create: `11_数据结构/系统/题库表/view.js`, `view.css`
- Modify: `11_数据结构/总览.md`
- Create: `11_数据结构/真题/真题冲刺.md`
- Create: `11_数据结构/错题回炉.md`
- Create: `11_数据结构/训练数据校准.md`

- [ ] **Step 1: Write tests as exact DOM and query contracts**

Before creating the views, record these failing checks:

```bash
test -f 11_数据结构/系统/统计卡/view.js
test -f 11_数据结构/系统/复习队列/view.js
test -f 11_数据结构/系统/题库表/view.js
rg -q 'ds408-kpi-weighted' 11_数据结构/系统/统计卡/view.js
rg -q '证据待补' 11_数据结构/系统/复习队列/view.js
rg -q 'data-filter' 11_数据结构/系统/题库表/view.js
```

Expected: FAIL because the view files do not exist.

- [ ] **Step 2: Implement the KPI view with four non-ambiguous metrics**

`系统/统计卡/view.js` must calculate and label these separately:

```javascript
const pages = dv.pages('"11_数据结构/刷题记录"').where(p => p.题号).array();
const weight = { 未开始: 0, 学习中: 0.25, 已AC: 0.5, 已二刷: 0.75, 可纸笔: 1 };
const total = pages.length || 1;
const exams = pages.filter(p => p.来源 === "408真题");
const weighted = 100 * pages.reduce((sum, p) => sum + (weight[p.状态] ?? 0), 0) / total;
const acPlus = 100 * pages.filter(p => ["已AC", "已二刷", "可纸笔"].includes(p.状态)).length / total;
const paper = 100 * pages.filter(p => p.状态 === "可纸笔").length / total;
const examPaper = 100 * exams.filter(p => p.状态 === "可纸笔").length / (exams.length || 1);
```

Render `.ds408-kpi-weighted`, `.ds408-kpi-ac`, `.ds408-kpi-paper`, and `.ds408-kpi-exam`; each card includes its formula in a tooltip or subtitle.

- [ ] **Step 3: Implement today's three tasks and alerts**

`系统/复习队列/view.js` must build three candidates with this stable comparator:

```javascript
const priority = { "S+": 0, S: 1, A: 2, B: 3 };
const level = { 未开始: 0, 学习中: 1, 已AC: 2, 已二刷: 3, 可纸笔: 4 };
const dueTime = value => value ? dv.date(value)?.toMillis?.() ?? Infinity : Infinity;
const rank = p => [
  dueTime(p.下次复习),
  p.来源 === "408真题" ? 0 : 1,
  priority[p.优先级] ?? 9,
  p.手写 === "必须" ? 0 : 1,
  level[p.状态] ?? 9,
  Number(p.题号) || 999
];
```

Select one overdue/pending-evidence problem, one unfinished or pending 408 exam, and one due/unmastered mother pattern. When a category lacks a candidate, take the highest-ranked unused item from all problem/pattern candidates. Render overdue count, next-seven-days count, and evidence-pending count above the cards.

- [ ] **Step 4: Implement the filterable table using the shared engine**

Filters are mother (`全部`, M01–M12), status, priority, source, and evidence state. Sorting keys are problem number, mother, priority, state, next review, and paper score. Compact status buttons must load `系统/共享/state-engine.js` and call the same atomic `buildAction`; they must not contain a second transition implementation. A forward-state click opens that row's inline evidence editor, and confirmation passes the entered fields to `buildAction` in the same `processFrontMatter` call. A backward-state click shows a confirmation explaining that historical evidence is retained.

- [ ] **Step 5: Rewrite `总览.md` as a thin composition page**

Use this structure:

````markdown
---
tags: ["408", 数据结构, 总览]
cssclasses: [ds-408-dashboard]
---

# 408 数据结构训练总览

> [!tip] 使用方式
> 首页决定今天做什么；本页解释整体掌握度、到期复习、证据质量和每道题的阶段。

## 核心指标
```dataviewjs
await dv.view("11_数据结构/系统/统计卡");
```

## 今日训练与复习告警
```dataviewjs
await dv.view("11_数据结构/系统/复习队列");
```

## 全量题库
```dataviewjs
await dv.view("11_数据结构/系统/题库表");
```

## 母题索引
```dataview
TABLE 母题模块, 模板状态, 复写得分, 下次复写
FROM "11_数据结构/母题模板"
SORT 母题编号 ASC
```
````

- [ ] **Step 6: Create the three focused entry pages**

`真题/真题冲刺.md` queries `来源 = "408真题"` and displays year, mother, status, evidence state, paper score, next review. `错题回炉.md` includes `错误类型 != "无" OR 证据状态 = "待校准"`. `训练数据校准.md` includes only `证据状态 = "待校准"` and explains that a matching closed-book validation clears the flag; none of these pages stores duplicate status.

- [ ] **Step 7: Open all four pages and inspect runtime errors**

```bash
obsidian open path="11_数据结构/总览.md"
obsidian dev:errors
obsidian open path="11_数据结构/真题/真题冲刺.md"
obsidian dev:errors
obsidian open path="11_数据结构/错题回炉.md"
obsidian dev:errors
obsidian open path="11_数据结构/训练数据校准.md"
obsidian dev:errors
```

Expected: no new error whose stack or file path contains `11_数据结构/系统`.

- [ ] **Step 8: Commit overview and entry pages**

```bash
git add 11_数据结构/系统/统计卡 11_数据结构/系统/复习队列 11_数据结构/系统/题库表 11_数据结构/总览.md 11_数据结构/真题/真题冲刺.md 11_数据结构/错题回炉.md 11_数据结构/训练数据校准.md
git commit -m "feat: add 408 training analytics and queues"
```

### Task 7: Install and configure Apex Dashboard

**Files:**
- Create/modify through Obsidian: `.obsidian/plugins/apex-dashboard/**`
- Modify: `.obsidian/community-plugins.json`
- Create: `11_数据结构/408作战台.md`

Primary references: [Apex Dashboard README](https://github.com/PandoraReads/apex-dashboard/blob/main/README.md) and [Chinese README](https://github.com/PandoraReads/apex-dashboard/blob/main/README_ZH.md).

- [ ] **Step 1: Install the community plugin through the Obsidian CLI**

```bash
obsidian plugin:install id=apex-dashboard enable
obsidian plugin id=apex-dashboard
obsidian plugins:enabled filter=community versions | rg '^apex-dashboard\b'
```

Expected: Apex appears in the enabled community-plugin list. Do not download unversioned `main.js` manually when the community installation succeeds.

- [ ] **Step 2: Create the Apex data file with action-first DQL sections**

`408作战台.md` must use `dashboard: true`, a concise 408 quote banner, file quick actions for `总览`, `真题冲刺`, `错题回炉`, `训练数据校准`, and `使用说明`, followed by DQL sections in this order:

1. `今日首要任务`: overdue first, then evidence-pending, then 408/priority/handwriting/low-stage/problem-number ranking, `LIMIT 1`;
2. `到期复习`: overdue or pending-evidence problem, `LIMIT 1`;
3. `今日真题`: 408 exam not fully evidenced, `LIMIT 1`;
4. `模板复写`: mother pattern due or not `可复写`, `LIMIT 1`;
5. `复习告警`: all overdue, seven-day-due, and pending-evidence rows, `LIMIT 8`;
6. `本周得分链`: notes whose `最后训练 >= today - dur("7 days")`, showing four scores and total;
7. `训练阶段分布`: `GROUP BY 状态`, showing `length(rows)`.

Use Apex's frontmatter DQL form exactly:

```yaml
columns:
  - name: 今日首要任务
    color: "#6f8f72"
    type: dataview
    dataview:
      query: 'TABLE WITHOUT ID file.link AS "题目", 来源, 母题, 限时, 状态, 证据状态 FROM "11_数据结构/刷题记录" WHERE 状态 != "可纸笔" OR 证据状态 = "待校准" SORT choice(下次复习 AND date(下次复习) <= today, 0, choice(证据状态 = "待校准", 1, 2)) ASC, choice(下次复习, date(下次复习), date("2999-12-31")) ASC, choice(来源 = "408真题", 0, 1) ASC, choice(优先级 = "S+", 0, choice(优先级 = "S", 1, choice(优先级 = "A", 2, 3))) ASC, choice(手写 = "必须", 0, 1) ASC, 题号 ASC LIMIT 1'
      title: "今天先完成这一题"
      pageSize: 5
      density: compact
      striped: false
      rowNumbers: false
```

Each column needs a matching empty `##` heading in the body so Apex preserves order. Quick actions must use Vault-relative `.md` targets.

- [ ] **Step 3: Configure approved settings through the loaded plugin API**

```bash
obsidian eval code='(async()=>{const p=app.plugins.getPlugin("apex-dashboard");if(!p)throw new Error("Apex not loaded");p.settings={...p.settings,dashboardFile:"11_数据结构/408作战台.md",language:"zh",stylePreset:"matcha",recentDocCount:5,widgetWeatherEnabled:false,widgetLunarEnabled:false,readingEnabled:false,countdownEnabled:false};await p.saveSettings();p.refreshAllDashboards();return JSON.stringify({dashboardFile:p.settings.dashboardFile,language:p.settings.language,stylePreset:p.settings.stylePreset});})()'
```

Expected JSON contains `408作战台.md`, `zh`, and `matcha`.

- [ ] **Step 4: Open Apex and validate its rendered sections**

Run the command palette action `Apex Dashboard: Open dashboard`, then:

```bash
obsidian dev:dom selector=".apex-dashboard-root" total
obsidian dev:dom selector=".dashboard-dataview-section" total
obsidian dev:errors
```

Expected: one Apex root, seven DQL sections, and no new Apex/DQL parse errors. If a DQL expression is rejected, simplify only that expression while preserving its sort order and record the accepted query in `408作战台.md`.

- [ ] **Step 5: Test the approved degradation path**

```bash
obsidian plugin:disable id=apex-dashboard
obsidian open path="11_数据结构/总览.md"
obsidian dev:dom selector=".ds408-kpi-weighted" total
obsidian plugin:enable id=apex-dashboard
```

Expected: total remains usable with one weighted KPI while Apex is disabled; Apex re-enables afterward.

- [ ] **Step 6: Commit Apex installation and dashboard data separately**

```bash
git add .obsidian/plugins/apex-dashboard .obsidian/community-plugins.json 11_数据结构/408作战台.md
git commit -m "feat: add Matcha Apex 408 command center"
```

### Task 8: Add the scoped Matcha visual system

**Files:**
- Create: `.obsidian/snippets/408-training.css`
- Modify: `.obsidian/appearance.json`
- Modify frontmatter only: `11_数据结构/总览.md`, `408作战台.md`, `真题/真题冲刺.md`, `错题回炉.md`, `训练数据校准.md`, `使用说明.md`

- [ ] **Step 1: Create design tokens and strictly scoped selectors**

Use these tokens under the project classes; do not add a bare `.markdown-preview-view`, `.callout`, `table`, `pre`, or `code` selector:

```css
.ds-408-dashboard,
.ds-408-problem,
.ds-408-pattern {
  --ds408-matcha-900: #30483a;
  --ds408-matcha-700: #58725f;
  --ds408-matcha-500: #7f9d82;
  --ds408-matcha-200: #cbd8c7;
  --ds408-paper: #f6f2e7;
  --ds408-ink: #26342c;
  --h1-color: var(--ds408-matcha-900);
  --h2-color: var(--ds408-matcha-700);
  --link-color: var(--ds408-matcha-700);
  --interactive-accent: var(--ds408-matcha-700);
}

.theme-dark .ds-408-dashboard,
.theme-dark .ds-408-problem,
.theme-dark .ds-408-pattern {
  --ds408-paper: #202822;
  --ds408-ink: #e7eee8;
}

.ds-408-dashboard .markdown-preview-sizer,
.ds-408-problem .markdown-preview-sizer,
.ds-408-pattern .markdown-preview-sizer {
  max-width: 1120px;
}
```

Add scoped rules for KPI cards, action cards, state badges, callouts, tables, code blocks, property panels, navigation pills, focus rings, dark mode, and `@media (max-width: 700px)`. On narrow screens, cards become one column and tables use an internal scroll wrapper; page-level horizontal overflow is forbidden.

- [ ] **Step 2: Add `ds-408-dashboard` to every entry/help page**

Problem and mother classes already come from migration. Add only `ds-408-dashboard` to the six dashboard/help pages; do not set a global body class.

- [ ] **Step 3: Enable the snippet without losing existing appearance settings**

Update `.obsidian/appearance.json` by preserving `baseFontSize` and merging `"408-training"` into `enabledCssSnippets`. Reload Obsidian:

```bash
obsidian reload
obsidian dev:css selector=".ds-408-problem .callout" prop="border-radius"
```

Expected: the computed rule source is `.obsidian/snippets/408-training.css`.

- [ ] **Step 4: Prove the snippet does not leak into another project**

Open one note outside `11_数据结构`, query its body classes, and compare the scoped card selector count:

```bash
obsidian dev:dom selector=".markdown-preview-view.ds-408-problem, .markdown-preview-view.ds-408-pattern, .markdown-preview-view.ds-408-dashboard" total
```

Expected outside this project: `0`.

- [ ] **Step 5: Commit visual system**

```bash
git add .obsidian/snippets/408-training.css .obsidian/appearance.json 11_数据结构
git commit -m "style: add scoped Matcha 408 visual system"
```

### Task 9: Complete mother patterns M01–M04

**Files:**
- Modify: `11_数据结构/母题模板/M01_数组原地操作与一次扫描.md`
- Modify: `11_数据结构/母题模板/M02_有序数组双指针与多指针.md`
- Modify: `11_数据结构/母题模板/M03_单链表原地操作.md`
- Modify: `11_数据结构/母题模板/M04_栈与队列按定义模拟.md`
- Create: `11_数据结构/系统/母题复写/view.js`, `view.css`

- [ ] **Step 1: Implement mother rewrite evidence controls**

The view accepts a 0–10 score, sets `模板状态` to `可复写` only at 8 or above, writes `上次复写` today, schedules +14 days on pass or +1 day on failure, and never overwrites an existing historical date except `上次复写` because that field intentionally means the most recent attempt.

- [ ] **Step 2: Write M01 with two reusable templates**

Include in-place cyclic positioning/marking and one-pass suffix-state maintenance. State the invariant `扫描位置 i 之前的目标信息已经最终确定`, show duplicate protection in the positioning loop, prove the answer range `[1,n+1]`, and analyze O(n)/O(1) where applicable.

- [ ] **Step 3: Write M02 with the ordered multi-pointer template**

Define distance as `2(max-min)`, move a pointer currently attaining the minimum, and prove that retaining that minimum while increasing other values cannot reduce the current range. Code must accept three arrays and lengths and run in O(n1+n2+n3) time and O(1) extra space.

- [ ] **Step 4: Write M03 with find-middle, reverse, and alternating merge**

Use a headed singly linked list, explicitly sever the two halves before reversal, preserve `next` before every pointer rewrite, and cover lengths 0, 1, 2, odd, and even. Complexity is O(n) time and O(1) extra space.

- [ ] **Step 5: Write M04 with array stack and circular queue contracts**

Use one consistent circular-queue convention: one empty slot, empty when `front == rear`, full when `(rear + 1) % MaxSize == front`, size `(rear-front+MaxSize)%MaxSize`. Include push/pop and enqueue/dequeue failure returns and an expression/bracket simulation example.

- [ ] **Step 6: Validate and commit M01–M04**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=content 2>&1 | rg '母题模板/M0[1-4]' || true
git add 11_数据结构/母题模板/M0{1,2,3,4}_*.md 11_数据结构/系统/母题复写
git commit -m "docs: complete 408 mother patterns M01 to M04"
```

Expected targeted validation output: no M01–M04 missing-section or legacy-empty-prompt errors.

### Task 10: Complete mother patterns M05–M08

**Files:**
- Modify: `11_数据结构/母题模板/M05_二叉树遍历、递归与构造.md`
- Modify: `11_数据结构/母题模板/M06_并查集与堆.md`
- Modify: `11_数据结构/母题模板/M07_图的存储、度与遍历.md`
- Modify: `11_数据结构/母题模板/M08_最小生成树与最短路径.md`

- [ ] **Step 1: Write M05 traversal and reconstruction templates**

Include recursive preorder/inorder/postorder, iterative inorder with an explicit stack, level order with a queue, and preorder+inorder reconstruction. Every recursive function states its meaning and base case; reconstruction documents uniqueness when keys are distinct.

- [ ] **Step 2: Write M06 disjoint-set and heap templates**

Include parent-array initialization, path-compressed `Find`, union by rank/size, zero-based `AdjustDown`, bottom-up `BuildHeap`, and heap-sort extraction. State amortized disjoint-set bounds and O(n) heap construction versus O(n log n) heap sort.

- [ ] **Step 3: Write M07 graph storage, degree, DFS, and BFS templates**

Separate directed row=outdegree/column=indegree from undirected row-sum degree. Include adjacency-matrix DFS/BFS with disconnected-component outer loops and state O(n²) for a matrix and O(n+e) for an adjacency list.

- [ ] **Step 4: Write M08 Prim, Kruskal, Dijkstra, and Floyd templates**

For each algorithm, state input restrictions and invariant. Explicitly forbid Dijkstra with negative edges; distinguish MST from shortest path; include Floyd update `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])` and guard infinity addition.

- [ ] **Step 5: Validate and commit M05–M08**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=content 2>&1 | rg '母题模板/M0[5-8]' || true
git add 11_数据结构/母题模板/M0{5,6,7,8}_*.md
git commit -m "docs: complete 408 mother patterns M05 to M08"
```

### Task 11: Complete mother patterns M09–M12

**Files:**
- Modify: `11_数据结构/母题模板/M09_拓扑排序与关键路径.md`
- Modify: `11_数据结构/母题模板/M10_折半、散列与字符串查找.md`
- Modify: `11_数据结构/母题模板/M11_BST 与中序有序.md`
- Modify: `11_数据结构/母题模板/M12_经典排序、归并与划分.md`

- [ ] **Step 1: Write M09 Kahn and critical-path templates**

Include topological output, cycle detection by processed count, uniqueness detection by requiring exactly one zero-indegree vertex at every step, and earliest/latest event-time recurrences on an AOE network.

- [ ] **Step 2: Write M10 binary search, hash table, and KMP templates**

Use one closed interval `[left,right]` binary-search convention and derive loop/return boundaries. Include linear-probing search/insert with empty and deleted sentinels. Define the KMP `next` meaning before code and trace one mismatch fallback.

- [ ] **Step 3: Write M11 BST search, validation, insertion, deletion, and nearest-value templates**

State whether duplicate keys are forbidden. Include range-bound validation or strict-inorder validation, all three deletion cases, and explain why the search path brackets K; when a question asks for all tied nearest keys, retain both predecessor and successor candidates rather than assuming a single result.

- [ ] **Step 4: Write M12 insertion, quick, merge, heap, and radix/bucket principles**

Include stable/unstable and in-place/extra-space labels, worst/average time, quick-sort degeneration protection, merge-buffer boundaries, and the zero-based heap child indices `2i+1`, `2i+2`.

- [ ] **Step 5: Run mother-content acceptance and commit**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=content 2>&1 | rg '母题模板' || true
git add 11_数据结构/母题模板/M{09,10,11,12}_*.md
git commit -m "docs: complete 408 mother patterns M09 to M12"
```

Expected targeted output: no mother-pattern errors.

### Task 12: Write full-score answers for the linear-list exam questions

**Files:**
- Modify: `11_数据结构/刷题记录/005_M01_2018_最小未出现正整数.md`
- Modify: `11_数据结构/刷题记录/006_M01_2025_数组后缀最大乘积.md`
- Modify: `11_数据结构/刷题记录/008_M02_2020_三个有序数组的最小距离.md`
- Modify: `11_数据结构/刷题记录/015_M03_2019_单链表重排.md`

Every answer is inside `> [!success]- 满分参考答案（完成独立作答后展开）` and contains the exact bold labels `算法思想`, `正确性说明`, `C/C++ 代码`, `复杂度分析`, and `评分点`.

- [ ] **Step 1: Upgrade 2018 with a closed proof chain**

Retain the O(n)/O(1) cyclic-placement solution. Prove: the answer belongs to `[1,n+1]`; the while condition prevents duplicate-value loops; after termination, any present x in `[1,n]` occupies x−1; the first mismatch is therefore the minimum missing positive. Mention the auxiliary-array O(n) alternative only as a contrast, not the final solution.

- [ ] **Step 2: Write 2025 suffix maximum-product answer**

Scan right-to-left while maintaining suffix maximum and minimum. For each `A[i]`, choose `A[i]*suffixMax` when nonnegative and `A[i]*suffixMin` when negative; include `j=i` through the initialized suffix. Use `long long` internally if overflow is possible, state O(n) time and O(1) auxiliary space excluding output.

- [ ] **Step 3: Write 2020 three-array minimum-distance answer**

Use `D=2(max-min)`, update the answer, then advance one pointer attaining the current minimum. The proof must show that while the minimum remains fixed, increasing either nonminimum value cannot produce a smaller maximum-minus-minimum range; hence only advancing a minimum pointer can improve it. Complexity O(n1+n2+n3)/O(1).

- [ ] **Step 4: Write 2019 singly linked-list reorder answer**

Use fast/slow split, reverse the second half, and alternate merge. Code must match the headed-list interface, sever the first half, save both next pointers before splicing, and terminate correctly for odd/even lengths. Complexity O(n)/O(1).

- [ ] **Step 5: Validate and commit the four answers**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=content 2>&1 | rg '00[568]|015' || true
git add 11_数据结构/刷题记录/{005_M01_2018_最小未出现正整数.md,006_M01_2025_数组后缀最大乘积.md,008_M02_2020_三个有序数组的最小距离.md,015_M03_2019_单链表重排.md}
git commit -m "docs: add full-score linear-list exam answers"
```

### Task 13: Write full-score answers for the tree and graph exam questions

**Files:**
- Modify: `11_数据结构/刷题记录/028_M05_2017_表达式树转中缀表达式.md`
- Modify: `11_数据结构/刷题记录/033_M07_2021_判断无向图是否存在 EL 路径.md`
- Modify: `11_数据结构/刷题记录/034_M07_2023_邻接矩阵中寻找 K 顶点.md`
- Modify: `11_数据结构/刷题记录/041_M09_2024_判断有向图是否存在唯一拓扑序列.md`

- [ ] **Step 1: Write 2017 expression-tree answer**

Use a recursive function with a root flag. A leaf prints its operand; an internal nonroot subtree prints opening and closing parentheses; a unary minus has a null left child and prints `-` before its right subtree. The proof is structural induction on subtree height. Complexity O(n) time and O(h) recursion stack.

- [ ] **Step 2: Write 2021 EL-path answer**

Use the condition already granted by the question: the graph is nonempty and connected, so an Euler trail exists iff the odd-degree vertex count is 0 or 2. Sum each adjacency-matrix row for one vertex degree; do not divide the row sum by two. Return immediately if more than two odd vertices are found. Complexity O(n²)/O(1).

- [ ] **Step 3: Write 2023 K-vertex answer**

For each vertex i, sum row i as outdegree and column i as indegree, print `VerticesList[i]` when outdegree > indegree, and count printed vertices. Complexity O(n²)/O(1). State the row/column interpretation explicitly.

- [ ] **Step 4: Write 2024 unique-topological-order answer**

Compute indegrees from the matrix. At every Kahn iteration scan/count zero-indegree unprocessed vertices: zero means a cycle, more than one means multiple topological choices, exactly one is removed and its outgoing neighbors are decremented. Return 1 only after processing n vertices with uniqueness maintained. With a matrix, complexity is O(n²) and auxiliary space O(n).

- [ ] **Step 5: Validate and commit the four answers**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=content 2>&1 | rg '028|033|034|041' || true
git add '11_数据结构/刷题记录/028_M05_2017_表达式树转中缀表达式.md' '11_数据结构/刷题记录/033_M07_2021_判断无向图是否存在 EL 路径.md' '11_数据结构/刷题记录/034_M07_2023_邻接矩阵中寻找 K 顶点.md' '11_数据结构/刷题记录/041_M09_2024_判断有向图是否存在唯一拓扑序列.md'
git commit -m "docs: add full-score tree and graph exam answers"
```

### Task 14: Write full-score answers for the BST exam questions

**Files:**
- Modify: `11_数据结构/刷题记录/051_M11_2022_顺序存储二叉树判断 BST.md`
- Modify: `11_数据结构/刷题记录/052_M11_2026_BST 中寻找与 K 绝对差最小的所有结点.md`

- [ ] **Step 1: Write 2022 sequential-tree BST answer**

Perform inorder traversal using indices `2i+1` and `2i+2`, stop at `i >= ElemNum` or value −1, and compare each visited positive key with a mutable previous key. The sequence must be strictly increasing. The proof uses the BST inorder theorem in both directions for distinct keys. Complexity O(ElemNum) time in the array representation and O(h) recursion stack.

- [ ] **Step 2: Write 2026 all-nearest-nodes answer with a correct tie argument**

Do not claim that recursive descent into only one child can discover every tied key without qualification. Use BST search to maintain the greatest key `< K`, the least key `> K`, and an exact match. At termination: exact match gives difference 0 and one key; otherwise compute both boundary differences and output one or both boundary keys when tied. This is O(h) time and O(1) iterative auxiliary space, and it correctly returns all distinct BST keys at the minimum absolute difference.

- [ ] **Step 3: Validate all ten exam answers and commit**

```bash
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=content 2>&1 | rg '刷题记录' || true
git add '11_数据结构/刷题记录/051_M11_2022_顺序存储二叉树判断 BST.md' '11_数据结构/刷题记录/052_M11_2026_BST 中寻找与 K 绝对差最小的所有结点.md'
git commit -m "docs: add full-score BST exam answers"
```

Expected targeted output: no exam-answer errors for any of the 10 exam files.

### Task 15: Update instructions and run full functional verification

**Files:**
- Modify: `11_数据结构/使用说明.md`
- Modify if verification finds a project defect: files already introduced by Tasks 1–14

- [ ] **Step 1: Rewrite the guide to match the implemented workflow**

Document:

- Apex home versus total-overview responsibilities;
- five status definitions and evidence gates;
- first/second/paper/migration intervals;
- the 2+2+4+2 paper rubric;
- how `待校准` is cleared without losing old status;
- how to use the mother rewrite score;
- how Apex/Dataview degradation works;
- the rule that personal dates, errors, weak points, and scores must come from real training.

Remove the duplicated table header and the old statement that changing only `状态` is sufficient.

- [ ] **Step 2: Run all automated checks fresh**

```bash
node --test 11_数据结构/系统/测试/state-engine.test.mjs
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=all
rg -l "const statuses = \['未开始'" 11_数据结构/刷题记录 | wc -l
rg -l '在此手写核心模板代码' 11_数据结构/母题模板 | wc -l
obsidian links path="11_数据结构/408作战台.md"
obsidian links path="11_数据结构/总览.md"
obsidian dev:errors
```

Expected: Node tests all pass; Vault validator prints `PASS phase=all problems=57 exams=10 patterns=12` with the current pending count in `0..15`; both legacy-text counts are 0; links resolve; no new project error.

- [ ] **Step 3: Verify an actual state transition and rollback**

Use a temporary copy of one `未开始` problem note inside `11_数据结构/系统/测试/fixtures/`, open it, and verify:

1. attempting `可纸笔` with 7 points shows an error and changes no property;
2. entering `已AC` with `独立通过` fills only missing first-pass date, last training, review count, and +3-day review;
3. moving back to `学习中` preserves the date and result;
4. deleting the fixture after the check leaves the 57 production notes untouched.

Use a recoverable fixture deletion and confirm `validate-training-vault.rb --phase=all` still passes afterward.

- [ ] **Step 4: Commit the guide and any verified corrections**

```bash
git add 11_数据结构/使用说明.md 11_数据结构
git commit -m "docs: document the 408 evidence training workflow"
```

### Task 16: Perform screenshot-based visual QA and final repository audit

**Files:**
- Create verification artifacts only when useful: `11_数据结构/docs/qa/*.png`
- Modify only the responsible CSS/view file when a defect is observed

- [ ] **Step 1: Capture desktop light-mode pages**

Capture Apex, total overview, one LeetCode problem, one 408 exam, and one mother pattern:

```bash
obsidian command id=apex-dashboard:open-dashboard
obsidian dev:screenshot path="11_数据结构/docs/qa/apex-light.png"
obsidian open path="11_数据结构/总览.md"
obsidian dev:screenshot path="11_数据结构/docs/qa/overview-light.png"
obsidian open path="11_数据结构/刷题记录/001_M01_LC27_移除元素.md"
obsidian dev:screenshot path="11_数据结构/docs/qa/problem-light.png"
obsidian open path="11_数据结构/刷题记录/041_M09_2024_判断有向图是否存在唯一拓扑序列.md"
obsidian dev:screenshot path="11_数据结构/docs/qa/exam-light.png"
obsidian open path="11_数据结构/母题模板/M09_拓扑排序与关键路径.md"
obsidian dev:screenshot path="11_数据结构/docs/qa/pattern-light.png"
```

Inspect each with the local image viewer. Check hierarchy, clipping, line length, control focus, table overflow, and whether collapsed answers remain collapsed.

- [ ] **Step 2: Repeat in dark mode and mobile emulation**

Switch Obsidian to dark mode, capture Apex/overview/problem, then run `obsidian dev:mobile on` and capture Apex/overview/problem again. Return with `obsidian dev:mobile off`. Required outcome: no page-level horizontal overflow, no hidden buttons, and readable contrast.

- [ ] **Step 3: Re-run CSS-scope and error checks after visual fixes**

```bash
rg -n '^\s*(\.markdown-preview-view|\.callout|table|pre|code)(\s|\{|,)' .obsidian/snippets/408-training.css
obsidian dev:errors
git status --short --branch --untracked-files=all
git diff --check
```

Expected: the scope scan has no matches; errors have no new `11_数据结构` source; only intended project/Apex/snippet changes plus the pre-existing external dirty worktree appear; `git diff --check` is clean.

- [ ] **Step 4: Run the complete verification suite one final time**

```bash
node --test 11_数据结构/系统/测试/state-engine.test.mjs
ruby 11_数据结构/系统/脚本/validate-training-vault.rb --phase=all
obsidian plugins:enabled filter=community versions | rg '^(apex-dashboard|dataview)\b'
obsidian dev:dom selector=".apex-dashboard-root" total
obsidian dev:errors
```

Expected: all rule tests and Vault checks pass, both plugins are enabled, one Apex root is rendered, and no project error is present.

- [ ] **Step 5: Commit visual corrections and QA artifacts**

```bash
git add .obsidian/snippets/408-training.css 11_数据结构
git commit -m "test: complete 408 dashboard visual QA"
```

## Completion checklist

Before reporting completion, confirm each item from the approved design:

- 57 problem notes use one evidence contract and shared controls;
- exactly 15 old positive-state notes remain `待校准` until real validation;
- four dashboard rates have distinct names and formulas;
- Apex uses `408作战台.md`, Chinese, and Matcha;
- today's three categories and fallback rank are implemented;
- all 12 mother patterns contain executable handwritten templates and proofs;
- all 10 exam questions contain collapsed full-score answers;
- all 47 LeetCode pages preserve personal practice areas without invented history;
- Apex-disabled and Dataview-disabled reading paths remain usable;
- desktop light, desktop dark, and narrow/mobile screenshots have been inspected;
- no CSS selector changes pages outside `11_数据结构`;
- no unrelated dirty worktree is staged or committed.
