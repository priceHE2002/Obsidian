import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const dashboardPath = `${root}11_数据结构/408作战台.md`;
const bundlePath = `${root}.obsidian/plugins/apex-dashboard/main.js`;
const dashboard = readFileSync(dashboardPath, 'utf8');

function loadApex186Probe() {
  let source = readFileSync(bundlePath, 'utf8');
  // These symbols are private to the official, hash-pinned 1.8.6 bundle. The
  // probe appends an export in memory; it never changes the release asset.
  source += '\n;module.exports.__contractProbe={parse:y2,serialize:Os,executeDql:E1};';
  class Base { constructor() {} }
  const callable = function () {};
  const obsidian = new Proxy({
    Plugin: Base, Modal: Base, ItemView: Base, Component: Base, Menu: Base,
    Notice: Base, TFile: Base, TFolder: Base, Setting: Base,
    Platform: { isMobile: false }, normalizePath: (value) => value,
    parseYaml: () => ({}), setIcon: () => {}, requestUrl: async () => ({ json: {} }),
    MarkdownRenderer: { render: async () => {} }, debounce: (fn) => fn,
  }, { get: (target, key) => key in target ? target[key] : callable });
  const windowStub = new Proxy({ setTimeout, clearTimeout, setInterval, clearInterval }, { get: (target, key) => key in target ? target[key] : callable });
  const sandbox = {
    module: { exports: {} }, exports: {},
    require: (id) => { if (id === 'obsidian') return obsidian; throw new Error(`unexpected require: ${id}`); },
    console, setTimeout, clearTimeout, setInterval, clearInterval, window: windowStub,
    document: new Proxy({}, { get: () => callable }), navigator: {}, URL, Date,
    Math, JSON, Map, Set, WeakMap, WeakSet, Promise, RegExp, Error, Array,
    Object, String, Number, Boolean, Symbol, Intl, TextEncoder, TextDecoder,
  };
  vm.runInNewContext(source, sandbox, { filename: 'apex-dashboard/main.js' });
  return sandbox.module.exports.__contractProbe;
}

const apex = loadApex186Probe();
const parsed = apex.parse(dashboard);
const queries = Object.fromEntries(parsed.columns.map((column) => [column.name, column.dataviewConfig.query]));

function page(name, fields, folder = '11_数据结构/刷题记录') {
  const path = `${folder}/${name}.md`;
  return { file: { path, basename: name }, fields: { 'file.link': path, ...fields } };
}

function run(query, pages) {
  const outcome = apex.executeDql(query, pages);
  assert.equal(outcome.ok, true, outcome.error?.message);
  assert.equal(outcome.empty, false);
  return outcome.result.rows;
}

function withoutLimit(query) {
  return query.replace(/\s+LIMIT\s+\d+\s*$/i, '');
}

function localIso(dayOffset) {
  const date = new Date();
  date.setHours(12, 0, 0, 0);
  date.setDate(date.getDate() + dayOffset);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

test('Apex 1.8.6 round-trip proves display-only YAML keys are unsupported and the dashboard does not advertise them', () => {
  const fixture = `---
dashboard: true
banner:
  quote: x
  author: y
quickActions: []
columns:
  - name: Probe
    color: "#000000"
    type: dataview
    dataview:
      query: 'TABLE file.link FROM "X"'
      title: Probe
      pageSize: 5
      density: compact
      striped: false
      rowNumbers: false
---

## Probe
`;
  const result = apex.parse(fixture);
  assert.deepEqual(Object.keys(result.columns[0].dataviewConfig).sort(), ['query', 'title']);
  const roundTrip = apex.serialize(result);
  for (const key of ['pageSize', 'density', 'striped', 'rowNumbers']) {
    assert.equal(roundTrip.includes(`${key}:`), false, `${key} is discarded by Apex 1.8.6`);
    assert.equal(dashboard.includes(`${key}:`), false, `${key} must not imply unsupported persistence`);
  }
});

const problems = [
  page('empty', { '题号': 2, '来源': '408真题', '母题': 'M01', '限时': 30, '状态': '未开始', '证据状态': '待校准', '下次复习': '', '优先级': 'S', '手写': '必须' }),
  page('missing', { '题号': 1, '来源': '408真题', '母题': 'M01', '限时': 30, '状态': '未开始', '证据状态': '待校准', '优先级': 'S', '手写': '必须' }),
  page('overdue', { '题号': 3, '来源': 'LeetCode', '母题': 'M02', '限时': 20, '状态': '学习中', '证据状态': '有效', '下次复习': localIso(-1), '优先级': 'A', '手写': '建议' }),
  page('future', { '题号': 4, '来源': 'LeetCode', '母题': 'M02', '限时': 20, '状态': '已AC', '证据状态': '有效', '下次复习': localIso(3), '优先级': 'A', '手写': '建议' }),
  page('scored', { '题号': 5, '来源': 'LeetCode', '母题': 'M03', '限时': 25, '状态': '可纸笔', '证据状态': '有效', '最后训练': localIso(-2), '思想得分': 2, '正确性得分': 2, '代码得分': 4, '复杂度得分': 2 }),
];
const patterns = [
  page('M02', { '母题编号': 'M02', '母题模块': 'B', '模板状态': '未开始', '下次复写': '' }, '11_数据结构/母题模板'),
  page('M01', { '母题编号': 'M01', '母题模块': 'A', '模板状态': '未开始' }, '11_数据结构/母题模板'),
  page('M03', { '母题编号': 'M03', '母题模块': 'C', '模板状态': '学习中', '下次复写': localIso(-1) }, '11_数据结构/母题模板'),
];

test('all seven approved queries execute in the Apex built-in engine with empty, missing, and valid date fixtures', () => {

  for (const name of ['今日首要任务', '到期复习', '今日真题', '复习告警', '本周得分链', '训练阶段分布']) {
    assert.doesNotThrow(() => run(queries[name], problems), name);
  }
  assert.doesNotThrow(() => run(queries['模板复写'], patterns));

});

test('blank and missing review dates use the same safe fallback before stable numeric tie-breakers', () => {
  const pending = run(withoutLimit(queries['到期复习']), problems).filter((row) => ['empty', 'missing'].some((name) => row.page.file.path.endsWith(`/${name}.md`)));
  assert.deepEqual(Array.from(pending, (row) => row.page.file.basename), ['missing', 'empty'], 'blank dates share the far-future fallback, then 题号 breaks the tie');

  const rewrites = run(withoutLimit(queries['模板复写']), patterns).filter((row) => ['M01', 'M02'].includes(row.page.file.basename));
  assert.deepEqual(Array.from(rewrites, (row) => row.page.file.basename), ['M01', 'M02'], 'blank rewrite dates fall back safely before 母题编号 tie-break');
});

test('pending-evidence alert text survives an empty or missing review date in the eager choice evaluator', () => {
  const alerts = run(queries['复习告警'], problems);
  const emptyAlert = alerts.find((row) => row.page.file.basename === 'empty');
  const missingAlert = alerts.find((row) => row.page.file.basename === 'missing');
  assert.equal(emptyAlert.values[1], '待校准');
  assert.equal(missingAlert.values[1], '待校准');
});

test('unbounded weekly scores use an explicit dashboard cap while the five-state distribution stays naturally bounded', () => {
  assert.match(queries['本周得分链'], /\sLIMIT\s+50\s*$/i);
  assert.doesNotMatch(queries['训练阶段分布'], /\sLIMIT\s+/i);
});
