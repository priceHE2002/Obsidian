import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const paths = {
  kpi: `${root}11_数据结构/系统/统计卡/view.js`,
  queue: `${root}11_数据结构/系统/复习队列/view.js`,
  bank: `${root}11_数据结构/系统/题库表/view.js`,
  dashboard: `${root}11_数据结构/总览.md`,
  sprint: `${root}11_数据结构/真题/真题冲刺.md`,
  retry: `${root}11_数据结构/错题回炉.md`,
  calibration: `${root}11_数据结构/训练数据校准.md`,
};

function source(key) { return readFileSync(paths[key], 'utf8'); }
function flatPages(items) {
  return {
    where(predicate) { return flatPages(items.filter(predicate)); },
    array() { return [...items]; },
  };
}
class Element {
  constructor(tag = 'div', options = {}) { this.tag = tag; this.cls = options.cls || ''; this.children = []; this.attributes = {}; this.textContent = ''; this.hidden = false; this.classList = { toggle() {} }; }
  createEl(tag, options = {}) { const child = new Element(tag, options); this.children.push(child); return child; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener() {}
}
function collectText(node) { return [node.textContent, ...node.children.map(collectText)].join(' '); }

test('analytics custom views exist and compile as AsyncFunction', () => {
  for (const key of ['kpi', 'queue', 'bank']) {
    assert.equal(existsSync(paths[key]), true, `${key} view is missing`);
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    assert.doesNotThrow(() => new AsyncFunction('dv', 'input', 'window', 'Notice', source(key)));
  }
});

test('KPI uses the canonical problem query and precise metrics', async () => {
  const text = source('kpi');
  assert.match(text, /dv\.pages\('\"11_数据结构\/刷题记录\"'\)\.where\(p\s*=>\s*p\.题号\s*!==\s*undefined\s*&&\s*p\.题号\s*!==\s*null\)\.array\(\)/);
  for (const cls of ['weighted', 'ac', 'paper', 'exam']) assert.match(text, new RegExp(`ds408-kpi-${cls}`));
  const pages = [
    { 题号: 1, 状态: '未开始', 来源: 'LeetCode' },
    { 题号: 2, 状态: '已AC', 来源: 'LeetCode' },
    { 题号: 3, 状态: '可纸笔', 来源: '408真题' },
    { 题号: 4, 状态: '已AC', 来源: '408真题' },
  ];
  const roots = [];
  const dv = { pages: () => flatPages(pages), el(tag, value, options) { const e = new Element(tag, options); roots.push(e); return e; } };
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  await new AsyncFunction('dv', 'input', 'window', 'Notice', source('kpi'))(dv, {}, {}, class {});
  const output = roots.map(collectText).join(' ');
  assert.match(output, /50\.0%/); // (0 + .5 + 1 + .5) / 4
  assert.match(output, /75\.0%/);
});

test('queue and bank retain their safety and interaction contract', () => {
  const queue = source('queue');
  assert.match(queue, /ds408-review-queue/);
  assert.match(queue, /证据待补/);
  assert.match(queue, /<\s*today/);
  assert.match(queue, /<=\s*sevenDays/);
  assert.match(queue, /new Set/);
  const bank = source('bank');
  assert.match(bank, /await dv\.io\.load\("11_数据结构\/系统\/共享\/state-engine\.js"\)/);
  assert.equal((bank.match(/dv\.io\.load/g) || []).length, 1);
  assert.equal((bank.match(/processFrontMatter\(/g) || []).length >= 1, true);
  assert.equal(/innerHTML/.test(bank), false);
  for (const filter of ['mother', 'status', 'priority', 'source', 'evidence']) assert.equal(bank.includes(`"${filter}"`), true, filter);
  assert.match(bank, /window\.confirm\(/);
  assert.match(bank, /engine\.buildAction/);
});

test('analytics CSS is locally scoped, token-only, and touch friendly', () => {
  for (const [key, cssRoot] of [['kpi', 'ds408-kpi'], ['queue', 'ds408-review-queue'], ['bank', 'ds408-bank']]) {
    const cssPath = paths[key].replace(/view\.js$/, 'view.css');
    assert.equal(existsSync(cssPath), true);
    const css = readFileSync(cssPath, 'utf8');
    assert.equal(/#[0-9a-f]{3,8}\b|\brgba?\s*\(/i.test(css), false);
    assert.match(css, new RegExp(`\\.${cssRoot}[^{}]*:focus-visible`));
    assert.match(css, /44px/);
    for (const selector of css.matchAll(/(^|\n)\s*(\.[^{\n]+)\{/g)) assert.match(selector[2].trim(), new RegExp(`^\\.${cssRoot}`));
  }
});

test('dashboard and supporting query pages use the prescribed thin Dataview composition', () => {
  const dashboard = source('dashboard');
  for (const view of ['统计卡', '复习队列', '题库表']) assert.equal(dashboard.includes(`dv.view("11_数据结构/系统/${view}")`), true);
  assert.match(dashboard, /TABLE 母题模块, 模板状态, 复写得分, 下次复写/);
  assert.match(source('sprint'), /来源\s*=\s*"408真题"/);
  assert.match(source('retry'), /错误类型\s*!=\s*"无"\s+OR\s+证据状态\s*=\s*"待校准"/);
  assert.match(source('calibration'), /回退阶段不会清除/);
  assert.match(source('calibration'), /证据状态\s*=\s*"待校准"/);
});
