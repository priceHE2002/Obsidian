import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const paths = Object.fromEntries(['kpi', 'queue', 'bank'].map((key) => [key, `${root}11_数据结构/系统/${{ kpi: '统计卡', queue: '复习队列', bank: '题库表' }[key]}/view.js`]));
const engineSource = readFileSync(`${root}11_数据结构/系统/共享/state-engine.js`, 'utf8');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const source = (key) => readFileSync(paths[key], 'utf8');

class Element {
  constructor(tag = 'div', options = {}) {
    this.tag = tag; this.children = []; this.parentElement = null; this.attributes = {}; this.handlers = new Map();
    this.textContent = ''; this.value = ''; this.type = ''; this.disabled = false; this.hidden = false; this.removed = false; this.className = options.cls || '';
    this.classList = { toggle: (name, force) => { const names = new Set(this.className.split(/\s+/).filter(Boolean)); force ? names.add(name) : names.delete(name); this.className = [...names].join(' '); } };
  }
  createEl(tag, options = {}) { const child = new Element(tag, options); this.appendChild(child); return child; }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  replaceChildren(...children) { this.children.forEach((child) => { child.parentElement = null; }); this.children = []; children.forEach((child) => this.appendChild(child)); }
  insertAdjacentElement(_position, child) { const parent = this.parentElement; const index = parent.children.indexOf(this); child.parentElement = parent; parent.children.splice(index + 1, 0, child); return child; }
  remove() { if (!this.parentElement) return; const index = this.parentElement.children.indexOf(this); if (index >= 0) this.parentElement.children.splice(index, 1); this.parentElement = null; this.removed = true; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(type, handler) { const list = this.handlers.get(type) || []; list.push(handler); this.handlers.set(type, list); }
  dispatch(type) { if (type === 'click' && this.disabled) return undefined; return (this.handlers.get(type) || []).map((handler) => handler({ target: this, preventDefault() {} })); }
  click() { return this.dispatch('click'); }
  change() { return this.dispatch('change'); }
  input() { return this.dispatch('input'); }
  submit() { return this.dispatch('submit'); }
  focus() { globalThis.document.activeElement = this; }
  get isConnected() { return !this.removed; }
  get valueAsNumber() { return Number(this.value); }
  querySelectorAll(selector) { return all(this).filter((node) => matches(node, selector)); }
}
function all(node) { return node.children.flatMap((child) => [child, ...all(child)]); }
function matches(node, selector) {
  const part = selector.trim(); const tag = part.match(/^[a-z]+/i)?.[0]; const cls = part.match(/\.([\w-]+)/)?.[1]; const attr = part.match(/\[([^=\]]+)(?:=["']?([^\]"']+)["']?)?\]/);
  return (!tag || node.tag === tag) && (!cls || node.className.split(/\s+/).includes(cls)) && (!attr || (attr[2] === undefined ? Object.hasOwn(node.attributes, attr[1]) : node.getAttribute(attr[1]) === attr[2]));
}
function find(rootNode, selector) { const node = rootNode.querySelectorAll(selector)[0]; assert.ok(node, `missing ${selector}`); return node; }
function textOf(node) { return [node.textContent, ...node.children.map(textOf)].join(' '); }
function flatPages(items) { return { where(predicate) { return flatPages(items.filter(predicate)); }, array() { return [...items]; } }; }
function moment(today = '2026-08-19') {
  return (value) => ({ format: () => value || today, add: (days) => ({ format: () => { const date = new Date(`${value || today}T00:00:00Z`); date.setUTCDate(date.getUTCDate() + days); return date.toISOString().slice(0, 10); } }) });
}
async function runView(key, { problems = [], patterns = [], today = '2026-08-19', load = engineSource, processFrontMatter, confirm = () => true } = {}) {
  const roots = []; const document = { createElement: (tag) => new Element(tag) }; const files = new Map(problems.map((p) => [p.file.path, { ...p }]));
  const dv = {
    pages(path) { return flatPages(path.includes('母题模板') ? patterns : problems); },
    date() { return { toISODate: () => today }; },
    el(tag, _value, options) { const element = new Element(tag, options); roots.push(element); return element; },
    io: { load: async () => load },
    app: { vault: { getAbstractFileByPath: (path) => files.has(path) ? { path } : null }, fileManager: { processFrontMatter: processFrontMatter || (async (file, callback) => callback(files.get(file.path))) } },
  };
  globalThis.document = document;
  await new AsyncFunction('dv', 'input', 'window', 'Notice', source(key))(dv, {}, { moment: moment(today), confirm }, class {});
  return { root: roots[0], files, dv, document };
}
function problem(overrides = {}) { const id = overrides.题号 ?? 1; return { 题号: id, 母题: 'M01', 标题: `题${id}`, 优先级: 'S', 来源: 'LeetCode', 手写: '必须', 状态: '学习中', 证据状态: '有效', 复习次数: 0, file: { path: `11_数据结构/刷题记录/${id}.md` }, ...overrides }; }
function flush() { return new Promise((resolve) => setTimeout(resolve, 0)); }

test('views compile and KPI executes the specified formula on known data', async () => {
  for (const key of Object.keys(paths)) { assert.equal(existsSync(paths[key]), true); assert.doesNotThrow(() => new AsyncFunction('dv', 'input', 'window', 'Notice', source(key))); }
  const { root: kpi } = await runView('kpi', { problems: [problem({ 题号: 1, 状态: '未开始' }), problem({ 题号: 2, 状态: '已AC' }), problem({ 题号: 3, 状态: '可纸笔', 来源: '408真题' }), problem({ 题号: 4, 状态: '已AC', 来源: '408真题' })] });
  assert.match(textOf(kpi), /50\.0%/); assert.match(textOf(kpi), /75\.0%/);
});

test('queue treats impossible calendar dates as invalid for alerts and stable ranking', async () => {
  const { root: queue } = await runView('queue', { problems: [problem({ 题号: 1, 下次复习: '2026-02-31', 状态: '未开始' }), problem({ 题号: 2, 下次复习: '2026-08-18', 状态: '未开始' }), problem({ 题号: 3, 下次复习: '2026-08-20', 状态: '未开始' })] });
  const alerts = find(queue, '.ds408-review-queue__alerts').children;
  assert.equal(alerts[0].children[0].textContent, '1');
  assert.equal(alerts[1].children[0].textContent, '1');
  assert.match(textOf(find(queue, '.ds408-review-queue__list').children[0]), /2 · 题2/);
});

test('bank filters are AND-composed and reset through dispatched change/click events', async () => {
  const items = [problem({ 题号: 1 }), problem({ 题号: 2, 优先级: 'A' }), problem({ 题号: 3, 母题: 'M02' })];
  const { root: bank } = await runView('bank', { problems: items });
  for (const [filter, value] of [['mother', 'M01'], ['status', '学习中'], ['priority', 'S'], ['source', 'LeetCode'], ['evidence', '有效']]) { const select = find(bank, `select[data-filter=${filter}]`); select.value = value; select.change(); }
  assert.match(textOf(find(bank, '.ds408-bank__summary')), /显示 1 \/ 3 题/);
  find(bank, '.ds408-bank__clear').click();
  assert.match(textOf(find(bank, '.ds408-bank__summary')), /显示 3 \/ 3 题/);
});

test('bank puts aria-sort on six column headers and updates sort direction', async () => {
  const { root: bank } = await runView('bank', { problems: [problem({ 题号: 2, 下次复习: '2026-02-31' }), problem({ 题号: 1, 下次复习: '2026-08-18' })] });
  let headers = bank.querySelectorAll('th'); assert.equal(headers.filter((node) => node.getAttribute('aria-sort') !== null).length, 6);
  assert.equal(headers.find((node) => node.children.some((child) => child.textContent === '题号')).getAttribute('aria-sort'), 'ascending');
  find(bank, '.ds408-bank__sort').click(); headers = bank.querySelectorAll('th'); assert.equal(headers.find((node) => node.children.some((child) => child.textContent === '题号')).getAttribute('aria-sort'), 'descending');
  const reviewButton = bank.querySelectorAll('.ds408-bank__sort').find((node) => node.textContent === '下次复习'); reviewButton.click();
  const body = find(bank, 'tbody'); assert.equal(textOf(body.children[0].children[0]).trim(), '1');
});

test('bank keeps missing values last for every sortable key in both directions', async () => {
  const valid = problem({ 题号: 1, 母题: 'M01', 优先级: 'S', 状态: '学习中', 下次复习: '2026-08-18', 纸笔得分: 8 });
  const missing = problem({ 题号: 'NaN', 母题: '', 优先级: '未知', 状态: '未知', 下次复习: '2026-02-31', 纸笔得分: 'NaN', file: { path: '11_数据结构/刷题记录/missing.md' } });
  const { root: bank } = await runView('bank', { problems: [valid, missing] });
  for (const label of ['题号', '母题', '优先级', '状态', '下次复习', '纸笔分']) {
    for (let i = 0; i < 2; i += 1) {
      bank.querySelectorAll('.ds408-bank__sort').find((node) => node.textContent === label).click();
      const body = find(bank, 'tbody'); assert.match(textOf(body.children.at(-1).children[0]), /NaN/);
    }
  }
});

test('bank status interaction opens editor without writing and saves numeric evidence once', async () => {
  let writes = 0; const records = [problem()];
  const { root: bank } = await runView('bank', { problems: records, processFrontMatter: async (file, callback) => { writes += 1; callback(records.find((item) => item.file.path === file.path)); } });
  const ac = bank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC'); ac.click(); assert.equal(writes, 0);
  const editor = find(bank, '.ds408-bank__editor'); const fields = editor.querySelectorAll('select, input');
  find(editor, 'select').value = '独立通过'; find(editor, 'input').value = '12'; find(editor, '.ds408-bank__save').click(); await flush();
  assert.equal(writes, 1); assert.equal(typeof records[0].首刷耗时, 'number'); assert.equal(globalThis.document.activeElement, bank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC'));
});

test('bank keeps one labelled editor, focuses it, and restores the triggering button on cancel', async () => {
  const { root: bank } = await runView('bank', { problems: [problem()] });
  const ac = bank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC'); ac.click();
  let editor = find(bank, '.ds408-bank__editor'); const firstId = editor.getAttribute('id'); assert.equal(ac.getAttribute('aria-expanded'), 'true'); assert.equal(ac.getAttribute('aria-controls'), firstId); assert.equal(globalThis.document.activeElement, find(editor, 'select'));
  const twice = bank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已二刷'); twice.click(); assert.equal(bank.querySelectorAll('.ds408-bank__editor').length, 1); editor = find(bank, '.ds408-bank__editor'); assert.notEqual(editor.getAttribute('id'), firstId);
  find(editor, '.ds408-bank__cancel').click(); assert.equal(bank.querySelectorAll('.ds408-bank__editor').length, 0); assert.equal(globalThis.document.activeElement, twice); assert.equal(twice.getAttribute('aria-expanded'), 'false');
});

test('bank permits same pending state editing, disables a valid current state, and protects frontmatter on engine throw', async () => {
  const pending = problem({ 状态: '已AC', 证据状态: '待校准' }); const { root: pendingBank } = await runView('bank', { problems: [pending] });
  const pendingCurrent = pendingBank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC'); assert.equal(pendingCurrent.disabled, false); pendingCurrent.click(); assert.ok(pendingBank.querySelectorAll('.ds408-bank__editor').length);
  const valid = problem({ 状态: '已AC', 证据状态: '有效' }); const { root: validBank } = await runView('bank', { problems: [valid] }); assert.equal(validBank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC').disabled, true);
  const fm = { ...problem() }; const before = structuredClone(fm); const throwing = 'var api={STATUSES:["未开始","学习中","已AC","已二刷","可纸笔"],buildAction:function(){throw new Error("boom")}};';
  const { root: throwBank } = await runView('bank', { problems: [fm], load: throwing, processFrontMatter: async (_file, callback) => callback(fm) }); throwBank.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC').click(); find(throwBank, '.ds408-bank__editor').querySelectorAll('.ds408-bank__save')[0].click(); await flush(); assert.deepEqual(fm, before);
});

test('bank backward respects confirmation and save locking prevents duplicate confirm writes', async () => {
  const completed = problem({ 状态: '已AC', 证据状态: '有效', 首刷日期: '2026-08-01', 首刷结果: '独立通过' }); let backwardWrites = 0; let allow = false;
  const one = await runView('bank', { problems: [completed], confirm: () => allow, processFrontMatter: async (file, callback) => { backwardWrites += 1; callback(completed); } });
  const back = one.root.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '未开始'); back.click(); assert.equal(backwardWrites, 0); allow = true; back.click(); await flush(); assert.equal(backwardWrites, 1);
  const locked = problem(); let writes = 0; let release; const gate = new Promise((resolve) => { release = resolve; });
  const two = await runView('bank', { problems: [locked], processFrontMatter: async (file, callback) => { writes += 1; callback(locked); await gate; } });
  const ac = two.root.querySelectorAll('.ds408-bank__status').find((node) => node.textContent === '已AC'); ac.click(); find(two.root, 'select').value = '独立通过'; const save = find(two.root, '.ds408-bank__save'); save.click(); save.click(); await flush(); assert.equal(writes, 1); release(); await flush();
});

function topLevelCommaSplit(value) { const parts = []; let start = 0; let depth = 0; for (let i = 0; i < value.length; i += 1) { if (value[i] === '(') depth += 1; if (value[i] === ')') depth -= 1; if (value[i] === ',' && depth === 0) { parts.push(value.slice(start, i)); start = i + 1; } } parts.push(value.slice(start)); return parts; }
function cssSelectors(css) { const found = []; function walk(part) { let cursor = 0; while (cursor < part.length) { const open = part.indexOf('{', cursor); if (open < 0) return; const head = part.slice(cursor, open).trim(); let depth = 1; let close = open + 1; while (close < part.length && depth) { if (part[close] === '{') depth += 1; if (part[close] === '}') depth -= 1; close += 1; } const body = part.slice(open + 1, close - 1); if (head.startsWith('@')) walk(body); else if (head) found.push(...topLevelCommaSplit(head).map((item) => item.trim()).filter(Boolean)); cursor = close; } } walk(css.replace(/\/\*[\s\S]*?\*\//g, '')); return found; }
test('views retain fully scoped token-only CSS and exact dashboard/query Dataview structure', () => {
  for (const [folder, cssRoot] of [['统计卡', 'ds408-kpi'], ['复习队列', 'ds408-review-queue'], ['题库表', 'ds408-bank']]) {
    const css = readFileSync(`${root}11_数据结构/系统/${folder}/view.css`, 'utf8');
    assert.equal(/#[0-9a-f]{3,8}\b|\brgba?\s*\(/i.test(css), false); assert.ok(cssSelectors(css).every((selector) => selector.startsWith(`.${cssRoot}`)));
  }
  const dashboard = readFileSync(`${root}11_数据结构/总览.md`, 'utf8'); for (const view of ['统计卡', '复习队列', '题库表']) assert.equal((dashboard.match(new RegExp(`await dv\\.view\\("11_数据结构/系统/${view}"`, 'g')) || []).length, 1);
  const sprint = readFileSync(`${root}11_数据结构/真题/真题冲刺.md`, 'utf8'); const retry = readFileSync(`${root}11_数据结构/错题回炉.md`, 'utf8'); const calibration = readFileSync(`${root}11_数据结构/训练数据校准.md`, 'utf8');
  for (const page of [dashboard, sprint, retry, calibration]) assert.ok([...page.matchAll(/```dataview\n([\s\S]*?)```/g)].every((block) => /FROM\s+"11_数据结构\//.test(block[1])));
  assert.match(sprint, /WHERE\s+来源\s*=\s*"408真题"/); assert.match(retry, /WHERE\s+错误类型\s*!=\s*"无"\s+OR\s+证据状态\s*=\s*"待校准"/); assert.match(calibration, /WHERE\s+证据状态\s*=\s*"待校准"/);
});
