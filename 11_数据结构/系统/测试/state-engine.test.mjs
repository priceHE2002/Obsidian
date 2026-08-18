import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const modulePath = new URL('../共享/state-engine.js', import.meta.url);
const sourcePath = new URL('../共享/state-engine.js', import.meta.url);
const { STATUSES, buildAction, paperTotal, evidenceSatisfied } = require(fileURLToPath(modulePath));
const TODAY = '2026-08-18';

function current(overrides = {}) {
  return {
    状态: '学习中',
    复习次数: 0,
    证据状态: '待校准',
    ...overrides,
  };
}

test('CommonJS 与 Dataview Function 加载均公开相同 API', () => {
  const source = readFileSync(sourcePath, 'utf8');
  const dataviewApi = Function(`${source}\nreturn api;`)();
  assert.deepEqual(Object.keys(dataviewApi).sort(), ['STATUSES', 'buildAction', 'evidenceSatisfied', 'paperTotal']);
  assert.deepEqual(dataviewApi.STATUSES, STATUSES);
  assert.equal(typeof dataviewApi.buildAction, 'function');
});

test('transition：独立通过首刷设置三天后的复习日期', () => {
  const patch = buildAction(current(), {
    kind: 'transition', target: '已AC', evidence: { 首刷结果: '独立通过' },
  }, TODAY);
  assert.deepEqual(patch, {
    首刷结果: '独立通过', 最后训练: TODAY, 复习次数: 1, 状态: '已AC',
    首刷日期: TODAY, 下次复习: '2026-08-21', 证据状态: '有效',
  });
});

test('transition：订正通过首刷设置一天后的复习日期', () => {
  const patch = buildAction(current(), {
    kind: 'transition', target: '已AC', evidence: { 首刷结果: '订正通过' },
  }, TODAY);
  assert.equal(patch.下次复习, '2026-08-19');
  assert.equal(patch.状态, '已AC');
});

test('transition：二刷独立通过设置七天后复习且不重写旧首刷日期', () => {
  const patch = buildAction(current({
    状态: '已AC', 首刷日期: '2026-08-01', 首刷结果: '独立通过',
  }), {
    kind: 'transition', target: '已二刷', evidence: { 二刷结果: '独立通过' },
  }, TODAY);
  assert.equal(patch.二刷日期, TODAY);
  assert.equal(patch.下次复习, '2026-08-25');
  assert.equal(Object.hasOwn(patch, '首刷日期'), false);
});

test('transition：纸笔总分不足八分拒绝', () => {
  assert.throws(() => buildAction(current({ 状态: '已二刷' }), {
    kind: 'transition', target: '可纸笔',
    evidence: { 思想得分: 2, 正确性得分: 2, 代码得分: 3, 复杂度得分: 0 },
  }, TODAY), /纸笔得分/);
});

test('transition：纸笔四项得分派生总分并设置十四天后复习', () => {
  const patch = buildAction(current({ 状态: '已二刷' }), {
    kind: 'transition', target: '可纸笔',
    evidence: { 思想得分: 2, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 },
  }, TODAY);
  assert.equal(patch.纸笔得分, 8);
  assert.equal(patch.纸笔日期, TODAY);
  assert.equal(patch.下次复习, '2026-09-01');
});

test('transition：回退至学习中不清空历史证据', () => {
  const patch = buildAction(current({
    状态: '已二刷', 首刷日期: '2026-08-01', 首刷结果: '独立通过', 二刷日期: '2026-08-10', 二刷结果: '独立通过',
  }), { kind: 'transition', target: '学习中' }, TODAY);
  assert.equal(patch.状态, '学习中');
  assert.equal(Object.hasOwn(patch, '首刷日期'), false);
  assert.equal(Object.hasOwn(patch, '二刷日期'), false);
  assert.equal(Object.hasOwn(patch, '首刷结果'), false);
  assert.equal(Object.hasOwn(patch, '二刷结果'), false);
});

test('failure：首刷失败不推进状态且次日复习', () => {
  const patch = buildAction(current(), { kind: 'failure', stage: '首刷' }, TODAY);
  assert.deepEqual(patch, { 首刷结果: '失败', 最后训练: TODAY, 复习次数: 1, 下次复习: '2026-08-19' });
});

test('failure：二刷失败记录对应结果而不改变状态', () => {
  const patch = buildAction(current({ 状态: '已AC' }), { kind: 'failure', stage: '二刷' }, TODAY);
  assert.equal(patch.二刷结果, '失败');
  assert.equal(Object.hasOwn(patch, '状态'), false);
  assert.equal(patch.下次复习, '2026-08-19');
});

test('retention：可纸笔通过延后30天，失败延后1天', () => {
  const accepted = buildAction(current({ 状态: '可纸笔' }), { kind: 'retention', passed: true }, TODAY);
  const rejected = buildAction(current({ 状态: '可纸笔' }), { kind: 'retention', passed: false }, TODAY);
  assert.equal(accepted.下次复习, '2026-09-17');
  assert.equal(rejected.下次复习, '2026-08-19');
  assert.equal(Object.hasOwn(accepted, '状态'), false);
});

test('retention：非可纸笔状态拒绝', () => {
  assert.throws(() => buildAction(current(), { kind: 'retention', passed: true }, TODAY), /可纸笔/);
});

test('同阶段待校准已AC可补证据并转为有效', () => {
  const patch = buildAction(current({ 状态: '已AC' }), {
    kind: 'transition', target: '已AC', evidence: { 首刷结果: '独立通过' },
  }, TODAY);
  assert.equal(patch.首刷日期, TODAY);
  assert.equal(patch.证据状态, '有效');
});

test('输入 current、action 与 evidence 均不被修改', () => {
  const item = current();
  const evidence = { 首刷结果: '独立通过' };
  const action = { kind: 'transition', target: '已AC', evidence };
  const before = JSON.stringify({ item, action, evidence });
  buildAction(item, action, TODAY);
  assert.equal(JSON.stringify({ item, action, evidence }), before);
});

test('paperTotal 拒绝缺失、空白、NaN、Infinity、小数及越界评分', () => {
  const invalid = [
    {},
    { 思想得分: ' ', 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 },
    { 思想得分: NaN, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 },
    { 思想得分: Infinity, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 },
    { 思想得分: 1.5, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 },
    { 思想得分: 3, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 },
    { 思想得分: 2, 正确性得分: 2, 代码得分: 5, 复杂度得分: 1 },
  ];
  for (const scores of invalid) assert.throws(() => paperTotal(scores), /评分/);
  assert.equal(paperTotal({ 思想得分: 2, 正确性得分: 2, 代码得分: 3, 复杂度得分: 1 }), 8);
});

test('evidenceSatisfied 按阶段日期、结果和纸笔总分判定', () => {
  assert.equal(evidenceSatisfied({}, '未开始'), true);
  assert.equal(evidenceSatisfied({}, '学习中'), true);
  assert.equal(evidenceSatisfied({ 首刷日期: TODAY, 首刷结果: '独立通过' }, '已AC'), true);
  assert.equal(evidenceSatisfied({ 首刷日期: TODAY, 首刷结果: '失败' }, '已AC'), false);
  assert.equal(evidenceSatisfied({ 二刷日期: TODAY, 二刷结果: '独立通过' }, '已二刷'), true);
  assert.equal(evidenceSatisfied({ 纸笔日期: TODAY, 纸笔得分: 8 }, '可纸笔'), true);
  assert.equal(evidenceSatisfied({ 纸笔日期: TODAY, 纸笔得分: 7 }, '可纸笔'), false);
  assert.equal(evidenceSatisfied({}, '不存在'), false);
});

test('非法状态、动作、stage、today 与复习次数均拒绝', () => {
  assert.throws(() => buildAction(current({ 状态: '错误' }), { kind: 'transition', target: '学习中' }, TODAY), /状态/);
  assert.throws(() => buildAction(current(), { kind: 'transition', target: '错误' }, TODAY), /状态/);
  assert.throws(() => buildAction(current(), { kind: 'unknown' }, TODAY), /动作/);
  assert.throws(() => buildAction(current(), { kind: 'failure', stage: '三刷' }, TODAY), /阶段/);
  assert.throws(() => buildAction(current(), { kind: 'failure', stage: '首刷' }, '2026-02-29'), /日期/);
  assert.throws(() => buildAction(current({ 复习次数: -1 }), { kind: 'failure', stage: '首刷' }, TODAY), /复习次数/);
  assert.throws(() => buildAction(current({ 复习次数: '0' }), { kind: 'failure', stage: '首刷' }, TODAY), /复习次数/);
  assert.throws(() => buildAction(null, { kind: 'failure', stage: '首刷' }, TODAY), /current/);
  assert.throws(() => buildAction(current(), null, TODAY), /action/);
});

test('evidence 注入未知、状态及派生字段均拒绝', () => {
  for (const evidence of [{ 状态: '可纸笔' }, { 复习次数: 99 }, { 纸笔日期: TODAY }, { 未知字段: 'x' }]) {
    assert.throws(() => buildAction(current(), { kind: 'failure', stage: '首刷', evidence }, TODAY), /证据/);
  }
});

test('日期加法以 UTC 处理月末与闰年', () => {
  const monthEnd = buildAction(current(), { kind: 'failure', stage: '首刷' }, '2026-01-31');
  const leapYear = buildAction(current({ 状态: '可纸笔' }), { kind: 'retention', passed: true }, '2024-02-01');
  assert.equal(monthEnd.下次复习, '2026-02-01');
  assert.equal(leapYear.下次复习, '2024-03-02');
});
