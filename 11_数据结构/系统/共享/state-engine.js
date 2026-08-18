/* 训练状态的纯函数 patch 构造器：可由 CommonJS 与 Dataview Function 直接加载。 */
var api = (function () {
  'use strict';

  var STATUSES = Object.freeze(['未开始', '学习中', '已AC', '已二刷', '可纸笔']);
  var EVIDENCE_FIELDS = Object.freeze([
    '首刷结果', '首刷耗时', '二刷结果',
    '思想得分', '正确性得分', '代码得分', '复杂度得分',
  ]);
  var SCORE_RULES = Object.freeze({
    思想得分: [0, 2],
    正确性得分: [0, 2],
    代码得分: [0, 4],
    复杂度得分: [0, 2],
  });

  function fail(message) {
    throw new Error(message);
  }

  function isPlainRecord(value) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
    var prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
  }

  function hasOwn(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
  }

  function ownValue(object, key) {
    return hasOwn(object, key) ? object[key] : undefined;
  }

  function copyOwnRecord(record) {
    var copy = Object.create(null);
    Object.keys(record).forEach(function (key) {
      copy[key] = record[key];
    });
    return copy;
  }

  function statusIndex(status) {
    return STATUSES.indexOf(status);
  }

  function validDate(dateText) {
    if (typeof dateText !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(dateText)) {
      fail('日期必须是严格的 YYYY-MM-DD 格式');
    }
    var date = new Date(dateText + 'T00:00:00.000Z');
    if (Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== dateText) {
      fail('日期必须是真实的日历日期');
    }
    return date;
  }

  function addDays(dateText, amount) {
    var date = validDate(dateText);
    date.setUTCDate(date.getUTCDate() + amount);
    return date.toISOString().slice(0, 10);
  }

  function nonEmpty(value) {
    return value !== undefined && value !== null && !(typeof value === 'string' && value.trim() === '');
  }

  function paperTotal(evidence) {
    if (!isPlainRecord(evidence)) fail('纸笔评分必须为对象');

    var total = 0;
    Object.keys(SCORE_RULES).forEach(function (field) {
      var range = SCORE_RULES[field];
      var value = ownValue(evidence, field);
      if (!nonEmpty(value)) fail(field + '评分不能为空');
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        fail(field + '评分必须是有限数值');
      }
      if (!Number.isInteger(value)) fail(field + '评分必须是整数');
      if (value < range[0] || value > range[1]) {
        fail(field + '评分必须在 ' + range[0] + ' 到 ' + range[1] + ' 之间');
      }
      total += value;
    });
    return total;
  }

  function evidenceSatisfied(evidence, status) {
    if (status === '未开始' || status === '学习中') return true;
    if (!isPlainRecord(evidence)) return false;
    if (status === '已AC') {
      return nonEmpty(ownValue(evidence, '首刷日期'))
        && (ownValue(evidence, '首刷结果') === '独立通过' || ownValue(evidence, '首刷结果') === '订正通过');
    }
    if (status === '已二刷') {
      return nonEmpty(ownValue(evidence, '二刷日期')) && ownValue(evidence, '二刷结果') === '独立通过';
    }
    if (status === '可纸笔') {
      return nonEmpty(ownValue(evidence, '纸笔日期'))
        && typeof ownValue(evidence, '纸笔得分') === 'number'
        && Number.isFinite(ownValue(evidence, '纸笔得分'))
        && ownValue(evidence, '纸笔得分') >= 8;
    }
    return false;
  }

  function validateCurrent(current) {
    if (!isPlainRecord(current)) fail('current 必须是对象');
    var status = ownValue(current, '状态');
    var reviewCount = ownValue(current, '复习次数');
    if (statusIndex(status) < 0) fail('current 的状态非法');
    if (typeof reviewCount !== 'number' || !Number.isSafeInteger(reviewCount)
      || reviewCount < 0 || reviewCount >= Number.MAX_SAFE_INTEGER) {
      fail('current 的复习次数必须是非负整数');
    }
    return { status: status, reviewCount: reviewCount, evidenceStatus: ownValue(current, '证据状态') };
  }

  function validateEvidenceValue(field, value) {
    if (typeof value === 'object' || typeof value === 'function') fail('证据字段必须是标量：' + field);
    if (field === '首刷结果') {
      if (['未完成', '独立通过', '订正通过', '失败'].indexOf(value) < 0) fail('证据字段首刷结果取值非法');
      return;
    }
    if (field === '二刷结果') {
      if (['未完成', '独立通过', '失败'].indexOf(value) < 0) fail('证据字段二刷结果取值非法');
      return;
    }
    if (field === '首刷耗时') {
      if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value < 0) {
        fail('证据字段首刷耗时必须是非负整数');
      }
      return;
    }
    var range = SCORE_RULES[field];
    if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value)
      || value < range[0] || value > range[1]) {
      fail('证据字段' + field + '取值非法');
    }
  }

  function applyEvidence(draft, patch, evidence) {
    if (evidence === undefined) return;
    if (!isPlainRecord(evidence)) fail('证据必须是对象');
    Object.keys(evidence).forEach(function (field) {
      if (EVIDENCE_FIELDS.indexOf(field) < 0) fail('证据字段不被允许：' + field);
      var value = ownValue(evidence, field);
      if (nonEmpty(value)) {
        validateEvidenceValue(field, value);
        draft[field] = value;
        patch[field] = value;
      }
    });
  }

  function setCommonTrainingPatch(patch, reviewCount, today) {
    patch.最后训练 = today;
    patch.复习次数 = reviewCount + 1;
  }

  function validateAction(action) {
    if (!isPlainRecord(action)) fail('action 必须是对象');
    var kind = ownValue(action, 'kind');
    if (typeof kind !== 'string') fail('动作类型非法');
    if (kind === 'transition') {
      var target = ownValue(action, 'target');
      if (statusIndex(target) < 0) fail('目标状态非法');
      return { kind: kind, target: target };
    }
    if (kind === 'failure') {
      var stage = ownValue(action, 'stage');
      if (stage !== '首刷' && stage !== '二刷') fail('失败阶段必须是首刷或二刷');
      return { kind: kind, stage: stage };
    }
    if (kind === 'retention') {
      var passed = ownValue(action, 'passed');
      if (!hasOwn(action, 'passed') || typeof passed !== 'boolean') fail('retention 的 passed 必须是布尔值');
      return { kind: kind, passed: passed };
    }
    fail('动作类型非法');
  }

  function setTransitionEvidence(patch, draft, target, today) {
    if (target === '学习中') {
      if (!nonEmpty(draft.首刷日期)) {
        draft.首刷日期 = today;
        patch.首刷日期 = today;
      }
      return;
    }
    if (target === '已AC') {
      if (draft.首刷结果 !== '独立通过' && draft.首刷结果 !== '订正通过') {
        fail('进入已AC时首刷结果必须为独立通过或订正通过');
      }
      if (!nonEmpty(draft.首刷日期)) {
        draft.首刷日期 = today;
        patch.首刷日期 = today;
      }
      patch.下次复习 = addDays(today, draft.首刷结果 === '独立通过' ? 3 : 1);
      return;
    }
    if (target === '已二刷') {
      if (draft.二刷结果 !== '独立通过') fail('进入已二刷时二刷结果必须为独立通过');
      if (!nonEmpty(draft.二刷日期)) {
        draft.二刷日期 = today;
        patch.二刷日期 = today;
      }
      patch.下次复习 = addDays(today, 7);
      return;
    }
    if (target === '可纸笔') {
      var total = paperTotal(draft);
      if (total < 8) fail('纸笔得分必须不低于 8 分');
      draft.纸笔得分 = total;
      patch.纸笔得分 = total;
      if (!nonEmpty(draft.纸笔日期)) {
        draft.纸笔日期 = today;
        patch.纸笔日期 = today;
      }
      patch.下次复习 = addDays(today, 14);
    }
  }

  function buildAction(current, action, today) {
    var currentInfo = validateCurrent(current);
    validDate(today);
    var actionInfo = validateAction(action);

    var draft = copyOwnRecord(current);
    var patch = {};
    applyEvidence(draft, patch, ownValue(action, 'evidence'));

    if (actionInfo.kind === 'transition') {
      var currentIndex = statusIndex(currentInfo.status);
      var targetIndex = statusIndex(actionInfo.target);
      var isPending = currentInfo.evidenceStatus === '待校准';
      var needsEvidence = targetIndex > currentIndex || (actionInfo.target === currentInfo.status && isPending);
      var canClearPending = isPending && targetIndex >= currentIndex;

      setCommonTrainingPatch(patch, currentInfo.reviewCount, today);
      patch.状态 = actionInfo.target;
      if (needsEvidence) setTransitionEvidence(patch, draft, actionInfo.target, today);
      if (canClearPending && evidenceSatisfied(draft, actionInfo.target)) {
        patch.证据状态 = '有效';
      }
      return patch;
    }

    if (actionInfo.kind === 'failure') {
      patch[actionInfo.stage === '首刷' ? '首刷结果' : '二刷结果'] = '失败';
      setCommonTrainingPatch(patch, currentInfo.reviewCount, today);
      patch.下次复习 = addDays(today, 1);
      return patch;
    }

    if (actionInfo.kind === 'retention') {
      if (currentInfo.status !== '可纸笔') fail('retention 仅适用于可纸笔状态');
      setCommonTrainingPatch(patch, currentInfo.reviewCount, today);
      patch.下次复习 = addDays(today, actionInfo.passed ? 30 : 1);
      return patch;
    }

    fail('动作类型非法');
  }

  return Object.freeze({
    STATUSES: STATUSES,
    buildAction: buildAction,
    paperTotal: paperTotal,
    evidenceSatisfied: evidenceSatisfied,
  });
}());

if (typeof module === 'object' && module.exports) module.exports = api;
