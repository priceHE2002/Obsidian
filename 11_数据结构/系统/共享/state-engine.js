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

  function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function hasOwn(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
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
    if (!isRecord(evidence)) fail('纸笔评分必须为对象');

    var total = 0;
    Object.keys(SCORE_RULES).forEach(function (field) {
      var range = SCORE_RULES[field];
      var value = evidence[field];
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
    if (!isRecord(evidence)) return false;
    if (status === '已AC') {
      return nonEmpty(evidence.首刷日期)
        && (evidence.首刷结果 === '独立通过' || evidence.首刷结果 === '订正通过');
    }
    if (status === '已二刷') {
      return nonEmpty(evidence.二刷日期) && evidence.二刷结果 === '独立通过';
    }
    if (status === '可纸笔') {
      return nonEmpty(evidence.纸笔日期)
        && typeof evidence.纸笔得分 === 'number'
        && Number.isFinite(evidence.纸笔得分)
        && evidence.纸笔得分 >= 8;
    }
    return false;
  }

  function validateCurrent(current) {
    if (!isRecord(current)) fail('current 必须是对象');
    if (statusIndex(current.状态) < 0) fail('current 的状态非法');
    if (typeof current.复习次数 !== 'number' || !Number.isFinite(current.复习次数) || current.复习次数 < 0) {
      fail('current 的复习次数必须是非负有限数值');
    }
  }

  function applyEvidence(draft, patch, evidence) {
    if (evidence === undefined) return;
    if (!isRecord(evidence)) fail('证据必须是对象');
    Object.keys(evidence).forEach(function (field) {
      if (EVIDENCE_FIELDS.indexOf(field) < 0) fail('证据字段不被允许：' + field);
      if (nonEmpty(evidence[field])) {
        draft[field] = evidence[field];
        patch[field] = evidence[field];
      }
    });
  }

  function setCommonTrainingPatch(patch, current, today) {
    patch.最后训练 = today;
    patch.复习次数 = current.复习次数 + 1;
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
    validateCurrent(current);
    validDate(today);
    if (!isRecord(action)) fail('action 必须是对象');
    if (typeof action.kind !== 'string') fail('动作类型非法');

    var draft = Object.assign({}, current);
    var patch = {};
    applyEvidence(draft, patch, action.evidence);

    if (action.kind === 'transition') {
      if (statusIndex(action.target) < 0) fail('目标状态非法');
      var currentIndex = statusIndex(current.状态);
      var targetIndex = statusIndex(action.target);
      var needsEvidence = targetIndex > currentIndex
        || (action.target === current.状态 && current.证据状态 === '待校准');

      setCommonTrainingPatch(patch, current, today);
      patch.状态 = action.target;
      if (needsEvidence) setTransitionEvidence(patch, draft, action.target, today);
      if (current.证据状态 === '待校准' && evidenceSatisfied(draft, action.target)) {
        patch.证据状态 = '有效';
      }
      return patch;
    }

    if (action.kind === 'failure') {
      if (action.stage !== '首刷' && action.stage !== '二刷') fail('失败阶段必须是首刷或二刷');
      patch[action.stage === '首刷' ? '首刷结果' : '二刷结果'] = '失败';
      setCommonTrainingPatch(patch, current, today);
      patch.下次复习 = addDays(today, 1);
      return patch;
    }

    if (action.kind === 'retention') {
      if (current.状态 !== '可纸笔') fail('retention 仅适用于可纸笔状态');
      setCommonTrainingPatch(patch, current, today);
      patch.下次复习 = addDays(today, action.passed === true ? 30 : 1);
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
