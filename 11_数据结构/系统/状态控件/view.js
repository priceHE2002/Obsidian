const root = dv.el("section", "", {
  cls: `ds408-control${input?.compact ? " is-compact" : ""}`,
});
root.setAttribute("role", "region");
root.setAttribute("aria-label", "408 训练状态控件");

function addText(parent, tag, text, className) {
  const element = parent.createEl(tag, className ? { cls: className } : undefined);
  element.textContent = text;
  return element;
}

function showError(message) {
  const error = addText(root, "p", message, "ds408-control__message is-error");
  error.setAttribute("role", "alert");
}

let engine;
try {
  const engineSource = await dv.io.load("11_数据结构/系统/共享/state-engine.js");
  engine = Function(`${engineSource}\nreturn api;`)();
} catch (error) {
  showError("状态规则加载失败，请稍后重新打开此笔记。");
  return;
}

if (!engine || !Array.isArray(engine.STATUSES) || typeof engine.buildAction !== "function") {
  showError("状态规则加载失败，请稍后重新打开此笔记。");
  return;
}

const page = input?.path ? dv.page(input.path) : dv.current();
if (!page || !page.file || !page.file.path) {
  showError("未能识别当前题目，无法写入训练状态。");
  return;
}

const file = dv.app.vault.getAbstractFileByPath(page.file.path);
if (!file || file.path !== page.file.path || !file.extension) {
  showError("当前题目文件不存在，未执行任何写入。");
  return;
}

const snapshot = { ...page };
const fieldPrefix = `ds408-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const statusRow = root.createEl("div", { cls: "ds408-control__status-row" });
const stateBadge = addText(statusRow, "span", "", "ds408-control__badge");
const warningBadge = addText(statusRow, "span", "证据待补", "ds408-control__badge is-warning");
warningBadge.hidden = true;

const statusGroup = root.createEl("div", { cls: "ds408-control__status-group" });
statusGroup.setAttribute("role", "group");
statusGroup.setAttribute("aria-label", "训练状态");
const statusButtons = new Map();

const form = root.createEl("form", { cls: "ds408-control__form" });
form.addEventListener("submit", (event) => event.preventDefault());
const fields = new Map();

function makeField(labelText, field, config) {
  const item = form.createEl("div", { cls: "ds408-control__field" });
  const id = `${fieldPrefix}-${field}`;
  const label = addText(item, "label", labelText);
  label.setAttribute("for", id);

  let control;
  if (config.type === "select") {
    control = item.createEl("select");
    config.options.forEach((optionText) => {
      const option = control.createEl("option");
      option.value = optionText;
      option.textContent = optionText;
    });
    control.value = String(snapshot[field] ?? config.options[0]);
    if (!config.options.includes(control.value)) control.value = config.options[0];
  } else {
    control = item.createEl("input");
    control.type = "number";
    control.min = String(config.min);
    if (config.max !== undefined) control.max = String(config.max);
    control.step = String(config.step ?? 1);
    if (snapshot[field] !== undefined && snapshot[field] !== null && snapshot[field] !== "") {
      control.value = String(snapshot[field]);
    }
  }
  control.id = id;
  control.name = field;
  fields.set(field, control);
}

makeField("首刷结果", "首刷结果", {
  type: "select",
  options: ["未完成", "独立通过", "订正通过", "失败"],
});
makeField("首刷耗时（分钟）", "首刷耗时", { type: "number", min: 0, step: 1 });
makeField("二刷结果", "二刷结果", {
  type: "select",
  options: ["未完成", "独立通过", "失败"],
});
makeField("思想得分（0–2）", "思想得分", { type: "number", min: 0, max: 2, step: 1 });
makeField("正确性得分（0–2）", "正确性得分", { type: "number", min: 0, max: 2, step: 1 });
makeField("代码得分（0–4）", "代码得分", { type: "number", min: 0, max: 4, step: 1 });
makeField("复杂度得分（0–2）", "复杂度得分", { type: "number", min: 0, max: 2, step: 1 });

const actionGroup = root.createEl("div", { cls: "ds408-control__actions" });
actionGroup.setAttribute("role", "group");
actionGroup.setAttribute("aria-label", "训练记录动作");
const buttons = [];

function makeButton(parent, text, className, onClick) {
  const button = addText(parent, "button", text, className);
  button.type = "button";
  button.addEventListener("click", onClick);
  buttons.push(button);
  return button;
}

const summary = root.createEl("dl", { cls: "ds408-control__summary" });
summary.setAttribute("aria-label", "训练证据摘要");
const summaryFields = [
  ["首刷日期", "首刷日期"],
  ["二刷日期", "二刷日期"],
  ["纸笔日期", "纸笔日期"],
  ["纸笔总分", "纸笔得分"],
  ["复习次数", "复习次数"],
  ["下次复习", "下次复习"],
];
const summaryValues = new Map();
summaryFields.forEach(([labelText, field]) => {
  const item = summary.createEl("div", { cls: "ds408-control__summary-item" });
  addText(item, "dt", labelText);
  summaryValues.set(field, addText(item, "dd", "—"));
});

const result = addText(root, "p", "", "ds408-control__message");
result.setAttribute("aria-live", "polite");
result.setAttribute("aria-atomic", "true");

function visibleEvidence() {
  const evidence = {};
  fields.forEach((control, field) => {
    if (control.type === "number") {
      if (control.value.trim() !== "") evidence[field] = control.valueAsNumber;
    } else if (control.value.trim() !== "") {
      evidence[field] = control.value;
    }
  });
  return evidence;
}

function present(value) {
  return value === undefined || value === null || value === "" ? "—" : String(value);
}

function renderCurrentState() {
  stateBadge.textContent = `当前状态：${present(snapshot.状态)}`;
  warningBadge.hidden = snapshot.证据状态 !== "待校准";
  statusButtons.forEach((button, status) => {
    const current = snapshot.状态 === status;
    button.setAttribute("aria-pressed", String(current));
    button.classList.toggle("is-current", current);
  });
  summaryValues.forEach((value, field) => {
    value.textContent = present(snapshot[field]);
  });
}

function setBusy(busy) {
  root.setAttribute("aria-busy", String(busy));
  buttons.forEach((button) => {
    button.disabled = busy;
  });
}

async function applyAction(action, includeEvidence = false) {
  const actionWithEvidence = includeEvidence ? { ...action, evidence: visibleEvidence() } : action;
  setBusy(true);
  result.textContent = "正在保存训练记录…";
  try {
    let patch;
    await dv.app.fileManager.processFrontMatter(file, (fm) => {
      patch = engine.buildAction({ ...fm }, actionWithEvidence, window.moment().format("YYYY-MM-DD"));
      Object.assign(fm, patch);
    });
    Object.assign(snapshot, patch);
    Object.entries(patch).forEach(([field, value]) => {
      const control = fields.get(field);
      if (control) control.value = String(value);
    });
    renderCurrentState();
    result.textContent = "训练记录已保存。";
    new Notice("训练记录已保存。");
  } catch (error) {
    const detail = error instanceof Error && error.message ? error.message : "请检查填写内容后重试";
    result.textContent = `未保存：${detail}`;
    new Notice(`训练记录未保存：${detail}`);
  } finally {
    setBusy(false);
  }
}

engine.STATUSES.forEach((status) => {
  const button = makeButton(statusGroup, status, "ds408-control__status-button", () => {
    applyAction({ kind: "transition", target: status }, true);
  });
  button.setAttribute("aria-pressed", "false");
  statusButtons.set(status, button);
});

makeButton(actionGroup, "记录首刷失败", "ds408-control__action-button is-secondary", () => {
  applyAction({ kind: "failure", stage: "首刷" });
});
makeButton(actionGroup, "记录二刷失败", "ds408-control__action-button is-secondary", () => {
  applyAction({ kind: "failure", stage: "二刷" });
});
makeButton(actionGroup, "迁移通过", "ds408-control__action-button", () => {
  applyAction({ kind: "retention", passed: true });
});
makeButton(actionGroup, "迁移失败", "ds408-control__action-button is-secondary", () => {
  applyAction({ kind: "retention", passed: false });
});

renderCurrentState();
