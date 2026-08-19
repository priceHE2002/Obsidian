const root = dv.el("section", "", { cls: "ds408-bank" });
root.setAttribute("aria-label", "可筛选 408 题库表");
function text(parent, tag, value, cls) { const node = parent.createEl(tag, cls ? { cls } : undefined); node.textContent = value; return node; }
function message(value, error = false) { live.textContent = value; if (error) new Notice(value); }
const live = root.createEl("p", { cls: "ds408-bank__live" }); live.setAttribute("aria-live", "polite"); live.setAttribute("aria-atomic", "true");
let engine;
try {
  const source = await dv.io.load("11_数据结构/系统/共享/state-engine.js");
  engine = Function(source+'\nreturn api;')();
} catch (error) { message("状态规则加载失败，题库仅可稍后重试。", true); return; }
if (!engine || !Array.isArray(engine.STATUSES) || typeof engine.buildAction !== "function") { message("状态规则加载失败，题库仅可稍后重试。", true); return; }

const priorityOrder = { "S+": 0, S: 1, A: 2, B: 3 };
const stateOrder = Object.fromEntries(engine.STATUSES.map((state, index) => [state, index]));
const state = { filters: { mother: "全部", status: "全部", priority: "全部", source: "全部", evidence: "全部" }, sort: { key: "题号", direction: "asc" }, busy: false };
let pages = dv.pages('"11_数据结构/刷题记录"').where(p=>p.题号!==undefined&&p.题号!==null).array();
const controls = root.createEl("div", { cls: "ds408-bank__controls" });
const summary = text(controls, "p", "", "ds408-bank__summary");
const tableWrap = root.createEl("div", { cls: "ds408-bank__table-wrap" });

function value(p, field) { return p && p[field] !== undefined && p[field] !== null && String(p[field]).trim() !== "" ? p[field] : ""; }
function pathOf(p) { return p && p.file && p.file.path ? String(p.file.path) : ""; }
function dateKey(value) {
  if (value && typeof value.toISODate === "function") value = value.toISODate();
  const raw = value === undefined || value === null ? "" : String(value).slice(0, 10);
  const matched = /^(\d{4})-(\d{2})-(\d{2})$/.exec(raw);
  if (!matched) return "9999-12-31";
  const year = Number(matched[1]); const month = Number(matched[2]); const date = Number(matched[3]);
  const utc = new Date(Date.UTC(year, month - 1, date));
  return utc.getUTCFullYear() === year && utc.getUTCMonth() === month - 1 && utc.getUTCDate() === date ? raw : "9999-12-31";
}
function sortValue(p, key) {
  if (key === "题号") return Number(value(p, "题号")) || Number.MAX_SAFE_INTEGER;
  if (key === "母题") return String(value(p, "母题"));
  if (key === "优先级") return priorityOrder[value(p, "优先级")] ?? 4;
  if (key === "状态") return stateOrder[value(p, "状态")] ?? 5;
  if (key === "下次复习") return dateKey(value(p, "下次复习"));
  if (key === "纸笔得分") return value(p, "纸笔得分") === "" ? -1 : Number(value(p, "纸笔得分"));
  return "";
}
function compare(a, b) { const av = sortValue(a, state.sort.key); const bv = sortValue(b, state.sort.key); if (av !== bv) { const raw = av < bv ? -1 : 1; return state.sort.direction === "asc" ? raw : -raw; } return pathOf(a).localeCompare(pathOf(b), "zh-Hans-CN"); }
function matches(p) { return (state.filters.mother === "全部" || value(p, "母题") === state.filters.mother)
  && (state.filters.status === "全部" || value(p, "状态") === state.filters.status)
  && (state.filters.priority === "全部" || value(p, "优先级") === state.filters.priority)
  && (state.filters.source === "全部" || value(p, "来源") === state.filters.source)
  && (state.filters.evidence === "全部" || value(p, "证据状态") === state.filters.evidence); }
function button(parent, label, cls, handler) { const node = text(parent, "button", label, cls); node.type = "button"; node.disabled = state.busy; node.addEventListener("click", handler); return node; }
function makeFilter(labelText, filter, options) {
  const item = controls.createEl("label", { cls: "ds408-bank__filter" }); text(item, "span", labelText);
  const select = item.createEl("select"); select.setAttribute("data-filter", filter); select.value = state.filters[filter]; select.disabled = state.busy;
  options.forEach(option => { const node = select.createEl("option"); node.value = option; node.textContent = option; });
  select.addEventListener("change", () => { state.filters[filter] = select.value; render(); });
}
makeFilter("母题", "mother", ["全部", ...Array.from({ length: 12 }, (_, i) => `M${String(i + 1).padStart(2, "0")}`)]);
makeFilter("状态", "status", ["全部", ...engine.STATUSES]);
makeFilter("优先级", "priority", ["全部", "S+", "S", "A", "B"]);
makeFilter("来源", "source", ["全部", "408真题", "LeetCode"]);
makeFilter("证据", "evidence", ["全部", "有效", "待校准"]);
button(controls, "清空筛选", "ds408-bank__clear", () => { if (state.busy) return; Object.keys(state.filters).forEach(key => { state.filters[key] = "全部"; }); render(); });

function makeLink(parent, page, label) { const link = text(parent, "a", label, "internal-link ds408-bank__link"); link.setAttribute("data-href", pathOf(page)); link.setAttribute("href", pathOf(page)); }
function rowText(row, content) { text(row, "td", content === "" ? "—" : String(content)); }
function setBusy(busy) {
  state.busy = busy; root.setAttribute("aria-busy", String(busy));
  root.querySelectorAll("button").forEach(node => { node.disabled = busy; });
  root.querySelectorAll("select").forEach(node => { node.disabled = busy; });
  render();
}
async function saveAction(page, action) {
  if (state.busy) return;
  const file = page && page.file && page.file.path ? dv.app.vault.getAbstractFileByPath(page.file.path) : null;
  if (!file || !file.path) { message("题目文件不存在，未执行任何写入。", true); return; }
  setBusy(true); message("正在保存训练记录…");
  try {
    let patch;
    await dv.app.fileManager.processFrontMatter(file, fm=>{ patch=engine.buildAction({...fm}, action, window.moment().format('YYYY-MM-DD')); Object.assign(fm,patch); });
    pages = pages.map(item => pathOf(item) === pathOf(page) ? { ...item, ...patch } : item);
    message("训练记录已保存。"); new Notice("训练记录已保存。");
  } catch (error) { message(`未保存：${error instanceof Error && error.message ? error.message : "请检查证据"}`, true); }
  finally { setBusy(false); }
}
function editor(row, page, target) {
  const editorRow = document.createElement("tr"); editorRow.className = "ds408-bank__editor-row"; row.insertAdjacentElement("afterend", editorRow);
  const editorCell = document.createElement("td"); editorCell.className = "ds408-bank__editor-cell"; editorCell.setAttribute("colspan", "10"); editorRow.appendChild(editorCell);
  const box = editorCell.createEl("div", { cls: "ds408-bank__editor" }); box.setAttribute("role", "region"); box.setAttribute("aria-label", `补充${target}训练证据`);
  const fields = [];
  function field(labelText, name, config) {
    const id = `ds408-bank-${Date.now()}-${name}`; const label = text(box, "label", labelText); label.setAttribute("for", id);
    const input = box.createEl(config.select ? "select" : "input"); input.id = id; input.name = name;
    if (config.select) { config.select.forEach(option => { const item = input.createEl("option"); item.value = option; item.textContent = option || "请选择"; }); }
    else { input.type = "number"; input.min = String(config.min); input.max = String(config.max); input.step = "1"; }
    fields.push([name, input]);
  }
  if (target === "学习中") text(box, "p", "确认开始本次学习；保存后将记录训练日期。", "ds408-bank__hint");
  if (target === "已AC") { field("首刷结果", "首刷结果", { select: ["", "独立通过", "订正通过"] }); field("首刷耗时（分钟，可空）", "首刷耗时", { min: 0, max: 9999 }); }
  if (target === "已二刷") field("二刷结果", "二刷结果", { select: ["", "独立通过"] });
  if (target === "可纸笔") { field("思想得分（0–2）", "思想得分", { min: 0, max: 2 }); field("正确性得分（0–2）", "正确性得分", { min: 0, max: 2 }); field("代码得分（0–4）", "代码得分", { min: 0, max: 4 }); field("复杂度得分（0–2）", "复杂度得分", { min: 0, max: 2 }); }
  const actions = box.createEl("div", { cls: "ds408-bank__editor-actions" });
  button(actions, "取消", "ds408-bank__cancel", () => { editorRow.remove(); });
  button(actions, "确认保存", "ds408-bank__save", () => {
    const evidence = {}; fields.forEach(([name, input]) => { if (input.type === "number") { if (input.value.trim() !== "") evidence[name] = input.valueAsNumber; } else if (input.value.trim() !== "") evidence[name] = input.value; });
    saveAction(page, { kind: "transition", target, evidence });
  });
}
function statusButtons(row, page) {
  const group = row.createEl("td", { cls: "ds408-bank__status-cell" }); const currentIndex = stateOrder[value(page, "状态")] ?? 0;
  engine.STATUSES.forEach((target, targetIndex) => {
    const current = targetIndex === currentIndex; const pending = value(page, "证据状态") === "待校准";
    const action = button(group, target, "ds408-bank__status", () => {
      if (state.busy) return;
      if (targetIndex < currentIndex) { if (window.confirm("将回退训练阶段；历史训练证据会保留。是否继续？")) saveAction(page, { kind: "transition", target }); return; }
      editor(row, page, target);
    });
    action.disabled = state.busy || (current && !pending); action.setAttribute("aria-pressed", String(current));
  });
}
function render() {
  controls.querySelectorAll && controls.querySelectorAll("select[data-filter]").forEach(select => { select.value = state.filters[select.getAttribute("data-filter")]; });
  tableWrap.replaceChildren(); const filtered = pages.filter(matches).sort(compare); summary.textContent = `显示 ${filtered.length} / ${pages.length} 题`;
  if (!filtered.length) { text(tableWrap, "p", "没有符合当前组合筛选条件的题目。", "ds408-bank__empty"); return; }
  const table = tableWrap.createEl("table", { cls: "ds408-bank__table" }); const head = table.createEl("thead").createEl("tr");
  const columns = [["题号", "题号"], ["母题", "母题"], ["标题", null], ["优先级", "优先级"], ["来源", null], ["证据", null], ["下次复习", "下次复习"], ["纸笔分", "纸笔得分"], ["状态", "状态"], ["状态操作", null]];
  columns.forEach(([label, key]) => { const th = head.createEl("th"); if (!key) { text(th, "span", label); return; } th.setAttribute("aria-sort", state.sort.key === key ? (state.sort.direction === "asc" ? "ascending" : "descending") : "none"); const sort = button(th, label, "ds408-bank__sort", () => { state.sort = { key, direction: state.sort.key === key && state.sort.direction === "asc" ? "desc" : "asc" }; render(); }); sort.setAttribute("aria-label", `按${label}排序`); });
  const body = table.createEl("tbody"); filtered.forEach(page => { const row = body.createEl("tr"); const first = row.createEl("td"); makeLink(first, page, String(value(page, "题号") || "—")); rowText(row, value(page, "母题")); rowText(row, value(page, "标题")); rowText(row, value(page, "优先级")); rowText(row, value(page, "来源")); rowText(row, value(page, "证据状态")); rowText(row, value(page, "下次复习")); rowText(row, value(page, "纸笔得分")); rowText(row, value(page, "状态")); statusButtons(row, page); });
}
render();
