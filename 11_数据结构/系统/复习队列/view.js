const root = dv.el("section", "", { cls: "ds408-review-queue" });
root.setAttribute("aria-label", "今日训练与复习队列");
const problems = dv.pages('"11_数据结构/刷题记录"').where(p=>p.题号!==undefined&&p.题号!==null).array();
const patterns = dv.pages('"11_数据结构/母题模板"').array();
const priorityOrder = { "S+": 0, S: 1, A: 2, B: 3 };
const stateOrder = { "未开始": 0, "学习中": 1, "已AC": 2, "已二刷": 3, "可纸笔": 4 };

function text(parent, tag, value, cls) { const node = parent.createEl(tag, cls ? { cls } : undefined); node.textContent = value; return node; }
function day(value) {
  if (value && typeof value.toISODate === "function") return value.toISODate() || null;
  const raw = value === undefined || value === null ? "" : String(value).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(raw) && !Number.isNaN(Date.parse(`${raw}T00:00:00Z`)) ? raw : null;
}
const today = day(typeof dv.date === "function" ? dv.date("today") : null) || (window.moment ? window.moment().format("YYYY-MM-DD") : new Date().toISOString().slice(0, 10));
const sevenDays = window.moment ? window.moment(today).add(7, "days").format("YYYY-MM-DD") : today;
function value(p, key) { return p && p[key] !== undefined && p[key] !== null ? String(p[key]) : ""; }
function pathOf(p) { return p && p.file && p.file.path ? String(p.file.path) : ""; }
function rankProblem(p) {
  const due = day(p.下次复习);
  return [due || "9999-12-31", p.来源 === "408真题" ? 0 : 1, priorityOrder[p.优先级] ?? 4,
    p.手写 === "必须" ? 0 : 1, stateOrder[p.状态] ?? 5, Number(p.题号) || Number.MAX_SAFE_INTEGER, pathOf(p)];
}
function rankPattern(p) { return [day(p.下次复写) || "9999-12-31", 2, 4, 1, 5, Number.MAX_SAFE_INTEGER, pathOf(p)]; }
function compare(a, b) { for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1; return 0; }
function sorted(items, rank) { return [...items].sort((a, b) => compare(rank(a), rank(b))); }
function overdue(p) { const due = day(p.下次复习); return due !== null && due < today; }
function incompleteExam(p) { return p.来源 === "408真题" && (p.状态 !== "可纸笔" || p.证据状态 === "待校准"); }
function patternDue(p) { const due = day(p.下次复写); return due !== null && due <= today || p.模板状态 !== "可复写"; }

const overdueCount = problems.filter(overdue).length;
const nextSevenCount = problems.filter(p => { const due = day(p.下次复习); return due !== null && due >= today && due <= sevenDays; }).length;
const pendingCount = problems.filter(p => p.证据状态 === "待校准").length;
const alerts = root.createEl("div", { cls: "ds408-review-queue__alerts" });
[["逾期复习", overdueCount], ["未来7天复习", nextSevenCount], ["证据待补", pendingCount]].forEach(([label, count]) => {
  const item = alerts.createEl("div", { cls: "ds408-review-queue__alert" }); text(item, "strong", String(count)); text(item, "span", label);
});

const used = new Set();
function pick(candidates, fallbackProblems = true) {
  const pool = sorted(candidates, item => item.kind === "pattern" ? rankPattern(item.page) : rankProblem(item.page));
  const fallback = sorted([
    ...problems.map(page => ({ kind: "problem", page })),
    ...(fallbackProblems ? patterns.map(page => ({ kind: "pattern", page })) : []),
  ], item => item.kind === "pattern" ? rankPattern(item.page) : rankProblem(item.page));
  const selected = [...pool, ...fallback].find(item => !used.has(`${item.kind}:${pathOf(item.page)}`));
  if (selected) used.add(`${selected.kind}:${pathOf(selected.page)}`);
  return selected;
}
const selected = [
  { title: "优先处理逾期与待校准", reason: "逾期复习或证据待补", item: pick(problems.filter(p => overdue(p) || p.证据状态 === "待校准").map(page => ({ kind: "problem", page }))) },
  { title: "408 真题闭卷证据", reason: "真题尚未完整纸笔验证", item: pick(problems.filter(incompleteExam).map(page => ({ kind: "problem", page }))) },
  { title: "母题闭卷复写", reason: "模板到期或尚不可复写", item: pick(patterns.filter(patternDue).map(page => ({ kind: "pattern", page }))) },
];
const list = root.createEl("div", { cls: "ds408-review-queue__list" });
function makeLink(parent, page, label) { const link = text(parent, "a", label, "internal-link ds408-review-queue__link"); link.setAttribute("data-href", pathOf(page)); link.setAttribute("href", pathOf(page)); }
selected.forEach(({ title, reason, item }) => {
  const card = list.createEl("article", { cls: "ds408-review-queue__card" }); text(card, "h3", title); text(card, "p", reason, "ds408-review-queue__reason");
  if (!item) { text(card, "p", "题库暂时没有可安排的候选。", "ds408-review-queue__empty"); return; }
  const p = item.page; const label = item.kind === "pattern" ? `${value(p, "母题编号") || value(p, "母题")} ${value(p, "母题模块")}` : `${value(p, "题号")} · ${value(p, "标题")}`;
  makeLink(card, p, label.trim() || "打开记录");
  const due = item.kind === "pattern" ? value(p, "下次复写") : value(p, "下次复习");
  text(card, "p", `${item.kind === "pattern" ? "母题" : "题目"}｜状态：${value(p, item.kind === "pattern" ? "模板状态" : "状态") || "未填写"}｜下次：${due || "未安排"}`, "ds408-review-queue__meta");
});
if (!problems.length && !patterns.length) text(root, "p", "尚无题目和母题模板；队列会在资料建立后自动出现。", "ds408-review-queue__empty");
