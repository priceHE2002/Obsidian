const records = dv.pages('"11_数据结构/刷题记录"')
  .where((p) => p.题号 !== undefined && p.题号 !== null)
  .sort((p) => Number(p.题号), "asc")
  .array();
const current = input?.path ? dv.page(input.path) : dv.current();
const root = dv.el("nav", "", { cls: "ds408-problem-nav" });
root.setAttribute("aria-label", "题目导航");

function addText(parent, tag, text, className) {
  const element = parent.createEl(tag, className ? { cls: className } : undefined);
  element.textContent = text;
  return element;
}

function addLink(target, label) {
  const link = addText(root, "a", label, "internal-link ds408-problem-nav__link");
  link.setAttribute("data-href", target);
  link.setAttribute("href", target);
  return link;
}

if (!current || !current.file || !current.file.path) {
  addText(root, "span", "未能识别当前题目。", "ds408-problem-nav__hint");
  addLink("11_数据结构/总览", "训练总览");
  return;
}

const currentIndex = records.findIndex((record) => record.file && record.file.path === current.file.path);
if (currentIndex < 0) {
  addText(root, "span", "当前题目未纳入训练序列。", "ds408-problem-nav__hint");
}

if (currentIndex > 0) {
  addLink(records[currentIndex - 1].file.path, "← 上一题");
}

const hasMother = current.母题 !== undefined && current.母题 !== null && String(current.母题).trim() !== ""
  && current.母题模块 !== undefined && current.母题模块 !== null && String(current.母题模块).trim() !== "";
if (hasMother) {
  const mother = String(current.母题).trim();
  const module = String(current.母题模块).trim();
  addLink(`11_数据结构/母题模板/${mother}_${module}`, `母题 ${mother}`);
} else {
  addText(root, "span", "未配置母题链接。", "ds408-problem-nav__hint");
}

addLink("11_数据结构/总览", "训练总览");

if (currentIndex >= 0 && currentIndex < records.length - 1) {
  addLink(records[currentIndex + 1].file.path, "下一题 →");
}
