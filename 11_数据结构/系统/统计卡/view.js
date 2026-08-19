const records = dv.pages('"11_数据结构/刷题记录"').where(p=>p.题号!==undefined&&p.题号!==null).array();
const root = dv.el("section", "", { cls: "ds408-kpi" });
root.setAttribute("aria-label", "408 训练核心指标");

const weights = { "未开始": 0, "学习中": 0.25, "已AC": 0.5, "已二刷": 0.75, "可纸笔": 1 };
const total = records.length;
const denominator = total || 1;
const countAtLeastAc = records.filter(p => ["已AC", "已二刷", "可纸笔"].includes(p.状态)).length;
const paperCount = records.filter(p => p.状态 === "可纸笔").length;
const exams = records.filter(p => p.来源 === "408真题");
const examPaper = exams.filter(p => p.状态 === "可纸笔").length;
const percent = value => `${(100 * value).toFixed(1)}%`;

function text(parent, tag, value, cls) {
  const node = parent.createEl(tag, cls ? { cls } : undefined);
  node.textContent = value;
  return node;
}
function card(cls, title, value, detail, formula) {
  const node = root.createEl("article", { cls: `ds408-kpi__card ${cls}` });
  node.setAttribute("title", formula);
  text(node, "h3", title, "ds408-kpi__title");
  text(node, "strong", value, "ds408-kpi__value");
  text(node, "p", detail, "ds408-kpi__detail");
}

card("ds408-kpi-weighted", "加权掌握度", percent(records.reduce((sum, p) => sum + (weights[p.状态] ?? 0), 0) / denominator),
  `${total ? total : 0} 题；未开始/学习中/已AC/已二刷/可纸笔 = 0/0.25/0.5/0.75/1`, "100×Σ状态权重÷题目总数");
card("ds408-kpi-ac", "AC及以上率", percent(countAtLeastAc / denominator), `${countAtLeastAc}/${total || 0} 题`, "100×状态≥已AC的题数÷题目总数");
card("ds408-kpi-paper", "纸笔达标率", percent(paperCount / denominator), `${paperCount}/${total || 0} 题`, "100×状态为可纸笔的题数÷题目总数");
card("ds408-kpi-exam", "408真题纸笔率", percent(examPaper / (exams.length || 1)), `${examPaper}/${exams.length} 题`, "100×408真题中可纸笔题数÷408真题总数（空分母按1）");

if (!total) text(root, "p", "题库尚无带题号的刷题记录；指标将在加入题目后自动计算。", "ds408-kpi__empty");
