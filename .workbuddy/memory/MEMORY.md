# Obsidian Vault · 项目长期约定

> 与具体课题无关的偏好在 `~/.workbuddy/MEMORY.md`；这里只记**本仓库**的结构与体例约定。

## 目录编号体系

`00_Inbox` / `01_数学` / `02_DSP` / `03_计算机系统` / `04_数学大观` / `05_英语` / `06_工具` /
`07_DL基础及论文进阶` / `08_考研打卡` / **`09_考研政治`**。
新科目按 `NN_科目名` 递增编号，不要插空、不要改名。

## 笔记体例（写任何新笔记前先找母本）

- **先找同类母本照抄体例**，再写内容。现有母本：
  - `05_英语/美国政治背景/` + `05_英语/英国政治背景/` —— 三段式专题（历史制度 / 词汇概念 / 冲突与真题对照），
    用 callout + mermaid + 表格，**明确不用 LaTeX 排长句概念**。
  - `04_数学大观/线代大观/` —— `MOC + 1_总纲 + N_专题`，LaTeX 只用于真正的数学表达式。
  - `09_考研政治/` —— `MOC + 政治01…05`，方法论类专题（总论 / 分题型 / 分模块 / 工具模板）。
- frontmatter 用 `title / tags / type / updated`；H2 用中文数字 `一、二、`；
  导航用 `> [!abstract] 本笔记导航`，结尾用 `> [!abstract] 继续阅读`；
  自测用 `<details><summary>🎯 自测：…</summary>…</details>`。
- **文件名必须带学科/模块前缀**（如 `政治01 …`、`英国01 …`），否则仓库内同名笔记会让
  `[[01 历史与制度骨架]]` 这类链接产生歧义。跨文件夹链接写**仓库根全路径 + 别名**。

## 交付前体检

```bash
PY=/Users/heyuhang/.workbuddy/binaries/python/envs/default/bin/python
$PY ~/.workbuddy/skills/exam-background-to-obsidian/scripts/check_notes.py \
  "<vault>/<新目录>" --vault "<vault>"
```

查四件事，**问题数必须为 0**：mermaid 引号/方括号配对与裸括号、callout 下一行是否以 `>` 续写、
`<details>/<summary>` 配对、wiki 链接是否存在及是否歧义。

> ⚠️ 踩过的坑：**只有标题行、没有正文的 callout**会被判「下一行未续 `>`」。
> 必须补正文行：`> [!important] 标题` → `> **正文**`。

## 关联技能

- `exam-background-to-obsidian` —— 应试**背景知识**长文 → 三段式专题（含 `check_notes.py`）。
- `exam-method-to-obsidian` —— 应试**方法论**长文 → MOC + 5 篇专题。
