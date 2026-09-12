# 项目约定 · 计算机组成原理（408）

## 目录结构

```
03_计算机系统/1_计算机组成原理/
├── 0_基础考点讲解/<N>_第N章 xxx/          # 王道基础班讲义 PDF
├── 1_王道强化/                             # 王道强化班讲义 PDF（P1～P6，均为纯扫描图）
├── 2_原书PDF/                              # 教材
├── 3_题本/                                 # 做题本
└── 4_强化笔记补充/                         # ← 自建笔记（2026-09-11 由「我的笔记/」整体移入）
    ├── 计算机组成原理总览（含新大纲考点）.pdf
    └── <章名>/
        ├── <章名>（完整笔记）.md
        └── assets/                         # 讲义抽出的图
```

> ⚠️ **笔记路径已变**：原先是 `我的笔记/<章名>/`，2026-09-11 起改为
> `4_强化笔记补充/<章名>/`。改路径后脚本里的硬编码路径要一并改
> （`make_acceptance.py` 的 NOTE、本文件下面的验收命令都指向旧路径）。

## 笔记撰写约定（用户已确认）

1. **一章一篇完整笔记**，不要拆成一章一套多篇（用户明确要求合并）。
   - 单篇内结构：1 个 H1 标题 + 若干 H1 大节（`# 3.1` `# 3.2` …），节内用 H2 细分。
   - **编号必须是一条单调链**：整章用 `3.x` 一套编号（`3.0 导读` + `3.1–3.6 正文` + `3.7/3.8 专题` + `3.9 附录`），
     H2 继承为 `3.x.y`，H3 用 `①②③` 或描述性标题（不再叠第四层数字）。
     **切忌在 H1 混用 level-1（`0.`/`7.`/`8.`）与 level-2（`3.x`）两种编号，也不要跳号。**
     改标题后必须全文回扫 `第 X 节` 交叉引用一并同步。
2. **按 408 考研标准补齐细节**，不只照抄讲义的图；讲义只有图、没文字的部分要把每一步文字化。
3. **讲义原图必须嵌入笔记**：`![[NN_图N_标题.jpg]]`，紧跟文字说明；图片放在笔记同级的 `assets/`。
4. 关键数值要**自行重算验证自洽**后再写进笔记（图上的十六进制/二进制串是模糊扫描，易误读）。
5. 每节末尾给「自测」+ `<details>` 折叠答案；易错点用 `> [!warning]` / `> [!important]`。
6. 数学公式用 `$...$` / `$$...$$`（Obsidian 支持 MathJax）。
   ⚠️ **行内 `$...$` 的定界符内侧不能有空格**：`$ x$`、`$x $`、`$ x $` 都会被 Obsidian
   当成普通文本**原样显示**（用户会以为"公式坏了"）。写「= … = 答案」的推导链时最易踩：
   `$= \log_2 4\text{K} = $ **12 根**` ✗ → `$= \log_2 4\text{K} =$ **12 根**` ✓（只差一个空格）。
   排查脚本：`~/.workbuddy/skills/lecture-pdf-to-obsidian-notes/scripts/scan_math_delims.py <目录>`；
   第三章笔记曾一次查出 8 处，集中在一段推导里，看起来像"整屏公式都坏了"。
7. **frontmatter 的 `tags` 必须是字符串**：纯数字 tag 一律加引号（`- "408"`），
   否则 YAML 会解析成数字，Obsidian 属性面板报类型 ⚠ 警告。`created` / `updated` 用 `YYYY-MM-DD` 即可（日期属性）。
8. **排版规范**（用户会要求"优化排版"）：
   - callout 标题必须是**短标签**（≤12 字），不要把整句当标题；标题末尾不加句号/冒号。
   - 篇首给一行「标注约定」图例说明各 callout 颜色含义。
   - 块公式 `$$` 开栅栏前必须空行；**列表项内不要嵌块公式**，用行内 `$...$`。
   - 中英之间加空格（`Cache 块`、`TLB 标记`）。
   - 图号写 `图 1` / `图 10`，区间 `图 1～图 9`；**`![[NN_图N_xxx.jpg]]` 文件名不能动**。
   - H1 大节之间用 `---` 分隔；节首出处统一 `> 📎 **对应讲义**：\`xxx.pdf\``。
   - **代码块里的 ASCII 图必须列对齐**。对齐模型：西文字符/框线/箭头 = **1 列**，中文/全角/`①-⑳` = **2 列**
     （依据：等宽西文 Quiet Mono West 步进 0.6em，等宽中文 Noto Sans SC Wide 1.2em）。
     改图时按此模型逐行算宽度，`┌─┐│└┘` 的右边框、流程图的右侧「尺子」、目录树的 `→` 注释列都要落在同一列上。
     详见 `~/.workbuddy/skills/obsidian-custom-fonts/references/monospace-2to1.md`
     与 `.../references/glyph-patching.md`。
     判定「某字符算 1 列还是 2 列」的可靠办法：逐行算「最后一个框线字符所在列」，
     若在某种假设下每一行都恒等于同一个列号，则该假设正确（本仓库的 `①` 就是这样确认必须占 2 列的）。

## 环境

- 无 `pdftotext` / `pdfinfo`；PDF 处理用 PyMuPDF：
  `/Users/heyuhang/.workbuddy/binaries/python/envs/default/bin/python`，`import pymupdf`（勿用 `fitz`）。
  同环境已装 `fontTools` 与 `Pillow`（字体度量 / 渲染验收用）。
- 验收字体有没有真的生效：**必须分两层，缺一不可**（一条命令：`scripts/verify_chromium.py`）——
  - **字体层（真值）**：读 `~/Library/Fonts/QuietMonoWest-*.ttf` 的 `cmap`+`hmtx` 逐码点核对步进；
  - **浏览器层（端到端）**：本机装有 Google Chrome，用
    `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-sandbox --disable-gpu --dump-dom <url>`
    在真实 Chromium 里量 `measureText`。
  - ⚠️ **两个会骗人的信号**：① `ui-monospace`（本机 = SF Mono）步进**也正好 0.6000em**，
    所以「量出来 0.6em」**证明不了**字形来自 Quiet Mono West，覆盖只能靠字体层查 `cmap`；
    ② 字体自带的 `ccmp`/`liga` 会把 `ŉ ﬀ ﬁ ﬂ`（U+0149/U+FB00-02）分解成两个字形 → 量出 1.2em，
    看着像破格其实是对的（关掉这两个 feature 才回 0.6em）。判据要按类别分派，别一律断言 0.6em。
  - 测试页必须补 Obsidian 内置的 `pre { font-family: var(--font-monospace) }`，
    否则 `pre` 退回浏览器默认 Menlo（0.6021em），得到一片假阴性。

- **字号**（2026-09-11 应「字体调大一些，数字公式也放大」调过；改在 `quiet-notes-unified.css`）：
  - 正文 `--font-text-size: 17px`（原 15）；H1~H4 都是 `em`，自动跟着放大。
  - 公式：行内 `mjx-container` **1.16em** ≈ 19.7px，行间 `[display="true"]` **1.26em** ≈ 21.4px。
  - **代码块 `--code-size: 15px` 单独钉死，不跟随正文。** 因为等宽步进恰好 0.6em，
    「一列多少像素」完全由字号决定，取 **5 的倍数**才落在整数像素上
    （15px → 9.00px/列 ✓；17px → 10.20px/列 ✗，长 ASCII 框图会发虚）。
  - Obsidian 核心的代码字号默认是 `--code-size: var(--font-smaller)`（= 0.875em），
    必须显式声明才生效；**验收页 / 探针里要读 `var(--code-size, 15px)` 而不是 `var(--font-text-size)`**，
    否则量错字号 —— 而栅格判据是比例判据，量错**不会报错**，得另测绝对像素。
  - 验收页生成器 `~/.workbuddy/skills/obsidian-custom-fonts/scripts/make_acceptance.py`
    → 输出 `.workbuddy/preview/09_字体方案实时验收.html`（现含「字号一览」实时实测 + 整数像素断言）。

## 相关技能

- `~/.workbuddy/skills/lecture-pdf-to-obsidian-notes/` —— 扫描版讲义 PDF → Obsidian 笔记 的完整流程。
