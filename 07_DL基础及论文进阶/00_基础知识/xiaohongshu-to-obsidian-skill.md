---
name: xiaohongshu-to-obsidian
description: 小红书笔记整理 Skill。给定小红书 explore 链接，自动下载笔记中的全部图片，OCR 识别内容，提取标题/描述/标签，按知识库规范整理为 Obsidian 笔记写入 `00_基础知识/` 下。触发词：小红书链接、小红书笔记、整理小红书、xhs 链接、xiaohongshu。
---

# 小红书笔记整理 Skill

## 概述

将小红书（xiaohongshu.com）上的图文笔记自动整理为 Obsidian 知识库中的结构化笔记。适用于深度学习/技术类小红书教程笔记。

## 适用场景

- 用户发来一个小红书 `explore` 链接，要求"下载图片"、"整理到知识库"、"插入到基础知识中"
- 用户分享技术教程类小红书笔记，希望纳入 Obsidian DL 知识库

## 完整工作流程

### 第一步：抓取页面，提取元信息

使用 `mcp__workspace__bash` 的 `curl` 命令抓取页面 HTML（必须带 User-Agent 和 Referer 头，否则会被反爬）：

```bash
curl -sL '{用户给的URL}' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' \
  -H 'Referer: https://www.xiaohongshu.com/' \
  2>/dev/null
```

从返回的 HTML 中提取以下信息：

1. **标题**：`grep -oP '"title":"[^"]*"'` → 找到非页脚类的 title（排除"小红书_沪ICP备"等）
2. **描述**：`grep -oP '"desc":"[^"]*"'` → 含话题标签如 `#大模型[话题]#`
3. **话题标签**：从 desc 或 `"tagList"` JSON 中解析
4. **所有图片 URL**：`grep -oP 'http[^"<>\s]*1040[^"<>\s]*' | grep -v '\.js\|\.css\|\.ico\|\.svg\|\.woff' | sort -u`

**过滤图片 URL 的关键规则：**
- 只看 **sns-webpic-qc.xhscdn.com** 域名下的链接
- 优先取 `!nd_dft_wlteh_jpg_3` 后缀的（默认尺寸，高质量）
- 去重：同一 `1040g...` ID 的不同后缀只保留一个
- 排除 JS/CSS/ICO/SVG/WOFF 等非图片资源
- 排除 avatar（头像图）

### 第二步：下载图片

对每个去重后的图片 URL，用 curl 下载到临时目录：

```bash
OUT="/sessions/{session}/mnt/outputs/xhs_images"
mkdir -p "$OUT"
curl -sL -o "$OUT/{序号}_{描述名}.jpg" '{图片URL}' \
  -H 'Referer: https://www.xiaohongshu.com/' \
  -H 'User-Agent: Mozilla/5.0'
```

**文件命名建议：** `01_概览.jpg`、`02_定义.jpg`、`03_性质.jpg`... 先按下载顺序编号，后续 OCR 后再修正为有意义的名字。

### 第三步：OCR 识别图片内容

对每张下载的图片运行 OCR：

```python
import pytesseract
from PIL import Image

img = Image.open('图片路径')
text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 3')
```

PSM 3（全自动页面分割）通常效果最好。如果效果差，尝试 PSM 6（统一文本块）。

**OCR 的目的不是逐字提取，而是：**
1. 理解每张图讲什么主题（概览、定义、性质、应用等）
2. 提取关键公式和概念名称
3. 确定 4 张图的逻辑顺序

**重要：** OCR 对中文手写风格图片效果有限，需要结合图片标题和上下文推断内容结构。

### 第四步：确定知识库归属

用 AskUserQuestion 确认（如果信息不足）：

1. **归属模块**：是放在 `00_基础知识/` 下的新模块？还是归属于现有模块（如 `02_深度学习面试题/`）的子主题？
2. **模块序号**：`00_基础知识/` 下已有 `00_PyTorch语法`、`01_NumPy语法`、`02_深度学习面试题`，新模块用下一个可用序号 `03_`。
3. **是否需要与现有内容关联**：双链到哪些已有笔记？

**默认策略（无需每次都问）：**
- 新主题 → 在 `00_基础知识/` 下创建 `{序号}_{主题名}/` 新文件夹
- 序号按现有模块递增
- 图片放 `assets/` 子文件夹

### 第五步：编写 Obsidian 笔记

#### 文件夹结构

```
00_基础知识/{序号}_{主题名}/
├── {主题名}.md              ← 主笔记
└── assets/
    ├── {主题名}_01_{描述}.jpg
    ├── {主题名}_02_{描述}.jpg
    └── ...
```

#### Frontmatter 模板

```yaml
---
title: {主题名}
tags:
  - 基础知识
  - 深度学习
  - {从XHS话题标签提取的标签}
  - {技术标签如 RLHF/KL散度/信息论}
source: 小红书-{笔记标题} ({笔记ID})
created: YYYY-MM-DD
---
```

#### 正文结构

笔记应包含以下部分（不必严格六个章节，因为是知识模块而非论文）：

```markdown
# {主题名}

> {一句话核心定位——这篇笔记讲什么，为什么重要}

## 一、概览

![[{图片名1}]]

{概括整篇笔记的核心要点}

## 二、核心概念

![[{图片名2}]]

{围绕图片内容展开解释，含 LaTeX 公式}

## 三、关键性质/深入理解

![[{图片名3}]]

{深入讲解，含表格对比}

## 四、实际应用/计算案例

![[{图片名4}]]

{具体例子，含 PyTorch 代码}

## 来源

- 图片来自小红书笔记：[{标题}]({原始链接})
- 话题标签：{标签列表}
```

**每张图片的引用方式：** 用 Obsidian 的 `![[图片文件名]]` 语法嵌入（不加 `assets/` 前缀，Obsidian 会自动在子目录中搜索）。

#### 内容编写原则

1. **图片 + 文字互补**：图片展示视觉化概念，文字做深入解释和代码实现
2. **具体代码示例**：每个概念至少配一个可运行的 PyTorch/NumPy 代码块
3. **用表格对比**：对比不同场景/方法时用表格
4. **LaTeX 公式**：所有数学公式用 `$...$` 或 `$$...$$`
5. **与已有知识关联**：适当 `[[双链]]` 到已有笔记
6. **中文为主，术语保留英文**

### 第六步：复制文件到知识库

```bash
cp 图片文件 "/sessions/{session}/mnt/08_DL基础及论文进阶/00_基础知识/{序号}_{主题名}/assets/"
```

用 Write 工具写 Markdown 笔记到目标路径。

### 第七步：验证

```bash
ls -lhR "/sessions/{session}/mnt/08_DL基础及论文进阶/00_基础知识/{序号}_{主题名}/"
```

确保：
- `.md` 笔记存在且内容完整
- `assets/` 下所有图片都存在且大小正常（通常 >100KB 每张）
- `![[...]]` 引用的图片名与 `assets/` 下的文件名一致

---

## 反爬与容错

1. **User-Agent 必须带**：模拟浏览器
2. **Referer 必须带**：`https://www.xiaohongshu.com/`
3. **链接可能被转义**：HTML 中可能是 `/` 代替 `/`，需要 `python3 -c` 处理
4. **图片 CDN 域名变化**：常见的有 `sns-webpic-qc.xhscdn.com`、`ci.xiaohongshu.com`，需要灵活匹配
5. **如果 curl 失败**：尝试用 `mcp__workspace__web_fetch` 作为备选方案

## 关键注意事项

1. 图片文件命名要有意义（根据 OCR 结果命名），不只是数字编号
2. 不修改 `.obsidian/`、`.claude/` 等隐藏目录
3. 每次新建模块前，先 `ls` 确认 `00_基础知识/` 下已有模块的序号，避免冲突
4. OCR 结果不完美是正常的——需要人工推断图片主题来补充结构
5. 笔记末尾必须注明来源链接
6. 代码块要尽量完整可运行，方便用户在 Obsidian Execute Code 插件中直接执行
