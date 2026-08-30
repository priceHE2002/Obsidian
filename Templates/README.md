# 考研数学 Obsidian 教材级公式排版

本套资源包含：

- `textbook-math.css`：Obsidian 数学公式排版 CSS
- `kaoyan-math-template.md`：高数 + 线代常用 LaTeX 模板
- 本说明文件

## 一、安装 CSS

1. 打开你的 Obsidian 仓库。
2. 进入：

   `设置 → 外观 → CSS 代码片段`

3. 点击“打开代码片段文件夹”。
4. 把 `textbook-math.css` 放进去。
5. 回到 Obsidian，点击刷新。
6. 开启 `textbook-math`。

典型路径：

```text
你的仓库/
└── .obsidian/
    └── snippets/
        └── textbook-math.css
```

## 二、模板放置

你可以把：

```text
kaoyan-math-template.md
```

放到：

```text
你的仓库/Templates/
```

如果你使用 Obsidian 官方 Templates 插件：

1. 设置 → 核心插件 → Templates
2. 打开 Templates
3. 指定模板文件夹为 `Templates`

## 三、最推荐的公式规范

### 行内公式

```latex
$f(x)=x^2+1$
```

### 重要公式

```latex
$$
f(x)=x^2+1
$$
```

### 多步推导

```latex
$$
\begin{aligned}
f(x)
&=x^2+2x+1\\
&=(x+1)^2
\end{aligned}
$$
```

### 分数

普通：

```latex
\frac{a}{b}
```

教材式大分数：

```latex
\dfrac{a}{b}
```

### 积分

推荐：

```latex
\int f(x)\,\mathrm{d}x
```

### 三角函数

推荐：

```latex
\sin x,\quad \cos x,\quad \tan x
```

不要写：

```latex
sin x, cos x, tan x
```

### 自动括号

推荐：

```latex
\left(\frac{a}{b}\right)
```

## 四、为什么不强制替换 MathJax 数学字体

Obsidian 的数学公式由 MathJax 渲染，本身就采用接近 TeX / LaTeX 的数学排版逻辑。

如果直接用 CSS 强制把数学公式改成普通文本字体，例如：

```css
font-family: "Times New Roman";
```

很容易造成：

- 积分号比例异常
- 根号变形
- 大括号尺寸异常
- 希腊字母风格不统一
- 上下标位置不自然

因此本 CSS 只调整字号、公式间距和阅读节奏，不破坏 MathJax 的数学字体系统。

## 五、建议的考研数学笔记层级

```text
一级标题：章节
二级标题：知识点
三级标题：题型 / 方法

正文：定义、理解
Callout：定理 / 结论 / 易错点
$$...$$：核心公式
aligned：完整推导
```

## 六、兼容性

设计目标：

- Obsidian 阅读模式
- Obsidian Live Preview
- MathJax
- 常规浅色 / 深色主题
- Obsidian PDF 导出

如某些第三方主题对公式字号进行了强制覆盖，可适当把 CSS 中：

```css
font-size: 1.10em;
```

调整为：

```css
font-size: 1.14em;
```

建议不要超过 `1.18em`，否则公式会明显大于教材正文。