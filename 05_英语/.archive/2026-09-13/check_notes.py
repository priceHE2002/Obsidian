"""对新建的英国笔记做静态体检：
  1. mermaid 块内引号/方括号是否配对
  2. callout 形如 > [!x] 的下一行是否以 '> ' 续写（Obsidian 渲染要求）
  3. <details>/<summary> 是否配对
  4. [[wiki 链接]] 是否能落到仓库里真实存在的文件
"""
import os
import re
import glob

ROOT = "/Users/heyuhang/Documents/Obsidian Vault"
FILES = sorted(glob.glob(f"{ROOT}/05_英语/英国政治背景/*.md"))

# 仓库内所有文件名（不含扩展名）
basenames = {}
for p in glob.glob(f"{ROOT}/**/*.md", recursive=True):
    b = os.path.splitext(os.path.basename(p))[0]
    basenames.setdefault(b, []).append(p)
relpaths = {os.path.relpath(p, ROOT)[:-3]: p for p in glob.glob(f"{ROOT}/**/*.md", recursive=True)}

bad = 0
for f in FILES:
    t = open(f, encoding="utf-8").read()
    name = os.path.basename(f)
    print("=" * 8, name, len(t), "字符")

    # 1. mermaid 块
    for i, m in enumerate(re.findall(r"```mermaid\n(.*?)```", t, re.S), 1):
        q = m.count('"')
        ob, cb = m.count("["), m.count("]")
        ok = q % 2 == 0 and ob == cb
        if not ok:
            bad += 1
            print("  [mermaid#%d] 引号 %d / 方括号 %d,%d —— 不配对" % (i, q, ob, cb))
        # 每个节点标签内部不应有裸 ( ) 
        for lab in re.findall(r'\["([^"]*)"\]', m):
            if "(" in lab or ")" in lab:
                bad += 1
                print("  [mermaid#%d] 标签含裸括号: %s" % (i, lab))
    print("  mermaid 块 %d 个" % len(re.findall(r"```mermaid", t)))

    # 2. callout 续行
    lines = t.split("\n")
    for i, l in enumerate(lines):
        if re.match(r"^> \[!\w+\]", l):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not nxt.startswith(">"):
                bad += 1
                print("  [callout] 第 %d 行 %s 下一行未续 '>'：%r" % (i + 1, l, nxt[:60]))

    # 3. details
    d1 = len(re.findall(r"<details>", t))
    d2 = len(re.findall(r"</details>", t))
    s1 = len(re.findall(r"<summary>", t))
    s2 = len(re.findall(r"</summary>", t))
    if d1 != d2 or s1 != s2 or d1 != s1:
        bad += 1
        print("  [details] 不配对 %d/%d %d/%d" % (d1, d2, s1, s2))
    print("  details 块 %d 个" % d1)

    # 4. wiki 链接
    links = re.findall(r"\[\[([^\]]+)\]\]", t)
    for lk in links:
        target = lk.split("|")[0].strip()
        tgt = target[:-3] if target.endswith(".md") else target
        if "/" in tgt:
            exists = tgt in relpaths
        else:
            exists = tgt in basenames
        if not exists:
            bad += 1
            print("  [link] 指向不存在的文件：[[%s]]" % lk)
        elif "/" not in tgt and len(basenames[tgt]) > 1:
            bad += 1
            print("  [link] 歧义链接（%d 个同名文件）：[[%s]]" % (len(basenames[tgt]), lk))
    print("  wiki 链接 %d 条" % len(links))

print()
print("问题总数：%d" % bad)
