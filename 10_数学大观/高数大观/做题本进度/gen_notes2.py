# -*- coding: utf-8 -*-
import json, os, re
from collections import defaultdict

rows = json.load(open('/tmp/dataset.json'))
VROOT = "/sessions/adoring-bold-dijkstra/mnt/Obsidian Vault/10_数学大观/高数大观/做题本进度"
QDIR = os.path.join(VROOT, "00_题目")
MDIR = os.path.join(VROOT, "01_模块")
CH_FOLDERS = {'1':'第01章 函数极限连续','2':'第02章 一元微分','3':'第03章 一元积分','4':'第04章 微分方程','5':'第05章 多元微分','6':'第06章 二重积分'}
os.makedirs(QDIR, exist_ok=True); os.makedirs(MDIR, exist_ok=True)

def q(s): return '"%s"' % str(s or '').replace('"', '\\"')

seen = set()
conflicts = 0
for r in rows:
    ch = r['chapter']; mk = r['mod_key']
    canonical = f"C{ch}-{r['id']}"          # 全局唯一：C章-原书编号
    folder = os.path.join(QDIR, CH_FOLDERS[ch])
    os.makedirs(folder, exist_ok=True)
    fname = f"{canonical}.md"
    fpath = os.path.join(folder, fname)
    if fpath in seen: conflicts += 1; continue
    seen.add(fpath)
    preview = (r.get('preview') or '').strip()
    src = r['source'] or '—'
    tag = re.sub(r'[·、：:，,。.()（）]', '', str(r['chapter_name'])).strip()[:14]
    L = []
    L += ["---","tags:",'  - "数学大观"','  - "数一强化"',f'  - "{tag}"',
          f"章: {ch}", f"章名: {q(r['chapter_name'])}", f"模块: {q(mk)}", f"模块名: {q(r['mod_name'])}",
          f"题号: {q(canonical)}", f"源题号: {q(r['id'])}", f"序: {r['gidx']}",
          f"题源: {q(src)}", f"题源大类: {q(r['大类'])}", f"重点: {str(bool(r['重点'])).lower()}",
          f"页码: {r['page']}", '状态: "未开始"', "下次复习:", "备注:", "---"]
    L.append(f"# {r['id']} · {r['mod_name']}")
    L.append("")
    if preview:
        L.append("> [!note]- 题干预览（自动提取，仅供参考）")
        L.append("> " + preview[:180])
        L.append(">")
    L.append(f"> 题源：{src}　|　页码：p.{r['page']}　|　模块：{mk} {r['mod_name']}")
    L += ["", "```dataviewjs",
          "const st = ['未开始','已做','已订正','已二刷','已掌握'];",
          "const col = {'未开始':'#9aa4b2','已做':'#5b8cff','已订正':'#34d399','已二刷':'#2dd4bf','已掌握':'#fbbf24'};",
          "const cur = dv.current(); const root = dv.el('div','');",
          "root.innerHTML = '<div style=\"display:flex;gap:6px;flex-wrap:wrap;margin:4px 0\">' + st.map(s => `<button data-s=\"${s}\" style=\"padding:3px 14px;font-size:12.5px;border-radius:14px;border:1px solid var(--background-modifier-border);cursor:pointer;background:var(--background-primary);color:var(--text-normal);${cur.状态===s?`background:${col[s]};color:#fff;border-color:${col[s]};`:''}\">${s}</button>`).join('') + '</div>';",
          "root.addEventListener('click', async e => { const b = e.target.closest('button[data-s]'); if (!b) return; const s = b.dataset.s; const f = dv.app.vault.getAbstractFileByPath(cur.file.path); await dv.app.fileManager.processFrontMatter(f, fm => { fm['状态'] = s; }); const btns = root.querySelectorAll('button[data-s]'); btns.forEach(x => { x.style.background='var(--background-primary)'; x.style.color='var(--text-normal)'; x.style.borderColor='var(--background-modifier-border)'; }); b.style.background=col[s]; b.style.color='#fff'; b.style.borderColor=col[s]; });",
          "```", "", f"→ 所属模块：[[C{ch}-{mk} {r['mod_name']}|{mk} {r['mod_name']}]]"]
    open(fpath, 'w', encoding='utf-8').write('\n'.join(L))

print("生成题目笔记:", len(seen), "冲突跳过的:", conflicts)
