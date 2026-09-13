"""在「考研英语一阅读真题」PDF 里定位英国题材语料。

复用 obsidian-vocab-card-book/scripts/mine_exam_sentences.py 的抽取逻辑，
按英国背景关键词扫描 2010–2025 全部 Text，输出：
  - uk_passages.json : 每篇命中数 + 命中关键词
  - uk_hits.tsv      : 逐条命中（年份/Text/关键词/句子）
"""
import json
import re
import sys

sys.path.insert(0, "/Users/heyuhang/.workbuddy/skills/obsidian-vocab-card-book/scripts")
from mine_exam_sentences import extract_passages, to_sentences  # noqa: E402

PDF = "/Users/heyuhang/Documents/Obsidian Vault/05_英语/考研英语一阅读真题_A4装订版.pdf"
OUT_DIR = "/Users/heyuhang/Documents/Obsidian Vault/05_英语/.archive/2026-09-13"

# 关键词 -> 标签（按背景框架分组）
GROUPS = {
    "制度": ["Parliament", "parliamentary", "House of Commons", "House of Lords",
             "MPs", "MP ", "constitutional monarchy", "monarch", "monarchy",
             "the Crown", "Magna Carta", "Westminster", "general election",
             "by-election", "constituenc", "cabinet"],
    "阶级/精英教育": ["working class", "middle class", "upper class", "class system",
                      "class-conscious", "Oxbridge", "Oxford", "Cambridge",
                      "public school", "elite", "elitism", "social mobility",
                      "grammar school", "Eton", "aristocracy", "establishment",
                      "privilege", "inequality"],
    "帝国/殖民": ["the Empire", "British Empire", "colonial", "colonialism",
                  "imperial", "imperialism", "Commonwealth", "legacy",
                  "India", "imperial past"],
    "福利/NHS": ["NHS", "National Health Service", "welfare state", "welfare",
                 "public service", "austerity", "public spending", "the dole",
                 "benefits"],
    "撒切尔/市场": ["Thatcher", "Thatcherism", "privatis", "privatiz",
                    "deregulat", "nationalis", "nationaliz", "free market",
                    "trade union", "strike"],
    "苏格兰/权力下放": ["Scotland", "Scottish", "Wales", "Welsh", "Northern Ireland",
                        "devolution", "independence referendum", "separatis",
                        "nationalist", "SNP"],
    "脱欧": ["Brexit", "European Union", "the EU", "Brussels", "referendum",
             "sovereignty", "euro", "eurosceptic"],
    "媒体/公共机构": ["BBC", "broadcaster", "broadsheet", "tabloid", "press",
                      "public service broadcasting", "the Guardian", "Fleet Street",
                      "newspaper"],
    "国家指称": ["Britain", "British", "the UK", "United Kingdom", "England",
                 "English ", "Briton", "Anglo"],
}

YEAR_FIRST = {}


def sentences_for(pdf):
    corpus = []
    for u in extract_passages(pdf):
        for s in to_sentences(u["raw"]):
            corpus.append({"s": s, "year": u["year"], "text": u["text"]})
    return corpus


def main():
    corpus = sentences_for(PDF)
    keep = [c for c in corpus if 2010 <= c["year"] <= 2025]
    print("正文页 %d，句子 %d" % (len({(c['year'], c['text']) for c in keep}), len(keep)))

    per_passage, hits = {}, []
    for c in keep:
        key = (c["year"], c["text"])
        for group, words in GROUPS.items():
            for w in words:
                pat = re.compile(r"\b" + re.escape(w).strip() + r"\b", re.I)
                m = pat.search(c["s"])
                if not m:
                    continue
                per_passage.setdefault(key, {}).setdefault("groups", set()).add(group)
                per_passage[key].setdefault("kw", set()).add(m.group(0).strip())
                per_passage[key]["n"] = per_passage[key].get("n", 0) + 1
                hits.append({"year": c["year"], "text": c["text"], "group": group,
                             "kw": m.group(0).strip(), "s": c["s"]})

    summary = []
    for (y, t), v in sorted(per_passage.items()):
        summary.append({"year": y, "text": t, "hits": v.get("n", 0),
                        "groups": sorted(v["groups"]),
                        "keywords": sorted(v["kw"])})
    summary.sort(key=lambda x: -x["hits"])

    json.dump(summary, open(f"{OUT_DIR}/uk_passages.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(f"{OUT_DIR}/uk_hits.tsv", "w", encoding="utf-8") as f:
        f.write("year\ttext\tgroup\tkw\tsentence\n")
        for h in sorted(hits, key=lambda x: (x["year"], x["text"])):
            f.write("%s\t%s\t%s\t%s\t%s\n" % (h["year"], h["text"], h["group"],
                                              h["kw"], h["s"]))

    print("\n== 命中篇目（按命中密度）==")
    for s in summary:
        if s["hits"] >= 3:
            print("%s Text %s | %2d 次 | %s | %s"
                  % (s["year"], s["text"], s["hits"], "/".join(s["groups"]),
                     ", ".join(s["keywords"][:14])))
    print("\n总命中篇目 %d，总命中 %d 条" % (len(summary), len(hits)))


if __name__ == "__main__":
    main()
