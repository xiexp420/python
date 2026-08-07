# -*- coding: utf-8 -*-
"""merge_supp_news.py — 合并2025年1—7月补充新闻语料并重建分析面板
补充来源：/mnt/agents/upload/202*行业新闻.xls（14个文件，13,906条行业新闻）
处理：日期解析→与原始语料标题去重→稀土关键词过滤→BERT情感打分→并入原始语料
输出：
  /mnt/agents/upload/rare_earth_news_2020_2026_augmented.csv  扩展语料
  /mnt/agents/output/build/news_supp.pkl                      补充语料（含情感得分）
。
"""
import pandas as pd, numpy as np, glob, re, os

UP = "/mnt/agents/upload/"
BD = "/mnt/agents/output/build/"

RE_KW = ['稀土','永磁','钕','镨','镝','铽','钇','镧','铈','钐','铕','钆','钬','铒','铥','镱','镥','钪','磁材','磁体']
POLICY_KW = ["出口管制","出口限制","出口禁令","出口配额","出口许可","出口审批","出口审查","出口收紧",
             "出口调控","出口管理","出口监管","战略资源","关键矿产","断供","禁运","反制","配额","指标","整合",
             "稀土价格","开采","冶炼","收储","稀土集团","稀土磁体","镨钕","镝","铽","永磁","磁材"]

def load_supp():
    frames = []
    for f in sorted(glob.glob(UP + "202*行业新闻.xls")):
        df = pd.read_excel(f)
        df["dt"] = pd.to_datetime(df["新闻日期"], errors="coerce")
        frames.append(df[["dt", "新闻标题"]].dropna())
    sup = pd.concat(frames).rename(columns={"新闻标题": "title"})
    # 与原始语料标题去重
    orig = pd.read_csv(UP + "rare_earth_news_2020_2026.csv")
    sup = sup[~sup.title.fillna("").isin(set(orig.title.fillna("")))].copy()
    # 稀土主题过滤（与原语料主题口径一致）
    sup = sup[sup.title.fillna("").apply(lambda t: any(k in t for k in RE_KW))].copy()
    sup["date"] = sup.dt.dt.normalize()
    return sup.reset_index(drop=True)

def score_sentiment(titles):
    """BERT中文金融情感打分；失败则词典退化"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        name = "yiyanghkust/finbert-tone-chinese"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
        lab = {0: -1, 1: 0, 2: 1}  # Negative/Neutral/Positive
        out = []
        with torch.no_grad():
            for i in range(0, len(titles), 32):
                enc = tok(list(titles[i:i+32]), padding=True, truncation=True,
                          max_length=64, return_tensors="pt")
                pred = mdl(**enc).logits.argmax(-1).tolist()
                out += [lab[p] for p in pred]
        return np.array(out), "finbert-tone-chinese"
    except Exception as e:
        print("BERT不可用，退化为词典打分:", e)
        POS = ["上涨","涨停","大涨","回暖","突破","走强","活跃","利好","增长","批准","许可","回升","反弹","新高"]
        NEG = ["下滑","下跌","冷清","管制","禁令","短缺","停产","受阻","下跌","危机","收紧","断供","冲击"]
        def s(t):
            p = sum(k in t for k in POS); n = sum(k in t for k in NEG)
            return 1 if p > n else (-1 if n > p else 0)
        return titles.fillna("").map(s).values, "lexicon"

def build_augmented():
    sup = load_supp()
    print("补充稀土新闻:", len(sup))
    sup["s"], engine = score_sentiment(sup.title)
    print("情感打分引擎:", engine, "| 分布:",
          {"POS": int((sup.s==1).sum()), "NEU": int((sup.s==0).sum()), "NEG": int((sup.s==-1).sum())})
    sup["hit"] = sup.title.fillna("").apply(lambda t: any(k in t for k in POLICY_KW))
    sup.to_pickle(BD + "news_supp.pkl")
    # 合并进原始语料（与原csv同构：title/sentiment/publish_time）
    orig = pd.read_csv(UP + "rare_earth_news_2020_2026.csv")
    add = pd.DataFrame({
        "uuid": ["supp_%05d" % i for i in range(len(sup))],
        "title": sup.title,
        "sentiment": sup.s.map({1: "POS", 0: "NEU", -1: "NEG"}),
        "content": "", "host_source": "", "publish_source": "补充行业新闻",
        "publish_time": (sup.dt.astype("int64") // 10**9),
        "display_time": (sup.dt.astype("int64") // 10**9),
        "url": "", "importance": 1, "concepts": "稀土永磁", "industries": "",
        "fetch_time": "2026-08-01",
    })
    aug = pd.concat([orig, add], ignore_index=True)
    aug.to_csv(BD + "rare_earth_news_2020_2026_augmented.csv", index=False)
    print("扩展语料:", len(orig), "+", len(add), "=", len(aug))
    m = sup.date.dt.to_period("M").value_counts().sort_index()
    print("补充语料月度分布:\n", m.to_string())

if __name__ == "__main__":
    build_augmented()
