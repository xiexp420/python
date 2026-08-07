# -*- coding: utf-8 -*-
"""fig2.py — 复现论文图2 舆情新闻流量与情绪构成（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig2.py  （依赖 common_data.py；输出至 figs_v4/）
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common_data import build_daily_v5 as build_daily
plt.rcParams.update({"figure.dpi":150,"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.3})
TEAL,RED="#1b6e6e","#c0392b"
df = build_daily()
OUT="/mnt/agents/output/figs_v4/"

from common_data import load_news
news=load_news()
m=news.set_index("date").resample("ME")["s"].agg(["size",lambda x:(x==1).mean(),lambda x:(x==-1).mean()])
m.columns=["n","pos","neg"]
fig,ax=plt.subplots(figsize=(10,4.6))
ax.bar(m.index,m.n,width=20,color=TEAL,alpha=0.75,label="News flow")
ax2=ax.twinx(); ax2.plot(m.index,m.pos,color=RED,lw=1.5,label="POS share")
ax2.plot(m.index,m.neg,color="gray",lw=1.5,label="NEG share"); ax2.grid(False)
ax.set_ylabel("Monthly news count"); ax2.set_ylabel("Sentiment share")
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,frameon=False,loc="upper left",fontsize=9)
plt.tight_layout(); plt.savefig(OUT+"fig13_news.png"); print("fig2 saved")
