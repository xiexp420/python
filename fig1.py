# -*- coding: utf-8 -*-
"""fig1.py — 复现论文图1 稀土板块与市场基准累计净值（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig1.py  （依赖 common_data.py；输出至 figs_v4/）
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

nav=(1+df[["mkt_re","mkt_all"]]).cumprod()
fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(nav.index,nav.mkt_re,color=RED,lw=1.8,label="Rare-earth sector")
ax.plot(nav.index,nav.mkt_all,color="gray",lw=1.2,label="Market benchmark")
for d_ in ["2025-04-04","2025-10-09"]:
    ax.axvline(pd.Timestamp(d_),color=TEAL,ls="--",lw=1,alpha=0.7)
ax.legend(frameon=False,loc="upper left"); ax.set_ylabel("Cumulative NAV")
plt.tight_layout(); plt.savefig(OUT+"fig1_nav.png"); print("fig1 saved")
