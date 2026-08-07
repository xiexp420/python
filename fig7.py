# -*- coding: utf-8 -*-
"""fig7.py — 复现论文图7 GJR-GARCH-t条件波动率（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig7.py  （依赖 common_data.py；输出至 figs_v4/）
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

from arch import arch_model
r=df.mkt_re*100
gj=arch_model(r,vol="GARCH",p=1,o=1,q=1,dist="t").fit(disp="off")
cv=gj.conditional_volatility
fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(df.index,cv,color=RED,lw=1.0,label="GJR-GARCH(1,1,1)-t")
for d_ in df.index[df.GPR_Policy_news>0]:
    ax.axvline(d_,color=TEAL,alpha=0.12,lw=0.8)
ax.plot([],[],color=TEAL,alpha=0.4,lw=2,label="Policy-news days")
ax.legend(frameon=False,fontsize=9); ax.set_ylabel("Conditional volatility (%)")
plt.tight_layout(); plt.savefig(OUT+"fig6_garch.png"); print("fig7 saved")
