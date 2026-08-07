# -*- coding: utf-8 -*-
"""fig3.py — 复现论文图3 GPR-REE双分量指数（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig3.py  （依赖 common_data.py；输出至 figs_v4/）
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

fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(df.index,df.GPR5_N,color=TEAL,lw=1.2,label="News component (5d MA)")
ax2=ax.twinx(); ax2.plot(df.index,df.GPR5_P,color=RED,lw=1.4,label="Policy component (5d MA)"); ax2.grid(False)
ax.set_ylabel("News component"); ax2.set_ylabel("Policy component")
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,frameon=False,loc="upper left",fontsize=9)
plt.tight_layout(); plt.savefig(OUT+"fig2_index.png"); print("fig3 saved")
