# -*- coding: utf-8 -*-
"""fig5.py — 复现论文图5 分位系数谱（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig5.py  （依赖 common_data.py；输出至 figs_v4/）
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

import statsmodels.formula.api as smf
taus=[0.1,0.25,0.5,0.75,0.9]; cP,seP,cN,seN=[],[],[],[]
for t in taus:
    q=smf.quantreg("mkt_re ~ GPR5_P + GPR5_N + bsi + ei + log_amt + turn",df).fit(q=t,max_iter=5000)
    cP.append(q.params.GPR5_P); seP.append(q.bse.GPR5_P); cN.append(q.params.GPR5_N); seN.append(q.bse.GPR5_N)
fig,ax=plt.subplots(figsize=(8.6,5)); ax.axhline(0,color="gray",lw=0.8)
ax.errorbar(taus,cP,yerr=[1.645*s for s in seP],fmt="o-",color=RED,lw=2,ms=6,capsize=4,label="Policy component")
ax.errorbar(taus,cN,yerr=[1.645*s for s in seN],fmt="s-",color=TEAL,lw=2,ms=6,capsize=4,label="News component")
ax.set_xlabel("Quantile τ"); ax.set_ylabel("Coefficient on daily RE return")
ax.legend(frameon=False,loc="lower left")
plt.tight_layout(); plt.savefig(OUT+"fig4_quantile.png"); print("fig5 saved")
