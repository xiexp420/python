# -*- coding: utf-8 -*-
"""fig4.py — 复现论文图4 60日滚动相关（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig4.py  （依赖 common_data.py；输出至 figs_v4/）
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

rc=pd.DataFrame({"RE returns × Policy shocks":df.mkt_re.rolling(60).corr(df.GPR_Policy_news),
                 "RE returns × BSI":df.mkt_re.rolling(60).corr(df.bsi)})
fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(rc.index,rc.iloc[:,1],color=TEAL,lw=1.1,alpha=0.8,label=rc.columns[1])
ax.plot(rc.index,rc.iloc[:,0],color="#0f4c4c",lw=1.3,label=rc.columns[0])
ax.axhline(0,color="gray",lw=0.8); ax.legend(frameon=False,fontsize=9)
ax.set_ylabel("60-day rolling correlation")
plt.tight_layout(); plt.savefig(OUT+"fig3_rolling.png"); print("fig4 saved")
