# -*- coding: utf-8 -*-
"""fig6.py — 复现论文图6 局部投影脉冲响应（样本2022-04-18至2026-05-29，997个交易日）
运行：python3 fig6.py  （依赖 common_data.py；输出至 figs_v4/）
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

import statsmodels.api as sm
lpP,lpN=[],[]
for h in range(0,13):
    yh=df.mkt_re.shift(-h)
    X=sm.add_constant(pd.DataFrame({"GPR5_P":df.GPR5_P,"GPR5_N":df.GPR5_N,"bsi":df.bsi,"ei":df.ei,
        "log_amt":df.log_amt,"turn":df.turn,"y_l1":df.mkt_re.shift(1),"P_l1":df.GPR5_P.shift(1),"N_l1":df.GPR5_N.shift(1)}))
    d=pd.concat([yh.rename("y"),X],axis=1).dropna()
    m=sm.OLS(d.y,d.drop(columns="y")).fit(cov_type="HAC",cov_kwds={"maxlags":5})
    lpP.append((m.params.GPR5_P,1.645*m.bse.GPR5_P)); lpN.append((m.params.GPR5_N,1.645*m.bse.GPR5_N))
h=list(range(13))
fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,lp,c,t in [(axs[0],lpP,RED,"Policy component"),(axs[1],lpN,TEAL,"News component")]:
    b=[x[0] for x in lp]; e=[x[1] for x in lp]
    ax.axhline(0,color="gray",lw=0.8)
    ax.fill_between(h,[bb-ee for bb,ee in lp],[bb+ee for bb,ee in lp],color=c,alpha=0.18)
    ax.plot(h,b,"o-",color=c,lw=2,ms=5); ax.set_title(t); ax.set_xlabel("Trading days after shock (h)")
axs[0].set_ylabel("LP-IRF coefficient")
plt.tight_layout(); plt.savefig(OUT+"fig5_lpirf.png"); print("fig6 saved")
