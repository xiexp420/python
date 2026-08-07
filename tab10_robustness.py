# -*- coding: utf-8 -*-
"""tab10_robustness.py — 复现论文表10 稳健性检验（窗口敏感性/原始日度/子样本/GARCH-EVT）
运行：python3 tab10_robustness.py （依赖 common_data.py）
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.api as sm
from common_data import build_daily_v5
df = build_daily_v5()
Xc = ["bsi","ei","log_amt","turn","xq_sent","xq_attn"]

print("== 窗口敏感性（3日/10日移动平均） ==")
for w in [3, 10]:
    d = df.copy()
    d["P"] = d.GPR_Policy_news.rolling(w).mean(); d["N"] = d.GPR_News_total.rolling(w).mean()
    d = d.dropna(subset=["P","N"])
    X = sm.add_constant(d[["P","N"]+Xc])
    m = sm.OLS(d.mkt_re, X).fit(cov_type="HAC", cov_kwds={"maxlags":5})
    print(f"  {w}日窗口: GPR_P={m.params.P:+.5f}(p={m.pvalues.P:.3f})  xq_sent={m.params.xq_sent:+.5f}(p={m.pvalues.xq_sent:.4f})  R2={m.rsquared:.4f}")

print("== 原始日度数据（不平滑） ==")
d = df.copy(); d["P"] = d.GPR_Policy_news; d["N"] = d.GPR_News_total
X = sm.add_constant(d[["P","N"]+Xc])
m = sm.OLS(d.mkt_re, X).fit(cov_type="HAC", cov_kwds={"maxlags":5})
print(f"  GPR_P={m.params.P:+.5f}(p={m.pvalues.P:.3f})  xq_sent={m.params.xq_sent:+.5f}(p={m.pvalues.xq_sent:.4f})  R2={m.rsquared:.4f}")

print("== 剔除两轮管制事件窗的子样本 ==")
mask = ~(((df.index>="2025-04-01")&(df.index<="2025-05-15")) | ((df.index>="2025-10-08")&(df.index<="2025-11-15")))
ds = df[mask]
X = sm.add_constant(ds[["GPR5_P","GPR5_N"]+Xc])
m = sm.OLS(ds.mkt_re, X).fit(cov_type="HAC", cov_kwds={"maxlags":5})
print(f"  n={len(ds)}: GPR5_P={m.params.GPR5_P:+.5f}(p={m.pvalues.GPR5_P:.3f})  xq_sent={m.params.xq_sent:+.5f}(p={m.pvalues.xq_sent:.4f})  R2={m.rsquared:.4f}")

print("== GARCH-EVT VaR回测 ==")
from arch import arch_model
from scipy.stats import genpareto, chi2
r = df.mkt_re*100
gj = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
mu = gj.params["mu"]; sig = gj.conditional_volatility
z = (r-mu)/sig
u = np.percentile(z, 10)
ex = -z[z<u]+u
c, loc, scale = genpareto.fit(ex, floc=0)
def evt_var(alpha):
    qz = u-(scale/c)*((alpha/0.10)**(-c)-1)
    return mu+sig*qz
for alpha in [0.05, 0.01]:
    var = evt_var(alpha); hits = (r.values<var.values).astype(int); n = len(hits); x = int(hits.sum())
    lr = -2*np.log((alpha**x*(1-alpha)**(n-x))/((x/n)**x*(1-x/n)**(n-x)))
    print(f"  GARCH-EVT {int((1-alpha)*100)}%: x={x} rate={x/n:.4f} Kupiec_p={1-chi2.cdf(lr,1):.4f}")
print(f"  GPD形状参数xi={c:.4f}, 尺度={scale:.4f}")
