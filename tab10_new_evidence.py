# -*- coding: utf-8 -*-
"""tab10_new_evidence.py — 复现论文新增稳健性与经济价值证据
包含：(1)滞后一期股吧情绪 (2)上下游DID及500次安慰剂 (3)VaR下行保护经济价值
运行：python3 tab10_new_evidence.py （依赖 common_data.py）
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.api as sm
import scipy.stats as st
from common_data import build_daily_v5, XCOLS_V5
df = build_daily_v5()

print("="*50)
print("(1) 滞后一期股吧情绪（表3滞后列）")
Xl = sm.add_constant(df[XCOLS_V5].shift(1)).dropna()
ml = sm.OLS(df.mkt_re[Xl.index], Xl).fit(cov_type="HAC", cov_kwds={"maxlags":5})
print("  滞后xq_sent: %.5f (p=%.4f) | 滞后xq_attn: %.5f (p=%.4f)" %
      (ml.params.xq_sent, ml.pvalues.xq_sent, ml.params.xq_attn, ml.pvalues.xq_attn))

print("="*50)
print("(2) 上下游DID及安慰剂检验（表8）")
# 与 tab11_diagnostics.py 完全同口径：双估计窗[130/60]、各窗独立500次安慰剂、
# 伪事件抽样下界 w+1（保证伪估计窗长度完整），种子42。文档表8/表11数值以本口径为准。
EVENTS = {"2025-04-04": [130, 60], "2025-10-09": [130, 60]}
def did_one(df, loc, estwin):
    est = df.iloc[loc-estwin:loc-10]
    bu, au = np.polyfit(est.mkt_all, est.mkt_up, 1)
    bd, ad = np.polyfit(est.mkt_all, est.mkt_down, 1)
    cu = np.cumsum((df.mkt_up.iloc[loc:loc+16]-(au+bu*df.mkt_all.iloc[loc:loc+16])).values)
    cd = np.cumsum((df.mkt_down.iloc[loc:loc+16]-(ad+bd*df.mkt_all.iloc[loc:loc+16])).values)
    return cu[15]-cd[15]
rng = np.random.default_rng(42); n = len(df)
for ev, wins in EVENTS.items():
    e = pd.Timestamp(ev)
    loc = df.index.get_indexer([e], method="nearest")[0]
    for w in wins:
        dids = np.array([did_one(df, int(rng.integers(w + 1, n - 16)), w) for _ in range(500)])
        d = did_one(df, loc, w)
        p = (np.abs(dids) >= abs(d)).mean()
        print("  %s 估计窗[-%d,-11]: DID=%.4f, 安慰剂p=%.3f (500次)" % (ev, w, d, p))

print("="*50)
print("(3) VaR下行保护经济价值")
oos = pd.read_pickle("/mnt/agents/output/build/oos_v4.pkl")
from scipy.stats import t as stu
idx = oos["sig"].index; ry = df.mkt_re.reindex(idx).values
nu = oos["nu"].values; mu = oos["mu"].values/100; sg = oos["sig"].values/100
q = stu.ppf(0.05, nu)/np.sqrt(nu/(nu-2)); var = mu+sg*q
var_s = pd.Series(var, index=idx)
thr = var_s.expanding().quantile(0.10).shift(1)
high_risk = (var_s < thr).shift(1).fillna(False).values
loss_bh = ry[high_risk]; loss_strat = 0.5*ry[high_risk]
print("  高风险预警日 n=%d" % high_risk.sum())
print("  左尾(最差10%%)日均损失: 买入持有=%.4f%%, 降仓策略=%.4f%%" %
      (np.percentile(loss_bh, 10)*100, np.percentile(loss_strat, 10)*100))
