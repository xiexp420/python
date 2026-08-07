# -*- coding: utf-8 -*-
"""tab11_diagnostics.py — 复现论文表11 统计诊断与补充稳健性
内容：(1)ADF单位根检验 (2)Granger因果检验 (3)BH-FDR多重检验校正
     (4)Newey-West滞后阶敏感性 (5)收益1%缩尾稳健性
     (6)2025年10月事件估计窗污染检验（短窗[-60,-11] vs 基准窗[-130,-11]，500次安慰剂DID）
运行：python3 tab11_diagnostics.py （依赖 common_data.py）
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.multitest import multipletests
from common_data import build_daily_v5, XCOLS_V5
df = build_daily_v5()

print("="*60)
print("(1) ADF单位根检验（含常数项，AIC选滞后）")
rows = []
for v, name in [("mkt_re","板块收益"), ("GPR5_P","政策型分量"), ("GPR5_N","舆情型分量"),
                ("xq_sent","股吧净情绪"), ("xq_attn","股吧关注度"), ("bsi","情绪平衡度"),
                ("ei","情绪强度"), ("heat","新闻热度")]:
    s = df[v].dropna()
    stat, p, lags, nobs = adfuller(s, regression="c", autolag="AIC")[:4]
    rows.append((name, round(stat, 3), round(p, 4), lags))
    print(f"  {name:8s}: ADF={stat:7.3f}, p={p:.4f}, 滞后={lags}")
print("  → 全部序列在1%水平拒绝单位根原假设" if all(r[2] < 0.01 for r in rows) else "  → 注意存在非平稳序列")

print("="*60)
print("(1b) KPSS平稳性检验（原假设：平稳）——针对边界组变量")
from statsmodels.tsa.stattools import kpss as _kpss
for v, name in [("GPR5_N","舆情型分量"), ("ei","情绪强度"), ("heat","新闻热度"), ("xq_attn","股吧关注度")]:
    s2 = df[v].dropna()
    stat, p, _, _ = _kpss(s2, regression="c", nlags="auto")
    print(f"  {name:8s}: KPSS={stat:6.3f}, p={p:.4f}")

print("="*60)
print("(2) Granger因果检验：股吧净情绪 × 板块收益（滞后1-3阶）")
gdat = df[["mkt_re", "xq_sent"]].dropna()
for lag in [1, 2, 3]:
    r1 = grangercausalitytests(gdat[["mkt_re", "xq_sent"]], maxlag=lag, verbose=False)
    r2 = grangercausalitytests(gdat[["xq_sent", "mkt_re"]], maxlag=lag, verbose=False)
    p1 = r1[lag][0]["ssr_ftest"][1]
    p2 = r2[lag][0]["ssr_ftest"][1]
    print(f"  滞后{lag}: xq_sent→r  p={p1:.4f} | r→xq_sent  p={p2:.4f}")
print("  注：若r→xq_sent显著而反向不显著，提示同步反向因果主导；"
      "正文同期效应按同时性关系解释，不做因果断言。")

print("="*60)
print("(3) BH-FDR多重检验校正（基准回归式(10)全部解释变量）")
X = sm.add_constant(df[XCOLS_V5])
m = sm.OLS(df.mkt_re, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
pv = m.pvalues.drop("const")
rej, p_adj, _, _ = multipletests(pv.values, alpha=0.05, method="fdr_bh")
for name, p0, pa, rj in zip(pv.index, pv.values, p_adj, rej):
    print(f"  {name:10s}: 原始p={p0:.4f} → FDR校正p={pa:.4f} {'显著' if rj else '不显著'}")

print("="*60)
print("(4) Newey-West滞后阶敏感性（3/5/10阶）")
for L in [3, 5, 10]:
    mm = sm.OLS(df.mkt_re, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    print(f"  NW滞后{L}: xq_sent={mm.params.xq_sent:.5f} (p={mm.pvalues.xq_sent:.4f}), "
          f"GPR5_P={mm.params.GPR5_P:.6f} (p={mm.pvalues.GPR5_P:.4f})")

print("="*60)
print("(5) 收益1%双边缩尾稳健性")
lo, hi = df.mkt_re.quantile([0.01, 0.99])
rw = df.mkt_re.clip(lo, hi)
mw = sm.OLS(rw, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"  缩尾后: xq_sent={mw.params.xq_sent:.5f} (p={mw.pvalues.xq_sent:.4f}), "
      f"GPR5_P={mw.params.GPR5_P:.6f} (p={mw.pvalues.GPR5_P:.4f}), N={int(mw.nobs)}")

print("="*60)
print("(6) 2025年10月事件估计窗污染检验")
EVENTS = {"2025-04-04": [130, 60], "2025-10-09": [130, 60]}
def did_one(df, loc, estwin):
    est = df.iloc[loc - estwin:loc - 10]
    bu, au = np.polyfit(est.mkt_all, est.mkt_up, 1)
    bd, ad = np.polyfit(est.mkt_all, est.mkt_down, 1)
    cu = np.cumsum((df.mkt_up.iloc[loc:loc + 16] - (au + bu * df.mkt_all.iloc[loc:loc + 16])).values)
    cd = np.cumsum((df.mkt_down.iloc[loc:loc + 16] - (ad + bd * df.mkt_all.iloc[loc:loc + 16])).values)
    return cu[15] - cd[15]
rng = np.random.default_rng(42); n = len(df)
for ev, wins in EVENTS.items():
    e = pd.Timestamp(ev)
    loc = df.index.get_indexer([e], method="nearest")[0]
    for w in wins:
        dids = np.array([did_one(df, int(rng.integers(w + 1, n - 16)), w) for _ in range(500)])
        d = did_one(df, loc, w)
        p = (np.abs(dids) >= abs(d)).mean()
        print(f"  {ev} 估计窗[-{w},-11]: DID={d:.4f}, 安慰剂p={p:.3f} (500次)")
print("  注：2025-10-09事件的[-130,-11]估计窗覆盖4月事件窗，短窗[-60,-11]不受污染；"
      "两窗结论一致方可确认DID不显著非估计窗偏误所致。")
