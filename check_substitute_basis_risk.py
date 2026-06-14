"""
check_substitute_basis_risk.py
==============================

Quantifies the basis risk of swapping US ETFs for non-US-incorporated
single-stock substitutes (for non-US-person estate-tax planning).

Pairs analyzed:
  GDX  ↔ AEM   (Agnico Eagle Mines, Canada)
  XLE  ↔ SHEL  (Shell, UK)
  QQQ  ↔ TSM   (Taiwan Semiconductor, Taiwan)

For each pair, over three lookback windows (full / 5y / 2y), reports:
  - Daily return correlation
  - Daily return beta (slope of substitute vs original)
  - 20-day forward log return correlation (matches rotation horizon)
  - Tracking error (annualised stdev of residual after de-betaing)
  - Annualised volatility of each
  - Annualised return of each

A lower correlation / higher tracking error / very different vol means the
substitute is a poorer proxy for the original. Models trained on the
original ETF would predict signals less accurately for the substitute.

Usage:
  python check_substitute_basis_risk.py
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PAIRS = [
    ("GDX",  "AEM",  "Gold miners ETF",  "Agnico Eagle Mines (Canada)"),
    ("XLE",  "SHEL", "Energy sector ETF", "Shell (UK)"),
    ("QQQ",  "TSM",  "Nasdaq-100 ETF",   "Taiwan Semiconductor (Taiwan)"),
]

START_DATE   = "2010-01-01"
FORWARD_DAYS = 20


def download(ticker):
    df = yf.Ticker(ticker).history(
        start=START_DATE,
        end=datetime.today().strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": ticker})
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def stats(joined, original, substitute, label):
    daily_o = np.log(joined[original] / joined[original].shift(1)).dropna()
    daily_s = np.log(joined[substitute] / joined[substitute].shift(1)).dropna()
    fwd_o   = np.log(joined[original].shift(-FORWARD_DAYS) / joined[original])
    fwd_s   = np.log(joined[substitute].shift(-FORWARD_DAYS) / joined[substitute])

    aligned = pd.concat([daily_o, daily_s], axis=1).dropna()
    aligned.columns = ["o", "s"]
    daily_corr = aligned["o"].corr(aligned["s"])

    # Beta via OLS slope
    cov  = aligned["o"].cov(aligned["s"])
    var_o = aligned["o"].var()
    beta = cov / var_o if var_o > 0 else np.nan
    # Residual return: s - beta*o; annualised standard deviation = tracking error
    resid    = aligned["s"] - beta * aligned["o"]
    te_ann   = resid.std() * np.sqrt(252) * 100

    fwd_pair = pd.concat([fwd_o.rename("o"), fwd_s.rename("s")], axis=1).dropna()
    fwd_corr = fwd_pair["o"].corr(fwd_pair["s"]) if len(fwd_pair) > 20 else np.nan

    ann_vol_o = aligned["o"].std() * np.sqrt(252) * 100
    ann_vol_s = aligned["s"].std() * np.sqrt(252) * 100
    ann_ret_o = aligned["o"].mean() * 252 * 100
    ann_ret_s = aligned["s"].mean() * 252 * 100

    return {
        "label":      label,
        "n":          len(aligned),
        "daily_corr": daily_corr,
        "beta":       beta,
        "fwd20_corr": fwd_corr,
        "te_ann_pct": te_ann,
        "ann_vol_o":  ann_vol_o,
        "ann_vol_s":  ann_vol_s,
        "ann_ret_o":  ann_ret_o,
        "ann_ret_s":  ann_ret_s,
    }


def print_pair(original, substitute, desc_o, desc_s, joined):
    print(f"\n{'═'*64}")
    print(f"  {original}  →  {substitute}")
    print(f"  ({desc_o}  →  {desc_s})")
    print(f"{'═'*64}")

    today = joined.index[-1]
    windows = [
        ("Full sample",   joined.copy()),
        ("Last 5 years",  joined[joined.index >= today - pd.Timedelta(days=5*365)]),
        ("Last 2 years",  joined[joined.index >= today - pd.Timedelta(days=2*365)]),
    ]

    print(f"  {'Window':<14} {'n':>5} {'Corr':>7} {'Beta':>6} "
          f"{'FwdCorr':>8} {'TE':>7} {'Vol O→S':>14} {'Ret O→S':>14}")
    print(f"  {'─'*14} {'─'*5} {'─'*7} {'─'*6} {'─'*8} {'─'*7} {'─'*14} {'─'*14}")
    for win_name, win_df in windows:
        if len(win_df) < 50:
            print(f"  {win_name:<14}  insufficient data")
            continue
        s = stats(win_df, original, substitute, win_name)
        print(f"  {s['label']:<14} {s['n']:>5} "
              f"{s['daily_corr']:>+6.3f} "
              f"{s['beta']:>+5.2f} "
              f"{s['fwd20_corr']:>+7.3f} "
              f"{s['te_ann_pct']:>5.1f}% "
              f"{s['ann_vol_o']:>5.1f}% → {s['ann_vol_s']:>4.1f}% "
              f"{s['ann_ret_o']:>+5.1f}% → {s['ann_ret_s']:>+4.1f}%")


def main():
    print(f"\n{'═'*64}")
    print(f"  ETF → Stock Substitute Basis Risk Analysis")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"{'═'*64}")
    print(f"  Metric guide:")
    print(f"    Corr        — daily return correlation (higher = better proxy)")
    print(f"    Beta        — substitute's sensitivity to original (1.0 = same)")
    print(f"    FwdCorr     — 20-day forward log return correlation (model horizon)")
    print(f"    TE          — annualised tracking error of residual (lower = better)")
    print(f"    Vol O→S     — annualised volatility of original → substitute")
    print(f"    Ret O→S     — annualised return of original → substitute")

    for original, substitute, desc_o, desc_s in PAIRS:
        print(f"\n  Downloading {original} and {substitute}...")
        try:
            df_o = download(original)
            df_s = download(substitute)
            joined = df_o.join(df_s, how="inner").dropna()
            if len(joined) < 100:
                print(f"  ⚠ Not enough overlapping data for {original}/{substitute}")
                continue
            print_pair(original, substitute, desc_o, desc_s, joined)
        except Exception as e:
            print(f"  ✗ Failed to compare {original}/{substitute}: {e}")

    print(f"\n{'═'*64}")
    print(f"  Interpretation guide:")
    print(f"    Daily Corr  > 0.7 → strong proxy")
    print(f"                0.5-0.7 → moderate (acceptable with awareness)")
    print(f"                < 0.5 → weak (single-stock noise likely dominates)")
    print(f"    Beta close to 1.0 means like-for-like move magnitude.")
    print(f"    TE > 30% annualised means heavy single-stock idiosyncrasy.")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
