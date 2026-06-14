"""
find_gdx_diversifiers_stocks.py
================================

Stock-based version of the GDX diversifier analysis.

Why a separate version?
-----------------------
US-domiciled ETFs (UUP, USDU, even GDX itself) are US-situs assets and
expose non-US holders to US estate tax (up to 40% on amounts above the
$60k non-resident exemption, unless modified by treaty). Individual stocks
of non-US-incorporated companies — even when listed on NYSE — are NOT
US-situs and have NO US estate tax exposure.

This script runs the same diversifier analysis (conditional alpha when GDX
is losing/flat/weak, vol-weighted score, correlation) but on a curated
universe of ~35 non-US-incorporated single stocks spanning financials,
tech, defensive consumer, pharma, industrials, and commodities — across
Canada, Europe, Japan, and emerging markets.

Excludes:
  - All US-incorporated stocks (e.g., NEM, AAPL, JPM) — US-situs, estate tax
  - US-domiciled ETFs — covered by the ETF script
  - Gold miners (AEM, GOLD, WPM, FNV) — correlated with GDX by construction;
    not useful as diversifiers

Methodology, scoring, and output format match find_gdx_diversifiers.py.

Usage:
  python find_gdx_diversifiers_stocks.py

Outputs:
  - diversifier_analysis_stocks.csv  (full results)
  - console:  ranked top tables for each GDX regime
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
START_DATE          = "2010-01-01"
FORWARD_DAYS        = 20
LOSING_THRESHOLD    = -0.03
FLAT_LOW            = -0.03
FLAT_HIGH           = +0.03
WEAK_THRESHOLD      = +0.02
MIN_AVG_DOLLAR_VOL  = 25_000_000  # $25M ADV — same as ETF version
OUT_CSV             = "diversifier_analysis_stocks.csv"

# ── Candidate universe — non-US-incorporated stocks ──────────────────────────
# Each entry: (NYSE ticker, country of incorporation, description)
# Country matters for US estate tax — only non-US qualifies for the carve-out.
# All entries here are non-US-incorporated even when NYSE-listed.
CANDIDATES = [
    # Canadian financials (rate-sensitive — may benefit when GDX struggles)
    ("TD",    "Canada",   "Toronto-Dominion Bank"),
    ("RY",    "Canada",   "Royal Bank of Canada"),
    ("BNS",   "Canada",   "Bank of Nova Scotia"),
    ("BMO",   "Canada",   "Bank of Montreal"),
    ("CM",    "Canada",   "CIBC"),
    ("MFC",   "Canada",   "Manulife Financial"),

    # European banks (similar rate-sensitive logic)
    ("SAN",   "Spain",    "Banco Santander"),
    ("BCS",   "UK",       "Barclays"),
    ("ING",   "Netherlands", "ING Groep"),
    ("HSBC",  "UK",       "HSBC Holdings"),
    ("LYG",   "UK",       "Lloyds Banking"),

    # Japanese banks
    ("MUFG",  "Japan",    "Mitsubishi UFJ Financial"),
    ("SMFG",  "Japan",    "Sumitomo Mitsui Financial"),

    # European tech / industrial
    ("ASML",  "Netherlands", "ASML Holding"),
    ("SAP",   "Germany",  "SAP SE"),
    ("STLA",  "Netherlands", "Stellantis"),

    # Asian tech
    ("TSM",   "Taiwan",   "Taiwan Semiconductor"),
    ("BABA",  "China",    "Alibaba Group"),
    ("JD",    "China",    "JD.com"),
    ("BIDU",  "China",    "Baidu"),
    ("INFY",  "India",    "Infosys"),

    # Defensive consumer (non-US)
    ("UL",    "UK",       "Unilever"),
    ("NSRGY", "Switzerland", "Nestle (ADR)"),
    ("DEO",   "UK",       "Diageo"),
    ("BTI",   "UK",       "British American Tobacco"),

    # Pharma (non-US)
    ("NVO",   "Denmark",  "Novo Nordisk"),
    ("AZN",   "UK",       "AstraZeneca"),
    ("GSK",   "UK",       "GSK plc"),
    ("SNY",   "France",   "Sanofi"),
    ("NVS",   "Switzerland", "Novartis"),

    # Industrials / auto (non-US)
    ("TM",    "Japan",    "Toyota Motor"),
    ("HMC",   "Japan",    "Honda Motor"),

    # Commodity-linked (likely GDX-correlated — included for context/ranking)
    ("RIO",   "UK/Australia", "Rio Tinto"),
    ("BHP",   "Australia", "BHP Group"),
    ("SHEL",  "UK",       "Shell"),
    ("BP",    "UK",       "BP"),
    ("TTE",   "France",   "TotalEnergies"),
]


# ── Helpers (identical to ETF version) ───────────────────────────────────────
def download_ticker(ticker, start):
    try:
        df = yf.Ticker(ticker).history(
            start=start,
            end=datetime.today().strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if df is None or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].copy()
        df.columns = ["close", "volume"]
        df = df.dropna()
        df = df[df["close"] > 0]
        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        print(f"  ⚠ {ticker}: download failed ({e})")
        return None


def compute_metrics(cand_df, gdx_fwd, ticker, country, description):
    fwd   = np.log(cand_df["close"].shift(-FORWARD_DAYS) / cand_df["close"])
    daily = np.log(cand_df["close"] / cand_df["close"].shift(1))
    dv    = (cand_df["close"] * cand_df["volume"]).rolling(252).mean()
    avg_dv = dv.dropna().mean() if dv.dropna().any() else 0.0

    aligned = pd.concat([gdx_fwd.rename("gdx_fwd"),
                          fwd.rename("cand_fwd")], axis=1).dropna()
    if len(aligned) < 100:
        return None

    g = aligned["gdx_fwd"]
    c = aligned["cand_fwd"]
    overall_mean = c.mean()
    overall_std  = c.std()

    def regime_stats(mask):
        n = int(mask.sum())
        if n < 20:
            return {"n": n, "mean_pct": np.nan, "hit_rate_pct": np.nan,
                    "sharpe": np.nan, "alpha_pct": np.nan,
                    "std_pct": np.nan, "tradeable_score": np.nan}
        rets = c[mask]
        mean_ret = rets.mean()
        std_ret  = rets.std()
        alpha_pct = (mean_ret - overall_mean) * 100
        std_pct = std_ret * 100 if std_ret > 0 else 0.0
        sign    = 1 if alpha_pct >= 0 else -1
        tradeable_score = sign * (abs(alpha_pct) * np.sqrt(std_pct)) \
                          if std_pct > 0 else 0.0
        return {
            "n":               n,
            "mean_pct":        mean_ret * 100,
            "hit_rate_pct":    (rets > 0).mean() * 100,
            "sharpe":          mean_ret / std_ret if std_ret > 0 else 0.0,
            "alpha_pct":       alpha_pct,
            "std_pct":         std_pct,
            "tradeable_score": tradeable_score,
        }

    losing = regime_stats(g < LOSING_THRESHOLD)
    flat   = regime_stats((g >= FLAT_LOW) & (g <= FLAT_HIGH))
    weak   = regime_stats(g < WEAK_THRESHOLD)

    return {
        "ticker":              ticker,
        "country":             country,
        "description":         description,
        "n_obs":               len(aligned),
        "avg_daily_dollar_vol_M":
            avg_dv / 1_000_000 if avg_dv else 0.0,
        "passes_liquidity":    avg_dv >= MIN_AVG_DOLLAR_VOL,
        "overall_mean_20d_pct": overall_mean * 100,
        "overall_std_20d_pct":  overall_std * 100,

        "losing_n":            losing["n"],
        "losing_mean_pct":     losing["mean_pct"],
        "losing_hit_rate":     losing["hit_rate_pct"],
        "losing_sharpe":       losing["sharpe"],
        "losing_alpha_pct":    losing["alpha_pct"],
        "losing_std_pct":      losing["std_pct"],
        "losing_score":        losing["tradeable_score"],

        "flat_n":              flat["n"],
        "flat_mean_pct":       flat["mean_pct"],
        "flat_hit_rate":       flat["hit_rate_pct"],
        "flat_sharpe":         flat["sharpe"],
        "flat_alpha_pct":      flat["alpha_pct"],
        "flat_std_pct":        flat["std_pct"],
        "flat_score":          flat["tradeable_score"],

        "weak_n":              weak["n"],
        "weak_mean_pct":       weak["mean_pct"],
        "weak_hit_rate":       weak["hit_rate_pct"],
        "weak_sharpe":         weak["sharpe"],
        "weak_alpha_pct":      weak["alpha_pct"],
        "weak_std_pct":        weak["std_pct"],
        "weak_score":          weak["tradeable_score"],
    }


def print_top_table(df, regime, top_n=10, sort_by="score"):
    sort_col = f"{regime}_{sort_by}"
    sub = df[df["passes_liquidity"]].dropna(subset=[sort_col])
    sub = sub.sort_values(sort_col, ascending=False).head(top_n)
    label = {
        "losing": f"GDX LOSING  (fwd 20d < {LOSING_THRESHOLD*100:.0f}%)",
        "flat":   f"GDX FLAT    ({FLAT_LOW*100:+.0f}% to {FLAT_HIGH*100:+.0f}%)",
        "weak":   f"GDX WEAK    (fwd 20d < {WEAK_THRESHOLD*100:+.0f}%)",
    }[regime]
    sort_label = "VOL-WEIGHTED SCORE" if sort_by == "score" else "RAW ALPHA"
    print(f"\n  ── Top {top_n} by {sort_label} when {label} ──")
    print(f"  {'Ticker':<5} {'Country':<12} {'Description':<22} {'n':>5} "
          f"{'Mean':>7} {'Alpha':>7} {'Std':>6} {'Score':>7} {'Hit%':>5}")
    print(f"  {'─'*5} {'─'*12} {'─'*22} {'─'*5} {'─'*7} {'─'*7} {'─'*6} "
          f"{'─'*7} {'─'*5}")
    for _, r in sub.iterrows():
        print(f"  {r['ticker']:<5} {r['country'][:12]:<12} "
              f"{r['description'][:22]:<22} "
              f"{int(r[regime + '_n']):>5} "
              f"{r[regime + '_mean_pct']:>+6.2f}% "
              f"{r[regime + '_alpha_pct']:>+6.2f}% "
              f"{r[regime + '_std_pct']:>5.2f}% "
              f"{r[regime + '_score']:>+6.3f} "
              f"{r[regime + '_hit_rate']:>4.0f}%")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*60}")
    print(f"  GDX Diversifier Analysis — STOCK UNIVERSE")
    print(f"  (non-US-incorporated, no US estate tax exposure)")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"{'═'*60}")
    print(f"  Window:       2010-01-01 → today  ({FORWARD_DAYS}d forward)")
    print(f"  Candidates:   {len(CANDIDATES)}")
    print(f"  Liquidity:    ADV >= ${MIN_AVG_DOLLAR_VOL/1_000_000:.0f}M")

    print(f"\n  Downloading GDX...")
    gdx = download_ticker("GDX", START_DATE)
    if gdx is None:
        print(f"  ✗ GDX download failed — aborting")
        return
    print(f"  ✓ GDX: {len(gdx)} rows  ({gdx.index[0].date()} → {gdx.index[-1].date()})")

    gdx_fwd   = np.log(gdx["close"].shift(-FORWARD_DAYS) / gdx["close"])
    gdx_daily = np.log(gdx["close"] / gdx["close"].shift(1))

    print(f"\n  Downloading {len(CANDIDATES)} candidate stocks...")
    rows = []
    for ticker, country, desc in CANDIDATES:
        df = download_ticker(ticker, START_DATE)
        if df is None:
            print(f"  ✗ {ticker:<5} {country:<12} {desc[:30]:<30} — skipped")
            continue
        result = compute_metrics(df, gdx_fwd, ticker, country, desc)
        if result is None:
            print(f"  ⚠ {ticker:<5} {country:<12} {desc[:30]:<30} — insufficient overlap")
            continue

        cand_daily = np.log(df["close"] / df["close"].shift(1))
        corr_df = pd.concat([gdx_daily.rename("gdx"),
                             cand_daily.rename("cand")], axis=1).dropna()
        result["corr_with_gdx"] = corr_df.corr().iloc[0, 1] \
                                  if len(corr_df) > 30 else np.nan

        rows.append(result)
        print(f"  ✓ {ticker:<5} {country:<12} {desc[:30]:<30} "
              f"ADV=${result['avg_daily_dollar_vol_M']:>6.0f}M  "
              f"corr={result['corr_with_gdx']:+.2f}")

    if not rows:
        print(f"\n  ✗ No usable candidates — aborting")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n  ✓ Full results → {OUT_CSV}  ({len(df)} candidates)")

    print_top_table(df, "weak",   top_n=12, sort_by="score")
    print_top_table(df, "losing", top_n=10, sort_by="score")
    print_top_table(df, "flat",   top_n=10, sort_by="score")

    print(f"\n{'─'*60}")
    print(f"  Reference: same rankings sorted by RAW ALPHA")
    print(f"{'─'*60}")
    print_top_table(df, "losing", top_n=8, sort_by="alpha_pct")
    print_top_table(df, "weak",   top_n=8, sort_by="alpha_pct")

    print(f"\n  ── Top 10 by LOWEST correlation with GDX (daily returns) ──")
    print(f"  {'Ticker':<5} {'Country':<12} {'Description':<28} {'Corr':>7}")
    print(f"  {'─'*5} {'─'*12} {'─'*28} {'─'*7}")
    sub = df[df["passes_liquidity"]].dropna(subset=["corr_with_gdx"])
    sub = sub.sort_values("corr_with_gdx").head(10)
    for _, r in sub.iterrows():
        print(f"  {r['ticker']:<5} {r['country'][:12]:<12} "
              f"{r['description'][:28]:<28} {r['corr_with_gdx']:>+6.3f}")

    print(f"\n{'═'*60}")
    print(f"  Done.  See {OUT_CSV} for full ranked table.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
