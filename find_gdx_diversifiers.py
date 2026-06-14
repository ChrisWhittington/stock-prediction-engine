"""
find_gdx_diversifiers.py
========================

Screen high-volume ETFs for use as a fourth rotation instrument.

Methodology:
  1. Download GDX and a curated candidate universe via yfinance (2010-present).
  2. Compute 20-day forward log return for each instrument (matches the
     rotation's hold_days).
  3. Identify "GDX-weak" days using three thresholds:
       - "losing":  GDX forward return < -3%
       - "flat":    -3% <= GDX forward return <= +3%
       - "weak":    GDX forward return < +2%  (combined losing + flat)
  4. For each candidate, score:
       - mean forward return during GDX-weak days
       - hit rate (% positive forward returns) during GDX-weak days
       - Sharpe-like ratio during GDX-weak days
       - conditional alpha: mean return when GDX-weak minus mean overall
         (isolates "what does this asset add when GDX struggles" from
          "this asset just trends up generally")
       - overall correlation with GDX (using daily returns)
       - average daily dollar volume (liquidity filter)
  5. Rank candidates by conditional alpha and output CSV + console summary.

Usage:
  python find_gdx_diversifiers.py

Outputs:
  - diversifier_analysis.csv  (full results, all metrics)
  - console:  top-10 by conditional alpha for each regime
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
LOSING_THRESHOLD    = -0.03   # GDX 20d fwd log return < -3%
FLAT_LOW            = -0.03
FLAT_HIGH           = +0.03
WEAK_THRESHOLD      = +0.02   # broad "weak or flat" combined
MIN_AVG_DOLLAR_VOL  = 25_000_000   # $25M average daily volume (liquidity floor;
                                   # lowered from $50M to admit assets like UUP
                                   # that are still plenty tradeable at small
                                   # portfolio scale)
OUT_CSV             = "diversifier_analysis.csv"

# ── Candidate universe ───────────────────────────────────────────────────────
# Curated for high volume and breadth across asset classes. Excludes XLE and
# QQQ (already in rotation) but includes peers in those buckets for context.
CANDIDATES = [
    # Bonds
    ("TLT",  "Long Treasury (20+y)"),
    ("IEF",  "Intermediate Treasury (7-10y)"),
    ("SHY",  "Short Treasury (1-3y)"),
    ("BIL",  "T-Bills (1-3 month)"),
    ("HYG",  "High Yield Corporate"),
    ("LQD",  "Investment Grade Corporate"),
    ("TIP",  "TIPS (inflation-linked)"),
    ("AGG",  "US Aggregate Bond"),
    # US equity sectors (XLE and QQQ excluded; SMH for semi as XLK peer)
    ("XLK",  "Technology"),
    ("XLF",  "Financials"),
    ("XLI",  "Industrials"),
    ("XLP",  "Consumer Staples"),
    ("XLY",  "Consumer Discretionary"),
    ("XLU",  "Utilities"),
    ("XLV",  "Healthcare"),
    ("XLB",  "Materials"),
    ("XLRE", "Real Estate"),
    ("SMH",  "Semiconductors"),
    ("KRE",  "Regional Banks"),
    ("XBI",  "Biotech"),
    # Commodities
    ("SLV",  "Silver"),
    ("GLD",  "Gold (sanity check)"),
    ("USO",  "Oil (WTI)"),
    ("DBC",  "Broad Commodities"),
    ("DBA",  "Agriculture"),
    ("CPER", "Copper"),
    # International equity
    ("EFA",  "Developed ex-US"),
    ("EEM",  "Emerging Markets"),
    ("FXI",  "China Large Cap"),
    ("EWJ",  "Japan"),
    ("INDA", "India"),
    # FX / Defensive
    ("UUP",  "Long Dollar Index (K-1 tax)"),
    ("USDU", "Long Dollar Bullish (1099 tax)"),
    ("USMV", "Low Volatility US"),
    ("VYM",  "High Dividend US"),
    ("MTUM", "US Momentum Factor"),
    # Real / inflation
    ("VNQ",  "US REIT (broad)"),
    ("PDBC", "Optimum Yield Diversified Commodity"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def download_ticker(ticker, start):
    """Download adjusted close + volume. Returns DataFrame or None."""
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
        # Strip timezone for clean joins
        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        print(f"  ⚠ {ticker}: download failed ({e})")
        return None


def compute_metrics(cand_df, gdx_fwd, candidate_ticker, description):
    """
    Compute diversifier metrics for one candidate.
    Returns a dict suitable for a DataFrame row.
    """
    # Forward 20-day log return for the candidate
    fwd = np.log(cand_df["close"].shift(-FORWARD_DAYS) / cand_df["close"])

    # Daily log return (for correlation with GDX)
    daily = np.log(cand_df["close"] / cand_df["close"].shift(1))

    # Average daily dollar volume (rolling 252-day mean of close*volume)
    dv = (cand_df["close"] * cand_df["volume"]).rolling(252).mean()
    avg_dv = dv.dropna().mean() if dv.dropna().any() else 0.0

    # Align candidate forward returns with GDX forward returns
    aligned = pd.concat([gdx_fwd.rename("gdx_fwd"),
                          fwd.rename("cand_fwd")],
                         axis=1).dropna()

    if len(aligned) < 100:
        return None

    g = aligned["gdx_fwd"]
    c = aligned["cand_fwd"]

    # Overall stats
    overall_mean = c.mean()
    overall_std  = c.std()

    # Regime masks
    mask_losing = g < LOSING_THRESHOLD
    mask_flat   = (g >= FLAT_LOW) & (g <= FLAT_HIGH)
    mask_weak   = g < WEAK_THRESHOLD

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
        # Composite tradeability score: alpha × sqrt(vol). Rewards both
        # diversifier edge AND adequate volatility (small-move assets like
        # UUP get penalized vs higher-vol assets with similar alpha).
        # Negative alpha → negative score (i.e. an anti-diversifier).
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

    losing = regime_stats(mask_losing)
    flat   = regime_stats(mask_flat)
    weak   = regime_stats(mask_weak)

    # Correlation with GDX (daily returns)
    gdx_daily = gdx_fwd.index.to_series()  # placeholder
    # Reload daily GDX returns for correlation (handled in caller)

    return {
        "ticker":              candidate_ticker,
        "description":         description,
        "n_obs":               len(aligned),
        "avg_daily_dollar_vol_M":
            avg_dv / 1_000_000 if avg_dv else 0.0,
        "passes_liquidity":    avg_dv >= MIN_AVG_DOLLAR_VOL,
        "overall_mean_20d_pct": overall_mean * 100,
        "overall_std_20d_pct":  overall_std * 100,

        # Losing regime
        "losing_n":            losing["n"],
        "losing_mean_pct":     losing["mean_pct"],
        "losing_hit_rate":     losing["hit_rate_pct"],
        "losing_sharpe":       losing["sharpe"],
        "losing_alpha_pct":    losing["alpha_pct"],
        "losing_std_pct":      losing["std_pct"],
        "losing_score":        losing["tradeable_score"],

        # Flat regime
        "flat_n":              flat["n"],
        "flat_mean_pct":       flat["mean_pct"],
        "flat_hit_rate":       flat["hit_rate_pct"],
        "flat_sharpe":         flat["sharpe"],
        "flat_alpha_pct":      flat["alpha_pct"],
        "flat_std_pct":        flat["std_pct"],
        "flat_score":          flat["tradeable_score"],

        # Combined weak regime (used for primary ranking)
        "weak_n":              weak["n"],
        "weak_mean_pct":       weak["mean_pct"],
        "weak_hit_rate":       weak["hit_rate_pct"],
        "weak_sharpe":         weak["sharpe"],
        "weak_alpha_pct":      weak["alpha_pct"],
        "weak_std_pct":        weak["std_pct"],
        "weak_score":          weak["tradeable_score"],
    }


def print_top_table(df, regime, top_n=10, sort_by="score"):
    """
    Print a ranked table of candidates for one regime.
    sort_by: "score"  → vol-weighted composite (alpha × sqrt(std))
             "alpha"  → raw conditional alpha
    """
    sort_col = f"{regime}_{sort_by}"
    sub = df[df["passes_liquidity"]].dropna(subset=[sort_col])
    sub = sub.sort_values(sort_col, ascending=False).head(top_n)
    label = {
        "losing": f"GDX LOSING  (fwd 20d < {LOSING_THRESHOLD*100:.0f}%)",
        "flat":   f"GDX FLAT    ({FLAT_LOW*100:+.0f}% to {FLAT_HIGH*100:+.0f}%)",
        "weak":   f"GDX WEAK    (fwd 20d < {WEAK_THRESHOLD*100:+.0f}%)",
    }[regime]
    sort_label = "VOL-WEIGHTED SCORE (alpha × √std)" if sort_by == "score" \
                 else "RAW CONDITIONAL ALPHA"
    print(f"\n  ── Top {top_n} by {sort_label} when {label} ──")
    print(f"  {'Ticker':<6} {'Description':<28} {'n':>5} "
          f"{'Mean':>7} {'Alpha':>7} {'Std':>6} {'Score':>7} "
          f"{'Hit%':>5} {'Sharpe':>7}")
    print(f"  {'─'*6} {'─'*28} {'─'*5} {'─'*7} {'─'*7} {'─'*6} {'─'*7} "
          f"{'─'*5} {'─'*7}")
    for _, r in sub.iterrows():
        print(f"  {r['ticker']:<6} {r['description'][:28]:<28} "
              f"{int(r[regime + '_n']):>5} "
              f"{r[regime + '_mean_pct']:>+6.2f}% "
              f"{r[regime + '_alpha_pct']:>+6.2f}% "
              f"{r[regime + '_std_pct']:>5.2f}% "
              f"{r[regime + '_score']:>+6.3f} "
              f"{r[regime + '_hit_rate']:>4.0f}% "
              f"{r[regime + '_sharpe']:>+6.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*60}")
    print(f"  GDX Diversifier Analysis")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"{'═'*60}")
    print(f"  Window:       2010-01-01 → today  ({FORWARD_DAYS}d forward)")
    print(f"  Candidates:   {len(CANDIDATES)}")
    print(f"  Liquidity:    ADV >= ${MIN_AVG_DOLLAR_VOL/1_000_000:.0f}M")

    # 1. Download GDX
    print(f"\n  Downloading GDX...")
    gdx = download_ticker("GDX", START_DATE)
    if gdx is None:
        print(f"  ✗ GDX download failed — aborting")
        return
    print(f"  ✓ GDX: {len(gdx)} rows  ({gdx.index[0].date()} → {gdx.index[-1].date()})")

    # GDX forward 20-day log return
    gdx_fwd = np.log(gdx["close"].shift(-FORWARD_DAYS) / gdx["close"])
    gdx_daily = np.log(gdx["close"] / gdx["close"].shift(1))

    print(f"\n  Downloading {len(CANDIDATES)} candidate ETFs...")
    rows = []
    for ticker, desc in CANDIDATES:
        df = download_ticker(ticker, START_DATE)
        if df is None:
            print(f"  ✗ {ticker:<6} ({desc}) — skipped")
            continue
        result = compute_metrics(df, gdx_fwd, ticker, desc)
        if result is None:
            print(f"  ⚠ {ticker:<6} ({desc}) — insufficient overlap with GDX")
            continue

        # Correlation with GDX (daily returns)
        cand_daily = np.log(df["close"] / df["close"].shift(1))
        corr_df = pd.concat([gdx_daily.rename("gdx"),
                             cand_daily.rename("cand")], axis=1).dropna()
        result["corr_with_gdx"] = corr_df.corr().iloc[0, 1] \
                                  if len(corr_df) > 30 else np.nan

        rows.append(result)
        print(f"  ✓ {ticker:<6} {desc[:36]:<36} "
              f"ADV=${result['avg_daily_dollar_vol_M']:>6.0f}M  "
              f"corr={result['corr_with_gdx']:+.2f}")

    if not rows:
        print(f"\n  ✗ No usable candidates — aborting")
        return

    df = pd.DataFrame(rows)

    # 2. Save full CSV
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n  ✓ Full results → {OUT_CSV}  ({len(df)} candidates)")

    # 3. Print ranked tables for each regime — by vol-weighted score
    #    (rewards alpha AND adequate volatility for tradeability)
    print_top_table(df, "weak",   top_n=12, sort_by="score")
    print_top_table(df, "losing", top_n=10, sort_by="score")
    print_top_table(df, "flat",   top_n=10, sort_by="score")

    # Also print top by raw alpha (for those who don't care about vol)
    print(f"\n{'─'*60}")
    print(f"  Reference: same rankings sorted by RAW ALPHA (no vol penalty)")
    print(f"{'─'*60}")
    print_top_table(df, "weak",   top_n=8, sort_by="alpha_pct")
    print_top_table(df, "losing", top_n=8, sort_by="alpha_pct")

    # 4. Print lowest-correlation candidates (structural diversifiers)
    print(f"\n  ── Top 10 by LOWEST correlation with GDX (daily returns) ──")
    print(f"  {'Ticker':<6} {'Description':<32} {'Corr':>7}")
    print(f"  {'─'*6} {'─'*32} {'─'*7}")
    sub = df[df["passes_liquidity"]].dropna(subset=["corr_with_gdx"])
    sub = sub.sort_values("corr_with_gdx").head(10)
    for _, r in sub.iterrows():
        print(f"  {r['ticker']:<6} {r['description'][:32]:<32} "
              f"{r['corr_with_gdx']:>+6.3f}")

    print(f"\n{'═'*60}")
    print(f"  Done.  See {OUT_CSV} for the full ranked table with all metrics.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
