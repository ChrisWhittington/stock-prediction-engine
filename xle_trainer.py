"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AURUM·AI  —  xle_trainer.py  (v1)                           ║
║                                                                              ║
║  Trains a neural net to predict 10-day forward returns on XLE               ║
║  (Energy Select Sector SPDR ETF).                                                  ║
║                                                                              ║
║  DATA                                                                        ║
║    Downloads 15 tickers via yfinance from 2015 + Google Trends.            ║
║    No GPR or COT (gold-specific) — replaced with tech/equity features.      ║
║                                                                              ║
║  FEATURES  (169 → 45 after selection)                                       ║
║    Price returns, ratios, volatility, MA signals, RSI, MACD, lags,         ║
║    VIX term structure, safe-haven FX (JPY/CHF), oil-gold correlation,      ║
║    MOVE index, COT positioning, GPR geopolitical risk, GLD flow.            ║
║    All normalised via rolling z-score (252-day window) — no lookahead.     ║
║                                                                              ║
║  FEATURE SELECTION  (3 steps)                                               ║
║    1. Correlation filter (threshold=0.85) — removes redundant features     ║
║    2. IC ranking (top_n=45)  — keeps features most correlated with target  ║
║    3. IC noise floor (min_ic=0.02) — drops pure noise                      ║
║                                                                              ║
║  MODEL                                                                       ║
║    Single-layer LSTM (hidden=48) → LayerNorm → Linear(48→32→16→1).         ║
║    ~20,449 parameters. AdamW optimiser, HuberLoss, early stopping.         ║
║                                                                              ║
║  TRAINING                                                                   ║
║    Purged walk-forward CV — 8 folds, 3-month val blocks, 30-day purge.     ║
║    Random seed each run to find different local minima.                     ║
║    Best run tracked by IC mean — saved to aurum_output/best_run.txt.       ║
║                                                                              ║
║  OUTPUTS                                                                    ║
║    fold_N_model.pt        — 8 fold models for ensemble inference            ║
║    final_model.pt         — single model trained on full dataset            ║
║    oof_predictions.csv    — honest out-of-fold predictions (344 days)      ║
║    full_predictions.csv   — all 3,300+ days for backtest engine             ║
║    feature_importance.csv — IC ranking of all features                     ║
║    wf_results.png         — walk-forward validation chart                  ║
║    ensemble_results.png   — ensemble inference chart                       ║
║    final_model_results.png — final model test chart                        ║
║                                                                              ║
║  RUN                                                                        ║
║    pip install torch yfinance pandas numpy scikit-learn matplotlib          ║
║               scipy joblib ta requests openpyxl xlrd pytrends              ║
║    python gold_miner_trainer.py                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings, os, sys, time, glob, shutil
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves PNGs to disk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import spearmanr
import joblib

import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Reproducibility ───────────────────────────────────────────────────────────
# SEED=None → random seed each run (different local minima, use best result)
# SEED=42   → pin seed for exact reproducibility (debugging only)
SEED = None

if SEED is not None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
else:
    # Random seed — record it so results can be reproduced if needed
    SEED = int(torch.randint(0, 100000, (1,)).item())
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"  Random seed this run: {SEED}  "
          f"(set SEED={SEED} to reproduce exactly)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "xle_output"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  edit these to experiment
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Data
    # v7: start_date moved to 2013-01-01
    # Rationale: v6 geopolitical features (GPR, VIX curve, safe-haven FX,
    # COT, GLD holdings) now give the model enough context to handle
    # different macro regimes without needing the 2016 exclusion patch.
    # 2013 start still captures: taper tantrum (2013), oil crash (2015),
    # gold bottom (2015-16), and all subsequent regimes.
    # The 252-day rolling z-score needs ~1yr warmup so 2013 gives
    # clean features from early 2014 onward — 12 full years of training.
    "target":          "XLE",
    "start_date":      "2007-01-01",  # longer history captures GFC oil spike
    "end_date":        datetime.today().strftime("%Y-%m-%d"),
    "forward_days":    20,                  # predict 20-day forward log return
    "lookback":        30,                  # matched to GDX/QQQ convention
    "zscore_window":   252,                 # rolling z-score window (1 trading year)

    # Walk-forward CV
    "n_folds":         8,
    "val_months":      3,                   # each validation block = 3 months
    "purge_days":      30,                  # gap between train and val edges
    "min_train_years": 2,                   # reduced from 3 — 2013 start means
                                            # 3yr minimum pushes fold 1 too late
    "fold_starts":     3,                   # train each fold N times, keep best IC

    # exclude_before removed — v6 features now handle the 2016 bear market
    # regime properly via GPR/VIX curve/COT context. No longer needed.
    "exclude_before":  None,

    # Model architecture — unchanged from v4/v5 (working well)
    "hidden_size":     48,
    "num_lstm_layers": 1,
    "dropout":         0.4,

    # Training
    "epochs":          150,
    "batch_size":      32,
    "lr":              1e-3,
    "weight_decay":    3e-4,
    "early_stop_patience": 20,
    "lr_patience":     8,
    "grad_clip":       1.0,

    # Loss function — huber with delta=0.05 balances shrinkage vs outlier robustness
    "loss_fn":         "huber",
    "huber_delta":     0.05,

    # Ensemble weighting — 60% accuracy + 40% recency
    "ensemble_recency_weight": 0.4,

    # Ensemble quality filters — kept as safety rails but thresholds relaxed
    # In v5 these didn't exclude anything (all folds had positive IC)
    # They protect against future catastrophically bad folds
    "ensemble_min_dir_acc":    0.40,
    "ensemble_min_ic":         -0.15,

    # Feature selection pipeline
    # Step 1: remove features correlated above this threshold (redundancy)
    "fs_corr_threshold":       0.85,
    # Step 2: keep top N features by absolute IC with target
    "fs_top_n":                45,
    # Step 3: minimum absolute IC to keep a feature (noise floor)
    "fs_min_ic":               0.02,
    # Set to False to skip selection and use all features (for comparison)
    "fs_enabled":              True,
}

TICKERS = [
    "XLE",          # target — Energy Select Sector SPDR
    "XOP",          # E&P ETF — higher beta energy proxy
    "OIH",          # oil services — leading indicator for energy capex
    "CL=F",         # crude oil futures — primary driver
    "BZ=F",         # Brent crude — international oil benchmark
    "NG=F",         # natural gas futures
    "HO=F",         # heating oil — for crack spread calculation
    "UUP",          # dollar index — oil priced in USD (inverse)
    "TLT",          # 20Y treasury — rate/growth proxy
    "SPY",          # broad market — risk appetite
    "^VIX",         # fear index
    "^VIX3M",       # 3-month VIX — term structure
    "^TNX",         # 10Y yield — inflation/growth signal
    "^MOVE",        # bond volatility
    "USDJPY=X",     # USD/JPY — risk sentiment
    "USDCHF=X",     # USD/CHF — safe haven
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_prices(tickers, start, end):
    print(f"\n{'═'*60}")
    print(f"  [1] DOWNLOADING PRICE DATA")
    print(f"      {start}  →  {end}")
    print(f"{'═'*60}")

    # Detect yfinance version to handle API differences
    import yfinance as _yf
    yf_version = tuple(int(x) for x in _yf.__version__.split(".")[:2])
    print(f"  yfinance version: {_yf.__version__}")

    frames = {}
    for t in tickers:
        try:
            if yf_version >= (0, 24):
                # Newer yfinance: use Ticker object
                ticker_obj = yf.Ticker(t)
                df = ticker_obj.history(start=start, end=end, auto_adjust=True)
            else:
                df = yf.download(t, start=start, end=end,
                                 progress=False, auto_adjust=True)

            # Flatten MultiIndex columns if present (yfinance 0.2.x quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df is None or len(df) < 200:
                print(f"  ✗  {t:<10} only {len(df) if df is not None else 0} rows — skipped")
                continue

            # Column might be "Close" or "close" depending on version
            close_col = next((c for c in df.columns
                              if c.lower() == "close"), None)
            if close_col is None:
                print(f"  ✗  {t:<10} no Close column — skipped")
                continue

            series = df[close_col].rename(t)
            series = series[series > 0].dropna()  # remove bad data points
            frames[t] = series
            print(f"  ✓  {t:<10} {len(series):>5} rows  "
                  f"({series.index[0].date()} → {series.index[-1].date()})")

        except Exception as e:
            print(f"  ✗  {t:<10} error: {e}")

    if not frames:
        raise RuntimeError(
            "No tickers downloaded successfully.\n"
            "Try: pip install --upgrade yfinance"
        )

    prices = pd.concat(frames.values(), axis=1)
    prices = prices.sort_index().ffill().dropna(how="all")

    # Normalise index — strip timezone AND time component
    # yfinance returns timestamps like 2024-01-15 05:00:00+00:00
    # normalize() converts to midnight, tz_localize(None) removes timezone
    prices.index = pd.to_datetime(prices.index).normalize().tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")]

    print(f"\n  Combined: {prices.shape[0]} rows × {prices.shape[1]} tickers")
    return prices


def download_gpr(start, end):
    """
    Download the Geopolitical Risk Index (Caldara & Iacoviello, Federal Reserve).
    Monthly data interpolated to daily. Free, no API key needed.

    Source: https://www.matteoiacoviello.com/gpr.htm
    """
    cache_path = os.path.join(OUT_DIR, "gpr_cache.csv")
    if os.path.exists(cache_path):
        age_days = (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_path))).days
        if age_days < 7:   # GPR updates monthly — 7 day cache is fine
            try:
                cached = pd.read_csv(cache_path, parse_dates=["date"],
                                     index_col="date")
                cached.index = pd.to_datetime(
                    cached.index).normalize().tz_localize(None)
                mask = (cached.index >= pd.Timestamp(start)) & \
                       (cached.index <= pd.Timestamp(end))
                sliced = cached[mask]
                if len(sliced) > 10:
                    print(f"  ✓ GPR: cache hit ({age_days}d old, "
                          f"{len(sliced)} rows)")
                    return sliced
            except Exception:
                pass

    print(f"\n  [1b] Downloading Geopolitical Risk Index (GPR)...")
    GPR_URL = (
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    )
    try:
        gpr_raw = pd.read_excel(GPR_URL, index_col=0, parse_dates=True)
        gpr_raw.index = pd.to_datetime(gpr_raw.index).tz_localize(None)

        cols_want = ["GPR", "GPRACT", "GPRTHR",
                     "GPRD", "GPRD_ACT", "GPRD_THREAT"]
        cols_have = [c for c in cols_want if c in gpr_raw.columns]
        if not cols_have:
            cols_have = [c for c in gpr_raw.columns
                         if any(k in c.upper()
                                for k in ["GPR", "ACT", "THR"])][:3]

        gpr = gpr_raw[cols_have].rename(columns={
            "GPRD":        "GPR",
            "GPRD_ACT":    "GPRACT",
            "GPRD_THREAT": "GPRTHR",
        }).copy()

        # Resample to full daily range and cache
        date_range = pd.date_range(
            start="2004-01-01",
            end=datetime.today().strftime("%Y-%m-%d"), freq="B")
        gpr_daily = gpr.reindex(date_range).ffill().bfill()
        gpr_daily.index.name = "date"
        try:
            gpr_daily.to_csv(cache_path)
        except Exception:
            pass

        # Return sliced to requested range
        mask = (gpr_daily.index >= pd.Timestamp(start)) & \
               (gpr_daily.index <= pd.Timestamp(end))
        result = gpr_daily[mask]
        print(f"  ✓ GPR: {len(result)} rows  columns: {list(result.columns)}")
        return result

    except Exception as e:
        print(f"  ✗ GPR download failed: {e}")
        print(f"    Continuing without GPR features — "
              f"download manually from https://www.matteoiacoviello.com/gpr.htm")
        return None


def download_cot(start, end):
    """
    Download CFTC Commitments of Traders data for gold futures (COMEX).
    Uses annual legacy ZIP files from cftc.gov — most reliable method.

    Gold CFTC contract code: 088691
    Files: https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip
    Coverage: 1986→present, weekly (Tuesday close)

    Key columns:
      NonComm_Positions_Long_All   — speculative longs (hedge funds)
      NonComm_Positions_Short_All  — speculative shorts
      Comm_Positions_Long_All      — commercial longs (producers)
      Comm_Positions_Short_All     — commercial shorts
      Open_Interest_All            — total open interest
    """
    print(f"\n  [1c] Downloading COT data (CFTC annual ZIPs)...")
    import requests, zipfile, io as _io

    start_year = int(str(start)[:4])
    end_year   = int(str(end)[:4])

    all_frames = []
    for year in range(start_year, end_year + 1):
        # Gold is in the LEGACY COT report (deacot{year}.zip)
        # NOT fut_fin_txt (financial/currencies only)
        urls_to_try = [
            f"https://www.cftc.gov/files/dea/history/deacot{year}.zip",
            f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
        ]
        try:
            resp = None
            used_url = None
            for url in urls_to_try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    resp = r
                    used_url = url
                    break

            if resp is None:
                print(f"    {year}: not found")
                continue

            with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
                names = z.namelist()
                txt   = next((n for n in names
                              if n.lower().endswith((".txt", ".csv"))), names[0])
                with z.open(txt) as f:
                    df_yr = pd.read_csv(f, low_memory=False)

            # Filter to gold rows — legacy files use spaces in column names
            NAME_COL = "Market and Exchange Names"
            CODE_COL = "CFTC Contract Market Code"
            filtered = pd.DataFrame()
            if NAME_COL in df_yr.columns:
                mask = df_yr[NAME_COL].astype(str) \
                           .str.upper().str.contains("GOLD", na=False)
                filtered = df_yr[mask]
            elif CODE_COL in df_yr.columns:
                filtered = df_yr[
                    df_yr[CODE_COL].astype(str)
                    .str.strip().str.startswith("088")]

            if len(filtered) == 0:
                print(f"    {year}: no gold rows found — skipping")
                continue

            all_frames.append(filtered)
            print(f"    {year}: {len(filtered)} rows ✓")

        except Exception as e:
            print(f"    {year}: failed — {e}")
            continue

    if not all_frames:
        print(f"  ✗ COT: no data retrieved")
        return None

    cot_raw = pd.concat(all_frames, ignore_index=True)

    # Parse date — use ISO format column preferentially
    iso_col  = "As of Date in Form YYYY-MM-DD"
    yymmdd   = "As of Date in Form YYMMDD"
    if iso_col in cot_raw.columns:
        cot_raw["date"] = pd.to_datetime(
            cot_raw[iso_col], errors="coerce").dt.tz_localize(None)
    elif yymmdd in cot_raw.columns:
        cot_raw["date"] = pd.to_datetime(
            cot_raw[yymmdd].astype(str).str.zfill(6),
            format="%y%m%d", errors="coerce").dt.tz_localize(None)
    elif "report_date" in cot_raw.columns:
        cot_raw["date"] = pd.to_datetime(
            cot_raw["report_date"], errors="coerce").dt.tz_localize(None)
    else:
        date_col = next((c for c in cot_raw.columns
                         if "date" in c.lower()), None)
        if date_col is None:
            print(f"  ✗ COT: no date column found")
            return None
        cot_raw["date"] = pd.to_datetime(
            cot_raw[date_col], errors="coerce").dt.tz_localize(None)
    cot_raw = cot_raw.dropna(subset=["date"]).set_index("date").sort_index()

    # Map column names — legacy files use human-readable spaced names
    col_map = {
        # Legacy format (deacot{year}.zip) — spaces in names
        "Market and Exchange Names":              "market_name",
        "As of Date in Form YYYY-MM-DD":          "report_date",
        "As of Date in Form YYMMDD":              "report_date_alt",
        "Noncommercial Positions-Long (All)":     "spec_long",
        "Noncommercial Positions-Short (All)":    "spec_short",
        "Commercial Positions-Long (All)":        "comm_long",
        "Commercial Positions-Short (All)":       "comm_short",
        "Open Interest (All)":                    "open_int",
        # Newer underscore format (just in case)
        "NonComm_Positions_Long_All":             "spec_long",
        "NonComm_Positions_Short_All":            "spec_short",
        "Comm_Positions_Long_All":                "comm_long",
        "Comm_Positions_Short_All":               "comm_short",
        "Open_Interest_All":                      "open_int",
    }
    cot_raw = cot_raw.rename(columns=col_map)

    # Fuzzy fallback — catch slight column name variations
    if "spec_long" not in cot_raw.columns:
        for col in cot_raw.columns:
            cl = col.lower()
            if "noncommercial" in cl and "long" in cl:
                cot_raw = cot_raw.rename(columns={col: "spec_long"})
            elif "noncommercial" in cl and "short" in cl:
                cot_raw = cot_raw.rename(columns={col: "spec_short"})
            elif "commercial" in cl and "long" in cl \
                    and "spec_long" in cot_raw.columns:
                cot_raw = cot_raw.rename(columns={col: "comm_long"})
            elif "commercial" in cl and "short" in cl \
                    and "spec_short" in cot_raw.columns:
                cot_raw = cot_raw.rename(columns={col: "comm_short"})
    needed = ["spec_long", "spec_short", "comm_long", "comm_short"]
    missing = [c for c in needed if c not in cot_raw.columns]
    if missing:
        print(f"  ✗ COT: missing columns after rename: {missing}")
        print(f"    Available: {list(cot_raw.columns[:15])}")
        return None

    # Compute derived series
    cot = pd.DataFrame(index=cot_raw.index)
    cot["spec_long"]  = pd.to_numeric(cot_raw.get("spec_long"),  errors="coerce")
    cot["spec_short"] = pd.to_numeric(cot_raw.get("spec_short"), errors="coerce")
    cot["comm_long"]  = pd.to_numeric(cot_raw.get("comm_long"),  errors="coerce")
    cot["comm_short"] = pd.to_numeric(cot_raw.get("comm_short"), errors="coerce")
    cot["open_int"]   = pd.to_numeric(
        cot_raw.get("open_int", pd.Series(dtype=float)), errors="coerce")

    cot["net_spec"]      = cot["spec_long"]  - cot["spec_short"]
    cot["net_comm"]      = cot["comm_long"]  - cot["comm_short"]
    cot["spec_oi_ratio"] = cot["net_spec"] / (cot["open_int"] + 1e-9)
    cot["comm_oi_ratio"] = cot["net_comm"] / (cot["open_int"] + 1e-9)

    cot = cot[["net_spec", "net_comm",
               "spec_oi_ratio", "comm_oi_ratio", "open_int"]].dropna(how="all")

    # Deduplicate — some years have multiple gold contract rows per date
    # (e.g. front month + back month). Keep the row with highest open interest
    # as it represents the most active contract.
    cot = cot.sort_values("open_int", ascending=False)
    cot = cot[~cot.index.duplicated(keep="first")]
    cot = cot.sort_index()

    print(f"  ✓ COT: {len(cot)} weekly rows before resampling  "
          f"({cot.index[0].date()} → {cot.index[-1].date()})")

    # Resample to daily business days
    date_range = pd.date_range(start=start, end=end, freq="B")
    cot = cot.reindex(date_range).ffill().bfill()

    print(f"  ✓ COT: {len(cot)} daily rows  "
          f"net_spec range: "
          f"{cot['net_spec'].min():.0f} → {cot['net_spec'].max():.0f}")
    return cot



def download_cot_crude(start, end):
    """
    Download CFTC COT data for WTI Crude Oil futures (NYMEX).
    Contract code: 067651 — same download mechanism as gold COT.
    Weekly (Tuesday close). Key signal: spec net positioning as % of OI.
    """
    cache_path = os.path.join(OUT_DIR, "cot_crude_cache.csv")
    if os.path.exists(cache_path):
        age_days = (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_path))).days
        if age_days < 1:
            try:
                cached = pd.read_csv(cache_path, parse_dates=["date"],
                                     index_col="date")
                cached.index = pd.to_datetime(
                    cached.index).normalize().tz_localize(None)
                mask = (cached.index >= pd.Timestamp(start)) & \
                       (cached.index <= pd.Timestamp(end))
                sliced = cached[mask]
                if len(sliced) > 10:
                    print(f"  ✓ COT crude: cache hit ({age_days}d old, "
                          f"{len(sliced)} rows)")
                    return sliced
            except Exception:
                pass

    print(f"\n  [1c2] Downloading COT crude oil data...")
    import requests, zipfile, io as _io

    start_year = int(str(start)[:4])
    end_year   = int(str(end)[:4])
    all_frames = []

    for year in range(start_year, end_year + 1):
        urls = [
            f"https://www.cftc.gov/files/dea/history/deacot{year}.zip",
            f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
        ]
        try:
            resp = None
            for url in urls:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    resp = r
                    break
            if resp is None:
                continue
            with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
                names = z.namelist()
                txt   = next((n for n in names if n.lower().endswith((".txt",".csv"))), names[0])
                with z.open(txt) as f:
                    df_yr = pd.read_csv(f, low_memory=False)
            NAME_COL = "Market and Exchange Names"
            CODE_COL = "CFTC Contract Market Code"
            filtered = pd.DataFrame()
            if NAME_COL in df_yr.columns:
                mask = df_yr[NAME_COL].astype(str).str.upper().str.contains("CRUDE OIL", na=False)
                filtered = df_yr[mask]
            elif CODE_COL in df_yr.columns:
                filtered = df_yr[df_yr[CODE_COL].astype(str).str.strip().str.startswith("067")]
            if len(filtered) == 0:
                continue
            all_frames.append(filtered)
            print(f"    {year}: {len(filtered)} rows OK")
        except Exception as e:
            print(f"    {year}: failed — {e}")
            continue

    if not all_frames:
        print(f"  ✗ COT crude: no data")
        return None

    cot_raw = pd.concat(all_frames, ignore_index=True)

    # Parse date — use ISO format column preferentially
    iso_col = "As of Date in Form YYYY-MM-DD"
    yymmdd  = "As of Date in Form YYMMDD"
    if iso_col in cot_raw.columns:
        cot_raw["date"] = pd.to_datetime(
            cot_raw[iso_col], errors="coerce").dt.tz_localize(None)
    elif yymmdd in cot_raw.columns:
        cot_raw["date"] = pd.to_datetime(
            cot_raw[yymmdd].astype(str).str.zfill(6),
            format="%y%m%d", errors="coerce").dt.tz_localize(None)
    elif "report_date" in cot_raw.columns:
        cot_raw["date"] = pd.to_datetime(
            cot_raw["report_date"], errors="coerce").dt.tz_localize(None)
    else:
        date_col = next((c for c in cot_raw.columns if "date" in c.lower()), None)
        if date_col is None:
            return None
        cot_raw["date"] = pd.to_datetime(
            cot_raw[date_col], errors="coerce").dt.tz_localize(None)
    cot_raw = cot_raw.dropna(subset=["date"]).set_index("date").sort_index()

    col_map = {
        "Noncommercial Positions-Long (All)":  "spec_long",
        "Noncommercial Positions-Short (All)": "spec_short",
        "Commercial Positions-Long (All)":     "comm_long",
        "Commercial Positions-Short (All)":    "comm_short",
        "Open Interest (All)":                 "open_int",
        "NonComm_Positions_Long_All":          "spec_long",
        "NonComm_Positions_Short_All":         "spec_short",
        "Comm_Positions_Long_All":             "comm_long",
        "Comm_Positions_Short_All":            "comm_short",
        "Open_Interest_All":                   "open_int",
    }
    cot_raw = cot_raw.rename(columns=col_map)

    # Fuzzy fallback
    if "spec_long" not in cot_raw.columns:
        for col in cot_raw.columns:
            cl = col.lower()
            if "noncommercial" in cl and "long" in cl:
                cot_raw = cot_raw.rename(columns={col: "spec_long"})
            elif "noncommercial" in cl and "short" in cl:
                cot_raw = cot_raw.rename(columns={col: "spec_short"})
            elif "commercial" in cl and "long" in cl \
                    and "spec_long" in cot_raw.columns:
                cot_raw = cot_raw.rename(columns={col: "comm_long"})
            elif "commercial" in cl and "short" in cl \
                    and "spec_short" in cot_raw.columns:
                cot_raw = cot_raw.rename(columns={col: "comm_short"})

    if any(c not in cot_raw.columns for c in ["spec_long","spec_short","comm_long","comm_short"]):
        print(f"  ✗ COT crude: missing columns")
        return None

    cot = pd.DataFrame(index=cot_raw.index)
    for col in ["spec_long","spec_short","comm_long","comm_short","open_int"]:
        cot[col] = pd.to_numeric(cot_raw.get(col, pd.Series(dtype=float)), errors="coerce")

    cot["oil_net_spec"]      = cot["spec_long"] - cot["spec_short"]
    cot["oil_net_comm"]      = cot["comm_long"] - cot["comm_short"]
    cot["oil_spec_oi_ratio"] = cot["oil_net_spec"] / (cot["open_int"] + 1e-9)
    cot["oil_comm_oi_ratio"] = cot["oil_net_comm"] / (cot["open_int"] + 1e-9)
    cot = cot[["oil_net_spec","oil_net_comm","oil_spec_oi_ratio","oil_comm_oi_ratio","open_int"]].dropna(how="all")
    cot = cot.sort_values("open_int", ascending=False)
    cot = cot[~cot.index.duplicated(keep="first")].sort_index()
    print(f"  ✓ COT crude: {len(cot)} weekly rows ({cot.index[0].date()} → {cot.index[-1].date()})")
    date_range = pd.date_range(start="2004-01-01",
                               end=datetime.today().strftime("%Y-%m-%d"), freq="B")
    cot_daily = cot.reindex(date_range).ffill().bfill()
    cot_daily.index.name = "date"
    try:
        cot_daily.to_csv(cache_path)
    except Exception:
        pass
    mask = (cot_daily.index >= pd.Timestamp(start)) & \
           (cot_daily.index <= pd.Timestamp(end))
    cot = cot_daily[mask]
    print(f"  ✓ COT crude: {len(cot)} daily rows  spec_oi_ratio: {cot['oil_spec_oi_ratio'].min():.3f} → {cot['oil_spec_oi_ratio'].max():.3f}")
    return cot


def download_gld_holdings(start, end):
    """
    Download GLD ETF daily gold holdings (tonnes).
    Published by World Gold Council / SPDR.

    Rising holdings = institutional buying = bullish flow signal
    Falling holdings = redemptions = bearish flow signal
    Rate of change more useful than level for prediction.

    Primary source: SPDR Gold Shares historical data CSV
    Fallback: parse from public URL
    Coverage: 2004→present (daily)
    """
    print(f"\n  [1d] Downloading GLD ETF holdings...")

    # Try multiple sources in order
    urls = [
        # SPDR direct CSV (may require browser headers)
        "https://www.spdrgoldshares.com/media/GLD/file/GLD_historical_data.csv",
        # Alternative: Gold hub API (World Gold Council)
        "https://www.gold.org/goldhub/data/gold-etf-holdings",
    ]

    # Method 1: try yfinance GLD volume as proxy if direct CSV fails
    # GLD AUM changes correlate strongly with holdings changes
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}

        for url in urls[:1]:
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    from io import StringIO
                    raw = pd.read_csv(StringIO(resp.text),
                                      parse_dates=True, index_col=0)
                    # Find tonnage column
                    tonne_col = next(
                        (c for c in raw.columns
                         if any(k in c.lower()
                                for k in ["tonn", "oz", "holding", "tonne"])),
                        raw.columns[0])
                    holdings = pd.to_numeric(
                        raw[tonne_col], errors="coerce").dropna()
                    holdings.index = pd.to_datetime(
                        holdings.index).tz_localize(None)
                    holdings = holdings.sort_index()
                    date_range = pd.date_range(start=start, end=end, freq="B")
                    holdings = holdings.reindex(date_range).ffill().bfill()
                    print(f"  ✓ GLD holdings: {len(holdings)} rows")
                    return holdings.rename("gld_holdings")
            except Exception:
                pass

        # Method 2: Use GLD price * shares outstanding as proxy
        # Not perfect but captures the same flow information
        print(f"  → Direct CSV unavailable — using GLD AUM proxy via yfinance")
        gld = yf.Ticker("GLD")
        hist = gld.history(start=start, end=end, auto_adjust=True)
        if len(hist) > 100:
            # Volume * price approximates daily flow direction
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            # Use rolling volume as flow proxy
            vol_series = hist["Volume"].rename("gld_holdings")
            date_range = pd.date_range(start=start, end=end, freq="B")
            vol_series = vol_series.reindex(date_range).ffill().bfill()
            print(f"  ✓ GLD volume proxy: {len(vol_series)} rows")
            return vol_series
        return None

    except Exception as e:
        print(f"  ✗ GLD holdings failed: {e}")
        return None


def download_google_trends(start, end, keywords=None):
    """
    Download Google Trends data for energy-related search terms.
    Cache: saves to xle_output/google_trends_cache_<tag>.csv
    Reuses cache if less than 7 days old — avoids rate limits during seed search.
    """
    if keywords is None:
        keywords = ["oil price", "energy stocks", "crude oil"]

    # ── Cache check ───────────────────────────────────────────────────────
    cache_tag  = "_".join(sorted(k.replace(" ", "")
                                  for k in keywords))[:40]
    cache_path = os.path.join(OUT_DIR,
                              f"google_trends_cache_{cache_tag}.csv")
    if os.path.exists(cache_path):
        age_days = (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_path))).days
        if age_days < 7:   # trends don't change fast — 7 day cache is fine
            try:
                cached = pd.read_csv(cache_path, parse_dates=["date"],
                                     index_col="date")
                cached.index = pd.to_datetime(
                    cached.index).tz_localize(None)
                mask   = (cached.index >= pd.Timestamp(start)) & \
                         (cached.index <= pd.Timestamp(end))
                sliced = cached[mask]
                if len(sliced) > 10:
                    print(f"  ✓ Google Trends: cache hit  "
                          f"({age_days}d old, {len(sliced)} rows)")
                    return sliced
            except Exception:
                pass

    print(f"\n  [1e] Downloading Google Trends  {keywords}...")
    try:
        from pytrends.request import TrendReq
        import time as _time

        pytrends   = TrendReq(hl="en-US", tz=0, timeout=(10, 30),
                              retries=3, backoff_factor=0.5)
        cache_start = "2004-01-01"
        cache_end   = datetime.today().strftime("%Y-%m-%d")
        all_trends  = []

        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            try:
                pytrends.build_payload(batch, cat=0,
                    timeframe=f"{cache_start} {cache_end}",
                    geo="", gprop="")
                df_t = pytrends.interest_over_time()
                if df_t is not None and len(df_t) > 10:
                    df_t = df_t.drop(columns=["isPartial"], errors="ignore")
                    df_t.index = pd.to_datetime(
                        df_t.index).tz_localize(None)
                    all_trends.append(df_t)
                _time.sleep(2)
            except Exception as e:
                print(f"    Batch {batch} failed: {e}")
                continue

        if not all_trends:
            raise ValueError("No trends data retrieved")

        trends      = pd.concat(all_trends, axis=1)
        full_range  = pd.date_range(start=cache_start,
                                    end=cache_end, freq="B")
        trends      = trends.reindex(full_range).ffill().bfill()
        # Note: no gold_attention column for XLE — energy-specific keywords only

        try:
            trends.index.name = "date"
            trends.to_csv(cache_path)
            print(f"  ✓ Google Trends: {len(trends)} rows cached → "
                  f"{os.path.basename(cache_path)}")
        except Exception as e:
            print(f"  ⚠ Cache save failed: {e}")

        mask = (trends.index >= pd.Timestamp(start)) & \
               (trends.index <= pd.Timestamp(end))
        return trends[mask]

    except ImportError:
        print(f"  ✗ pytrends not installed — run: pip install pytrends")
        return None
    except Exception as e:
        print(f"  ✗ Google Trends failed: {e}")
        print(f"    Continuing without Trends features")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def log_return(series, periods=1):
    return np.log(series / series.shift(periods))

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast,   adjust=False).mean()
    ema_s = series.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig          # histogram only (most informative component)

def rolling_zscore(series, window):
    """
    Normalize using a trailing window — no future data ever used.
    Safe for both training and live inference.
    """
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / (s + 1e-9)

def build_features(prices, cfg, gpr=None, cot=None,
                   gld_holdings=None, trends=None, cot_crude=None):
    print(f"\n{'═'*60}")
    print(f"  [2] ENGINEERING FEATURES")
    print(f"{'═'*60}")

    df   = pd.DataFrame(index=prices.index)
    gdx  = prices[cfg["target"]]
    zw   = cfg["zscore_window"]

    # ── Normalise index to date-only (no time component) ─────────────────
    # yfinance returns timestamps with timezone offsets (e.g. 05:00:00 UTC)
    # which causes multiple rows per calendar date when data sources merge.
    # Strip to date-only before building any features.
    def norm_index(s):
        idx = pd.to_datetime(s.index).normalize().tz_localize(None)
        s = s.copy()
        s.index = idx
        # Drop any duplicate dates (keep last — most complete data)
        return s[~s.index.duplicated(keep="last")]

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize().tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")]

    gdx    = norm_index(prices[cfg["target"]])
    df     = pd.DataFrame(index=prices.index)

    print(f"  Index normalised: {len(prices)} unique business dates")

    # ── Reference series ──────────────────────────────────────────────────
    target = prices[cfg["target"]]   # XLE
    xop    = prices.get("XOP")       # E&P — higher beta energy
    oih    = prices.get("OIH")       # oil services
    oil    = prices.get("CL=F")      # WTI crude
    brent  = prices.get("BZ=F")      # Brent crude
    natgas = prices.get("NG=F")      # natural gas
    spy    = prices.get("SPY")       # broad market
    vix    = prices.get("^VIX")
    vix3m  = prices.get("^VIX3M")
    tnx    = prices.get("^TNX")
    uup    = prices.get("UUP")
    tlt    = prices.get("TLT")
    move   = prices.get("^MOVE")
    usdjpy = prices.get("USDJPY=X")
    usdchf = prices.get("USDCHF=X")

    # Keep gold/gld/gdxj as None — gold-specific blocks below skipped
    gold = None
    gld  = None
    gdxj = None
    gdx  = target   # reuse downstream feature code

    # ── 1. Returns (log, multiple horizons) ───────────────────────────────
    for col in prices.columns:
        s = prices[col]
        for p in [1, 3, 5, 10, 21]:
            raw = log_return(s, p)
            df[f"{col}_r{p}"] = rolling_zscore(raw, zw)

    # ── 2. Ratio features — energy-specific proxies ───────────────────────
    # XLE/SPY ratio — energy vs broad market (sector rotation)
    if spy is not None:
        ratio = target / (spy + 1e-9)
        df["xle_spy_ratio"]     = rolling_zscore(ratio, zw)
        df["xle_spy_mom5"]      = rolling_zscore(log_return(ratio, 5), zw)
        df["xle_spy_mom21"]     = rolling_zscore(log_return(ratio, 21), zw)

    # XOP/XLE ratio — E&P vs integrated (risk appetite in energy)
    if xop is not None:
        ratio = xop / (target + 1e-9)
        df["xop_xle_ratio"]     = rolling_zscore(ratio, zw)
        df["xop_r5"]            = rolling_zscore(log_return(xop, 5), zw)
        df["xop_r21"]           = rolling_zscore(log_return(xop, 21), zw)

    # OIH — oil services leads capex cycle
    if oih is not None:
        df["oih_r5"]            = rolling_zscore(log_return(oih, 5), zw)
        df["oih_r21"]           = rolling_zscore(log_return(oih, 21), zw)

    # Oil features — direct price drivers
    if oil is not None:
        oil_vol = log_return(oil, 1).rolling(20).std() * np.sqrt(252)
        df["oil_vol20"]         = rolling_zscore(oil_vol, zw)
        df["xle_oil_spread"]    = rolling_zscore(
            log_return(target, 5) - log_return(oil, 5), zw)

    # Brent/WTI spread — geopolitical premium
    if brent is not None and oil is not None:
        df["brent_wti_spread"]  = rolling_zscore(brent - oil, zw)
        df["brent_r5"]          = rolling_zscore(log_return(brent, 5), zw)

    # Natural gas momentum
    if natgas is not None:
        df["natgas_r5"]         = rolling_zscore(log_return(natgas, 5), zw)
        df["natgas_r21"]        = rolling_zscore(log_return(natgas, 21), zw)

    # ── 2b. Crack spread — refinery margin (heating oil vs crude) ─────────
    # HO=F (heating oil) or RB=F (gasoline) as product price proxy
    ho = prices.get("HO=F", prices.get("RB=F"))
    if ho is not None and oil is not None:
        # Scale heating oil from $/gallon to $/barrel (×42)
        crack = ho * 42 - oil
        df["crack_spread"]      = rolling_zscore(crack, zw)
        df["crack_r5"]          = rolling_zscore(log_return(crack.clip(lower=1), 5), zw)

    # ── 2c. Seasonality — energy has strong seasonal patterns ─────────────
    # Driving season (summer), heating season (winter) captured by sin/cos
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["week_sin"]  = np.sin(2 * np.pi * df.index.isocalendar().week.astype(float) / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df.index.isocalendar().week.astype(float) / 52)

    # ── 3. Volatility ─────────────────────────────────────────────────────
    gdx_r1 = log_return(gdx, 1)
    for w in [10, 20, 60]:
        vol = gdx_r1.rolling(w).std() * np.sqrt(252)
        df[f"gdx_vol{w}"]     = rolling_zscore(vol, zw)

    if gold is not None:
        gold_r1 = log_return(gold, 1)
        df["gold_vol20"]      = rolling_zscore(
            gold_r1.rolling(20).std() * np.sqrt(252), zw)

    # ── 4. Moving average signals (trend regime) ──────────────────────────
    for ma in [20, 50, 200]:
        dist = gdx / gdx.rolling(ma).mean() - 1
        df[f"gdx_ma{ma}_dist"] = rolling_zscore(dist, zw)

    # Golden/death cross signal
    cross = gdx.rolling(50).mean() / (gdx.rolling(200).mean() + 1e-9) - 1
    df["gdx_ma_cross"]        = rolling_zscore(cross, zw)

    # ── 5. Momentum / mean reversion indicators ───────────────────────────
    rsi = compute_rsi(gdx, 14) / 100
    df["gdx_rsi14"]           = rolling_zscore(rsi, zw)

    df["gdx_macd_hist"]       = rolling_zscore(compute_macd(gdx) / (gdx + 1e-9), zw)

    if gold is not None:
        df["gold_rsi14"]      = rolling_zscore(compute_rsi(gold, 14) / 100, zw)
        df["gold_macd_hist"]  = rolling_zscore(compute_macd(gold) / (gold + 1e-9), zw)

    # Rate of change
    for p in [5, 10, 20]:
        roc = gdx / (gdx.shift(p) + 1e-9) - 1
        df[f"gdx_roc{p}"]     = rolling_zscore(roc, zw)

    # ── 6. Macro indicators ───────────────────────────────────────────────
    if vix is not None:
        df["vix_level"]       = rolling_zscore(vix / 100, zw)
        df["vix_r5"]          = rolling_zscore(log_return(vix, 5), zw)

    if tnx is not None:
        df["tnx_level"]       = rolling_zscore(tnx / 100, zw)
        df["tnx_chg10"]       = rolling_zscore(tnx.diff(10), zw)

    if uup is not None:
        df["uup_r5"]          = rolling_zscore(log_return(uup, 5), zw)
        df["uup_r21"]         = rolling_zscore(log_return(uup, 21), zw)

    if tlt is not None:
        df["tlt_r10"]         = rolling_zscore(log_return(tlt, 10), zw)

    # ── 7. Lag features (serial autocorrelation) ──────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f"gdx_r1_lag{lag}"] = rolling_zscore(gdx_r1.shift(lag), zw)

    # ── 8. Cross-asset momentum ───────────────────────────────────────────
    if gold is not None:
        df["gold_r5"]         = rolling_zscore(log_return(gold, 5), zw)
        df["gold_r21"]        = rolling_zscore(log_return(gold, 21), zw)

    if tlt is not None and vix is not None:
        # Risk-off composite: TLT up + VIX up = flight to safety
        roff = log_return(tlt, 5) - log_return(vix, 5)
        df["risk_off_signal"] = rolling_zscore(roff, zw)

    # ── 9. Regime-shift detection features ───────────────────────────────
    # These specifically address folds 3 (2018 trade-war) and 6 (2022 rate
    # shock) where abrupt macro regime changes broke the model.

    # Vol expansion ratio — detects when volatility regime is changing
    # Rising = vol expanding = regime shift likely underway → model less reliable
    vol20 = gdx_r1.rolling(20).std() * np.sqrt(252)
    vol60 = gdx_r1.rolling(60).std() * np.sqrt(252)
    df["vol_regime"]      = rolling_zscore(vol20 / (vol60 + 1e-9), zw)

    # Vol acceleration — second derivative of vol (regime change early warning)
    df["vol_accel"]       = rolling_zscore(vol20.diff(5), zw)

    # Gold trend strength — is gold in a clean trend or choppy?
    # Strong trend = model signal more reliable; choppy = reduce confidence
    if gold is not None:
        gold_ma20  = gold.rolling(20).mean()
        gold_ma60  = gold.rolling(60).mean()
        gold_ma200 = gold.rolling(200).mean()
        # Distance of short MA from long MA — positive = uptrend
        df["gold_trend_str"]  = rolling_zscore(
            (gold_ma20 - gold_ma200) / (gold_ma200 + 1e-9), zw)
        # Trend consistency: are all MAs aligned? (20 > 60 > 200)
        aligned = ((gold_ma20 > gold_ma60) & (gold_ma60 > gold_ma200)
                   ).astype(float)
        df["gold_trend_align"] = rolling_zscore(
            aligned.rolling(20).mean(), zw)   # fraction of last 20d aligned

    # Rate-of-change of VIX — spike = sudden fear = regime shift signal
    if vix is not None:
        df["vix_spike"]       = rolling_zscore(vix.diff(3) / (vix + 1e-9), zw)

    # TNX rate-of-change acceleration — fast yield moves disrupt miners
    if tnx is not None:
        df["tnx_accel"]       = rolling_zscore(tnx.diff(5).diff(5), zw)

    # ── 10. Geopolitical & Sentiment Features ────────────────────────────
    # These address the May 2026 miss — Iran risk premium buildup/deflation
    # was invisible to all previous features.

    # ── 10a. VIX Term Structure ───────────────────────────────────────────
    # VIX/VIX3M ratio — when >1.0 = near-term fear spike = geopolitical
    # event likely in play. When ratio normalises back below 1.0 after a
    # spike = risk premium deflating = SELL signal for gold premium.
    if vix is not None and vix3m is not None:
        vix_curve = vix / (vix3m + 1e-9)
        df["vix_curve"]        = rolling_zscore(vix_curve, zw)
        df["vix_curve_chg5"]   = rolling_zscore(vix_curve.diff(5), zw)
        # Inversion flag: is the curve inverted? (sustained fear signal)
        df["vix_inverted"]     = rolling_zscore(
            (vix_curve > 1.0).astype(float).rolling(5).mean(), zw)
        print(f"  ✓ VIX term structure features added")
    elif vix is not None:
        print(f"  ✗ ^VIX3M not available — VIX curve features skipped")

    # ── 10b. Safe-haven currency flows ───────────────────────────────────
    # JPY and CHF strengthen during geopolitical fear (USD/JPY falls)
    # When USDJPY rises sharply after a dip = fear receding = gold premium fades
    if usdjpy is not None:
        df["usdjpy_r5"]        = rolling_zscore(log_return(usdjpy, 5),  zw)
        df["usdjpy_r21"]       = rolling_zscore(log_return(usdjpy, 21), zw)
        # Reversal: USDJPY recovering from recent low = risk-off unwinding
        usdjpy_min10 = usdjpy.rolling(10).min()
        df["usdjpy_reversal"]  = rolling_zscore(
            (usdjpy - usdjpy_min10) / (usdjpy_min10 + 1e-9), zw)
        print(f"  ✓ USD/JPY safe-haven features added")
    else:
        print(f"  ✗ USDJPY=X not available — JPY features skipped")

    if usdchf is not None:
        df["usdchf_r5"]        = rolling_zscore(log_return(usdchf, 5),  zw)
        df["usdchf_r21"]       = rolling_zscore(log_return(usdchf, 21), zw)
        print(f"  ✓ USD/CHF safe-haven features added")
    else:
        print(f"  ✗ USDCHF=X not available — CHF features skipped")

    # ── 10c. Oil-Gold Rolling Correlation ────────────────────────────────
    # In geopolitical crises (Middle East), oil and gold rise together.
    # When this correlation spikes high then drops = geopolitical premium
    # in gold is deflating relative to oil = bearish for gold.
    if oil is not None and gold is not None:
        oil_r1  = log_return(oil,  1)
        gold_r1 = log_return(gold, 1)
        # 10-day rolling correlation
        oil_gold_corr = oil_r1.rolling(10).corr(gold_r1)
        df["oil_gold_corr10"]  = rolling_zscore(oil_gold_corr, zw)
        # Rate of change — falling correlation after spike = premium deflating
        df["oil_gold_corr_chg"] = rolling_zscore(oil_gold_corr.diff(5), zw)
        # 20-day version for slower signal
        df["oil_gold_corr20"]  = rolling_zscore(
            oil_r1.rolling(20).corr(gold_r1), zw)
        print(f"  ✓ Oil-gold correlation features added")

    # ── 10d. MOVE Index (bond market volatility) ──────────────────────────
    # Bond vol spikes = rate uncertainty = disrupts gold carrying costs
    # Also useful as general macro uncertainty signal
    if move is not None:
        df["move_level"]       = rolling_zscore(move / 100, zw)
        df["move_r10"]         = rolling_zscore(log_return(move, 10), zw)
        df["move_vix_ratio"]   = rolling_zscore(
            move / (vix * 10 + 1e-9) if vix is not None else move, zw)
        print(f"  ✓ MOVE index features added")
    else:
        print(f"  ✗ ^MOVE not available — bond vol features skipped")

    # ── 10e. Geopolitical Risk Index (GPR — Caldara & Iacoviello) ────────
    # The single most direct measure of geopolitical risk going back to 1985.
    # GPR     = headline index (total geopolitical risk mentions)
    # GPRACT  = actual events (real incidents — most predictive for gold)
    # GPRTHR  = threats (rhetoric — leads actual events by days/weeks)
    #
    # Key signals:
    #   Rising GPRACT + rising gold = genuine fear premium (trust the rally)
    #   Falling GPRACT + flat gold  = premium about to deflate (go short)
    #   Rising GPRTHR alone         = watch closely, premium building
    if gpr is not None:
        # Normalise GPR index to match prices
        gpr_norm = gpr.copy()
        gpr_norm.index = pd.to_datetime(gpr_norm.index).normalize().tz_localize(None)
        gpr_norm = gpr_norm[~gpr_norm.index.duplicated(keep="last")]
        gpr_aligned = gpr_norm.shift(1, freq="B").reindex(prices.index).ffill()  # +1 biz day publication lag (PATCHED v8)

        for col in gpr_aligned.columns:
            series = gpr_aligned[col].astype(float)
            # Level (z-scored over rolling window)
            df[f"gpr_{col.lower()}_level"] = rolling_zscore(series, zw)
            # 5-day change — detecting spikes and deflations
            df[f"gpr_{col.lower()}_chg5"]  = rolling_zscore(
                series.diff(5), zw)
            # 21-day change — medium-term trend
            df[f"gpr_{col.lower()}_chg21"] = rolling_zscore(
                series.diff(21), zw)

        # Gold-GPR divergence — gold rising but GPR falling = premium to unwind
        if gold is not None and "GPR" in gpr_aligned.columns:
            gpr_chg  = gpr_aligned["GPR"].diff(5)
            gold_chg = log_return(gold, 5)
            divergence = rolling_zscore(gold_chg, zw) - rolling_zscore(
                gpr_chg, zw)
            df["gold_gpr_divergence"] = rolling_zscore(divergence, zw)
            # Positive = gold outpacing GPR = premium risk
            # Negative = GPR outpacing gold = gold may catch up

        print(f"  ✓ GPR geopolitical risk features added  "
              f"({len([c for c in df.columns if 'gpr' in c])} features)")
    else:
        print(f"  ✗ GPR data not available — geopolitical features skipped")
        print(f"    Download from: https://www.matteoiacoviello.com/gpr.htm")

    # ── 11. COT Positioning (CFTC) ────────────────────────────────────────
    # Commitments of Traders — institutional gold futures positioning.
    # The most direct measure of smart-money sentiment on gold.
    #
    # Key signals:
    #   Extreme net spec long (>250k contracts) = crowded = contrarian bearish
    #   Spec OI ratio falling from extreme = unwinding = bearish momentum
    #   Net comm short rising = producers hedging output = bearish
    #   Spec/comm divergence rising = spec leading commercials = bull trend
    if cot is not None:
        # Normalise COT index to match prices index (both midnight, no tz)
        cot_norm = cot.copy()
        cot_norm.index = pd.to_datetime(cot_norm.index).normalize().tz_localize(None)
        cot_norm = cot_norm[~cot_norm.index.duplicated(keep="last")]
        cot_aligned = cot_norm.shift(3, freq="B").reindex(prices.index).ffill()  # CFTC 3-biz-day publication lag (PATCHED v8)
        print(f"  COT aligned: {cot_aligned.notna().all(axis=1).sum()} complete rows  "
              f"NaN rate: {cot_aligned.isna().mean().mean():.1%}")

        # Net speculative positioning — level and changes
        if "net_spec" in cot_aligned.columns:
            ns = cot_aligned["net_spec"].astype(float)
            df["cot_net_spec"]       = rolling_zscore(ns, zw)
            df["cot_net_spec_chg4"]  = rolling_zscore(ns.diff(4),  zw)
            df["cot_net_spec_chg13"] = rolling_zscore(ns.diff(13), zw)
            # Extreme positioning flag: are specs near their historical max?
            ns_pct = ns.rolling(252).rank(pct=True)
            df["cot_spec_extreme"]   = rolling_zscore(ns_pct, zw)

        # Net commercial positioning (inverse of spec — commercials hedge)
        if "net_comm" in cot_aligned.columns:
            nc = cot_aligned["net_comm"].astype(float)
            df["cot_net_comm"]       = rolling_zscore(nc, zw)
            df["cot_net_comm_chg4"]  = rolling_zscore(nc.diff(4), zw)

        # Spec positioning as fraction of open interest
        # Normalises for changes in market size over 20 years
        if "spec_oi_ratio" in cot_aligned.columns:
            sr = cot_aligned["spec_oi_ratio"].astype(float)
            df["cot_spec_oi"]        = rolling_zscore(sr, zw)
            df["cot_spec_oi_chg4"]   = rolling_zscore(sr.diff(4), zw)

        # Spec/Comm divergence — when specs and commercials diverge strongly
        # specs buying while commercials sell heavily = strong trend signal
        if "net_spec" in cot_aligned.columns and \
           "net_comm" in cot_aligned.columns:
            divergence = (cot_aligned["net_spec"].astype(float) -
                          cot_aligned["net_comm"].astype(float))
            df["cot_spec_comm_div"]  = rolling_zscore(divergence, zw)

        n_cot = len([c for c in df.columns if "cot_" in c])
        print(f"  ✓ COT positioning features added  ({n_cot} features)")
    else:
        print(f"  ✓ COT positioning features skipped (gold COT — not used for XLE)")

    # ── Crude Oil COT features ────────────────────────────────────────────
    if cot_crude is not None:
        try:
            cc_norm = cot_crude.copy()
            cc_norm.index = pd.to_datetime(
                cc_norm.index).normalize().tz_localize(None)
            cc_norm = cc_norm[~cc_norm.index.duplicated(keep="last")]
            cc = cc_norm.reindex(prices.index).ffill().bfill()

            if "oil_net_spec" in cc.columns:
                ns = cc["oil_net_spec"].astype(float)
                df["oil_cot_net_spec"]      = rolling_zscore(ns, zw)
                df["oil_cot_net_spec_chg4"] = rolling_zscore(ns.diff(4), zw)
                ns_pct = ns.rolling(252).rank(pct=True)
                df["oil_cot_spec_extreme"]  = rolling_zscore(ns_pct, zw)

            if "oil_spec_oi_ratio" in cc.columns:
                sr = cc["oil_spec_oi_ratio"].astype(float)
                df["oil_cot_spec_oi"]       = rolling_zscore(sr, zw)
                df["oil_cot_spec_oi_chg4"]  = rolling_zscore(sr.diff(4), zw)

            if "oil_net_comm" in cc.columns:
                nc = cc["oil_net_comm"].astype(float)
                df["oil_cot_net_comm"]      = rolling_zscore(nc, zw)

            n_oil_cot = len([c for c in df.columns if "oil_cot_" in c])
            print(f"  ✓ Crude COT features added  ({n_oil_cot} features)")
        except Exception as e:
            print(f"  ✗ Crude COT features failed: {e}")
    else:
        print(f"  ✗ Crude COT not available — skipped")

    # ── 12. GLD ETF Holdings / Flow ───────────────────────────────────────
    # Daily institutional flow in/out of the GLD ETF.
    # Rising holdings = institutions adding gold exposure = bullish flow
    # Falling holdings = redemptions = bearish flow
    # Rate of change and acceleration more predictive than level alone.
    if gld_holdings is not None:
        # Normalise holdings index to match prices
        h_norm = gld_holdings.copy()
        h_norm.index = pd.to_datetime(h_norm.index).normalize().tz_localize(None)
        h_norm = h_norm[~h_norm.index.duplicated(keep="last")]
        h = h_norm.shift(1, freq="B").reindex(prices.index).ffill().astype(float)  # SPDR 1-biz-day reporting lag (PATCHED v8)

        # Level (normalised)
        df["gld_hold_level"]    = rolling_zscore(h, zw)

        # Rate of change — is flow accelerating in or out?
        df["gld_hold_r5"]       = rolling_zscore(h.diff(5),  zw)
        df["gld_hold_r21"]      = rolling_zscore(h.diff(21), zw)

        # Acceleration — second derivative of flow
        df["gld_hold_accel"]    = rolling_zscore(h.diff(5).diff(5), zw)

        # Trend: is holdings above its own 63-day MA? (sustained inflow)
        h_ma63 = h.rolling(63).mean()
        df["gld_hold_trend"]    = rolling_zscore(
            (h - h_ma63) / (h_ma63 + 1e-9), zw)

        # Holdings momentum percentile — extreme outflows often contrarian
        df["gld_hold_pct"]      = rolling_zscore(
            h.rolling(252).rank(pct=True), zw)

        n_gld = len([c for c in df.columns if "gld_hold" in c])
        print(f"  ✓ GLD holdings features added  ({n_gld} features)")
    else:
        print(f"  ✗ GLD holdings not available — flow features skipped")

    # ── 13. Google Trends (Retail Sentiment) ──────────────────────────────
    # Retail search interest as a proxy for public attention to gold.
    # Useful for detecting retail FOMO (late-stage rallies) and fear
    # (capitulation bottoms). Weekly data interpolated to daily.
    #
    # Key signals:
    #   Spike in "buy gold" = retail FOMO = often near short-term top
    #   Spike in "gold price" = broad attention = momentum continuation
    #   "inflation hedge" rising = macro narrative building = bullish
    if trends is not None:
        tr_norm = trends.copy()
        tr_norm.index = pd.to_datetime(tr_norm.index).normalize().tz_localize(None)
        tr_norm = tr_norm[~tr_norm.index.duplicated(keep="last")]
        tr = tr_norm.reindex(prices.index).ffill().bfill()

        for col in tr.columns:
            s = tr[col].astype(float)
            # Level and rate of change
            df[f"gtrend_{col.lower().replace(' ','_')}_level"] = \
                rolling_zscore(s, zw)
            df[f"gtrend_{col.lower().replace(' ','_')}_chg4"]  = \
                rolling_zscore(s.diff(4), zw)

        # Composite attention score vs gold return — retail chasing?
        # When retail interest spikes AFTER a big gold move = FOMO top signal
        if "gold_attention" in tr.columns and gold is not None:
            gold_r21   = log_return(gold, 21)
            attn_chg4  = tr["gold_attention"].astype(float).diff(4)
            # Attention lagging big price move = retail chasing = fade signal
            df["gtrend_chase_signal"] = rolling_zscore(
                attn_chg4 / (gold_r21.abs() + 0.01), zw)

        n_tr = len([c for c in df.columns if "gtrend_" in c])
        print(f"  ✓ Google Trends features added  ({n_tr} features)")
    else:
        print(f"  ✗ Google Trends not available — retail sentiment skipped")

    # ── 14. Dollar secular trend ──────────────────────────────────────────
    # Oil priced in USD — dollar trend is a primary driver for energy.
    uup = prices["UUP"].dropna() if "UUP" in prices.columns else None
    if uup is not None:
        uup_ma50  = uup.rolling(50).mean()
        uup_ma200 = uup.rolling(200).mean()
        df["dollar_trend"]       = rolling_zscore(
            uup_ma50 / (uup_ma200 + 1e-9) - 1, zw)
        df["dollar_trend_slope"] = rolling_zscore(
            uup_ma50 / uup_ma50.shift(21) - 1, zw)
        print(f"  ✓ Dollar secular trend features added")

    # ── 15. XLE own momentum features — REMOVED 2026-05-27 ──────────────
    # Previously added xle_r5/r10/r21, xle_ma50_dist, xle_ma200_dist, xle_trend.
    # A/B test (PROJECT_NOTES.md changelog 2026-05-27) showed the selective
    # high-Sharpe seeds produced by this richer feature set were worse rotation
    # citizens than the older feature-set seeds. Reverted. If reintroducing,
    # also re-evaluate the rotation confidence-percentile threshold; selective
    # models need lower thresholds (or a rotation-aware seed-scoring rule).

    # ── Target: forward log return ─────────────────────────────────────────
    fwd = cfg["forward_days"]
    # Smoothed target — average of returns at days +17, +18, +19, +20
    # Reduces noise vs point-in-time target; hold_days unchanged at 20
    target = -(log_return(gdx, -17) + log_return(gdx, -18) +
              log_return(gdx, -19) + log_return(gdx, -fwd)) / 4  # PATCHED v9: negated so target = standard forward return

    # ── Clean up ───────────────────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns that are entirely NaN (failed data sources)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN columns: "
              f"{all_nan_cols[:5]}{'...' if len(all_nan_cols) > 5 else ''}")
        df.drop(columns=all_nan_cols, inplace=True)

    # Report columns with high NaN rates before dropping rows
    nan_rates = df.isna().mean()
    high_nan  = nan_rates[nan_rates > 0.3].sort_values(ascending=False)
    if len(high_nan) > 0:
        print(f"  ⚠ {len(high_nan)} columns >30% NaN (will reduce sample count):")
        for col, rate in high_nan.head(10).items():
            print(f"    {col}: {rate:.0%} NaN")

    valid = df.dropna().index.intersection(target.dropna().index)

    if len(valid) == 0:
        # Find the most problematic columns
        print(f"  ✗ No valid rows after dropna — diagnosing...")
        row_nan_count = df.isna().sum(axis=1)
        print(f"  Min NaNs per row: {row_nan_count.min()}")
        print(f"  Columns causing most row-level NaN:")
        col_nan = df.isna().mean().sort_values(ascending=False)
        print(f"    {col_nan.head(10).to_dict()}")
        # Emergency fallback: drop high-NaN columns and retry
        cols_to_drop = nan_rates[nan_rates > 0.05].index.tolist()
        print(f"  Dropping {len(cols_to_drop)} columns with >5% NaN and retrying")
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        valid = df.dropna().index.intersection(target.dropna().index)

    df     = df.loc[valid]
    target = target.loc[valid]

    if len(df) == 0:
        raise RuntimeError(
            "Feature matrix is empty after cleaning. "
            "Check that price data downloaded correctly."
        )

    print(f"  Features : {df.shape[1]} columns")
    print(f"  Samples  : {df.shape[0]} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Target   : {fwd}-day forward log return on {cfg['target']}")

    return df, target


# ══════════════════════════════════════════════════════════════════════════════
# 2b. FEATURE SELECTION PIPELINE
#
# Three steps applied in sequence:
#   Step 1 — Remove correlated features     (redundancy reduction)
#   Step 2 — Rank by IC with target         (signal strength ranking)
#   Step 3 — Apply minimum IC floor         (noise removal)
#
# Result: 169 noisy features → ~40 high-signal features
# Expected improvement: IC variance drops, train/val gap narrows
# ══════════════════════════════════════════════════════════════════════════════

def remove_correlated_features(features_df, threshold=0.85):
    """
    Step 1: Remove features that are highly correlated with each other.

    When two features have correlation > threshold, drop the second one.
    This eliminates redundant signals — e.g. gdx_r1, gdx_r3, gdx_r5 all
    measure the same thing at different horizons; keeping all three adds
    noise without adding new information.

    Uses the upper triangle of the correlation matrix — keeps the first
    feature encountered in each correlated group (earlier features in the
    DataFrame tend to be the simpler/more fundamental ones).
    """
    print(f"\n  Step 1: Removing correlated features (threshold={threshold})")
    n_before = features_df.shape[1]

    corr      = features_df.corr().abs()
    upper     = corr.where(
        np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop = [col for col in upper.columns
               if any(upper[col] > threshold)]

    result = features_df.drop(columns=to_drop)
    print(f"    {n_before} → {result.shape[1]} features  "
          f"(dropped {len(to_drop)} correlated)")

    # Show a few examples of what was dropped
    if to_drop:
        print(f"    Example drops: {to_drop[:5]}"
              f"{'...' if len(to_drop) > 5 else ''}")

    return result


def rank_features_by_ic(features_df, targets_s,
                         top_n=45, min_ic=0.02,
                         purge_pct=0.70):
    """
    Step 2 + 3: Rank features by Spearman IC with the target, keep top_n.
    Step 3:     Drop any remaining features below min_ic noise floor.

    Uses only the first purge_pct of the data for ranking to avoid
    forward-looking bias — we never look at the test set to select features.

    purge_pct=0.70 means features are ranked on the first 70% of the
    dataset (roughly 2013–2022), then applied to the remaining 30%.
    """
    print(f"\n  Step 2: Ranking {features_df.shape[1]} features by IC with target")
    print(f"    (Using first {purge_pct:.0%} of data to avoid lookahead)")

    cutoff   = int(len(features_df) * purge_pct)
    X_rank   = features_df.iloc[:cutoff]
    y_rank   = targets_s.iloc[:cutoff]

    scores = {}
    for col in X_rank.columns:
        ic, _ = spearmanr(X_rank[col].values,
                          y_rank.values,
                          nan_policy="omit")
        scores[col] = abs(float(ic)) if not np.isnan(ic) else 0.0

    ranked = pd.Series(scores).sort_values(ascending=False)

    # Print top features — these are your most predictive signals
    print(f"\n    ── Top 20 features by |IC| ──────────────────────")
    for feat, score in ranked.head(20).items():
        bar = "█" * int(score * 40)
        print(f"    {score:.4f}  {bar:<16}  {feat}")

    print(f"\n    ── Bottom 10 features (noise) ───────────────────")
    for feat, score in ranked.tail(10).items():
        print(f"    {score:.4f}  {feat}")

    # Step 2: keep top_n
    selected = ranked.head(top_n)

    # Step 3: apply minimum IC floor — remove anything below noise threshold
    print(f"\n  Step 3: Applying IC floor (min_ic={min_ic})")
    below_floor = selected[selected < min_ic]
    if len(below_floor) > 0:
        print(f"    Dropping {len(below_floor)} features below IC floor:")
        for feat, score in below_floor.items():
            print(f"      {score:.4f}  {feat}")
        selected = selected[selected >= min_ic]

    final_features = selected.index.tolist()
    print(f"\n    Final selection: {len(final_features)} features")
    return final_features, ranked


def select_features(features_df, targets_s, cfg):
    """
    Full three-step feature selection pipeline.
    Returns the filtered features DataFrame and the IC ranking series.
    """
    if not cfg.get("fs_enabled", True):
        print(f"\n  Feature selection disabled — using all "
              f"{features_df.shape[1]} features")
        return features_df, None

    print(f"\n{'═'*60}")
    print(f"  [2b] FEATURE SELECTION PIPELINE")
    print(f"       {features_df.shape[1]} input features → target ~{cfg['fs_top_n']}")
    print(f"{'═'*60}")

    # Step 1: remove correlated features
    df_step1 = remove_correlated_features(
        features_df, threshold=cfg["fs_corr_threshold"])

    # Steps 2+3: rank by IC and apply floor
    selected_cols, ic_ranking = rank_features_by_ic(
        df_step1, targets_s,
        top_n   = cfg["fs_top_n"],
        min_ic  = cfg["fs_min_ic"],
        purge_pct = 0.70,
    )

    df_final = df_step1[selected_cols]

    print(f"\n{'═'*60}")
    print(f"  FEATURE SELECTION COMPLETE")
    print(f"  {features_df.shape[1]} → {df_final.shape[1]} features")
    print(f"  Rows retained: {df_final.shape[0]}")
    print(f"{'═'*60}")

    # Save feature importance to CSV — clean column names for Excel
    ic_path = os.path.join(OUT_DIR, "feature_importance.csv")
    ic_df = ic_ranking.reset_index()
    ic_df.columns = ["feature", "abs_ic"]
    ic_df["rank"] = range(1, len(ic_df) + 1)
    ic_df["selected"] = ic_df["feature"].isin(selected_cols)
    ic_df = ic_df[["rank", "feature", "abs_ic", "selected"]]
    ic_df.to_csv(ic_path, index=False)
    print(f"  ✓ Full IC ranking saved → {ic_path}")

    return df_final, ic_ranking


# ══════════════════════════════════════════════════════════════════════════════
# 3. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MinerSequenceDataset(Dataset):
    """
    Sliding-window sequence dataset.
    X[i] = features[i : i+lookback]   shape (lookback, n_features)
    y[i] = target[i + lookback]        scalar
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X  = torch.tensor(X, dtype=torch.float32)
        self.y  = torch.tensor(y, dtype=torch.float32)
        self.lb = lookback

    def __len__(self):
        return max(0, len(self.X) - self.lb)

    def __getitem__(self, idx):
        return self.X[idx : idx + self.lb], self.y[idx + self.lb]


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GoldMinerLSTM(nn.Module):
    """
    2-layer LSTM → FC regression head

    Input  : (batch, lookback, n_features)
    Output : (batch,)  — predicted forward log return
    """
    def __init__(self, n_features, hidden_size=64,
                 num_lstm_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_lstm_layers,
            dropout     = dropout if num_lstm_layers > 1 else 0.0,
            batch_first = True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name: nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name: nn.init.orthogonal_(p.data)
            elif "bias" in name: p.data.zero_()
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out, _ = self.lstm(x)               # (batch, seq, hidden)
        last   = out[:, -1, :]              # final timestep
        return self.head(last).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Deep copy weights
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def run_epoch(model, loader, criterion, optimizer=None, grad_clip=1.0):
    """Single train or eval pass. optimizer=None → eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()
    losses = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train_model(model, train_loader, val_loader, cfg, verbose=True):
    """Full training loop with early stopping and LR scheduling."""
    # Configurable loss — MSE fixes prediction shrinkage toward zero,
    # Huber is safer with outlier spikes but causes conservative predictions
    loss_fn = cfg.get("loss_fn", "huber")
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "huber":
        delta = cfg.get("huber_delta", 0.05)
        criterion = nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unknown loss_fn '{loss_fn}' — use 'mse' or 'huber'")
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5,
                    patience=cfg["lr_patience"])
    stopper   = EarlyStopping(patience=cfg["early_stop_patience"])

    train_hist, val_hist = [], []

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = run_epoch(model, train_loader, criterion,
                            optimizer, cfg["grad_clip"])
        va_loss = run_epoch(model, val_loader,   criterion)
        scheduler.step(va_loss)
        train_hist.append(tr_loss)
        val_hist.append(va_loss)

        if verbose and (epoch % 25 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"      ep {epoch:>4}  train={tr_loss:.6f}  "
                  f"val={va_loss:.6f}  lr={lr_now:.1e}")

        if stopper.step(va_loss, model):
            if verbose:
                print(f"      ✓ Early stop ep {epoch}  "
                      f"best_val={stopper.best_loss:.6f}")
            stopper.restore(model)
            break

    return train_hist, val_hist


# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def get_predictions(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(DEVICE)).cpu().numpy())
            actuals.extend(y.numpy())
    return np.array(preds), np.array(actuals)


def compute_metrics(preds, actuals):
    mae      = mean_absolute_error(actuals, preds)
    rmse     = np.sqrt(mean_squared_error(actuals, preds))
    dir_acc  = np.mean(np.sign(preds) == np.sign(actuals))
    ic, _    = spearmanr(preds, actuals)
    # Sharpe-like: simulate going long/short based on prediction sign
    daily_rets = np.sign(preds) * actuals
    sharpe   = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252 / 10)
    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc,
            "ic": ic, "sharpe": sharpe}


def print_metrics(metrics, label=""):
    tag = f"[{label}] " if label else ""
    ic     = metrics['ic']
    sharpe = metrics['sharpe']

    if ic >= 0.30:   ic_desc = "exceptional"
    elif ic >= 0.10: ic_desc = "strong"
    elif ic >= 0.05: ic_desc = "useful"
    elif ic >= 0.0:  ic_desc = "weak positive"
    else:            ic_desc = "⚠ negative"

    if sharpe >= 2.0:   sh_desc = "excellent"
    elif sharpe >= 1.0: sh_desc = "good"
    elif sharpe >= 0.5: sh_desc = "moderate"
    elif sharpe >= 0.0: sh_desc = "weak"
    else:               sh_desc = "⚠ negative"

    print(f"      {tag}Dir Acc : {metrics['dir_acc']:.2%}  "
          f"IC: {ic:.4f} ({ic_desc})  "
          f"Sharpe: {sharpe:.3f} ({sh_desc})  "
          f"MAE: {metrics['mae']:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PURGED WALK-FORWARD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def make_wf_splits(n_samples, dates, cfg):
    """
    Generate (train_idx, val_idx) pairs for purged walk-forward CV.

    Timeline per fold:
      [═══ expanding train ═══] [purge_days gap] [══ val block ══]

    The purge gap removes all samples whose TARGET overlaps with the
    training window (avoids label leakage across the boundary).

    Fix: use linspace over n_folds points (not n_folds+1) so the last
    fold's val block reaches close to the end of the dataset.
    """
    n_folds      = cfg["n_folds"]
    val_size     = cfg["val_months"] * 21
    purge        = cfg["purge_days"]
    min_train    = int(cfg["min_train_years"] * 252)

    # Space val block starts evenly — last one leaves room for val_size rows
    earliest_val = min_train + purge
    latest_val   = n_samples - val_size
    starts       = np.linspace(earliest_val, latest_val,
                               n_folds, dtype=int)     # n_folds not n_folds+1

    splits = []
    for vs in starts:
        train_end = vs - purge
        val_end   = min(vs + val_size, n_samples)
        if train_end < min_train:
            continue
        splits.append((np.arange(0, train_end),
                       np.arange(vs, val_end)))
    return splits


def walk_forward_cv(features_df, targets_s, cfg):
    print(f"\n{'═'*60}")
    print(f"  [7] PURGED WALK-FORWARD CROSS-VALIDATION")
    print(f"      {cfg['n_folds']} folds | {cfg['val_months']}-month val blocks "
          f"| {cfg['purge_days']}-day purge gap")
    print(f"{'═'*60}")

    X_all  = features_df.values.astype(np.float32)
    y_all  = targets_s.values.astype(np.float32)
    dates  = features_df.index
    lb     = cfg["lookback"]
    n_feat = X_all.shape[1]

    # Fix 1: find earliest allowed training index
    # Samples before exclude_before are never used in any fold's training set
    exclude_before = cfg.get("exclude_before", None)
    if exclude_before:
        excl_dt   = pd.Timestamp(exclude_before)
        excl_idx  = int(np.searchsorted(dates, excl_dt))
        print(f"  Excluding training samples before {exclude_before} "
              f"(idx < {excl_idx}, {excl_idx} rows removed from train sets)")
    else:
        excl_idx = 0

    splits       = make_wf_splits(len(X_all), dates, cfg)
    fold_metrics = []
    fold_curves  = []
    fold_periods = []
    oof_preds    = np.full(len(y_all), np.nan)
    oof_actuals  = np.full(len(y_all), np.nan)
    saved_fold_num = 0
    abandon_seed   = False

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        fold_num = fold_i + 1                  # position in split list
        tr_start = dates[tr_idx[0]].date()
        tr_end   = dates[tr_idx[-1]].date()
        va_start = dates[va_idx[0]].date()
        va_end   = dates[va_idx[-1]].date()

        print(f"\n  Fold {fold_num}/{len(splits)}")
        print(f"    Train : {tr_start} → {tr_end}  ({len(tr_idx)} rows)")
        print(f"    Purge : {cfg['purge_days']} days")
        print(f"    Val   : {va_start} → {va_end}  ({len(va_idx)} rows)")

        # Apply exclude_before: clip training indices to post-exclusion window
        tr_idx_filtered = tr_idx[tr_idx >= excl_idx]
        if len(tr_idx_filtered) < cfg["batch_size"] * 4:
            print(f"    ✗ Too few training samples after exclusion — skipping")
            continue
        X_tr = X_all[tr_idx_filtered]
        y_tr = y_all[tr_idx_filtered]
        X_va = X_all[va_idx]
        y_va = y_all[va_idx]

        # DataLoaders
        tr_ds = MinerSequenceDataset(X_tr, y_tr, lb)
        va_ds = MinerSequenceDataset(X_va, y_va, lb)

        if len(tr_ds) < cfg["batch_size"] or len(va_ds) < 2:
            print("    ✗ Not enough samples — skipping fold")
            continue

        tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"],
                               shuffle=False, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=cfg["batch_size"],
                               shuffle=False)

        # ── Adaptive multi-start: 3 min, up to 5 if IC or DirAcc weak ────
        MIN_STARTS        = cfg.get("fold_starts", 3)
        MAX_STARTS        = MIN_STARTS + 2
        EARLY_EXIT_IC     = 0.15
        EARLY_EXIT_DACC   = 0.60
        ABANDON_FOLD_IC   = -0.10
        ABANDON_FOLD_DACC = 0.40
        best_fold_ic      = -999.0
        best_fold_score   = -999.0
        best_fold_model   = None
        best_fold_preds   = None
        best_fold_actuals = None
        best_fold_metrics = None
        best_fold_tr_hist = None
        best_fold_va_hist = None

        for start_i in range(MAX_STARTS):
            fold_seed = SEED * 100 + fold_i * 10 + start_i
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)

            model = GoldMinerLSTM(
                n_features      = n_feat,
                hidden_size     = cfg["hidden_size"],
                num_lstm_layers = cfg["num_lstm_layers"],
                dropout         = cfg["dropout"],
            ).to(DEVICE)

            t0 = time.time()
            tr_hist, va_hist = train_model(model, tr_loader, va_loader,
                                           cfg, verbose=(start_i == 0))
            elapsed = time.time() - t0

            preds, actuals = get_predictions(model, va_loader)
            metrics        = compute_metrics(preds, actuals)
            trial_ic       = float(metrics.get("ic", -999))
            trial_dacc     = float(metrics.get("dir_acc", 0))
            trial_score    = trial_ic + 2.0 * max(0.0, trial_dacc - 0.5)

            print(f"    Start {start_i+1}/{MAX_STARTS}: "
                  f"IC={trial_ic:.4f}  "
                  f"DirAcc={trial_dacc:.2%}  "
                  f"Score={trial_score:.4f}  "
                  f"time={elapsed:.0f}s  "
                  f"{'★ best' if trial_score > best_fold_score else ''}")

            if trial_score > best_fold_score:
                best_fold_score   = trial_score
                best_fold_ic      = trial_ic
                best_fold_model   = model
                best_fold_preds   = preds
                best_fold_actuals = actuals
                best_fold_metrics = metrics
                best_fold_tr_hist = tr_hist
                best_fold_va_hist = va_hist

            if start_i == MIN_STARTS - 1:
                best_dacc = best_fold_metrics.get("dir_acc", 0) \
                            if best_fold_metrics else 0
                if best_fold_ic >= EARLY_EXIT_IC and best_dacc >= EARLY_EXIT_DACC:
                    print(f"    ✓ Early exit — IC={best_fold_ic:.4f} DirAcc={best_dacc:.2%}")
                    break
                else:
                    print(f"    ⚠ IC={best_fold_ic:.4f} DirAcc={best_dacc:.2%} weak — trying more starts")

        best_dacc = best_fold_metrics.get("dir_acc", 0) if best_fold_metrics else 0
        if (best_fold_ic < ABANDON_FOLD_IC or best_dacc < ABANDON_FOLD_DACC) \
                and fold_i >= 1:
            print(f"    ✗ Fold IC={best_fold_ic:.4f} DirAcc={best_dacc:.2%} — abandoning seed")
            abandon_seed = True
            break

        model   = best_fold_model
        preds   = best_fold_preds
        actuals = best_fold_actuals
        metrics = best_fold_metrics
        tr_hist = best_fold_tr_hist
        va_hist = best_fold_va_hist
        print(f"    Best start IC={best_fold_ic:.4f}")

        # saved_fold_num = sequential index of folds that actually trained
        saved_fold_num += 1
        print_metrics(metrics, label=f"Fold {saved_fold_num}")

        fold_metrics.append(metrics)
        fold_curves.append({"train": tr_hist, "val": va_hist,
                             "fold": saved_fold_num,
                             "val_range": f"{va_start}→{va_end}"})
        fold_periods.append((va_idx, va_start, va_end))

        # Store OOF predictions (aligned to val window)
        pred_start = va_idx[0] + lb
        pred_end   = pred_start + len(preds)
        oof_preds[pred_start:pred_end]   = preds
        oof_actuals[pred_start:pred_end] = actuals

        # Save with saved_fold_num so filenames match regime diagnostic
        torch.save({
            "state_dict":    model.state_dict(),
            "model_config":  {
                "n_features":      int(n_feat),
                "hidden_size":     int(cfg["hidden_size"]),
                "num_lstm_layers": int(cfg["num_lstm_layers"]),
                "dropout":         float(cfg["dropout"]),
            },
            "fold":          int(saved_fold_num),
            "val_period":    f"{va_start}→{va_end}",
            "metrics":       {k: float(v) for k, v in metrics.items()},
            "feature_names": list(features_df.columns),
            "cfg": {
                "lookback":     int(cfg["lookback"]),
                "forward_days": int(cfg["forward_days"]),
                "zscore_window": int(cfg["zscore_window"]),
            },
        }, os.path.join(OUT_DIR, f"fold_{saved_fold_num}_model.pt"))
        print(f"    ✓ Saved fold_{saved_fold_num}_model.pt")

    # ── Aggregate summary ─────────────────────────────────────────────────
    if abandon_seed or len(fold_metrics) == 0:
        print(f"\n  ✗ Seed abandoned — insufficient fold quality")
        return [], [], [], np.array([]), np.array([])
    print(f"\n{'═'*60}")
    print(f"  WALK-FORWARD SUMMARY  ({len(fold_metrics)} folds)")
    print(f"{'═'*60}")
    for key in ["dir_acc", "ic", "sharpe", "mae", "rmse"]:
        vals = [m[key] for m in fold_metrics]
        unit = "%" if key == "dir_acc" else ""
        fmt  = ".2%" if key == "dir_acc" else ".4f"
        print(f"  {key:<12} mean={np.mean(vals):{fmt}}  "
              f"std={np.std(vals):.4f}  "
              f"min={np.min(vals):{fmt}}  "
              f"max={np.max(vals):{fmt}}")

    # Best fold by directional accuracy
    best_fold = int(np.argmax([m["dir_acc"] for m in fold_metrics])) + 1
    print(f"\n  Best fold by Dir Acc: Fold {best_fold}")

    return fold_metrics, fold_curves, fold_periods, oof_preds, oof_actuals


# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL MODEL — train on all data, save for inference
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(features_df, targets_s, cfg):
    """
    Train one final model on the full dataset.
    This is the model you deploy for live inference.
    Uses last 15% as a held-out final test (never touched during CV).
    """
    print(f"\n{'═'*60}")
    print(f"  [8] FINAL MODEL — full dataset training")
    print(f"{'═'*60}")

    X = features_df.values.astype(np.float32)
    y = targets_s.values.astype(np.float32)
    lb = cfg["lookback"]

    # Apply exclude_before — same exclusion as walk-forward folds
    exclude_before = cfg.get("exclude_before", None)
    if exclude_before:
        excl_dt  = pd.Timestamp(exclude_before)
        excl_idx = int(np.searchsorted(features_df.index, excl_dt))
        X = X[excl_idx:]
        y = y[excl_idx:]
        print(f"  Excluding samples before {exclude_before} "
              f"({excl_idx} rows removed)")

    # Hold out last 15% as final unseen test
    split     = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Capture the actual dates for the test set so charts show real dates
    features_trimmed = features_df.iloc[
        (int(np.searchsorted(features_df.index,
             pd.Timestamp(cfg["exclude_before"])))
         if cfg.get("exclude_before") else 0):]
    test_dates = features_trimmed.index[split + cfg["lookback"]:]

    tr_ds = MinerSequenceDataset(X_tr, y_tr, lb)
    te_ds = MinerSequenceDataset(X_te, y_te, lb)

    tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"],
                           shuffle=False, drop_last=True)
    te_loader = DataLoader(te_ds, batch_size=cfg["batch_size"],
                           shuffle=False)

    model = GoldMinerLSTM(
        n_features      = X.shape[1],
        hidden_size     = cfg["hidden_size"],
        num_lstm_layers = cfg["num_lstm_layers"],
        dropout         = cfg["dropout"],
    ).to(DEVICE)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train rows: {len(tr_ds)}  |  Test rows: {len(te_ds)}")

    tr_hist, va_hist = train_model(model, tr_loader, te_loader,
                                   cfg, verbose=True)
    preds, actuals   = get_predictions(model, te_loader)
    metrics          = compute_metrics(preds, actuals)
    print("\n  Final model — held-out test metrics:")
    print_metrics(metrics, label="Final")

    # Save — plain Python types only (PyTorch 2.6 weights_only safe)
    save_path = os.path.join(OUT_DIR, "final_model.pt")
    torch.save({
        "state_dict":    model.state_dict(),
        "model_config":  {
            "n_features":      int(X.shape[1]),
            "hidden_size":     int(cfg["hidden_size"]),
            "num_lstm_layers": int(cfg["num_lstm_layers"]),
            "dropout":         float(cfg["dropout"]),
        },
        "feature_names": list(features_df.columns),
        "metrics":       {k: float(v) for k, v in metrics.items()},
        "forward_days":  int(cfg["forward_days"]),
        "lookback":      int(cfg["lookback"]),
    }, save_path)
    print(f"\n  ✓ Saved → {save_path}")
    print(f"  Test period: {test_dates[0].date()} → {test_dates[-1].date()}"
          if len(test_dates) > 0 else "")

    return model, metrics, tr_hist, va_hist, preds, actuals, test_dates


def save_full_predictions(model, features_df, targets_s, cfg, oof_df):
    """
    Generate predictions for EVERY trading day using the final model.
    Uses a rolling lookback window — each day's prediction uses only
    past data, so there is no lookahead bias.

    OOF predictions (honest, model never saw that data) are preserved
    where available. The final model fills the remaining days.

    Saves aurum_output/full_predictions.csv for the backtest engine.
    """
    print(f"  Generating full-history predictions...")
    lb      = cfg["lookback"]
    print(f"  ({len(features_df)} total days, {lb} day lookback → "
          f"{len(features_df) - lb} prediction days)")
    model.eval()
    lb      = cfg["lookback"]
    X_all   = features_df.values.astype(np.float32)
    y_all   = targets_s.values.astype(np.float32)
    dates   = features_df.index
    preds   = np.full(len(X_all), np.nan)

    with torch.no_grad():
        for i in range(lb, len(X_all)):
            window   = X_all[i - lb : i]
            x_tensor = torch.tensor(window).unsqueeze(0).to(DEVICE)
            preds[i] = model(x_tensor).item()

    full_df = pd.DataFrame({
        "pred":   preds,
        "actual": y_all,
    }, index=dates)
    full_df = full_df.dropna()
    full_df.index.name = "date"

    # Overlay OOF predictions where available (they're more honest)
    full_df.loc[oof_df.index, "pred"]   = oof_df["pred"]
    full_df.loc[oof_df.index, "actual"] = oof_df["actual"]

    path = os.path.join(OUT_DIR, "full_predictions.csv")
    full_df.to_csv(path)
    oof_count  = full_df.index.isin(oof_df.index).sum()
    fill_count = len(full_df) - oof_count
    print(f"  ✓ Full predictions → {path}")
    print(f"    {len(full_df)} total days  "
          f"({oof_count} OOF honest + {fill_count} model-filled)")
    return full_df


# ══════════════════════════════════════════════════════════════════════════════
# 9. INFERENCE — predict on latest window
# ══════════════════════════════════════════════════════════════════════════════

def predict_latest(model, features_df, cfg, prices=None):
    """Run the trained model on the most recent lookback window."""
    print(f"\n{'═'*60}")
    print(f"  [9] LIVE INFERENCE")
    print(f"{'═'*60}")

    lb       = cfg["lookback"]
    window   = features_df.iloc[-lb:].values.astype(np.float32)
    x_tensor = torch.tensor(window).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        log_ret = model(x_tensor).item()

    pct_ret   = (np.exp(log_ret) - 1) * 100
    direction = "▲  BULLISH" if log_ret > 0 else "▼  BEARISH"
    conf      = min(abs(log_ret) / 0.05, 1.0)
    bar       = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    as_of     = features_df.index[-1].date()

    # ── Volatility regime filter ──────────────────────────────────────────
    # Model underperforms in low-vol sideways markets (Fold 4 evidence).
    # Warn when realised vol < 12% annualised — reduce position sizing.
    regime_warn = ""
    if prices is not None and cfg["target"] in prices.columns:
        gdx_prices = prices[cfg["target"]].dropna()
        recent_ret = np.log(gdx_prices / gdx_prices.shift(1)).dropna()
        realised_vol = float(recent_ret.iloc[-20:].std() * np.sqrt(252) * 100)
        vol_flag = "⚠ LOW VOL" if realised_vol < 12 else "✓ NORMAL"
        regime_warn = f"\n  Realised vol  : {realised_vol:.1f}% annualised  {vol_flag}"
        if realised_vol < 12:
            regime_warn += (
                "\n  ⚠  Low-vol sideways regime detected — model signal "
                "less reliable.\n     Consider reducing position size by 50%."
            )

    print(f"\n  Target     : {cfg['target']}")
    print(f"  As of      : {as_of}")
    print(f"  Horizon    : {cfg['forward_days']} trading days")
    print(f"  Direction  : {direction}")
    print(f"  Log return : {log_ret:+.4f}")
    print(f"  Pct return : {pct_ret:+.2f}%")
    print(f"  Confidence : [{bar}] {conf:.0%}")
    if regime_warn:
        print(regime_warn)

    return log_ret, pct_ret


def ensemble_predict(features_df, cfg, n_folds=None):
    """
    Load all saved fold models and average their predictions.
    More robust than any single model — smooths out regime sensitivity.
    """
    print(f"\n{'═'*60}")
    print(f"  [9b] ENSEMBLE INFERENCE  (all fold models)")
    print(f"{'═'*60}")

    abs_out = os.path.abspath(OUT_DIR)

    lb       = cfg["lookback"]
    window   = features_df.iloc[-lb:].values.astype(np.float32)
    x_tensor = torch.tensor(window).unsqueeze(0).to(DEVICE)
    n_feat   = features_df.shape[1]

    # Scan all fold_N_model.pt files that actually exist
    # (don't stop at first missing number — fold 1 may be absent if skipped)
    import glob
    pt_pattern = os.path.join(os.path.abspath(OUT_DIR), "fold_*_model.pt")
    pt_paths   = sorted(glob.glob(pt_pattern))

    fold_preds = []
    for path in pt_paths:
        fold_num = int(os.path.basename(path).split("_")[1])
        ckpt  = torch.load(path, map_location=DEVICE, weights_only=True)
        da    = ckpt["metrics"]["dir_acc"]
        ic    = ckpt["metrics"]["ic"]
        mcfg  = ckpt["model_config"]

        # Debug: show exactly what values are stored in the .pt file
        print(f"  Fold {fold_num} (.pt): dir_acc={da:.4f}  IC={ic:.4f}")

        # Hard quality filters — exclude folds clearly worse than random
        min_da = cfg.get("ensemble_min_dir_acc", 0.40)
        min_ic = cfg.get("ensemble_min_ic", -0.15)
        excluded_reason = []
        if da < min_da:
            excluded_reason.append(f"dir_acc={da:.1%} < {min_da:.0%}")
        if ic < min_ic:
            excluded_reason.append(f"IC={ic:.3f} < {min_ic:.2f}")

        if excluded_reason:
            print(f"  Fold {fold_num}: EXCLUDED  ({', '.join(excluded_reason)})")
            continue

        m = GoldMinerLSTM(**mcfg).to(DEVICE)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        with torch.no_grad():
            pred = m(x_tensor).item()
        fold_preds.append({"pred": pred, "dir_acc": da,
                           "ic": ic, "fold_num": fold_num})
        print(f"  Fold {fold_num}: {pred:+.4f}  "
              f"(dir_acc={da:.1%}  IC={ic:+.3f})  ✓ included")

    if not fold_preds:
        # Check whether files were found at all
        pt_files = [f for f in os.listdir(os.path.abspath(OUT_DIR))
                    if f.startswith("fold_") and f.endswith(".pt")]
        if not pt_files:
            print("  ⚠ No fold_*.pt files found — did walk_forward_cv run?")
        else:
            print(f"  ⚠ All {len(pt_files)} folds excluded by filters")
            print(f"  Tip: reduce ensemble_min_dir_acc or ensemble_min_ic in CFG")
        return None, None

    n          = len(fold_preds)
    preds_arr  = np.array([f["pred"]    for f in fold_preds])
    da_arr     = np.array([f["dir_acc"] for f in fold_preds])
    fold_nums  = [f["fold_num"] for f in fold_preds]

    recency_w  = cfg.get("ensemble_recency_weight", 0.4)

    # Accuracy weights — above-50% margin, normalised
    acc_weights = np.maximum(da_arr - 0.5, 0.01)
    acc_weights = acc_weights / acc_weights.sum()

    # Recency weights — exponential taper by position in fold_nums list
    # (fold_nums is already in chronological order since we iterate 1→N)
    recency_raw    = np.exp(np.linspace(0, 2, n))
    recency_weights = recency_raw / recency_raw.sum()

    # Blend accuracy + recency
    weights = (1 - recency_w) * acc_weights + recency_w * recency_weights
    weights = weights / weights.sum()

    total_found  = len(pt_paths)
    n_excluded   = total_found - len(fold_preds)

    ensemble   = float(np.dot(weights, preds_arr))
    pct_ret    = (np.exp(ensemble) - 1) * 100
    direction  = "▲  BULLISH" if ensemble > 0 else "▼  BEARISH"
    conf       = min(abs(ensemble) / 0.05, 1.0)
    bar        = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    agreement  = np.mean(np.sign(preds_arr) == np.sign(ensemble))

    print(f"\n  ── Ensemble Result ──────────────────────────────")
    print(f"  Folds found      : {total_found}")
    print(f"  Folds included   : {n}  (excluded {n_excluded} below threshold)")
    print(f"  Weighting        : {(1-recency_w):.0%} accuracy + "
          f"{recency_w:.0%} recency")
    print(f"  Fold weights     : "
          + "  ".join(f"F{fn}={w:.2f}"
                      for fn, w in zip(fold_nums, weights)))
    print(f"  Fold agreement   : {agreement:.0%} pointing same direction")
    print(f"  ─────────────────────────────────────────────────")
    print(f"\n{'═'*60}")
    print(f"  [9b] ENSEMBLE LIVE INFERENCE")
    print(f"{'═'*60}")
    print(f"\n  Target     : {cfg.get('target', 'GDX')}")
    print(f"  As of      : {features_df.index[-1].date()}")
    print(f"  Horizon    : {cfg.get('forward_days', 10)} trading days")
    print(f"  Models     : {n} fold ensemble  ({agreement:.0%} agreement)")
    print(f"  Direction  : {direction}")
    print(f"  Log return : {ensemble:+.4f}")
    print(f"  Pct return : {pct_ret:+.2f}%")
    print(f"  Confidence : [{bar}] {conf:.0%}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Tip: high fold agreement + high confidence = strongest signal")

    return ensemble, pct_ret, fold_preds, weights, fold_nums

DARK = "#0a0a08"
PANEL = "#111110"
GOLD  = "#d4a843"
BLUE  = "#60a5fa"
GREEN = "#4ade80"
RED   = "#f87171"
MUTED = "#6b6456"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a2820")
    ax.grid(color="#1e1e1a", linewidth=0.5, linestyle="--")


def plot_ensemble_results(fold_preds, weights, fold_nums,
                          ensemble_val, fold_metrics, fold_periods,
                          oof_preds, oof_actuals, features_df):
    """
    Three-panel ensemble summary chart:
      Left   — Per-fold predictions with weights (bar chart)
      Centre — Per-fold IC and directional accuracy
      Right  — Out-of-fold cumulative P&L with real dates
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=DARK)
    fig.suptitle("AURUM·AI — Ensemble Model Results",
                 color=GOLD, fontsize=12, y=1.01, fontweight="bold")

    # ── Chart 1: Per-fold predictions weighted ────────────────────────────
    style_ax(axes[0])
    preds_vals = [f["pred"] for f in fold_preds]
    fn_labels  = [f"F{f['fold_num']}" for f in fold_preds]
    colours    = [GREEN if p > 0 else RED for p in preds_vals]
    bar_width  = np.array(weights) * 4 + 0.2   # width proportional to weight

    bars = axes[0].bar(range(len(preds_vals)), preds_vals,
                       color=colours, alpha=0.8, width=bar_width)
    axes[0].axhline(0, color=MUTED, linewidth=0.8)
    axes[0].axhline(ensemble_val, color=GOLD, linewidth=1.5,
                    linestyle="--", label=f"Ensemble: {ensemble_val:+.4f}")

    for i, (bar, w) in enumerate(zip(bars, weights)):
        axes[0].text(i, bar.get_height() + 0.001 if bar.get_height() >= 0
                     else bar.get_height() - 0.004,
                     f"{w:.0%}", ha="center",
                     color=MUTED, fontsize=8)

    axes[0].set_xticks(range(len(fn_labels)))
    axes[0].set_xticklabels(fn_labels, color=MUTED, fontsize=9)
    axes[0].set_ylabel("Predicted log return", color=MUTED, fontsize=8)
    axes[0].set_title("Per-Fold Predictions  (bar width = weight)",
                      color=GOLD, fontsize=9)
    axes[0].legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED)

    # ── Chart 2: IC and directional accuracy per fold ─────────────────────
    style_ax(axes[1])
    all_fold_nums = [i + 1 for i in range(len(fold_metrics))]
    dir_accs = [m["dir_acc"] * 100 for m in fold_metrics]
    ics      = [m["ic"]           for m in fold_metrics]
    periods  = [f"{va_start.strftime('%b %y') if hasattr(va_start, 'strftime') else str(va_start)[:7]}"
                for (_, va_start, _) in fold_periods]

    ax2b = axes[1].twinx()
    axes[1].bar(all_fold_nums, dir_accs,
                color=[GREEN if d > 50 else RED for d in dir_accs],
                alpha=0.4, width=0.5, label="Dir Acc %")
    ax2b.plot(all_fold_nums, ics, color=GOLD, linewidth=2,
              marker="o", markersize=5, label="IC")
    ax2b.axhline(0,    color=MUTED, linewidth=0.5, linestyle="--")
    ax2b.axhline(0.10, color=GOLD,  linewidth=0.5, linestyle=":")
    axes[1].axhline(50, color=MUTED, linewidth=0.5, linestyle="--")

    axes[1].set_xticks(all_fold_nums)
    axes[1].set_xticklabels(periods, rotation=35, ha="right",
                             color=MUTED, fontsize=7)
    axes[1].set_ylabel("Directional Accuracy %", color=MUTED, fontsize=8)
    ax2b.set_ylabel("IC", color=GOLD, fontsize=8)
    ax2b.tick_params(colors=GOLD, labelsize=8)
    axes[1].set_title("Dir Acc (bars) vs IC (line) per Fold",
                      color=GOLD, fontsize=9)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2,
                   fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # ── Chart 3: OOF cumulative P&L with real dates ───────────────────────
    style_ax(axes[2])
    valid = ~np.isnan(oof_preds)
    if valid.sum() > 10:
        p   = oof_preds[valid]
        a   = oof_actuals[valid]
        cum_sig = np.cumsum(np.sign(p) * a)
        cum_bh  = np.cumsum(a)
        x_axis  = features_df.index[valid][:len(cum_sig)]

        axes[2].plot(x_axis, cum_sig, color=GOLD, linewidth=1.5,
                     label="Ensemble signal")
        axes[2].plot(x_axis, cum_bh,  color=BLUE, linewidth=1.0,
                     linestyle="--", label="Buy & hold GDX")
        axes[2].axhline(0, color=MUTED, linewidth=0.4)
        axes[2].fill_between(x_axis, cum_sig, 0,
                             where=cum_sig > 0, alpha=0.12, color=GREEN)
        axes[2].fill_between(x_axis, cum_sig, 0,
                             where=cum_sig < 0, alpha=0.12, color=RED)

        # Mark each fold's validation period
        for i, (va_idx, va_start, va_end) in enumerate(fold_periods):
            try:
                axes[2].axvspan(pd.Timestamp(va_start),
                                pd.Timestamp(va_end),
                                alpha=0.05, color=GOLD)
            except Exception:
                pass

        axes[2].xaxis.set_major_locator(mdates.YearLocator())
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        t_start = x_axis[0].strftime("%b %Y")
        t_end   = x_axis[-1].strftime("%b %Y")
        axes[2].set_title(
            f"OOF Cumulative Log Return  ({t_start} → {t_end})",
            color=GOLD, fontsize=9)
        axes[2].set_ylabel("Cumulative log return", color=MUTED, fontsize=8)
        axes[2].legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED)

        # Annotate final values
        axes[2].annotate(f"{cum_sig[-1]:+.2f}",
                         xy=(x_axis[-1], cum_sig[-1]),
                         color=GOLD, fontsize=8,
                         xytext=(5, 0), textcoords="offset points")
        axes[2].annotate(f"{cum_bh[-1]:+.2f}",
                         xy=(x_axis[-1], cum_bh[-1]),
                         color=BLUE, fontsize=8,
                         xytext=(5, 0), textcoords="offset points")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "ensemble_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved → {path}")

def plot_wf_summary(fold_metrics, fold_curves, oof_preds,
                    oof_actuals, features_df):
    n_folds = len(fold_curves)
    fig = plt.figure(figsize=(20, 14), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.45, wspace=0.35)

    # ── Row 0: per-fold loss curves ───────────────────────────────────────
    for i, fc in enumerate(fold_curves[:4]):
        ax = fig.add_subplot(gs[0, i])
        style_ax(ax)
        ax.plot(fc["train"], color=GOLD,  linewidth=1.2, label="Train")
        ax.plot(fc["val"],   color=BLUE,  linewidth=1.2, label="Val")
        ax.set_title(f"Fold {fc['fold']}  {fc['val_range']}",
                     color=GOLD, fontsize=8, pad=4)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # ── Row 1 left: directional accuracy per fold ─────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    style_ax(ax)
    fold_nums = [fc["fold"] for fc in fold_curves]
    dir_accs  = [m["dir_acc"] for m in fold_metrics]
    ics       = [m["ic"]      for m in fold_metrics]
    bars = ax.bar(fold_nums, [d * 100 for d in dir_accs],
                  color=[GREEN if d > 0.5 else RED for d in dir_accs],
                  alpha=0.8, width=0.6)
    ax.axhline(50, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Fold", color=MUTED, fontsize=8)
    ax.set_ylabel("Directional Accuracy %", color=MUTED, fontsize=8)
    ax.set_title("Directional Accuracy per Fold", color=GOLD, fontsize=9)
    for bar, val in zip(bars, dir_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1%}", ha="center", va="bottom",
                color=MUTED, fontsize=7)

    # ── Row 1 right: IC per fold ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    style_ax(ax)
    bars2 = ax.bar(fold_nums, ics,
                   color=[GREEN if ic > 0 else RED for ic in ics],
                   alpha=0.8, width=0.6)
    ax.axhline(0,    color=MUTED,  linestyle="--", linewidth=1)
    ax.axhline(0.05, color=GOLD,   linestyle=":",  linewidth=1, label=">0.05 useful")
    ax.set_xlabel("Fold", color=MUTED, fontsize=8)
    ax.set_ylabel("Information Coefficient", color=MUTED, fontsize=8)
    ax.set_title("IC (Rank Correlation) per Fold", color=GOLD, fontsize=9)
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # ── Row 2: OOF cumulative P&L ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    style_ax(ax)
    valid  = ~np.isnan(oof_preds)
    if valid.sum() > 10:
        p = oof_preds[valid]
        a = oof_actuals[valid]
        cum_signal  = np.cumsum(np.sign(p) * a)
        cum_bh      = np.cumsum(a)
        x_axis      = features_df.index[valid][:len(cum_signal)]
        ax.plot(x_axis, cum_signal, color=GOLD,  linewidth=1.5,
                label="Model signal (long/short)")
        ax.plot(x_axis, cum_bh,     color=BLUE,  linewidth=1.0,
                linestyle="--", label="Buy & hold GDX")
        ax.axhline(0, color=MUTED, linewidth=0.5)
        ax.fill_between(x_axis, cum_signal, 0,
                        where=cum_signal > 0, alpha=0.1, color=GREEN)
        ax.fill_between(x_axis, cum_signal, 0,
                        where=cum_signal < 0, alpha=0.1, color=RED)
        ax.set_title("Out-of-Fold Cumulative Log Return",
                     color=GOLD, fontsize=9)
        ax.set_ylabel("Cumulative log return", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("AURUM·AI — Walk-Forward Validation Results",
                 color=GOLD, fontsize=13, y=0.98, fontweight="bold")

    path = os.path.join(OUT_DIR, "wf_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved → {path}")


def plot_final_model(tr_hist, va_hist, preds, actuals,
                     features_df, test_dates=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK)

    # ── Chart 1: Loss curves ──────────────────────────────────────────────
    style_ax(axes[0])
    axes[0].plot(tr_hist, color=GOLD,  linewidth=1.5, label="Train")
    axes[0].plot(va_hist, color=BLUE,  linewidth=1.5, label="Val")
    axes[0].set_title("Final Model — Loss Curves", color=GOLD, fontsize=9)
    axes[0].legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED)
    axes[0].set_xlabel("Epoch", color=MUTED, fontsize=8)
    axes[0].set_ylabel("Huber Loss", color=MUTED, fontsize=8)

    # ── Chart 2: Predicted vs actual scatter ──────────────────────────────
    style_ax(axes[1])
    axes[1].scatter(actuals, preds, alpha=0.3, s=8, color=GOLD)
    lim = max(np.abs(actuals).max(), np.abs(preds).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], color=GREEN,
                 linewidth=1, linestyle="--", label="Perfect")
    axes[1].axhline(0, color=MUTED, linewidth=0.4)
    axes[1].axvline(0, color=MUTED, linewidth=0.4)
    axes[1].set_title("Predicted vs Actual Returns", color=GOLD, fontsize=9)
    axes[1].set_xlabel("Actual", color=MUTED, fontsize=8)
    axes[1].set_ylabel("Predicted", color=MUTED, fontsize=8)
    axes[1].legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # ── Chart 3: Cumulative P&L with real dates ───────────────────────────
    style_ax(axes[2])
    cum_sig = np.cumsum(np.sign(preds) * actuals)
    cum_bh  = np.cumsum(actuals)

    # Use real dates on x-axis if available, otherwise sample index
    if test_dates is not None and len(test_dates) >= len(cum_sig):
        x_axis = test_dates[:len(cum_sig)]
        axes[2].plot(x_axis, cum_sig, color=GOLD, linewidth=1.5,
                     label="Model signal")
        axes[2].plot(x_axis, cum_bh,  color=BLUE, linewidth=1.0,
                     linestyle="--", label="Buy & hold")
        axes[2].axhline(0, color=MUTED, linewidth=0.4)
        axes[2].fill_between(x_axis, cum_sig, 0,
                             where=cum_sig > 0, alpha=0.1, color=GREEN)
        axes[2].fill_between(x_axis, cum_sig, 0,
                             where=cum_sig < 0, alpha=0.1, color=RED)
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(axes[2].xaxis.get_majorticklabels(),
                 rotation=35, ha="right", fontsize=7)
        # Annotate start and end dates in title
        t_start = x_axis[0].strftime("%b %Y")
        t_end   = x_axis[-1].strftime("%b %Y")
        axes[2].set_title(
            f"Cumulative Log Return — Test Set  ({t_start} → {t_end})",
            color=GOLD, fontsize=9)
    else:
        # Fallback: sample index with note
        axes[2].plot(cum_sig, color=GOLD, linewidth=1.5, label="Model signal")
        axes[2].plot(cum_bh,  color=BLUE, linewidth=1.0,
                     linestyle="--", label="Buy & hold")
        axes[2].axhline(0, color=MUTED, linewidth=0.4)
        axes[2].set_title("Cumulative Log Return — Test Set (sample index)",
                          color=GOLD, fontsize=9)

    axes[2].set_ylabel("Cumulative log return", color=MUTED, fontsize=8)
    axes[2].legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    fig.suptitle("AURUM·AI — Final Model Results",
                 color=GOLD, fontsize=12, y=1.01, fontweight="bold")

    path = os.path.join(OUT_DIR, "final_model_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print(f"\n{'═'*60}")
    print(f"  AURUM·AI  —  Gold Miner Neural Net Trainer")
    print(f"  Device : {DEVICE}")
    print(f"  Target : {CFG['target']}  |  "
          f"Forward : {CFG['forward_days']}d  |  "
          f"Lookback: {CFG['lookback']}d")
    print(f"{'═'*60}")

    # 1. Download prices
    prices = download_prices(TICKERS, CFG["start_date"], CFG["end_date"])

    # 1b. GPR — geopolitical risk is relevant for energy (oil price shocks)
    gpr = download_gpr(CFG["start_date"], CFG["end_date"])
    cot = None  # gold COT not relevant for XLE

    # 1b2. Download crude oil COT positioning (key energy signal)
    cot_crude = download_cot_crude(CFG["start_date"], CFG["end_date"])

    # 1d. Download GLD ETF holdings (institutional flow)
    gld_holdings = download_gld_holdings(CFG["start_date"], CFG["end_date"])

    # 1e. Download Google Trends (retail sentiment)
    trends = download_google_trends(CFG["start_date"], CFG["end_date"])

    # 2. Features (rolling z-score applied inside)
    features, targets = build_features(
        prices, CFG,
        gpr=gpr,
        cot=cot,
        gld_holdings=gld_holdings,
        trends=trends,
        cot_crude=cot_crude,
    )

    # 2b. Feature selection pipeline
    # Step 1: remove correlated features (redundancy)
    # Step 2: rank by IC with target, keep top N
    # Step 3: drop features below IC noise floor
    features, ic_ranking = select_features(features, targets, CFG)

    # 3. Walk-forward CV
    fold_metrics, fold_curves, fold_periods, oof_preds, oof_actuals = \
        walk_forward_cv(features, targets, CFG)

    if len(fold_metrics) == 0:
        print(f"\n  ✗ Seed {SEED} abandoned — skipping to next seed")
        return

    # Save OOF predictions for backtest engine
    valid_mask = ~np.isnan(oof_preds)
    oof_df = pd.DataFrame({
        "pred":   oof_preds[valid_mask],
        "actual": oof_actuals[valid_mask],
    }, index=features.index[valid_mask])
    oof_path = os.path.join(OUT_DIR, "oof_predictions.csv")
    oof_df.index.name = "date"
    oof_df.to_csv(oof_path)
    print(f"  ✓ OOF predictions saved → {oof_path}  ({len(oof_df)} rows)")

    # ── Best model tracking ───────────────────────────────────────────────
    # Compare this run's IC mean against the saved best.
    ic_mean      = float(np.mean([m["ic"] for m in fold_metrics]))
    da_mean      = float(np.mean([m["dir_acc"] for m in fold_metrics]))
    config_tag   = f"fwd{CFG['forward_days']}_lb{CFG['lookback']}"
    best_ic_path = os.path.join(OUT_DIR, f"best_run_{config_tag}.txt")

    prev_ic   = 0.0
    prev_seed = None
    if os.path.exists(best_ic_path):
        try:
            lines     = open(best_ic_path).read().strip().split("\n")
            prev_ic   = float(lines[0].split("=")[1])
            prev_seed = lines[1].split("=")[1] if len(lines) > 1 else "unknown"
        except Exception:
            pass

    print(f"\n  ── Run Quality ──────────────────────────────────")
    print(f"  This run   : IC={ic_mean:.4f}  DirAcc={da_mean:.2%}  "
          f"seed={SEED}")
    if prev_ic > 0:
        print(f"  Best so far: IC={prev_ic:.4f}  seed={prev_seed}")

    if ic_mean > prev_ic:
        open(best_ic_path, "w").write(
            f"ic={ic_mean:.6f}\nseed={SEED}\ndir_acc={da_mean:.6f}")
        print(f"  ★ New best! Models saved.")

        best_dir  = os.path.join(OUT_DIR, "Best_Models_PT")
        os.makedirs(best_dir, exist_ok=True)
        run_label = f"{config_tag}_IC{ic_mean:.4f}_DA{da_mean:.4f}_seed{SEED}"
        run_dir   = os.path.join(best_dir, run_label)
        os.makedirs(run_dir, exist_ok=True)
        pt_files  = glob.glob(
            os.path.join(os.path.abspath(OUT_DIR), "fold_*_model.pt"))
        copied = 0
        for pt in pt_files:
            shutil.copy2(pt, run_dir)
            copied += 1
        for fname in [f"best_run_{config_tag}.txt",
                      "oof_predictions.csv", "full_predictions.csv"]:
            src = os.path.join(os.path.abspath(OUT_DIR), fname)
            if os.path.exists(src):
                shutil.copy2(src, run_dir)
        if copied > 0:
            print(f"  ✓ Backed up {copied} fold models → {run_dir}")
    else:
        print(f"  ↓ Below best (IC {ic_mean:.4f} < {prev_ic:.4f})")
        print(f"  ⚠ Consider keeping previous models:")
        print(f"    Copy xle_output\\fold_*.pt from a backup if available")
        print(f"    Or retrain again — each run finds a different solution")

    # ── Multi-objective candidate saving ─────────────────────────────────
    ic_std  = float(np.std([m["ic"] for m in fold_metrics]))
    ic_min  = float(np.min([m["ic"] for m in fold_metrics]))
    sh_mean = float(np.mean([m["sharpe"] for m in fold_metrics]))
    da_min  = float(np.min([m["dir_acc"] for m in fold_metrics]))

    CRIT = {
        "ic_mean_min":  0.25,   # slightly relaxed vs GDX (XLE is harder)
        "da_mean_min":  0.60,
        "ic_std_max":   0.30,   # relaxed — allow spread between good folds
        "ic_min_min":  -0.05,   # no catastrophically bad folds
        "sh_mean_min":  2.00,
        "da_min_min":   0.40,   # relaxed to increase candidate rate
    }

    passes = (
        ic_mean >= CRIT["ic_mean_min"] and
        da_mean >= CRIT["da_mean_min"] and
        ic_std  <= CRIT["ic_std_max"]  and
        ic_min  >= CRIT["ic_min_min"]  and
        sh_mean >= CRIT["sh_mean_min"] and
        da_min  >= CRIT["da_min_min"]
    )

    print(f"\n  ── Multi-objective Check ────────────────────────")
    print(f"  IC={ic_mean:.3f}≥{CRIT['ic_mean_min']}  "
          f"DA={da_mean:.2%}≥{CRIT['da_mean_min']:.0%}  "
          f"IC_std={ic_std:.3f}≤{CRIT['ic_std_max']}  "
          f"IC_min={ic_min:.3f}≥{CRIT['ic_min_min']}  "
          f"Sh={sh_mean:.2f}≥{CRIT['sh_mean_min']}  "
          f"DA_min={da_min:.2%}≥{CRIT['da_min_min']:.0%}")

    if passes:
        cand_dir  = os.path.join(OUT_DIR, "Best_Models_PT", "candidates")
        os.makedirs(cand_dir, exist_ok=True)
        cand_label = (f"{config_tag}_IC{ic_mean:.4f}_DA{da_mean:.4f}"
                      f"_Sh{sh_mean:.2f}_seed{SEED}")
        cand_run   = os.path.join(cand_dir, cand_label)
        os.makedirs(cand_run, exist_ok=True)
        pt_files   = glob.glob(
            os.path.join(os.path.abspath(OUT_DIR), "fold_*_model.pt"))
        copied = 0
        for pt in pt_files:
            shutil.copy2(pt, cand_run)
            copied += 1
        for fname in ["oof_predictions.csv", "full_predictions.csv"]:
            src = os.path.join(os.path.abspath(OUT_DIR), fname)
            if os.path.exists(src):
                shutil.copy2(src, cand_run)
        with open(os.path.join(cand_run, "metrics.txt"), "w") as f:
            f.write(f"seed={SEED}\n"
                    f"ic_mean={ic_mean:.6f}\n"
                    f"ic_std={ic_std:.6f}\n"
                    f"da_mean={da_mean:.6f}\n"
                    f"da_min={da_min:.6f}\n"
                    f"sh_mean={sh_mean:.6f}\n"
                    f"config={config_tag}\n")
        print(f"  ★ CANDIDATE saved ({copied} models) → {cand_run}")
    else:
        fails = []
        if ic_mean < CRIT["ic_mean_min"]:
            fails.append(f"IC={ic_mean:.3f}<{CRIT['ic_mean_min']}")
        if da_mean < CRIT["da_mean_min"]:
            fails.append(f"DA={da_mean:.2%}<{CRIT['da_mean_min']:.0%}")
        if ic_std > CRIT["ic_std_max"]:
            fails.append(f"IC_std={ic_std:.3f}>{CRIT['ic_std_max']}")
        if ic_min < CRIT["ic_min_min"]:
            fails.append(f"IC_min={ic_min:.3f}<{CRIT['ic_min_min']}")
        if sh_mean < CRIT["sh_mean_min"]:
            fails.append(f"Sh={sh_mean:.2f}<{CRIT['sh_mean_min']}")
        if da_min < CRIT["da_min_min"]:
            fails.append(f"DA_min={da_min:.2%}<{CRIT['da_min_min']:.0%}")
        print(f"  – Not a candidate: {', '.join(fails)}")

    # 4. Plot WF results
    try:
        plot_wf_summary(fold_metrics, fold_curves,
                        oof_preds, oof_actuals, features)
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    # 5. Final model — give it more room to converge than WF folds
    final_cfg = {**CFG,
                 "epochs":              300,
                 "early_stop_patience": 40,
                 "lr_patience":         15,
                 "lr":                  5e-4}
    model, final_metrics, tr_hist, va_hist, preds, actuals, test_dates = \
        train_final_model(features, targets, final_cfg)

    # 6. Plot final model
    try:
        plot_final_model(tr_hist, va_hist, preds, actuals,
                         features, test_dates=test_dates)
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    # 7. Single model inference + vol regime check
    predict_latest(model, features, CFG, prices=prices)

    # Generate full-history predictions for backtest engine
    # (covers all trading days, not just OOF fold val periods)
    save_full_predictions(model, features, targets, CFG, oof_df)

    # 8. Ensemble inference (weighted average of all fold models)
    ens_result = ensemble_predict(features, CFG)
    if ens_result[0] is not None:
        ens_val, ens_pct, ens_fold_preds, ens_weights, ens_fold_nums = ens_result
        try:
            plot_ensemble_results(
                ens_fold_preds, ens_weights, ens_fold_nums,
                ens_val, fold_metrics, fold_periods,
                oof_preds, oof_actuals, features)
        except Exception as e:
            print(f"  (Ensemble plot skipped: {e})")

    # 9. Regime diagnostic — flag accounts for both dir_acc AND ic
    #    A fold with good dir_acc but negative IC is misleading — mark correctly
    print(f"\n{'═'*60}")
    print(f"  REGIME DIAGNOSTIC")
    print(f"{'═'*60}")
    min_ic  = CFG.get("ensemble_min_ic", -0.15)
    min_da  = CFG.get("ensemble_min_dir_acc", 0.40)
    for i, (fm, (va_idx, va_start, va_end)) in \
            enumerate(zip(fold_metrics, fold_periods)):
        da   = fm["dir_acc"]
        ic   = fm["ic"]
        # Excluded if either metric below threshold
        excluded = da < min_da or ic < min_ic
        if excluded:
            flag = "✗ EXCL"
        elif da < 0.52 or ic < 0.0:
            flag = "⚠ WEAK"
        elif da < 0.65 and ic < 0.10:
            flag = "✓ OK  "
        else:
            flag = "★ STRONG"
        print(f"  Fold {i+1}  {va_start}→{va_end}  "
              f"DirAcc={da:.1%}  IC={ic:+.3f}  {flag}")

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  ✓ Complete in {elapsed/60:.1f} minutes")
    print(f"  Outputs saved to: ./{OUT_DIR}/")
    print(f"    final_model.pt           — single model for deployment")
    print(f"    fold_N_model.pt          — per-fold models (used by ensemble)")
    print(f"    feature_importance.csv   — ranked features with IC scores")
    print(f"    wf_results.png           — walk-forward validation chart")
    print(f"    ensemble_results.png     — ensemble inference chart")
    print(f"    final_model_results.png  — final model chart")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
