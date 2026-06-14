"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AURUM·AI  —  aurum_signal.py                                       ║
║                                                                              ║
║  Weekly live signal tracker. Runs in ~60 seconds, no retraining needed.    ║
║  Run every Monday after US market close.                                    ║
║                                                                              ║
║  INPUT                                                                       ║
║    fold_N_model.pt   — saved fold models from gold_miner_trainer.py        ║
║    Downloads fresh: prices (18 tickers), GPR, COT, GLD volume, Trends     ║
║                                                                              ║
║  PROCESS                                                                    ║
║    Builds full feature set → aligns to model's trained feature names →     ║
║    runs weighted ensemble across all fold models → sizes position by        ║
║    confidence percentile → fetches relevant news headlines.                ║
║                                                                              ║
║  OUTPUTS  (printed to console)                                              ║
║    GDX market data    — last close, daily change, 5/20-day range           ║
║    Ensemble signal    — direction, confidence, fold agreement               ║
║    Trading decision   — action, position size, entry date, review date     ║
║    Signal history     — last 8 weeks with outcomes filled retrospectively  ║
║    News headlines     — last 1-3 days filtered for gold/macro relevance    ║
║    signal_log.csv     — appended each run, outcomes filled after 10 days   ║
║                                                                              ║
║  RUN                                                                        ║
║    python aurum_signal.py                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, glob, shutil, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR    = "aurum_output"
LOG_PATH   = os.path.join(OUT_DIR, "signal_log.csv")
CASH_RATE  = 0.0    # annual cash/SGOV rate — set 0% to avoid optimism bias
                    # (historical rates varied: ~0% 2010-2015, ~5% 2023-2026)
CASH_DAILY = CASH_RATE / 252

# ── Deployed model seeds ───────────────────────────────────────────────────────
# Change seed here to switch models — signal auto-copies from candidates folder
DEPLOYED_SEEDS = {
    # v9 champions (May 2026), bootstrap-validated
    "GDX": {"seed":   162, "out_dir": "aurum_output"},
    "XLE": {"seed": 18264, "out_dir": "xle_output"},
    "QQQ": {"seed": 32928, "out_dir": "qqq_output"},
}


def deploy_seed_models(ticker):
    """
    Auto-copy fold models + predictions from candidates folder to output dir.
    Searches for a subfolder matching the seed in Best_Models_PT/candidates/.
    Returns True if models are ready, False if seed not found.

    Robustness (patched 2026-05-28): previously used mtime comparison to skip
    redundant copies, but that produced false-positive "already current" when
    the destination held files from concurrent training or a previously
    deployed different seed (mtime older than configured seed's source).
    Now uses content hash comparison of fold_1_model.pt — definitive, fast
    (~88 KB MD5 ≈ ms), self-healing against any kind of stale-file contamination.
    """
    import hashlib
    cfg      = DEPLOYED_SEEDS.get(ticker)
    if cfg is None:
        return True  # no seed configured — use whatever is in out_dir
    seed     = cfg["seed"]
    out_dir  = cfg["out_dir"]
    cand_dir = os.path.join(out_dir, "Best_Models_PT", "candidates")

    if not os.path.exists(cand_dir):
        print(f"  ⚠ {ticker}: candidates folder not found at {cand_dir}")
        return False

    # Find subfolder matching seed
    matches = [d for d in os.listdir(cand_dir)
               if os.path.isdir(os.path.join(cand_dir, d))
               and f"_seed{seed}" in d]

    if not matches:
        print(f"  ✗ {ticker}: seed={seed} not found in {cand_dir}")
        print(f"    Available: {[d for d in os.listdir(cand_dir) if os.path.isdir(os.path.join(cand_dir, d))]}")
        return False

    src_dir  = os.path.join(cand_dir, matches[0])
    pt_files = sorted(glob.glob(os.path.join(src_dir, "fold_*_model.pt")))

    if not pt_files:
        print(f"  ✗ {ticker}: no .pt files found in {src_dir}")
        return False

    # Content-based check: compare source vs destination fold_1 by MD5 hash.
    # Force-copy if destination is missing, hash differs, or comparison fails.
    src_fold1 = pt_files[0]
    dst_fold1 = os.path.join(out_dir, os.path.basename(src_fold1))
    needs_copy = True
    reason     = "missing"
    if os.path.exists(dst_fold1):
        try:
            with open(src_fold1, "rb") as f:
                src_hash = hashlib.md5(f.read()).hexdigest()
            with open(dst_fold1, "rb") as f:
                dst_hash = hashlib.md5(f.read()).hexdigest()
            if src_hash == dst_hash:
                needs_copy = False
            else:
                reason = "hash mismatch"
        except Exception as e:
            reason = f"compare failed ({e})"

    if needs_copy:
        copied = 0
        for pt in pt_files:
            shutil.copy2(pt, out_dir)
            copied += 1
        for fname in ["oof_predictions.csv", "full_predictions.csv"]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, out_dir)
        print(f"  ✓ {ticker}: deployed seed={seed} "
              f"({copied} models, {reason}) from {matches[0]}")
    else:
        print(f"  ✓ {ticker}: seed={seed} already current  (hash verified)")

    return True

# ── Signal config — must match BCFG in aurum_backtest.py ─────────────────────
SCFG = {
    "target":            "GDX",
    "forward_days":      20,        # v10: 20-day prediction horizon (was 10)
    "lookback":          30,        # v10: LSTM sequence length (was 20)
    "zscore_window":     252,       # rolling z-score window
    "min_signal":        0.008,     # v10: scaled for 20d predictions (was 0.005)
    "max_position_frac": 1.00,      # maximum position as fraction of capital
    "conf_tiers": [
        (0.00, 0.25, 0.25),
        (0.25, 0.50, 0.50),
        (0.50, 0.75, 0.75),
        (0.75, 1.01, 1.00),
    ],
    "ensemble_recency_weight": 0.4,
    "ensemble_min_dir_acc":    0.40,
    "ensemble_min_ic":         -0.15,
    # Data window — extended for longer lookback + 200d MA warmup
    # Need: 252 zscore + 200 MA + 30 lookback + 20 forward + buffer
    "data_days":         900,       # ~3 calendar years — covers 370d chart + zscore warmup
    "hold_days":         20,        # matches forward_days
}

TICKERS = [
    # GDX core
    "GDX", "GDXJ", "GLD", "GC=F", "SI=F", "HG=F",
    "NEM", "GOLD", "WPM",
    # Macro
    "UUP", "TLT", "SPY", "QQQ", "XLK", "IWM", "HYG",
    "^VIX", "^VIX3M", "^TNX", "^MOVE",
    "USDJPY=X", "USDCHF=X",
    # Energy — needed for XLE model (NG=F is top feature)
    "XLE", "XOP", "OIH",
    "CL=F", "BZ=F", "NG=F", "HO=F",
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — must match gold_miner_trainer.py exactly
# ══════════════════════════════════════════════════════════════════════════════

class GoldMinerLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64,
                 num_lstm_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = num_lstm_layers,
            dropout      = dropout if num_lstm_layers > 1 else 0.0,
            batch_first  = True,
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

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FOLD MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_fold_models():
    """Load all saved fold models from aurum_output/."""
    pt_paths = sorted(glob.glob(
        os.path.join(os.path.abspath(OUT_DIR), "fold_*_model.pt")))

    if not pt_paths:
        print(f"  ✗ No fold models found in {OUT_DIR}/")
        print(f"    Run gold_miner_trainer.py first to train models")
        return None, None

    fold_models = []
    feature_names = None

    for path in pt_paths:
        fold_num = int(os.path.basename(path).split("_")[1])
        ckpt     = torch.load(path, map_location="cpu", weights_only=True)
        mcfg     = ckpt["model_config"]
        da       = ckpt["metrics"]["dir_acc"]
        ic       = ckpt["metrics"]["ic"]

        # Quality filter
        if da < SCFG["ensemble_min_dir_acc"]:
            print(f"  Fold {fold_num}: skipped  (dir_acc={da:.1%} < threshold)")
            continue
        if ic < SCFG["ensemble_min_ic"]:
            print(f"  Fold {fold_num}: skipped  (IC={ic:.3f} < threshold)")
            continue

        model = GoldMinerLSTM(**mcfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        fold_models.append({
            "model":      model,
            "fold_num":   fold_num,
            "dir_acc":    da,
            "ic":         ic,
            "val_period": ckpt.get("val_period", "unknown"),
        })

        # Feature names from first valid fold
        if feature_names is None:
            feature_names = ckpt.get("feature_names", None)

    print(f"  ✓ Loaded {len(fold_models)} fold models")
    if feature_names:
        print(f"  ✓ Feature names: {len(feature_names)} features")

    return fold_models, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# 2. DOWNLOAD LATEST PRICES
# ══════════════════════════════════════════════════════════════════════════════

def download_recent_prices(cfg):
    """Download enough price history to compute all features."""
    end   = datetime.today()
    start = end - timedelta(days=cfg["data_days"])

    print(f"\n  Downloading prices  "
          f"({start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')})...")

    frames = {}
    for t in TICKERS:
        try:
            ticker_obj = yf.Ticker(t)
            df = ticker_obj.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df is None or len(df) < 50:
                continue

            close_col = next((c for c in df.columns
                              if c.lower() == "close"), None)
            if close_col is None:
                continue

            s = df[close_col].rename(t)
            s = s[s > 0].dropna()
            frames[t] = s
        except Exception:
            continue

    prices = pd.concat(frames.values(), axis=1)
    prices = prices.sort_index().ffill().dropna(how="all")
    # Normalise — strip timezone AND time component (same fix as trainer)
    prices.index = pd.to_datetime(prices.index).normalize().tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")]

    print(f"  ✓ Prices: {prices.shape[0]} rows × {prices.shape[1]} tickers")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# 3. DOWNLOAD AUXILIARY DATA (GPR, COT, GLD holdings)
# ══════════════════════════════════════════════════════════════════════════════

def download_gpr_recent(start, end):
    """Download GPR index — tries xls then xlsx format."""
    urls = [
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
    ]
    try:
        import requests
        for url in urls:
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code != 200:
                    print(f"    GPR HTTP {resp.status_code} for {url.split('/')[-1]}")
                    continue
                from io import BytesIO
                gpr_raw = pd.read_excel(BytesIO(resp.content),
                                        index_col=0, parse_dates=True)
                gpr_raw.index = pd.to_datetime(
                    gpr_raw.index).normalize().tz_localize(None)
                cols = [c for c in ["GPR", "GPRACT", "GPRTHR",
                                    "GPRD", "GPRD_ACT", "GPRD_THREAT"]
                        if c in gpr_raw.columns]
                if not cols:
                    print(f"    GPR columns not found — got: "
                          f"{list(gpr_raw.columns[:5])}")
                    continue
                gpr = gpr_raw[cols].rename(columns={
                    "GPRD":         "GPR",
                    "GPRD_ACT":     "GPRACT",
                    "GPRD_THREAT":  "GPRTHR",
                })
                date_range = pd.date_range(start=start, end=end, freq="B")
                return gpr.reindex(date_range).ffill().bfill()
            except Exception as e:
                print(f"    GPR error ({url.split('/')[-1]}): {e}")
                continue
        return None
    except Exception as e:
        print(f"    GPR outer error: {e}")
        return None


def download_cot_recent(start, end):
    """Download most recent COT data for gold futures."""
    import requests, zipfile, io as _io
    start_year = int(str(start)[:4])
    end_year   = int(str(end)[:4])
    all_frames = []

    for year in range(start_year, end_year + 1):
        try:
            url  = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
                names = z.namelist()
                txt   = next((n for n in names
                              if n.lower().endswith((".txt", ".csv"))), names[0])
                with z.open(txt) as f:
                    df_yr = pd.read_csv(f, low_memory=False)
            NAME_COL = "Market and Exchange Names"
            if NAME_COL in df_yr.columns:
                df_yr = df_yr[df_yr[NAME_COL].astype(str)
                              .str.upper().str.contains("GOLD", na=False)]
            if len(df_yr) == 0:
                continue
            all_frames.append(df_yr)
        except Exception:
            continue

    if not all_frames:
        return None

    cot_raw = pd.concat(all_frames, ignore_index=True)
    col_map  = {
        "As of Date in Form YYYY-MM-DD":       "report_date",
        "Noncommercial Positions-Long (All)":  "spec_long",
        "Noncommercial Positions-Short (All)": "spec_short",
        "Commercial Positions-Long (All)":     "comm_long",
        "Commercial Positions-Short (All)":    "comm_short",
        "Open Interest (All)":                 "open_int",
    }
    cot_raw = cot_raw.rename(columns=col_map)
    if "report_date" not in cot_raw.columns:
        date_col = next((c for c in cot_raw.columns
                         if "date" in c.lower()), None)
        if date_col is None:
            return None
        cot_raw = cot_raw.rename(columns={date_col: "report_date"})

    cot_raw["date"] = pd.to_datetime(
        cot_raw["report_date"], errors="coerce").dt.tz_localize(None)
    cot_raw = cot_raw.dropna(subset=["date"]).set_index("date").sort_index()

    needed = ["spec_long", "spec_short", "comm_long", "comm_short", "open_int"]
    if any(c not in cot_raw.columns for c in needed):
        return None

    cot = pd.DataFrame(index=cot_raw.index)
    for col in needed:
        cot[col] = pd.to_numeric(cot_raw[col], errors="coerce")

    cot["net_spec"]      = cot["spec_long"]  - cot["spec_short"]
    cot["net_comm"]      = cot["comm_long"]  - cot["comm_short"]
    cot["spec_oi_ratio"] = cot["net_spec"]   / (cot["open_int"] + 1e-9)
    cot["comm_oi_ratio"] = cot["net_comm"]   / (cot["open_int"] + 1e-9)
    cot = cot[["net_spec", "net_comm", "spec_oi_ratio",
               "comm_oi_ratio", "open_int"]].dropna(how="all")
    cot = cot.sort_values("open_int", ascending=False)
    cot = cot[~cot.index.duplicated(keep="first")].sort_index()

    date_range = pd.date_range(start=start, end=end, freq="B")
    return cot.reindex(date_range).ffill().bfill()


def download_cot_crude_recent(start, end):
    """Download most recent COT data for WTI crude oil futures (contract 067651)."""
    import requests, zipfile, io as _io
    start_year = int(str(start)[:4])
    end_year   = int(str(end)[:4])
    all_frames = []

    for year in range(start_year, end_year + 1):
        try:
            url  = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
                names = z.namelist()
                txt   = next((n for n in names
                              if n.lower().endswith((".txt", ".csv"))), names[0])
                with z.open(txt) as f:
                    df_yr = pd.read_csv(f, low_memory=False)
            NAME_COL = "Market and Exchange Names"
            CODE_COL = "CFTC Contract Market Code"
            if NAME_COL in df_yr.columns:
                df_yr = df_yr[df_yr[NAME_COL].astype(str)
                              .str.upper().str.contains("CRUDE OIL", na=False)]
            elif CODE_COL in df_yr.columns:
                df_yr = df_yr[df_yr[CODE_COL].astype(str)
                              .str.strip().str.startswith("067")]
            if len(df_yr) == 0:
                continue
            all_frames.append(df_yr)
        except Exception:
            continue

    if not all_frames:
        return None

    cot_raw = pd.concat(all_frames, ignore_index=True)
    col_map  = {
        "As of Date in Form YYYY-MM-DD":       "report_date",
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
    if "report_date" not in cot_raw.columns:
        date_col = next((c for c in cot_raw.columns
                         if "date" in c.lower()), None)
        if date_col is None:
            return None
        cot_raw = cot_raw.rename(columns={date_col: "report_date"})

    cot_raw["date"] = pd.to_datetime(
        cot_raw["report_date"], errors="coerce").dt.tz_localize(None)
    cot_raw = cot_raw.dropna(subset=["date"]).set_index("date").sort_index()

    needed = ["spec_long", "spec_short", "comm_long", "comm_short", "open_int"]
    if any(c not in cot_raw.columns for c in needed):
        return None

    cot = pd.DataFrame(index=cot_raw.index)
    for col in needed:
        cot[col] = pd.to_numeric(cot_raw[col], errors="coerce")

    cot["oil_net_spec"]      = cot["spec_long"]  - cot["spec_short"]
    cot["oil_net_comm"]      = cot["comm_long"]  - cot["comm_short"]
    cot["oil_spec_oi_ratio"] = cot["oil_net_spec"] / (cot["open_int"] + 1e-9)
    cot["oil_comm_oi_ratio"] = cot["oil_net_comm"] / (cot["open_int"] + 1e-9)
    cot = cot[["oil_net_spec", "oil_net_comm", "oil_spec_oi_ratio",
               "oil_comm_oi_ratio", "open_int"]].dropna(how="all")
    cot = cot.sort_values("open_int", ascending=False)
    cot = cot[~cot.index.duplicated(keep="first")].sort_index()

    date_range = pd.date_range(start=start, end=end, freq="B")
    return cot.reindex(date_range).ffill().bfill()


def download_gld_holdings_recent(start, end):
    """Download GLD ETF holdings proxy via yfinance volume."""
    try:
        gld  = yf.Ticker("GLD")
        hist = gld.history(start=start, end=end, auto_adjust=True)
        if len(hist) < 50:
            return None
        hist.index = pd.to_datetime(hist.index).normalize().tz_localize(None)
        vol = hist["Volume"].rename("gld_holdings")
        date_range = pd.date_range(start=start, end=end, freq="B")
        return vol.reindex(date_range).ffill().bfill()
    except Exception:
        return None


def download_google_trends_recent(start, end,
                                   keywords=None, tag=None):
    """Download Google Trends with caching. tag differentiates GDX/XLE/QQQ caches."""
    if keywords is None:
        keywords = ["gold price", "buy gold", "inflation hedge"]

    # Build cache path from tag or keywords
    if tag:
        cache_tag  = tag
    else:
        cache_tag  = "".join(sorted(
            k.replace(" ", "").replace("/", "")[:8] for k in keywords))[:40]
    cache_path = os.path.join(OUT_DIR, f"google_trends_cache_{cache_tag}.csv")

    if os.path.exists(cache_path):
        age_days = (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_path))).days
        if age_days < 7:   # trends don't change fast — 7 day cache is fine
            try:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                cached.index = pd.to_datetime(
                    cached.index).tz_localize(None).normalize()
                mask = (cached.index >= pd.Timestamp(start)) & \
                       (cached.index <= pd.Timestamp(end))
                sliced = cached[mask]
                if len(sliced) > 10:
                    print(f"  ✓ Google Trends ({cache_tag}): "
                          f"cache hit  ({age_days}d old, {len(sliced)} rows)")
                    return sliced
            except Exception as e:
                print(f"  ⚠ Trends cache read failed: {e}")

    try:
        from pytrends.request import TrendReq
        import time as _time
        pytrends  = TrendReq(hl="en-US", tz=0, timeout=(10, 30),
                             retries=2, backoff_factor=0.5)
        all_trends = []
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            try:
                pytrends.build_payload(
                    batch, cat=0,
                    timeframe=f"{str(start)[:10]} {str(end)[:10]}",
                    geo="", gprop="")
                df_t = pytrends.interest_over_time()
                if df_t is not None and len(df_t) > 10:
                    df_t = df_t.drop(columns=["isPartial"], errors="ignore")
                    df_t.index = pd.to_datetime(df_t.index).tz_localize(None)
                    all_trends.append(df_t)
                _time.sleep(2)
            except Exception:
                continue
        if not all_trends:
            return None
        trends    = pd.concat(all_trends, axis=1)
        date_range = pd.date_range(start=start, end=end, freq="B")
        trends    = trends.reindex(date_range).ffill().bfill()
        try:
            trends.to_csv(cache_path)
        except Exception:
            pass
        return trends
    except ImportError:
        return None
    except Exception:
        return None
    """Download most recent 2 years of COT data."""
    import requests, zipfile, io as _io
    start_year = int(str(start)[:4])
    end_year   = int(str(end)[:4])
    all_frames = []

    for year in range(start_year, end_year + 1):
        try:
            url  = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
                names = z.namelist()
                txt   = next((n for n in names
                              if n.lower().endswith((".txt", ".csv"))), names[0])
                with z.open(txt) as f:
                    df_yr = pd.read_csv(f, low_memory=False)

            NAME_COL = "Market and Exchange Names"
            if NAME_COL in df_yr.columns:
                mask    = df_yr[NAME_COL].astype(str).str.upper().str.contains(
                    "GOLD", na=False)
                df_yr   = df_yr[mask]
            if len(df_yr) == 0:
                continue
            all_frames.append(df_yr)
        except Exception:
            continue

    if not all_frames:
        return None

    cot_raw = pd.concat(all_frames, ignore_index=True)
    col_map  = {
        "As of Date in Form YYYY-MM-DD":       "report_date",
        "Noncommercial Positions-Long (All)":  "spec_long",
        "Noncommercial Positions-Short (All)": "spec_short",
        "Commercial Positions-Long (All)":     "comm_long",
        "Commercial Positions-Short (All)":    "comm_short",
        "Open Interest (All)":                 "open_int",
    }
    cot_raw = cot_raw.rename(columns=col_map)

    if "report_date" not in cot_raw.columns:
        date_col = next((c for c in cot_raw.columns
                         if "date" in c.lower()), None)
        if date_col is None:
            return None
        cot_raw = cot_raw.rename(columns={date_col: "report_date"})

    cot_raw["date"] = pd.to_datetime(
        cot_raw["report_date"], errors="coerce").dt.tz_localize(None)
    cot_raw = cot_raw.dropna(subset=["date"]).set_index("date").sort_index()

    needed = ["spec_long", "spec_short", "comm_long",
              "comm_short", "open_int"]
    for col in needed:
        if col not in cot_raw.columns:
            return None

    cot = pd.DataFrame(index=cot_raw.index)
    for col in needed:
        cot[col] = pd.to_numeric(cot_raw[col], errors="coerce")

    cot["net_spec"]      = cot["spec_long"]  - cot["spec_short"]
    cot["net_comm"]      = cot["comm_long"]  - cot["comm_short"]
    cot["spec_oi_ratio"] = cot["net_spec"]   / (cot["open_int"] + 1e-9)
    cot["comm_oi_ratio"] = cot["net_comm"]   / (cot["open_int"] + 1e-9)
    cot = cot[["net_spec", "net_comm",
               "spec_oi_ratio", "comm_oi_ratio", "open_int"]].dropna(how="all")
    cot = cot.sort_values("open_int", ascending=False)
    cot = cot[~cot.index.duplicated(keep="first")].sort_index()

    date_range = pd.date_range(start=start, end=end, freq="B")
    return cot.reindex(date_range).ffill().bfill()


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD FEATURES — mirrors gold_miner_trainer.py build_features()
# ══════════════════════════════════════════════════════════════════════════════

def log_return(series, periods=1):
    return np.log(series / series.shift(periods))

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast,   adjust=False).mean()
    ema_s = series.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    return macd - macd.ewm(span=signal, adjust=False).mean()

def rolling_zscore(series, window):
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / (s + 1e-9)

def build_features_live(prices, cfg, gpr=None, cot=None,
                        gld_holdings=None, trends=None, cot_crude=None):
    """
    Build features from recent price data.
    Mirrors build_features() in the trainer exactly.
    Returns DataFrame with same column names as training features.
    """
    df   = pd.DataFrame(index=prices.index)
    gdx  = prices["GDX"]
    zw   = cfg["zscore_window"]

    # Normalise index
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize().tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")]
    gdx    = prices["GDX"]
    df     = pd.DataFrame(index=prices.index)

    gold   = prices.get("GC=F",     prices.get("GLD"))
    oil    = prices.get("CL=F")
    vix    = prices.get("^VIX")
    vix3m  = prices.get("^VIX3M")
    tnx    = prices.get("^TNX")
    uup    = prices.get("UUP")
    tlt    = prices.get("TLT")
    gld    = prices.get("GLD")
    gdxj   = prices.get("GDXJ")
    move   = prices.get("^MOVE")
    usdjpy = prices.get("USDJPY=X")
    usdchf = prices.get("USDCHF=X")

    # 1. Returns
    for col in prices.columns:
        s = prices[col]
        for p in [1, 3, 5, 10, 21]:
            df[f"{col}_r{p}"] = rolling_zscore(log_return(s, p), zw)

    # 2. Ratios
    if gold is not None and oil is not None:
        df["gold_oil_ratio"]    = rolling_zscore(gold / (oil + 1e-9), zw)
    if gld is not None:
        df["gdx_gld_ratio"]     = rolling_zscore(gdx / (gld + 1e-9), zw)
        df["gdx_gld_mom5"]      = rolling_zscore(log_return(gdx / gld, 5), zw)
    if gold is not None:
        df["gold_silver_ratio"] = rolling_zscore(
            gold / (prices.get("SI=F", gold) + 1e-9), zw)
    if gdxj is not None:
        df["gdx_gdxj_ratio"]    = rolling_zscore(gdx / (gdxj + 1e-9), zw)

    # 3. Volatility
    gdx_r1 = log_return(gdx, 1)
    for w in [10, 20, 60]:
        df[f"gdx_vol{w}"] = rolling_zscore(
            gdx_r1.rolling(w).std() * np.sqrt(252), zw)
    if gold is not None:
        df["gold_vol20"] = rolling_zscore(
            log_return(gold, 1).rolling(20).std() * np.sqrt(252), zw)

    # 4. Moving averages
    for ma in [20, 50, 200]:
        df[f"gdx_ma{ma}_dist"] = rolling_zscore(
            gdx / gdx.rolling(ma).mean() - 1, zw)
    df["gdx_ma_cross"] = rolling_zscore(
        gdx.rolling(50).mean() / (gdx.rolling(200).mean() + 1e-9) - 1, zw)

    # 5. Momentum
    df["gdx_rsi14"]     = rolling_zscore(compute_rsi(gdx, 14) / 100, zw)
    df["gdx_macd_hist"] = rolling_zscore(compute_macd(gdx) / (gdx + 1e-9), zw)
    if gold is not None:
        df["gold_rsi14"]     = rolling_zscore(compute_rsi(gold, 14) / 100, zw)
        df["gold_macd_hist"] = rolling_zscore(
            compute_macd(gold) / (gold + 1e-9), zw)
    for p in [5, 10, 20]:
        df[f"gdx_roc{p}"] = rolling_zscore(
            gdx / (gdx.shift(p) + 1e-9) - 1, zw)

    # 6. Macro
    if vix is not None:
        df["vix_level"] = rolling_zscore(vix / 100, zw)
        df["vix_r5"]    = rolling_zscore(log_return(vix, 5), zw)
    if tnx is not None:
        df["tnx_level"]  = rolling_zscore(tnx / 100, zw)
        df["tnx_chg10"]  = rolling_zscore(tnx.diff(10), zw)
    if uup is not None:
        df["uup_r5"]  = rolling_zscore(log_return(uup, 5),  zw)
        df["uup_r21"] = rolling_zscore(log_return(uup, 21), zw)
        # v10: dollar secular trend (50d vs 200d MA) — ranked IC=0.101
        uup_ma50  = uup.rolling(50).mean()
        uup_ma200 = uup.rolling(200).mean()
        df["dollar_trend"]       = rolling_zscore(
            uup_ma50 / (uup_ma200 + 1e-9) - 1, zw)
        df["dollar_trend_slope"] = rolling_zscore(
            uup_ma50 / uup_ma50.shift(21) - 1, zw)
    if tlt is not None:
        df["tlt_r10"] = rolling_zscore(log_return(tlt, 10), zw)

    # 7. Lags
    for lag in [1, 2, 3, 5, 10]:
        df[f"gdx_r1_lag{lag}"] = rolling_zscore(gdx_r1.shift(lag), zw)

    # 8. Cross-asset
    if gold is not None:
        df["gold_r5"]  = rolling_zscore(log_return(gold, 5),  zw)
        df["gold_r21"] = rolling_zscore(log_return(gold, 21), zw)
    if tlt is not None and vix is not None:
        df["risk_off_signal"] = rolling_zscore(
            log_return(tlt, 5) - log_return(vix, 5), zw)

    # 9. Regime
    vol20 = gdx_r1.rolling(20).std() * np.sqrt(252)
    vol60 = gdx_r1.rolling(60).std() * np.sqrt(252)
    df["vol_regime"] = rolling_zscore(vol20 / (vol60 + 1e-9), zw)
    df["vol_accel"]  = rolling_zscore(vol20.diff(5), zw)
    if gold is not None:
        gold_ma20  = gold.rolling(20).mean()
        gold_ma60  = gold.rolling(60).mean()
        gold_ma200 = gold.rolling(200).mean()
        df["gold_trend_str"]   = rolling_zscore(
            (gold_ma20 - gold_ma200) / (gold_ma200 + 1e-9), zw)
        df["gold_trend_align"] = rolling_zscore(
            ((gold_ma20 > gold_ma60) & (gold_ma60 > gold_ma200)
             ).astype(float).rolling(20).mean(), zw)
    if vix is not None:
        df["vix_spike"] = rolling_zscore(vix.diff(3) / (vix + 1e-9), zw)
    if tnx is not None:
        df["tnx_accel"] = rolling_zscore(tnx.diff(5).diff(5), zw)

    # 10. Geopolitical
    if vix is not None and vix3m is not None:
        vix_curve = vix / (vix3m + 1e-9)
        df["vix_curve"]      = rolling_zscore(vix_curve, zw)
        df["vix_curve_chg5"] = rolling_zscore(vix_curve.diff(5), zw)
        df["vix_inverted"]   = rolling_zscore(
            (vix_curve > 1.0).astype(float).rolling(5).mean(), zw)
    if usdjpy is not None:
        df["usdjpy_r5"]       = rolling_zscore(log_return(usdjpy, 5),  zw)
        df["usdjpy_r21"]      = rolling_zscore(log_return(usdjpy, 21), zw)
        usdjpy_min10 = usdjpy.rolling(10).min()
        df["usdjpy_reversal"] = rolling_zscore(
            (usdjpy - usdjpy_min10) / (usdjpy_min10 + 1e-9), zw)
    if usdchf is not None:
        df["usdchf_r5"]  = rolling_zscore(log_return(usdchf, 5),  zw)
        df["usdchf_r21"] = rolling_zscore(log_return(usdchf, 21), zw)
    if oil is not None and gold is not None:
        oil_r1   = log_return(oil,  1)
        gold_r1b = log_return(gold, 1)
        oilgold  = oil_r1.rolling(10).corr(gold_r1b)
        df["oil_gold_corr10"]   = rolling_zscore(oilgold, zw)
        df["oil_gold_corr_chg"] = rolling_zscore(oilgold.diff(5), zw)
        df["oil_gold_corr20"]   = rolling_zscore(
            oil_r1.rolling(20).corr(gold_r1b), zw)
    if move is not None:
        df["move_level"]     = rolling_zscore(move / 100, zw)
        df["move_r10"]       = rolling_zscore(log_return(move, 10), zw)
        df["move_vix_ratio"] = rolling_zscore(
            move / (vix * 10 + 1e-9) if vix is not None else move, zw)

    # 11. GPR
    if gpr is not None:
        gpr_norm = gpr.copy()
        gpr_norm.index = pd.to_datetime(
            gpr_norm.index).normalize().tz_localize(None)
        gpr_norm = gpr_norm[~gpr_norm.index.duplicated(keep="last")]
        gpr_aligned = gpr_norm.reindex(prices.index).ffill().bfill()
        for col in gpr_aligned.columns:
            s = gpr_aligned[col].astype(float)
            df[f"gpr_{col.lower()}_level"] = rolling_zscore(s, zw)
            df[f"gpr_{col.lower()}_chg5"]  = rolling_zscore(s.diff(5), zw)
            df[f"gpr_{col.lower()}_chg21"] = rolling_zscore(s.diff(21), zw)
        if gold is not None and "GPR" in gpr_aligned.columns:
            gpr_chg = gpr_aligned["GPR"].diff(5)
            gold_chg = log_return(gold, 5)
            df["gold_gpr_divergence"] = rolling_zscore(
                rolling_zscore(gold_chg, zw) - rolling_zscore(gpr_chg, zw), zw)

    # 12. COT
    if cot is not None:
        cot_norm = cot.copy()
        cot_norm.index = pd.to_datetime(
            cot_norm.index).normalize().tz_localize(None)
        cot_norm = cot_norm[~cot_norm.index.duplicated(keep="last")]
        cot_aligned = cot_norm.reindex(prices.index).ffill().bfill()
        if "net_spec" in cot_aligned.columns:
            ns = cot_aligned["net_spec"].astype(float)
            df["cot_net_spec"]       = rolling_zscore(ns, zw)
            df["cot_net_spec_chg4"]  = rolling_zscore(ns.diff(4),  zw)
            df["cot_net_spec_chg13"] = rolling_zscore(ns.diff(13), zw)
            df["cot_spec_extreme"]   = rolling_zscore(
                ns.rolling(252).rank(pct=True), zw)
        if "net_comm" in cot_aligned.columns:
            nc = cot_aligned["net_comm"].astype(float)
            df["cot_net_comm"]      = rolling_zscore(nc, zw)
            df["cot_net_comm_chg4"] = rolling_zscore(nc.diff(4), zw)
        if "spec_oi_ratio" in cot_aligned.columns:
            sr = cot_aligned["spec_oi_ratio"].astype(float)
            df["cot_spec_oi"]      = rolling_zscore(sr, zw)
            df["cot_spec_oi_chg4"] = rolling_zscore(sr.diff(4), zw)
        if "net_spec" in cot_aligned.columns and "net_comm" in cot_aligned.columns:
            df["cot_spec_comm_div"] = rolling_zscore(
                cot_aligned["net_spec"].astype(float) -
                cot_aligned["net_comm"].astype(float), zw)

    # 13. GLD Holdings
    if gld_holdings is not None:
        h_norm = gld_holdings.copy()
        h_norm.index = pd.to_datetime(
            h_norm.index).normalize().tz_localize(None)
        h_norm = h_norm[~h_norm.index.duplicated(keep="last")]
        h = h_norm.reindex(prices.index).ffill().bfill().astype(float)
        df["gld_hold_level"] = rolling_zscore(h, zw)
        df["gld_hold_r5"]    = rolling_zscore(h.diff(5),  zw)
        df["gld_hold_r21"]   = rolling_zscore(h.diff(21), zw)
        df["gld_hold_accel"] = rolling_zscore(h.diff(5).diff(5), zw)
        h_ma63 = h.rolling(63).mean()
        df["gld_hold_trend"] = rolling_zscore(
            (h - h_ma63) / (h_ma63 + 1e-9), zw)
        df["gld_hold_pct"]   = rolling_zscore(
            h.rolling(252).rank(pct=True), zw)

    # 14. Google Trends
    if trends is not None:
        tr_norm = trends.copy()
        tr_norm.index = pd.to_datetime(
            tr_norm.index).normalize().tz_localize(None)
        tr_norm = tr_norm[~tr_norm.index.duplicated(keep="last")]
        tr = tr_norm.reindex(prices.index).ffill().bfill()
        for col in tr.columns:
            s = tr[col].astype(float)
            safe = col.lower().replace(" ", "_")
            df[f"gtrend_{safe}_level"] = rolling_zscore(s, zw)
            df[f"gtrend_{safe}_chg4"]  = rolling_zscore(s.diff(4), zw)
        if "gold_attention" in tr.columns and gold is not None:
            gold_r21  = log_return(gold, 21)
            attn_chg4 = tr["gold_attention"].astype(float).diff(4)
            df["gtrend_chase_signal"] = rolling_zscore(
                attn_chg4 / (gold_r21.abs() + 0.01), zw)

    # 15. Energy-specific features (XLE model)
    xle    = prices.get("XLE")
    xop    = prices.get("XOP")
    oih    = prices.get("OIH")
    brent  = prices.get("BZ=F")
    natgas = prices.get("NG=F")
    ho     = prices.get("HO=F")
    spy_e  = prices.get("SPY")

    if xle is not None:
        # XLE/SPY ratio — sector rotation
        if spy_e is not None:
            ratio = xle / (spy_e + 1e-9)
            df["xle_spy_ratio"]  = rolling_zscore(ratio, zw)
            df["xle_spy_mom5"]   = rolling_zscore(log_return(ratio, 5), zw)
            df["xle_spy_mom21"]  = rolling_zscore(log_return(ratio, 21), zw)
        # XLE vs oil spread
        if oil is not None:
            df["xle_oil_spread"] = rolling_zscore(
                log_return(xle, 5) - log_return(oil, 5), zw)

    if xop is not None and xle is not None:
        ratio = xop / (xle + 1e-9)
        df["xop_xle_ratio"]  = rolling_zscore(ratio, zw)
        df["xop_r5"]         = rolling_zscore(log_return(xop, 5), zw)
        df["xop_r21"]        = rolling_zscore(log_return(xop, 21), zw)

    if oih is not None:
        df["oih_r5"]         = rolling_zscore(log_return(oih, 5), zw)
        df["oih_r21"]        = rolling_zscore(log_return(oih, 21), zw)

    if oil is not None:
        oil_vol = log_return(oil, 1).rolling(20).std() * np.sqrt(252)
        df["oil_vol20"]      = rolling_zscore(oil_vol, zw)

    if brent is not None and oil is not None:
        df["brent_wti_spread"] = rolling_zscore(brent - oil, zw)
        df["brent_r5"]         = rolling_zscore(log_return(brent, 5), zw)

    if natgas is not None:
        df["natgas_r5"]      = rolling_zscore(log_return(natgas, 5), zw)
        df["natgas_r21"]     = rolling_zscore(log_return(natgas, 21), zw)

    if ho is not None and oil is not None:
        crack = ho * 42 - oil
        df["crack_spread"]   = rolling_zscore(crack, zw)
        df["crack_r5"]       = rolling_zscore(
            log_return(crack.clip(lower=1), 5), zw)

    # Seasonality — energy has strong annual patterns
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["week_sin"]  = np.sin(2 * np.pi *
                             df.index.isocalendar().week.astype(float) / 52)
    df["week_cos"]  = np.cos(2 * np.pi *
                             df.index.isocalendar().week.astype(float) / 52)

    # Crude oil COT positioning
    if cot_crude is not None:
        cc = cot_crude.copy()
        cc.index = pd.to_datetime(cc.index).normalize().tz_localize(None)
        cc = cc[~cc.index.duplicated(keep="last")]
        cc = cc.reindex(prices.index).ffill().bfill()
        if "oil_net_spec" in cc.columns:
            ns = cc["oil_net_spec"].astype(float)
            df["oil_cot_net_spec"]      = rolling_zscore(ns, zw)
            df["oil_cot_net_spec_chg4"] = rolling_zscore(ns.diff(4), zw)
            df["oil_cot_spec_extreme"]  = rolling_zscore(
                ns.rolling(252).rank(pct=True), zw)
        if "oil_spec_oi_ratio" in cc.columns:
            sr = cc["oil_spec_oi_ratio"].astype(float)
            df["oil_cot_spec_oi"]       = rolling_zscore(sr, zw)
            df["oil_cot_spec_oi_chg4"]  = rolling_zscore(sr.diff(4), zw)
        if "oil_net_comm" in cc.columns:
            df["oil_cot_net_comm"]      = rolling_zscore(
                cc["oil_net_comm"].astype(float), zw)

    # XLE own momentum features — energy sector trend persistence.
    # Mirrors section 15 in xle_trainer.py (build_features). Required by
    # post-2026-05 XLE models; older XLE checkpoints simply won't reference
    # these columns and align_features will leave them unread.
    if xle is not None:
        xle_ma50  = xle.rolling(50).mean()
        xle_ma200 = xle.rolling(200).mean()
        df["xle_r5"]         = rolling_zscore(log_return(xle,  5), zw)
        df["xle_r10"]        = rolling_zscore(log_return(xle, 10), zw)
        df["xle_r21"]        = rolling_zscore(log_return(xle, 21), zw)
        df["xle_ma50_dist"]  = rolling_zscore(
            xle / (xle_ma50 + 1e-9) - 1, zw)
        df["xle_ma200_dist"] = rolling_zscore(
            xle / (xle_ma200 + 1e-9) - 1, zw)
        df["xle_trend"]      = rolling_zscore(
            xle_ma50 / (xle_ma200 + 1e-9) - 1, zw)

    # QQQ own momentum features — tech sector trend persistence.
    # Mirrors section 15 in qqq_trainer.py. Required by QQQ models trained
    # after 2026-05-27 (which include qqq_r63 as the strongest QQQ-specific
    # feature, plus the long-window variants for long-climb capture).
    # Older QQQ checkpoints (e.g. seed 74174) won't reference these columns
    # and align_features will leave them unread.
    qqq_p = prices.get("QQQ")
    if qqq_p is not None:
        qqq_ma50  = qqq_p.rolling(50).mean()
        qqq_ma200 = qqq_p.rolling(200).mean()
        df["qqq_r5"]         = rolling_zscore(log_return(qqq_p,  5), zw)
        df["qqq_r10"]        = rolling_zscore(log_return(qqq_p, 10), zw)
        df["qqq_r21"]        = rolling_zscore(log_return(qqq_p, 21), zw)
        df["qqq_ma50_dist"]  = rolling_zscore(
            qqq_p / (qqq_ma50 + 1e-9) - 1, zw)
        df["qqq_ma200_dist"] = rolling_zscore(
            qqq_p / (qqq_ma200 + 1e-9) - 1, zw)
        df["qqq_trend"]      = rolling_zscore(
            qqq_ma50 / (qqq_ma200 + 1e-9) - 1, zw)
        # Long-window momentum — for long-climb capture
        df["qqq_r63"]        = rolling_zscore(log_return(qqq_p, 63),  zw)
        df["qqq_r126"]       = rolling_zscore(log_return(qqq_p, 126), zw)
        # Momentum acceleration
        qqq_r21_raw          = log_return(qqq_p, 21)
        df["qqq_r21_chg5"]   = rolling_zscore(qqq_r21_raw.diff(5), zw)
        # Volatility-adjusted 21d momentum
        qqq_daily_ret        = log_return(qqq_p, 1)
        qqq_vol20            = qqq_daily_ret.rolling(20).std() * np.sqrt(252)
        df["qqq_vol_adj_r21"] = rolling_zscore(
            qqq_r21_raw / (qqq_vol20 + 1e-9), zw)

    # XLK tech sector peer momentum — for QQQ models trained post 2026-05-27.
    # XLK was added to TICKERS to support this; older models that don't
    # reference these columns will simply not see them via align_features.
    xlk_p = prices.get("XLK")
    if xlk_p is not None:
        df["xlk_r10"]   = rolling_zscore(log_return(xlk_p, 10), zw)
        df["xlk_r21"]   = rolling_zscore(log_return(xlk_p, 21), zw)
        df["xlk_r63"]   = rolling_zscore(log_return(xlk_p, 63), zw)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Earnings season features — for QQQ model compatibility
    # QQQ earnings season = last 3 weeks of Jan/Apr/Jul/Oct
    idx        = prices.index
    month      = pd.Series(idx.month, index=idx)
    day        = pd.Series(idx.day,   index=idx)
    in_earn    = ((month.isin([1, 4, 7, 10])) & (day >= 10)).astype(float)
    pre_earn   = ((month.isin([12, 3, 6, 9])) & (day >= 20)).astype(float)
    df["pre_earnings"]    = pre_earn
    df["earnings_season"] = in_earn
    df["earnings_phase"]  = rolling_zscore(in_earn.rolling(5).sum(), zw)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. ALIGN FEATURES TO MODEL'S EXPECTED COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def align_features(features_df, feature_names):
    """
    Reorder and filter features to match exactly what the model was trained on.
    Missing features are filled with 0 (z-score neutral).
    Extra features are dropped.
    """
    aligned = pd.DataFrame(index=features_df.index)
    missing = []
    for col in feature_names:
        if col in features_df.columns:
            aligned[col] = features_df[col]
        else:
            aligned[col] = 0.0
            missing.append(col)
    if missing:
        if len(missing) > 5:
            print(f"  ⚠ {len(missing)} features missing (filled with 0): "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        else:
            # Minor version difference — suppress noisy warning for small gaps
            pass
    return aligned


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble(fold_models, features_aligned, cfg):
    """Run all fold models on the latest lookback window."""
    lb     = cfg["lookback"]
    window = features_aligned.iloc[-lb:].values.astype(np.float32)

    if len(window) < lb:
        print(f"  ✗ Not enough data for lookback window "
              f"(need {lb}, have {len(window)})")
        return None

    x_tensor = torch.tensor(window).unsqueeze(0)
    n        = len(fold_models)
    preds    = []

    for fm in fold_models:
        with torch.no_grad():
            pred = fm["model"](x_tensor).item()
        preds.append({
            "pred":     pred,
            "dir_acc":  fm["dir_acc"],
            "ic":       fm["ic"],
            "fold_num": fm["fold_num"],
        })

    # Ensemble weights — 60% accuracy + 40% recency
    da_arr       = np.array([p["dir_acc"] for p in preds])
    recency_w    = cfg["ensemble_recency_weight"]
    acc_weights  = np.maximum(da_arr - 0.5, 0.01)
    acc_weights  = acc_weights / acc_weights.sum()
    rec_raw      = np.exp(np.linspace(0, 2, n))
    rec_weights  = rec_raw / rec_raw.sum()
    weights      = (1 - recency_w) * acc_weights + recency_w * rec_weights
    weights      = weights / weights.sum()

    preds_arr = np.array([p["pred"] for p in preds])
    ensemble  = float(np.dot(weights, preds_arr))
    agreement = float(np.mean(np.sign(preds_arr) == np.sign(ensemble)))

    return {
        "ensemble":   ensemble,
        "pct_ret":    (np.exp(ensemble) - 1) * 100,
        "direction":  1 if ensemble > 0 else -1,
        "agreement":  agreement,
        "n_folds":    n,
        "weights":    weights,
        "fold_preds": preds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRADING DECISION
# ══════════════════════════════════════════════════════════════════════════════

def make_trading_decision(result, cfg, prices):
    """Convert ensemble output into a concrete trading decision."""
    ensemble  = result["ensemble"]
    agreement = result["agreement"]
    min_sig   = cfg["min_signal"]
    max_pos   = cfg["max_position_frac"]

    # Confidence percentile — approximate using the ensemble magnitude
    # relative to the historical OOF distribution
    oof_path = os.path.join(OUT_DIR, "oof_predictions.csv")
    conf_pct  = 0.5   # default if no history
    if os.path.exists(oof_path):
        oof = pd.read_csv(oof_path)
        if "pred" in oof.columns:
            conf_pct = float((oof["pred"].abs() < abs(ensemble)).mean())

    # Position size from confidence tier
    size_frac = 0.0
    for (lo, hi, sz) in cfg["conf_tiers"]:
        if lo <= conf_pct < hi:
            size_frac = sz
            break

    # No trade if below minimum signal
    if abs(ensemble) < min_sig:
        return {
            "action":    "HOLD CASH",
            "position":  0.0,
            "conf_pct":  conf_pct,
            "size_frac": 0.0,
            "reason":    "signal below minimum threshold",
        }

    # Direction
    if ensemble > 0:
        position = size_frac * max_pos
        action   = f"BUY GDX"
    else:
        position = 0.0
        action   = "HOLD CASH"   # cash_long mode

    # Volatility check
    gdx_prices  = prices["GDX"].dropna()
    recent_rets = np.log(gdx_prices / gdx_prices.shift(1)).dropna()
    realised_vol= float(recent_rets.iloc[-20:].std() * np.sqrt(252) * 100)
    vol_warning = realised_vol < 12

    return {
        "action":       action,
        "position":     position,
        "conf_pct":     conf_pct,
        "size_frac":    size_frac,
        "realised_vol": realised_vol,
        "vol_warning":  vol_warning,
        "reason":       f"conf_pct={conf_pct:.0%}  agree={agreement:.0%}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. SIGNAL LOG
# ══════════════════════════════════════════════════════════════════════════════

def load_signal_log():
    """Load existing signal history."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_PATH, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()


def save_signal(result, decision, as_of_date):
    """Append today's signal to the log."""
    log = load_signal_log()
    row = {
        "date":        as_of_date,
        "direction":   "BULLISH" if result["direction"] > 0 else "BEARISH",
        "ensemble":    round(result["ensemble"], 4),
        "pct_ret":     round(result["pct_ret"], 2),
        "agreement":   round(result["agreement"], 2),
        "n_folds":     result["n_folds"],
        "action":      decision["action"],
        "position":    round(decision["position"], 2),
        "conf_pct":    round(decision["conf_pct"], 2),
        "realised_vol":round(decision.get("realised_vol", 0), 1),
        "outcome_actual": None,
        "outcome_correct": None,
    }
    new_row = pd.DataFrame([row])
    log     = pd.concat([log, new_row], ignore_index=True)

    # Check outcomes for rows from ~10 trading days ago
    log = check_outcomes(log)
    log.to_csv(LOG_PATH, index=False)
    return log


def check_outcomes(log):
    """
    For signals older than 10 trading days, check if GDX moved
    in the predicted direction. Fill outcome columns retrospectively.
    """
    if len(log) < 2:
        return log

    try:
        today = pd.Timestamp.today().normalize()
        cutoff = today - pd.tseries.offsets.BusinessDay(12)

        gdx = yf.Ticker("GDX").history(
            start=(today - pd.DateOffset(days=60)).strftime("%Y-%m-%d"),
            auto_adjust=True)
        gdx.index = pd.to_datetime(gdx.index).normalize().tz_localize(None)
        gdx_close = gdx["Close"]

        for i, row in log.iterrows():
            if pd.isna(row.get("outcome_actual")) and \
               pd.notna(row.get("date")):
                signal_date = pd.Timestamp(row["date"])
                if signal_date <= cutoff:
                    future_dates = gdx_close.index[
                        gdx_close.index > signal_date]
                    if len(future_dates) >= 10:
                        try:
                            entry_loc = gdx_close.index.get_loc(
                                gdx_close.index[
                                    gdx_close.index >= signal_date][0])
                            exit_loc  = min(entry_loc + 10,
                                            len(gdx_close) - 1)
                            price_entry = gdx_close.iloc[entry_loc]
                            price_exit  = gdx_close.iloc[exit_loc]
                            actual_ret  = (price_exit / price_entry - 1) * 100
                            predicted_dir = (1 if row["direction"] == "BULLISH"
                                             else -1)
                            correct = (np.sign(actual_ret) == predicted_dir)
                            log.at[i, "outcome_actual"]  = round(actual_ret, 2)
                            log.at[i, "outcome_correct"] = correct
                        except Exception:
                            pass
    except Exception:
        pass

    return log


# ══════════════════════════════════════════════════════════════════════════════
# 9. PRINT SIGNAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 9b. GENERATE RECENT PREDICTIONS
# Slides the feature window back day by day to reconstruct what the
# ensemble would have said on each of the last N trading days.
# Adds ~30 seconds to the run but gives a complete current history.
# ══════════════════════════════════════════════════════════════════════════════

def generate_recent_predictions(fold_models, features_aligned, cfg,
                                 n_days=10):
    """
    For each of the last n_days trading days, slide the lookback window
    back one day at a time and run the ensemble.

    Returns a DataFrame with columns:
        date, pred, conf_pct, direction, position
    indexed newest-first.
    """
    lb        = cfg["lookback"]
    max_pos   = cfg["max_position_frac"]
    min_sig   = cfg["min_signal"]
    all_preds_abs = features_aligned.index  # just need length

    # We need at least lb + n_days rows of features
    if len(features_aligned) < lb + n_days:
        return pd.DataFrame()

    rows = []
    # features_aligned is sorted oldest→newest
    # index -1 = today, -2 = yesterday, etc.
    total = len(features_aligned)

    for day_offset in range(n_days):
        # Window ending at (today - day_offset)
        end_idx   = total - day_offset
        start_idx = end_idx - lb
        if start_idx < 0:
            break

        window   = features_aligned.iloc[start_idx:end_idx].values.astype(np.float32)
        date_idx = features_aligned.index[end_idx - 1]
        x_tensor = torch.tensor(window).unsqueeze(0)

        fold_preds_vals = []
        for fm in fold_models:
            with torch.no_grad():
                p = fm["model"](x_tensor).item()
            fold_preds_vals.append({"pred": p, "dir_acc": fm["dir_acc"],
                                    "ic": fm["ic"]})

        # Weighted ensemble
        da_arr      = np.array([f["dir_acc"] for f in fold_preds_vals])
        rec_w       = cfg["ensemble_recency_weight"]
        acc_w       = np.maximum(da_arr - 0.5, 0.01)
        acc_w      /= acc_w.sum()
        rec_raw     = np.exp(np.linspace(0, 2, len(fold_preds_vals)))
        rec_wts     = rec_raw / rec_raw.sum()
        weights     = (1 - rec_w) * acc_w + rec_w * rec_wts
        weights    /= weights.sum()
        preds_arr   = np.array([f["pred"] for f in fold_preds_vals])
        ensemble    = float(np.dot(weights, preds_arr))

        # Confidence percentile — rank vs all predictions seen so far
        # Use window of known predictions from full_predictions.csv if available
        conf_pct = 0.5   # default

        # Direction and position size
        if abs(ensemble) < min_sig:
            direction = 0
            pos       = 0.0
        elif ensemble > 0:
            direction = 1
            size_frac = 0.0
            for (lo, hi, sz) in cfg["conf_tiers"]:
                if lo <= conf_pct < hi:
                    size_frac = sz
                    break
            pos = size_frac * max_pos
        else:
            direction = -1
            pos       = 0.0

        rows.append({
            "date":      date_idx,
            "pred":      ensemble,
            "conf_pct":  conf_pct,
            "direction": direction,
            "position":  pos,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index(ascending=False)

    # Now calibrate conf_pct using the distribution across all days we computed
    all_abs = df["pred"].abs().values
    for i in range(len(df)):
        p = abs(df.iloc[i]["pred"])
        df.iloc[i, df.columns.get_loc("conf_pct")] = float(
            (all_abs < p).mean())

    # Re-apply position sizing with calibrated confidence
    for i in range(len(df)):
        row       = df.iloc[i]
        conf      = float(row["conf_pct"])
        ensemble  = float(row["pred"])
        if abs(ensemble) < min_sig or ensemble <= 0:
            df.iloc[i, df.columns.get_loc("position")] = 0.0
        else:
            size_frac = 0.0
            for (lo, hi, sz) in cfg["conf_tiers"]:
                if lo <= conf < hi:
                    size_frac = sz
                    break
            df.iloc[i, df.columns.get_loc("position")] = size_frac * max_pos

    # Deduplicate — keep only one prediction per date (newest window wins)
    df = df[~df.index.duplicated(keep="first")]

    return df
# ══════════════════════════════════════════════════════════════════════════════
# 10. PORTFOLIO TRACKER
# Replays every trade in rebalance_log.csv against real GDX prices.
# Gives live paper-trading P&L independent of the backtest simulation.
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 100_000.0   # starting paper portfolio value

def compute_live_portfolio(prices, recent_preds=None):
    """
    Reads daily_equity.csv (generated by aurum_backtest.py) for historical
    portfolio values, then marks the current position to today's GDX price.

    This is the correct approach — the backtest already computed the proper
    daily P&L including mark-to-market gains between trades.
    """
    equity_path = os.path.join("aurum_backtest", "daily_equity.csv")
    rebal_path  = os.path.join("aurum_backtest", "rebalance_log.csv")

    if not os.path.exists(equity_path):
        return None

    try:
        eq = pd.read_csv(equity_path, parse_dates=["date"], index_col="date")
        eq.index = pd.to_datetime(eq.index).normalize().tz_localize(None)
        eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    except Exception:
        return None

    if len(eq) == 0:
        return None

    # ── Extend equity curve to today using fresh GDX prices ──────────────
    # The equity file ends ~10 days before today (forward return lag).
    # We extend it forward day by day using the last known position and
    # fresh GDX daily returns — no retraining needed.
    try:
        last_eq_date  = eq.index[-1]
        today         = pd.Timestamp.today().normalize()

        if last_eq_date < today - pd.tseries.offsets.BusinessDay(1):
            # Download GDX from day after last equity date to today
            gdx_ext = yf.Ticker("GDX").history(
                start=(last_eq_date + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True)
            gdx_ext.index = pd.to_datetime(
                gdx_ext.index).normalize().tz_localize(None)
            gdx_ext = gdx_ext["Close"].dropna()
            gdx_ext = gdx_ext[~gdx_ext.index.duplicated(keep="last")].sort_index()

            if len(gdx_ext) > 0:
                # Get GDX price at last equity date for return calculation
                gdx_base = yf.Ticker("GDX").history(
                    start=(last_eq_date - pd.DateOffset(days=5)).strftime("%Y-%m-%d"),
                    end=(last_eq_date + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True)
                gdx_base.index = pd.to_datetime(
                    gdx_base.index).normalize().tz_localize(None)
                gdx_base_close = float(gdx_base["Close"].dropna().iloc[-1]) \
                                 if len(gdx_base) > 0 else None

                if gdx_base_close:
                    new_rows  = []
                    last_port = float(eq["portfolio_$"].iloc[-1])
                    prev_gdx  = gdx_base_close

                    for date_i, gdx_price in gdx_ext.items():
                        # Position from recent_preds if available
                        if recent_preds is not None and \
                           len(recent_preds) > 0 and \
                           date_i in recent_preds.index:
                            day_pos = float(recent_preds.loc[date_i, "position"])
                        else:
                            day_pos = float(eq["position"].iloc[-1])

                        # Mark to market on invested days; earn SGOV on flat days
                        SGOV_DAILY = CASH_DAILY
                        if day_pos > 0:
                            daily_ret = (gdx_price / prev_gdx - 1) * day_pos
                            last_port = last_port * (1 + daily_ret)
                        # Cash portion always earns SGOV
                        last_port = last_port + last_port * (1 - day_pos) * SGOV_DAILY

                        stock_v = last_port * day_pos
                        cash_v  = last_port * (1 - day_pos)
                        new_rows.append({
                            "date":        date_i,
                            "portfolio_$": round(last_port, 2),
                            "position":    day_pos,
                            "stock_$":     round(stock_v, 2),
                            "cash_$":      round(cash_v, 2),
                        })
                        prev_gdx = gdx_price

                    if new_rows:
                        ext_df = pd.DataFrame(new_rows).set_index("date")
                        eq     = pd.concat([eq, ext_df])
                        eq     = eq[~eq.index.duplicated(keep="last")].sort_index()
                        print(f"    Extended equity curve by {len(new_rows)} days "
                              f"to {eq.index[-1].date()}")
    except Exception as e:
        pass   # extension is best-effort — not critical
    last_eq_date  = eq.index[-1]
    last_port     = float(eq["portfolio_$"].iloc[-1])
    last_position = float(eq["position"].iloc[-1])

    # Get current GDX price
    gdx      = prices["GDX"].dropna() if "GDX" in prices.columns else None
    if gdx is None:
        return None
    gdx_last = float(gdx.iloc[-1])

    # The extended equity already reflects today's GDX price via daily returns.
    # Use it directly — no need to re-mark shares.
    port_today  = last_port
    stock_today = last_port * last_position
    cash        = last_port * (1 - last_position)

    # Share count for display only
    gdx_at_last = gdx_last if last_position > 0 else gdx_last
    gdx_shares  = stock_today / (gdx_last + 1e-9)

    # Get first trade date from rebalance log
    first_date = eq.index[0]
    if os.path.exists(rebal_path):
        try:
            trades     = pd.read_csv(rebal_path, parse_dates=["date"])
            first_date = pd.to_datetime(trades["date"].min()).normalize()
        except Exception:
            pass

    # Returns
    last_date  = pd.Timestamp.today()
    cal_years  = max((last_date - first_date).days / 365.25, 0.01)
    total_ret  = (port_today / INITIAL_CAPITAL - 1) * 100
    ann_ret    = (np.exp(np.log(max(port_today, 1) / INITIAL_CAPITAL)
                         / cal_years) - 1) * 100

    # GDX buy-and-hold from same start date — download full history
    try:
        gdx_full = yf.Ticker("GDX").history(
            start=(first_date - pd.DateOffset(days=5)).strftime("%Y-%m-%d"),
            auto_adjust=True)
        gdx_full.index = pd.to_datetime(
            gdx_full.index).normalize().tz_localize(None)
        gdx_full = gdx_full["Close"].dropna()
        gdx_full = gdx_full[~gdx_full.index.duplicated(keep="last")].sort_index()
        gdx_start_avail = gdx_full[gdx_full.index >= first_date]
        gdx_start = float(gdx_start_avail.iloc[0]) if len(gdx_start_avail) > 0 \
                    else gdx_last
        print(f"    B&H: GDX {first_date.date()} ${gdx_start:.2f} → "
              f"today ${gdx_last:.2f}")
    except Exception as e:
        gdx_start_avail = gdx[gdx.index >= first_date]
        gdx_start = float(gdx_start_avail.iloc[0]) if len(gdx_start_avail) > 0 \
                    else gdx_last
    bh_shares = INITIAL_CAPITAL / gdx_start
    bh_final  = bh_shares * gdx_last
    bh_ret    = (bh_final / INITIAL_CAPITAL - 1) * 100

    # Trade count
    n_trades = 0
    if os.path.exists(rebal_path):
        try:
            n_trades = len(pd.read_csv(rebal_path))
        except Exception:
            pass

    return {
        "initial":      INITIAL_CAPITAL,
        "port_final":   port_today,
        "stock_val":    max(stock_today, 0),
        "cash_val":     cash,
        "current_pos":  last_position,
        "gdx_last":     gdx_last,
        "gdx_shares":   gdx_shares,
        "n_trades":     n_trades,
        "first_trade":  first_date,
        "last_trade":   last_eq_date,
        "total_ret":    total_ret,
        "ann_ret":      ann_ret,
        "cal_years":    cal_years,
        "bh_ret":       bh_ret,
        "bh_final":     bh_final,
        "last_eq_date": last_eq_date,
        "last_port_bs": last_port,
        "equity_df":    eq,
    }
def compute_rotation_portfolio(recent_preds, xle_preds, qqq_preds,
                               prices, start_port, scfg):
    """
    Replay the combined GDX+XLE rotation strategy day by day.

    Rules:
      - GDX position from recent_preds (as normal)
      - Remaining idle cash: if XLE bullish conf>=40% → deploy into XLE
      - Cash earns SGOV_RATE when not invested

    Returns a dict mapping date → {portfolio_$, gdx_$, xle_$, cash_$, gdx_pos, xle_pos}
    sorted oldest→newest.
    """
    SGOV_RATE    = CASH_RATE          # ~5.2% annual T-bill rate
    SGOV_DAILY   = SGOV_RATE / 252
    MIN_ROT_CONF = 0.40           # minimum confidence to deploy rotation

    if recent_preds is None or len(recent_preds) == 0:
        return {}

    preds_sorted = recent_preds.sort_index()   # oldest first

    # Build price series for GDX and XLE
    gdx_p = xle_p = None
    if prices is not None:
        if "GDX" in prices.columns:
            gdx_p = prices["GDX"].dropna().copy()
            gdx_p.index = pd.to_datetime(gdx_p.index).normalize()
        if "XLE" in prices.columns:
            xle_p = prices["XLE"].dropna().copy()
            xle_p.index = pd.to_datetime(xle_p.index).normalize()

    # Helper: get price on or before date
    def get_price(series, date):
        if series is None:
            return None
        avail = series[series.index <= date]
        return float(avail.iloc[-1]) if len(avail) > 0 else None

    # Helper: get XLE position for date
    def get_xle_pos(date, idle_cash_frac):
        """Return fraction of TOTAL portfolio to put in XLE on this date."""
        if xle_preds is None or idle_cash_frac <= 0.01:
            return 0.0
        avail = xle_preds[xle_preds.index <= date]
        if len(avail) == 0 or (date - avail.index[-1]).days > 5:
            return 0.0
        v = float(avail.iloc[-1])
        if v <= scfg.get("min_signal", 0.002):
            return 0.0
        # Estimate confidence from oof distribution
        oof_path = os.path.join("xle_output", "oof_predictions.csv")
        try:
            oof = pd.read_csv(oof_path)
            conf = float((oof["pred"].abs() < abs(v)).mean()) \
                   if "pred" in oof.columns else 0.5
        except Exception:
            conf = 0.5
        if conf < MIN_ROT_CONF:
            return 0.0
        # Size from tiers, capped at idle cash fraction
        size = 0.0
        for (lo, hi, sz) in scfg["conf_tiers"]:
            if lo <= conf < hi:
                size = sz
                break
        return min(size, idle_cash_frac)

    # ── Replay ────────────────────────────────────────────────────────────
    portfolio   = {}
    running     = start_port
    prev_gdx    = get_price(gdx_p, preds_sorted.index[0] - pd.DateOffset(days=5))
    prev_xle    = get_price(xle_p, preds_sorted.index[0] - pd.DateOffset(days=5))

    for date_idx in preds_sorted.index:
        gdx_pos  = float(preds_sorted.loc[date_idx, "position"])
        curr_gdx = get_price(gdx_p, date_idx)
        curr_xle = get_price(xle_p, date_idx)

        # XLE position from remaining idle cash
        idle = 1.0 - gdx_pos
        xle_pos = get_xle_pos(date_idx, idle)
        cash_pos = idle - xle_pos

        # Daily returns
        gdx_ret = 0.0
        if prev_gdx and curr_gdx and gdx_pos > 0:
            gdx_ret = (curr_gdx / prev_gdx - 1) * gdx_pos

        xle_ret = 0.0
        if prev_xle and curr_xle and xle_pos > 0:
            xle_ret = (curr_xle / prev_xle - 1) * xle_pos

        cash_ret = cash_pos * SGOV_DAILY

        running = running * (1 + gdx_ret + xle_ret) + running * cash_ret

        portfolio[date_idx] = {
            "portfolio_$": round(running, 2),
            "gdx_$":       round(running * gdx_pos, 2),
            "xle_$":       round(running * xle_pos, 2),
            "cash_$":      round(running * cash_pos, 2),
            "gdx_pos":     gdx_pos,
            "xle_pos":     xle_pos,
        }

        if curr_gdx:
            prev_gdx = curr_gdx
        if curr_xle:
            prev_xle = curr_xle

    return portfolio


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL HISTORY CHART
# GDX candlesticks + portfolio value + position shading + trade markers
# ══════════════════════════════════════════════════════════════════════════════

def plot_signal_history(recent_preds, equity_df, prices, as_of_date,
                        actuals_s=None, port_series=None, rot_portfolio=None,
                        rot_gdx_xle=None, rot_gdx_tlt=None,
                        xle_recent_preds=None, qqq_recent_preds=None):
    """
    4-panel chart:
      1. GDX candlesticks with position shading + buy/sell markers
      2. Portfolio value vs GDX buy-and-hold
      3. Position % bar (how much capital is invested each day)
      4. Model signal (predicted return)
    Saved to aurum_output/signal_history_chart.png
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print("  ✗ matplotlib not available for chart")
        return

    if recent_preds is None or len(recent_preds) == 0:
        return
    if prices is None or "GDX" not in prices.columns:
        return

    # ── Colours ───────────────────────────────────────────────────────────
    DARK  = "#0d1117"
    PANEL = "#161b22"
    GREEN = "#3fb950"
    RED   = "#f85149"
    GOLD  = "#d4a939"
    BLUE  = "#58a6ff"
    MUTED = "#8b949e"

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.yaxis.label.set_color(MUTED)

    # ── Align data ────────────────────────────────────────────────────────
    preds = recent_preds.sort_index()   # oldest→newest

    gdx = prices["GDX"].dropna().copy()
    gdx.index = pd.to_datetime(gdx.index).normalize()
    gdx = gdx[~gdx.index.duplicated(keep="last")].sort_index()

    # Restrict to preds date range — ensure tz-naive
    start_date = pd.Timestamp(preds.index[0]).tz_localize(None).normalize()
    end_date   = pd.Timestamp(preds.index[-1]).tz_localize(None).normalize()
    gdx_window = gdx[(gdx.index >= start_date) & (gdx.index <= end_date)]

    # Download OHLCV for candles
    try:
        import yfinance as yf
        ohlcv = yf.Ticker("GDX").history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + pd.DateOffset(days=2)).strftime("%Y-%m-%d"),
            auto_adjust=True)
        ohlcv.index = pd.to_datetime(ohlcv.index).normalize().tz_localize(None)
        ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()
        ohlcv = ohlcv[(ohlcv.index >= start_date) & (ohlcv.index <= end_date)]
    except Exception:
        ohlcv = None

    if ohlcv is None or len(ohlcv) < 5:
        print("  ✗ Could not download OHLCV for chart")
        return

    # Use integer x-axis (no weekend gaps)
    n          = len(ohlcv)
    xs         = np.arange(n)
    date_to_x  = {d: i for i, d in enumerate(ohlcv.index)}

    # Align preds and equity to OHLCV dates — no ffill, missing = 0 (flat)
    pos_series  = preds["position"].reindex(ohlcv.index).fillna(0)
    pred_series = preds["pred"].reindex(ohlcv.index).fillna(0)

    # XLE & QQQ position series — strength-ranked cascade (patched 2026-05-28).
    # On each day, rank XLE vs QQQ by confidence percentile and deploy the
    # stronger one first from idle cash; the weaker one gets what's left.
    # Mirrors the cascade in run_full_history_simulation().
    xle_pos_series = pd.Series(0.0, index=ohlcv.index)
    qqq_pos_series = pd.Series(0.0, index=ohlcv.index)

    xle_raw  = None
    xle_conf = None
    if xle_recent_preds is not None and len(xle_recent_preds) > 0:
        xle_pos_df = xle_recent_preds["position"].copy()
        xle_pos_df.index = pd.to_datetime(xle_pos_df.index).normalize()
        xle_pos_df = xle_pos_df[~xle_pos_df.index.duplicated(keep="last")]
        xle_raw = xle_pos_df.reindex(ohlcv.index).ffill().fillna(0)
        if "conf_pct" in xle_recent_preds.columns:
            xle_conf_df = xle_recent_preds["conf_pct"].copy()
            xle_conf_df.index = pd.to_datetime(xle_conf_df.index).normalize()
            xle_conf_df = xle_conf_df[~xle_conf_df.index.duplicated(keep="last")]
            xle_conf = xle_conf_df.reindex(ohlcv.index).ffill().fillna(0)

    qqq_raw  = None
    qqq_conf = None
    if qqq_recent_preds is not None and len(qqq_recent_preds) > 0:
        qqq_pos_df = qqq_recent_preds["position"].copy()
        qqq_pos_df.index = pd.to_datetime(qqq_pos_df.index).normalize()
        qqq_pos_df = qqq_pos_df[~qqq_pos_df.index.duplicated(keep="last")]
        qqq_raw = qqq_pos_df.reindex(ohlcv.index).ffill().fillna(0)
        if "conf_pct" in qqq_recent_preds.columns:
            qqq_conf_df = qqq_recent_preds["conf_pct"].copy()
            qqq_conf_df.index = pd.to_datetime(qqq_conf_df.index).normalize()
            qqq_conf_df = qqq_conf_df[~qqq_conf_df.index.duplicated(keep="last")]
            qqq_conf = qqq_conf_df.reindex(ohlcv.index).ffill().fillna(0)

    for i in range(len(ohlcv)):
        gdx_p = float(pos_series.iloc[i])
        idle  = max(0.0, 1.0 - gdx_p)
        # Build candidate list with (name, target_position, confidence)
        cands = []
        if xle_raw is not None:
            xc = float(xle_conf.iloc[i]) if xle_conf is not None else 1.0
            cands.append(("xle", float(xle_raw.iloc[i]), xc))
        if qqq_raw is not None:
            qc = float(qqq_conf.iloc[i]) if qqq_conf is not None else 1.0
            cands.append(("qqq", float(qqq_raw.iloc[i]), qc))
        cands.sort(key=lambda c: c[2], reverse=True)
        remaining = idle
        deploy = {"xle": 0.0, "qqq": 0.0}
        for name, target, _ in cands:
            if remaining > 0.0 and target > 0:
                alloc = min(target, remaining)
                deploy[name] = alloc
                remaining -= alloc
        xle_pos_series.iloc[i] = deploy["xle"]
        qqq_pos_series.iloc[i] = deploy["qqq"]

    eq_series      = None
    rot_xle_series = None

    # rot_gdx_xle can be a scalar (start value) or a date→value dict
    if isinstance(rot_gdx_xle, dict):
        start_val = 100_000.0  # fallback
    else:
        start_val = float(rot_gdx_xle) if rot_gdx_xle is not None else 100_000.0

    # Build position lookup from recent_preds (same source as buy/sell markers)
    pos_lookup = {}
    for d, row in preds.iterrows():
        pos_lookup[pd.Timestamp(d).normalize().date()] = float(row["position"])

    # GDX-only equity line — computed from recent_preds × daily returns
    ohlcv_ret = ohlcv["Close"].pct_change().fillna(0)
    capital   = start_val
    last_pos  = 0.0
    eq_vals   = []
    for i, d in enumerate(ohlcv.index):
        dk  = pd.Timestamp(d).normalize().date()
        pos = pos_lookup.get(dk, last_pos)
        last_pos = pos
        capital = capital * (1 + float(ohlcv_ret.iloc[i]) * pos)
        eq_vals.append(capital)
    eq_series = pd.Series(eq_vals, index=ohlcv.index)

    # GDX+XLE rotation line — recompute fresh from recent_preds + xle_recent_preds
    # Same approach as GDX-only line — both start at start_val, compound daily
    # This avoids scaling issues from reading sim_df (which starts in 2010)
    if xle_recent_preds is not None and len(xle_recent_preds) > 0:
        xle_pos_lookup = {}
        for d, row in xle_recent_preds.iterrows():
            dk = pd.Timestamp(d).normalize().date()
            xle_pos_lookup[dk] = float(row["position"])

        xle_p = prices["XLE"].dropna() if prices is not None and \
                "XLE" in prices.columns else None
        if xle_p is not None:
            xle_p.index = pd.to_datetime(xle_p.index).normalize()
            xle_ret_lkp = {pd.Timestamp(d).normalize().date(): float(v)
                           for d, v in xle_p.pct_change().fillna(0).items()}
            cap_rot   = start_val
            last_gpos = 0.0
            last_xpos = 0.0
            rot_vals  = []
            for i, d in enumerate(ohlcv.index):
                dk    = pd.Timestamp(d).normalize().date()
                gpos  = pos_lookup.get(dk, last_gpos)
                xpos  = xle_pos_lookup.get(dk, last_xpos)
                # Cap XLE by idle cash
                xpos  = min(xpos, max(0.0, 1.0 - gpos))
                last_gpos, last_xpos = gpos, xpos
                gr    = float(ohlcv_ret.iloc[i]) * gpos
                xr    = xle_ret_lkp.get(dk, 0.0) * xpos
                cap_rot = cap_rot * (1 + gr + xr)
                rot_vals.append(cap_rot)
            rot_xle_series = pd.Series(rot_vals, index=ohlcv.index)
    else:
        rot_xle_series = None

    # GDX+XLE+QQQ rotation line
    rot_qqq_series = None
    if xle_recent_preds is not None and qqq_recent_preds is not None and \
            len(xle_recent_preds) > 0 and len(qqq_recent_preds) > 0:
        qqq_pos_lookup = {}
        for d, row in qqq_recent_preds.iterrows():
            qqq_pos_lookup[pd.Timestamp(d).normalize().date()] = \
                float(row["position"])

        qqq_p = prices["QQQ"].dropna() if prices is not None and \
                "QQQ" in prices.columns else None
        if qqq_p is not None and rot_xle_series is not None:
            qqq_p.index = pd.to_datetime(qqq_p.index).normalize()
            qqq_ret_lkp = {pd.Timestamp(d).normalize().date(): float(v)
                           for d, v in qqq_p.pct_change().fillna(0).items()}
            # Re-build XLE lookup for position capping
            xle_pos_lkp2 = {}
            for d, row in xle_recent_preds.iterrows():
                xle_pos_lkp2[pd.Timestamp(d).normalize().date()] = \
                    float(row["position"])
            cap_rqq  = start_val
            last_gp  = 0.0
            last_xp  = 0.0
            last_qp  = 0.0
            rqq_vals = []
            for i, d in enumerate(ohlcv.index):
                dk   = pd.Timestamp(d).normalize().date()
                gpos = pos_lookup.get(dk, last_gp)
                xpos = min(xle_pos_lkp2.get(dk, last_xp),
                           max(0.0, 1.0 - gpos))
                qpos = min(qqq_pos_lookup.get(dk, last_qp),
                           max(0.0, 1.0 - gpos - xpos))
                last_gp, last_xp, last_qp = gpos, xpos, qpos
                gr   = float(ohlcv_ret.iloc[i]) * gpos
                xr   = xle_ret_lkp.get(dk, 0.0) * xpos
                qr   = qqq_ret_lkp.get(dk, 0.0) * qpos
                cap_rqq = cap_rqq * (1 + gr + xr + qr)
                rqq_vals.append(cap_rqq)
            rot_qqq_series = pd.Series(rqq_vals, index=ohlcv.index)

    # XLE position series for stacked bars
    xle_pos_series_chart = xle_pos_series  # already built above

    # TLT position series (rule-based: above 20d MA)
    tlt_pos_arr = np.zeros(n)
    tlt_prices = prices["TLT"].dropna() if prices is not None and \
                 "TLT" in prices.columns else None
    if tlt_prices is not None:
        tlt_prices.index = pd.to_datetime(tlt_prices.index).normalize()
        for i, d in enumerate(ohlcv.index):
            avail = tlt_prices[tlt_prices.index <= d]
            if len(avail) >= 20:
                ct  = float(avail.iloc[-1])
                ma  = float(avail.iloc[-20:].mean())
                gdx_p_i = float(pos_series.iloc[i])
                xle_p_i = float(xle_pos_series.iloc[i]) \
                          if i < len(xle_pos_series) else 0.0
                idle = max(1.0 - gdx_p_i - xle_p_i, 0.0)
                tlt_pos_arr[i] = idle if ct > ma and idle > 0.01 else 0.0

    # XLE signal series for overlay
    xle_sig_series = None
    if xle_recent_preds is not None and len(xle_recent_preds) > 0:
        xs_df = xle_recent_preds["pred"].copy()
        xs_df.index = pd.to_datetime(xs_df.index).normalize()
        xle_sig_series = xs_df.reindex(ohlcv.index)

    prev_pos    = pos_series.shift(1).fillna(0)
    buy_dates   = ohlcv.index[(pos_series > 0) & (prev_pos == 0)]
    sell_dates  = ohlcv.index[(pos_series == 0) & (prev_pos > 0)]

    chart_start_val = 100_000
    if eq_series is not None and eq_series.dropna().shape[0] > 0:
        chart_start_val = float(eq_series.dropna().iloc[0])
    gdx_bh = (ohlcv["Close"] / ohlcv["Close"].iloc[0]) * chart_start_val

    # ── Layout ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        6, 1, figsize=(22, 15),
        gridspec_kw={"height_ratios": [4, 2, 1.2, 1.2, 1.0, 1.0]},
        facecolor=DARK, sharex=True)
    fig.suptitle(
        f"AURUM·AI — GDX Signal History  "
        f"({start_date.strftime('%b %Y')} → {end_date.strftime('%b %Y')})",
        color=GOLD, fontsize=13, fontweight="bold", y=0.995)

    # ── Panel 1: Candlesticks ─────────────────────────────────────────────
    ax1 = axes[0]
    style_ax(ax1)

    # Position shading — GDX (green) and XLE rotation (orange)
    for i in range(n):
        p     = float(pos_series.iloc[i])
        xle_p = float(xle_pos_series.iloc[i]) if i < len(xle_pos_series) else 0.0
        if p > 0:
            alpha = 0.05 + p * 0.08
            ax1.axvspan(i - 0.5, i + 0.5, alpha=alpha, color=GREEN, zorder=0)
        if xle_p > 0:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="#f0a500",
                        zorder=0)  # orange for XLE days

    # Candles
    for i, row in enumerate(ohlcv.itertuples()):
        o, h, l, c = row.Open, row.High, row.Low, row.Close
        col = GREEN if c >= o else RED
        ax1.bar(i, abs(c - o), 0.6, bottom=min(o, c),
                color=col, alpha=0.9, zorder=2)
        ax1.plot([i, i], [l, h], color=col,
                 linewidth=0.8, alpha=0.8, zorder=2)

    # Buy markers ▲
    for d in buy_dates:
        if d in date_to_x:
            xi = date_to_x[d]
            ax1.annotate("▲", xy=(xi, ohlcv.loc[d, "Low"] * 0.983),
                         color=GREEN, fontsize=10, ha="center",
                         fontweight="bold", zorder=5)

    # Sell markers ▼
    for d in sell_dates:
        if d in date_to_x:
            xi = date_to_x[d]
            ax1.annotate("▼", xy=(xi, ohlcv.loc[d, "High"] * 1.017),
                         color=RED, fontsize=10, ha="center",
                         fontweight="bold", zorder=5)

    ax1.set_ylabel("GDX Price ($)", color=MUTED, fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax1.set_xlim(-1, n)

    legend_els = [
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=GREEN, markersize=9, label="Buy GDX"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor=RED, markersize=9, label="Sell GDX"),
        mpatches.Patch(facecolor=GREEN, alpha=0.25, label="GDX long"),
        mpatches.Patch(facecolor="#f0a500", alpha=0.25, label="XLE rotation"),
    ]
    ax1.legend(handles=legend_els, fontsize=8,
               facecolor=PANEL, labelcolor=MUTED, loc="upper left")

    # ── Panel 2: Portfolio (indexed to 100 at chart start) ───────────────
    ax2 = axes[1]
    style_ax(ax2)

    idx_base = start_val / 100.0  # divisor to convert $ → index

    if eq_series is not None:
        port_vals = [float(eq_series.iloc[i]) / idx_base
                     if i < len(eq_series) and pd.notna(eq_series.iloc[i])
                     else np.nan for i in range(n)]
        non_nan = [v for v in port_vals if not np.isnan(v)]
        ax2.plot(xs, port_vals, color=GOLD, linewidth=1.8,
                 label="GDX only", zorder=4)
        ax2.fill_between(xs, min(non_nan) * 0.998, port_vals,
                         alpha=0.15, color=GOLD)

    if rot_xle_series is not None:
        xle_v = [float(rot_xle_series.iloc[i]) / idx_base
                 if i < len(rot_xle_series) and pd.notna(rot_xle_series.iloc[i])
                 else np.nan for i in range(n)]
        ax2.plot(xs, xle_v, color="#f0a500", linewidth=1.4,
                 linestyle="--", label="GDX+XLE", zorder=5)

    if rot_qqq_series is not None:
        qqq_v = [float(rot_qqq_series.iloc[i]) / idx_base
                 if i < len(rot_qqq_series) and pd.notna(rot_qqq_series.iloc[i])
                 else np.nan for i in range(n)]
        ax2.plot(xs, qqq_v, color="#7b68ee", linewidth=1.4,
                 linestyle="--", label="GDX+XLE+QQQ", zorder=6)

    # TLT line removed — disabled strategy

    ax2.axhline(100, color=MUTED, linewidth=0.5, linestyle=":")
    ax2.set_ylabel(f"Indexed (={start_date.strftime('%b %Y')}→100)",
                   color=MUTED, fontsize=8)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, loc="upper left")
    ax2.set_xlim(-1, n)

    # ── Panel 3: Stacked allocation bars ─────────────────────────────────
    ax3 = axes[2]
    style_ax(ax3)

    gdx_arr  = np.array([float(pos_series.iloc[i]) * 100 for i in range(n)])
    xle_arr  = np.array([float(xle_pos_series.iloc[i]) * 100
                         if i < len(xle_pos_series) else 0.0 for i in range(n)])
    qqq_arr  = np.array([float(qqq_pos_series.iloc[i]) * 100
                         if i < len(qqq_pos_series) else 0.0 for i in range(n)])
    cash_arr = np.maximum(100 - gdx_arr - xle_arr - qqq_arr, 0)

    ax3.bar(xs, gdx_arr,  color=GREEN,     alpha=0.85, width=0.8, label="GDX")
    ax3.bar(xs, xle_arr,  color="#f0a500", alpha=0.85, width=0.8,
            bottom=gdx_arr, label="XLE")
    ax3.bar(xs, qqq_arr,  color="#7b68ee", alpha=0.85, width=0.8,
            bottom=gdx_arr + xle_arr, label="QQQ")
    ax3.bar(xs, cash_arr, color=MUTED,     alpha=0.20, width=0.8,
            bottom=gdx_arr + xle_arr + qqq_arr, label="Cash/SGOV")

    ax3.axhline(0, color=MUTED, linewidth=0.5)
    ax3.set_ylabel("Allocation", color=MUTED, fontsize=8)
    ax3.set_ylim(-5, 115)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED,
               loc="upper left", ncol=5)
    ax3.set_xlim(-1, n)

    # ── Panel 4: GDX signal + XLE signal overlay ─────────────────────────
    ax4 = axes[3]
    style_ax(ax4)

    sig_vals  = [float(pred_series.iloc[i]) * 100 for i in range(n)]
    bull_mask = np.array(sig_vals) > 0
    bear_mask = ~bull_mask
    ax4.bar(np.array(xs)[bull_mask], np.array(sig_vals)[bull_mask],
            color=GREEN, alpha=0.7, width=0.8, label="GDX")
    ax4.bar(np.array(xs)[bear_mask], np.array(sig_vals)[bear_mask],
            color=RED,   alpha=0.7, width=0.8)

    ax4.axhline(0, color=MUTED, linewidth=0.5)
    ax4.axhline( 0.5, color=GOLD, linewidth=0.6, linestyle=":")
    ax4.axhline(-0.5, color=GOLD, linewidth=0.6, linestyle=":")
    ax4.set_ylabel("GDX Signal %", color=MUTED, fontsize=8)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED, loc="upper left")
    ax4.set_xlim(-1, n)

    # ── Panel 5: XLE price ───────────────────────────────────────────────
    ax5 = axes[4]
    style_ax(ax5)

    xle_price_chart = None
    if prices is not None and "XLE" in prices.columns:
        xp = prices["XLE"].dropna().copy()
        xp.index = pd.to_datetime(xp.index).normalize()
        xle_price_chart = xp.reindex(ohlcv.index).ffill()

    if xle_price_chart is not None:
        xle_pv = [float(xle_price_chart.iloc[i])
                  if i < len(xle_price_chart) and pd.notna(xle_price_chart.iloc[i])
                  else np.nan for i in range(n)]
        ax5.plot(xs, xle_pv, color="#f0a500", linewidth=1.2, zorder=4)
        ax5.fill_between(xs, np.nanmin(xle_pv), xle_pv,
                         alpha=0.15, color="#f0a500")

        # XLE buy/sell markers — transitions in xle_pos_series
        xle_prev = xle_pos_series.shift(1).fillna(0)
        xle_buys  = ohlcv.index[(xle_pos_series > 0) & (xle_prev == 0)]
        xle_sells = ohlcv.index[(xle_pos_series == 0) & (xle_prev > 0)]
        for d in xle_buys:
            i = ohlcv.index.get_loc(d)
            if i < len(xle_pv) and not np.isnan(xle_pv[i]):
                ax5.plot(i, xle_pv[i] * 0.992, marker="^",
                         color="#f0a500", markersize=7,
                         markeredgecolor=DARK, markeredgewidth=0.5, zorder=5)
        for d in xle_sells:
            i = ohlcv.index.get_loc(d)
            if i < len(xle_pv) and not np.isnan(xle_pv[i]):
                ax5.plot(i, xle_pv[i] * 1.008, marker="v",
                         color=RED, markersize=7,
                         markeredgecolor=DARK, markeredgewidth=0.5, zorder=5)

        # XLE signal overlay — secondary y-axis
        if xle_sig_series is not None:
            ax5b = ax5.twinx()
            ax5b.set_facecolor("none")
            xle_sv = [float(xle_sig_series.iloc[i]) * 100
                      if i < len(xle_sig_series) and
                      pd.notna(xle_sig_series.iloc[i])
                      else np.nan for i in range(n)]
            bull_xle = np.array(xle_sv)
            bear_xle = np.array(xle_sv)
            bull_xle[bull_xle <= 0] = np.nan
            bear_xle[bear_xle > 0]  = np.nan
            ax5b.bar(xs, bull_xle, color="#f0a500", alpha=0.4,
                     width=0.8, zorder=1)
            ax5b.bar(xs, bear_xle, color=RED, alpha=0.4,
                     width=0.8, zorder=1)
            ax5b.axhline(0, color=MUTED, linewidth=0.4)
            ax5b.set_ylabel("Signal", color=MUTED, fontsize=7)
            ax5b.tick_params(colors=MUTED, labelsize=7)
            ax5b.spines[:].set_color("#30363d")
            # Keep signal bars small relative to price
            sv_max = max(abs(v) for v in xle_sv if not np.isnan(v)) \
                     if any(not np.isnan(v) for v in xle_sv) else 1
            ax5b.set_ylim(-sv_max * 0.8, sv_max * 0.8)

        # Shade XLE long days
        for i in range(n):
            xp2 = float(xle_pos_series.iloc[i]) \
                  if i < len(xle_pos_series) else 0.0
            if xp2 > 0:
                ax5.axvspan(i - 0.5, i + 0.5, alpha=0.25,
                            color="#f0a500", zorder=0)

    ax5.set_ylabel("XLE $", color=MUTED, fontsize=8)
    ax5.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax5.set_xlim(-1, n)

    # ── Panel 6: QQQ price + signal bars ─────────────────────────────────
    ax6 = axes[5]
    style_ax(ax6)

    # Build QQQ signal series
    qqq_sig_series = None
    if qqq_recent_preds is not None and len(qqq_recent_preds) > 0:
        qs_df = qqq_recent_preds["pred"].copy()
        qs_df.index = pd.to_datetime(qs_df.index).normalize()
        qs_df = qs_df[~qs_df.index.duplicated(keep="last")]
        qqq_sig_series = qs_df.reindex(ohlcv.index)

    qqq_price_chart = None
    if prices is not None and "QQQ" in prices.columns:
        qp = prices["QQQ"].dropna()
        qp.index = pd.to_datetime(qp.index).normalize().tz_localize(None)
        qqq_price_chart = qp.reindex(ohlcv.index).ffill()

    if qqq_price_chart is not None:
        qqq_pv = [float(qqq_price_chart.iloc[i])
                  if i < len(qqq_price_chart) and pd.notna(qqq_price_chart.iloc[i])
                  else np.nan for i in range(n)]
        ax6.plot(xs, qqq_pv, color="#7b68ee", linewidth=1.2, zorder=4)
        ax6.fill_between(xs, np.nanmin(qqq_pv), qqq_pv,
                         alpha=0.15, color="#7b68ee")

        # QQQ buy/sell markers
        qqq_prev = qqq_pos_series.shift(1).fillna(0)
        qqq_buys  = ohlcv.index[(qqq_pos_series > 0) & (qqq_prev == 0)]
        qqq_sells = ohlcv.index[(qqq_pos_series == 0) & (qqq_prev > 0)]
        for d in qqq_buys:
            i = ohlcv.index.get_loc(d)
            if i < len(qqq_pv) and not np.isnan(qqq_pv[i]):
                ax6.plot(i, qqq_pv[i] * 0.992, marker="^",
                         color="#7b68ee", markersize=7,
                         markeredgecolor=DARK, markeredgewidth=0.5, zorder=5)
        for d in qqq_sells:
            i = ohlcv.index.get_loc(d)
            if i < len(qqq_pv) and not np.isnan(qqq_pv[i]):
                ax6.plot(i, qqq_pv[i] * 1.008, marker="v",
                         color=RED, markersize=7,
                         markeredgecolor=DARK, markeredgewidth=0.5, zorder=5)

        # QQQ signal overlay
        if qqq_sig_series is not None:
            ax6b = ax6.twinx()
            ax6b.set_facecolor("none")
            qqq_sv = [float(qqq_sig_series.iloc[i]) * 100
                      if i < len(qqq_sig_series) and
                      pd.notna(qqq_sig_series.iloc[i])
                      else np.nan for i in range(n)]
            bull_qqq = np.array(qqq_sv)
            bear_qqq = np.array(qqq_sv)
            bull_qqq[bull_qqq <= 0] = np.nan
            bear_qqq[bear_qqq > 0]  = np.nan
            ax6b.bar(xs, bull_qqq, color="#7b68ee", alpha=0.4, width=0.8, zorder=1)
            ax6b.bar(xs, bear_qqq, color=RED,       alpha=0.4, width=0.8, zorder=1)
            ax6b.axhline(0, color=MUTED, linewidth=0.4)
            ax6b.set_ylabel("Signal", color=MUTED, fontsize=7)
            ax6b.tick_params(colors=MUTED, labelsize=7)
            ax6b.spines[:].set_color("#30363d")
            sv_max = max(abs(v) for v in qqq_sv if not np.isnan(v)) \
                     if any(not np.isnan(v) for v in qqq_sv) else 1
            ax6b.set_ylim(-sv_max * 0.8, sv_max * 0.8)

        # Shade QQQ long days
        for i in range(n):
            qp2 = float(qqq_pos_series.iloc[i]) \
                  if i < len(qqq_pos_series) else 0.0
            if qp2 > 0:
                ax6.axvspan(i - 0.5, i + 0.5, alpha=0.25,
                            color="#7b68ee", zorder=0)

    ax6.set_ylabel("QQQ $", color=MUTED, fontsize=8)
    ax6.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax6.set_xlim(-1, n)

    # ── X-axis month labels (on bottom panel) ────────────────────────────
    tick_pos, tick_lbl = [], []
    prev_month = None
    for i, d in enumerate(ohlcv.index):
        if d.month != prev_month:
            tick_pos.append(i)
            tick_lbl.append(d.strftime("%b '%y") if d.month in (1, 4, 7, 10)
                            or i == 0 else d.strftime("%b"))
            prev_month = d.month
    ax6.set_xticks(tick_pos)
    ax6.set_xticklabels(tick_lbl, rotation=35, ha="right",
                        color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    out_path = os.path.join(OUT_DIR, "signal_history_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"  ✓ Signal history chart → {out_path}")
    print(f"    {n} trading days  |  "
          f"{len(buy_dates)} buys  |  {len(sell_dates)} sells")
    print(f"  (Close the chart window to continue...)")
    plt.show()   # blocks until window is closed
    plt.close()



# ══════════════════════════════════════════════════════════════════════════════
# 11. NEWS AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

NEWS_FEEDS = [
    ("Gold/Miners",  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GDX,GLD,GOLD,NEM&region=US&lang=en-US"),
    ("Gold/Miners",  "https://www.mining.com/feed/"),
    ("Macro",        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ETNX,%5EVIX,UUP,TLT&region=US&lang=en-US"),
    ("Geopolitical", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Geopolitical", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
]

GOLD_KEYWORDS = [
    "gold", "silver", "miner", "gdx", "bullion", "precious metal",
    "fed", "federal reserve", "rate", "inflation", "dollar", "usd",
    "iran", "geopolit", "conflict", "war", "sanction",
    "china", "treasury", "yield", "vix", "risk",
]


def fetch_news(lookback_days=3):
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime
    except ImportError:
        return []

    cutoff   = datetime.now() - timedelta(days=lookback_days)
    articles = []

    for category, url in NEWS_FEEDS:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root  = ET.fromstring(xml_data)
            ns    = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)

            for item in items[:20]:
                title_el = item.find("title")
                title    = title_el.text if title_el is not None else ""
                if not title:
                    continue
                pub_el  = (item.find("pubDate") or
                           item.find("atom:published", ns))
                pub_str = pub_el.text if pub_el is not None else ""
                try:
                    pub_date = parsedate_to_datetime(pub_str).replace(tzinfo=None)
                except Exception:
                    try:
                        pub_date = pd.to_datetime(pub_str).to_pydatetime().replace(
                            tzinfo=None)
                    except Exception:
                        pub_date = datetime.now()

                if pd.isna(pub_date) or pub_date < cutoff:
                    continue

                link_el = item.find("link")
                link    = link_el.text if link_el is not None else ""
                title_lower = title.lower()
                if not any(kw in title_lower for kw in GOLD_KEYWORDS):
                    continue

                bullish_words = ["rise","rally","surge","gain","high",
                                 "bull","support","strong","jump","up"]
                bearish_words = ["fall","drop","slump","decline","low",
                                 "bear","weak","slide","down","sell"]
                b = sum(1 for w in bullish_words if w in title_lower)
                s = sum(1 for w in bearish_words if w in title_lower)
                sentiment = "▲" if b > s else "▼" if s > b else "–"

                articles.append({"category": category, "title": title.strip(),
                                  "date": pub_date, "sentiment": sentiment,
                                  "url": link})
        except Exception:
            continue

    articles.sort(key=lambda x: x["date"], reverse=True)
    seen, unique = set(), []
    for a in articles:
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique[:20]


def print_portfolio(port):
    """Print live portfolio tracker section."""
    if port is None:
        print(f"\n  ── Portfolio Tracker ────────────────────────────")
        print(f"  Run aurum_backtest.py first to generate daily_equity.csv")
        return

    print(f"\n  ── Portfolio Tracker  (paper trading) ───────────")
    print(f"  Start capital    : ${port['initial']:>12,.0f}  "
          f"({port['first_trade'].strftime('%Y-%m-%d')})")
    print(f"  Backtest to      : {port['last_eq_date'].strftime('%Y-%m-%d')}  "
          f"(${port['last_port_bs']:>10,.0f})")
    print(f"  Live value today : ${port['port_final']:>12,.0f}  "
          f"(marked to GDX ${port['gdx_last']:.2f})")
    print(f"  Total return     : {port['total_ret']:>+10.2f}%  "
          f"(over {port['cal_years']:.1f} years)")
    print(f"  Annual return    : {port['ann_ret']:>+10.2f}%  (CAGR)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  GDX last close   : ${port['gdx_last']:>8.2f}")
    print(f"  Current position : {port['current_pos']:.0%}  "
          f"({port['gdx_shares']:.0f} shares)")
    print(f"  Stock value      : ${port['stock_val']:>12,.0f}")
    print(f"  Cash value       : ${port['cash_val']:>12,.0f}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  vs GDX B&H       : ${port['bh_final']:>12,.0f}  "
          f"({port['bh_ret']:>+.1f}%)")
    alpha = port['total_ret'] - port['bh_ret']
    print(f"  Alpha vs B&H     : {alpha:>+10.2f}%")
    print(f"  Total trades     : {port['n_trades']:>10}")


def print_news(articles, lookback_days=3):
    print(f"\n  ── News  (last {lookback_days} days) ────────────────────────")
    if not articles:
        print(f"  No relevant articles found")
        return
    current_cat = None
    for a in articles:
        if a["category"] != current_cat:
            current_cat = a["category"]
            print(f"\n  [{current_cat}]")
        date_str = (a["date"].strftime("%a %d %b %H:%M")
                    if pd.notna(a["date"]) else "date unknown")
        title = a["title"]
        if len(title) > 72:
            title = title[:69] + "..."
        print(f"  {a['sentiment']}  {date_str}  {title}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. SIGNAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_signal_report(result, decision, as_of_date, log,
                        prices=None, articles=None, news_days=3,
                        fold_models=None, features_aligned=None,
                        port=None, recent_preds=None,
                        xle_recent=None, qqq_recent=None,
                        n_history=370, sim_df=None):
    bar_full      = int(result["agreement"] * 20)
    bar           = "█" * bar_full + "░" * (20 - bar_full)
    conf_bar_full = int(decision["conf_pct"] * 20)
    conf_bar      = "█" * conf_bar_full + "░" * (20 - conf_bar_full)

    print(f"\n{'═'*60}")
    print(f"  AURUM·AI — Live Signal   {as_of_date.strftime('%Y-%m-%d')}")
    print(f"{'═'*60}")

    # GDX market data
    if prices is not None and "GDX" in prices.columns:
        gdx    = prices["GDX"].dropna()
        close  = float(gdx.iloc[-1])
        prev   = float(gdx.iloc[-2]) if len(gdx) >= 2 else close
        chg    = close - prev
        chg_pct= (chg / prev) * 100
        high5  = float(gdx.iloc[-5:].max())  if len(gdx) >= 5  else close
        low5   = float(gdx.iloc[-5:].min())  if len(gdx) >= 5  else close
        high20 = float(gdx.iloc[-20:].max()) if len(gdx) >= 20 else close
        low20  = float(gdx.iloc[-20:].min()) if len(gdx) >= 20 else close
        arrow  = "▲" if chg >= 0 else "▼"
        col    = "+" if chg >= 0 else ""
        range20      = high20 - low20
        pos_in_range = (close - low20) / (range20 + 1e-9) * 100
        rbar = "█" * int(pos_in_range/5) + "░" * (20 - int(pos_in_range/5))
        print(f"\n  ── GDX Market Data  ({as_of_date.strftime('%Y-%m-%d')}) ───────────────")
        print(f"  Last close   : ${close:.2f}  {arrow} {col}{chg:.2f}  ({col}{chg_pct:.2f}%)")
        print(f"  5-day range  : ${low5:.2f} – ${high5:.2f}")
        print(f"  20-day range : ${low20:.2f} – ${high20:.2f}")
        print(f"  20d position : [{rbar}] {pos_in_range:.0f}%")

    # Portfolio tracker
    # Portfolio tracker replaced by full history simulation (run before this)

    # Ensemble signal
    direction_str = "▲  BULLISH" if result["direction"] > 0 else "▼  BEARISH"
    print(f"\n  ── Ensemble Signal ──────────────────────────────")
    print(f"  Direction    : {direction_str}")
    print(f"  Log return   : {result['ensemble']:+.4f}")
    print(f"  Pct return   : {result['pct_ret']:+.2f}%  (predicted over {SCFG['forward_days']} days)")
    print(f"  Confidence   : [{conf_bar}] {decision['conf_pct']:.0%}")
    print(f"  Fold agree   : [{bar}] {result['agreement']:.0%}  "
          f"({int(result['agreement']*result['n_folds'])} of {result['n_folds']} folds)")

    # Fold breakdown
    print(f"\n  ── Fold Breakdown ───────────────────────────────")
    for fp in result["fold_preds"]:
        arrow = "▲" if fp["pred"] > 0 else "▼"
        print(f"  F{fp['fold_num']}  {arrow} {fp['pred']:+.4f}  "
              f"(acc={fp['dir_acc']:.0%}  IC={fp['ic']:+.3f})")

    # Trading decision
    ensemble = result["ensemble"]
    conf     = decision["conf_pct"]
    position = decision["position"]
    agree    = result["agreement"]
    pred_pct = result["pct_ret"]
    min_sig  = SCFG["min_signal"]

    if abs(ensemble) < min_sig:
        dir_reason = f"predicted {pred_pct:+.2f}% below {min_sig*100:.1f}% threshold"
    elif ensemble > 0:
        dir_reason = (f"model predicts +{pred_pct:.2f}% over 20 days  "
                      f"({int(agree*result['n_folds'])} of {result['n_folds']} folds agree)")
    else:
        dir_reason = (f"model predicts {pred_pct:.2f}% over 20 days  "
                      f"({int(agree*result['n_folds'])} of {result['n_folds']} folds agree)")

    if position == 0:
        size_reason = "bearish or below threshold → hold cash"
    else:
        if conf >= 0.75:   tier = "top quartile"
        elif conf >= 0.50: tier = "3rd quartile"
        elif conf >= 0.25: tier = "2nd quartile"
        else:              tier = "bottom quartile"
        size_reason = f"{tier} confidence — stronger than {conf:.0%} of historical predictions"

    if agree >= 0.75 and conf >= 0.60:   quality = "★ HIGH CONVICTION"
    elif agree >= 0.60 and conf >= 0.40: quality = "✓ MODERATE"
    elif agree >= 0.50:                  quality = "~ LOW CONVICTION"
    else:                                quality = "⚠ MIXED — folds disagree"

    print(f"\n  ── Trading Decision ─────────────────────────────")
    print(f"  Action       : {decision['action']}")
    print(f"  Signal       : {dir_reason}")
    print(f"  Position     : {position:.0%} of capital  ({size_reason})")
    print(f"  Quality      : {quality}")

    # ── TLT rule-based signal ─────────────────────────────────────────────
    def get_tlt_signal(prices):
        """
        Rule-based TLT signal — no ML model needed.
        Bullish when TLT is above its 20-day MA and momentum is positive.
        Returns dict with direction, pct_ret, action.
        """
        if prices is None or "TLT" not in prices.columns:
            return None
        tlt = prices["TLT"].dropna()
        if len(tlt) < 25:
            return None
        close    = float(tlt.iloc[-1])
        ma20     = float(tlt.iloc[-20:].mean())
        ma5      = float(tlt.iloc[-5:].mean())
        ret5     = float((tlt.iloc[-1] / tlt.iloc[-6] - 1) * 100)
        ret21    = float((tlt.iloc[-1] / tlt.iloc[-22] - 1) * 100)
        above_ma = close > ma20
        mom_pos  = ma5 > ma20
        bullish  = above_ma and mom_pos
        return {
            "direction": 1 if bullish else -1,
            "pct_ret":   ret5,
            "ret21":     ret21,
            "close":     close,
            "ma20":      ma20,
            "action":    "→ deploy" if bullish else "→ skip (bearish)",
        }

    tlt_sig = get_tlt_signal(prices)

    # XLE and TLT current signals — shown right after GDX decision
    print(f"\n  ── Rotation Signals (today) ─────────────────────")
    def rot_quality(conf):
        """Simple conviction label based on confidence percentile alone."""
        if conf >= 0.70:   return "★ HIGH"
        elif conf >= 0.45: return "✓ MODERATE"
        elif conf >= 0.25: return "~ LOW"
        else:              return "⚠ WEAK"

    # XLE (ML model)
    if xle_recent is not None and len(xle_recent) > 0:
        latest = xle_recent.index[-1]
        rr     = xle_recent.loc[latest]
        rpred  = float(rr["pred"])
        rconf  = float(rr["conf_pct"])
        rpos   = float(rr["position"])
        rdir   = "▲ BULL" if rr["direction"] == 1 else \
                 "▼ BEAR" if rr["direction"] == -1 else "–  FLAT"
        action = f"→ deploy {rpos:.0%}" if rpos > 0 else "→ hold"
        print(f"  XLE   {rdir}  {rpred*100:>+.2f}%  conf={rconf:.0%}  "
              f"{rot_quality(rconf)}  {action}")
    else:
        print(f"  XLE   (model not available — run xle_trainer.py)")

    # QQQ (ML model)
    if qqq_recent is not None and len(qqq_recent) > 0:
        latest = qqq_recent.index[-1]
        qr     = qqq_recent.loc[latest]
        qpred  = float(qr["pred"])
        qconf  = float(qr["conf_pct"])
        qpos   = float(qr["position"])
        qdir   = "▲ BULL" if qr["direction"] == 1 else \
                 "▼ BEAR" if qr["direction"] == -1 else "–  FLAT"
        qaction = f"→ deploy {qpos:.0%}" if qpos > 0 else "→ hold"
        print(f"  QQQ   {qdir}  {qpred*100:>+.2f}%  conf={qconf:.0%}  "
              f"{rot_quality(qconf)}  {qaction}")
    else:
        print(f"  QQQ   (model not available — run qqq_trainer.py)")

    # TLT disabled — idle cash earns SGOV
    # (TLT was dragging performance from 2021 onwards — rate hiking cycle)

    if position > 0:
        next_day    = as_of_date + pd.tseries.offsets.BusinessDay(1)
        review_date = as_of_date + pd.tseries.offsets.BusinessDay(10)
        idle_pct    = 1 - position
        print(f"\n  ── Instruction ──────────────────────────────────")
        print(f"  Buy GDX at open  : {next_day.strftime('%Y-%m-%d')}")
        print(f"  Allocate         : {position:.0%} of portfolio to GDX")

        # Deploy idle cash via strength-ranked cascade (patched 2026-05-28):
        # Collect XLE/QQQ candidates that pass the deployment gate, then
        # deploy in confidence-percentile order — strongest signal gets first
        # dibs on remaining idle. Replaces the fixed XLE-first ordering.
        remaining = idle_pct
        rot_candidates = []
        if xle_recent is not None and len(xle_recent) > 0:
            xle_row  = xle_recent.iloc[-1]
            xle_pred = float(xle_row["pred"])
            xle_conf = float(xle_row["conf_pct"])
            xle_pos  = float(xle_row["position"])
            if xle_pred > 0 and xle_conf >= 0.40 and xle_pos > 0:
                rot_candidates.append(("XLE", xle_pos, xle_conf, xle_pred))
        if qqq_recent is not None and len(qqq_recent) > 0:
            qqq_row  = qqq_recent.iloc[-1]
            qqq_pred = float(qqq_row["pred"])
            qqq_conf = float(qqq_row["conf_pct"])
            qqq_pos  = float(qqq_row["position"])
            if qqq_pred > 0 and qqq_conf >= 0.40 and qqq_pos > 0:
                rot_candidates.append(("QQQ", qqq_pos, qqq_conf, qqq_pred))

        rot_candidates.sort(key=lambda c: c[2], reverse=True)
        for name, pos, conf, pred in rot_candidates:
            if remaining > 0.01:
                alloc      = min(pos, remaining)
                remaining -= alloc
                print(f"  Also buy {name}     : {alloc:.0%}  "
                      f"(bullish {pred*100:+.2f}%  conf={conf:.0%})")

        if remaining > 0.01 and tlt_sig is not None and \
                tlt_sig["direction"] > 0:
            # TLT disabled — idle cash goes to SGOV
            pass

        if remaining > 0.01:
            print(f"  Remainder        : {remaining:.0%} SGOV/cash")

        print(f"  Review signal    : {review_date.strftime('%Y-%m-%d')}")
        print(f"  Exit if          : signal flips bearish OR GDX falls 8% from entry")
    else:
        print(f"\n  ── Instruction ──────────────────────────────────")
        print(f"  Hold cash — do not buy GDX")
        if ensemble < 0:
            print(f"  Bearish signal — if currently long GDX, consider exiting")
        else:
            print(f"  Signal too weak — wait for stronger conviction")

        # Check rotation opportunities — XLE and QQQ
        print(f"\n  ── Rotation Check ───────────────────────────────")
        for rot_ticker, rot_preds in [("XLE", xle_recent)]:
            if rot_preds is not None and len(rot_preds) > 0:
                rr   = rot_preds.iloc[-1]
                rdir = "▲ BULL" if rr["direction"] == 1 else \
                       "▼ BEAR" if rr["direction"] == -1 else "–  FLAT"
                print(f"  {rot_ticker:<4}  {rdir}  {float(rr['pred'])*100:>+.2f}%  "
                      f"conf={float(rr['conf_pct']):.0%}")
            else:
                print(f"  {rot_ticker:<4}  (model not available — "
                      f"run {rot_ticker.lower()}_trainer.py)")

        # Check rotation opportunities — strength-ranked cascade
        # (patched 2026-05-28). Rank XLE/QQQ by confidence percentile; deploy
        # strongest first to remaining capital; weaker one gets what's left.
        xle_row = xle_recent.iloc[-1] if xle_recent is not None and \
                  len(xle_recent) > 0 else None
        qqq_row = qqq_recent.iloc[-1] if qqq_recent is not None and \
                  len(qqq_recent) > 0 else None

        rot_candidates = []
        if xle_row is not None and float(xle_row["pred"]) > 0 and \
                float(xle_row["conf_pct"]) >= 0.40:
            rot_candidates.append(
                ("XLE", float(xle_row["position"]),
                 float(xle_row["conf_pct"]), float(xle_row["pred"])))
        if qqq_row is not None and float(qqq_row["pred"]) > 0 and \
                float(qqq_row["conf_pct"]) >= 0.40:
            rot_candidates.append(
                ("QQQ", float(qqq_row["position"]),
                 float(qqq_row["conf_pct"]), float(qqq_row["pred"])))

        rot_candidates.sort(key=lambda c: c[2], reverse=True)

        remaining = 1.0
        rotated   = False
        for name, target, conf, pred in rot_candidates:
            if remaining > 0.01:
                alloc      = min(target, remaining)
                remaining -= alloc
                rotated    = True
                print(f"\n  Rotate to       : {name}  {alloc:.0%} of capital")

        if remaining > 0.01:
            print(f"  {'Also' if rotated else 'Rotate to':<14}: "
                  f"{remaining:.0%} SGOV/cash")

    vol      = decision.get("realised_vol", 0)
    vol_flag = "⚠ LOW VOL — reduce size by 50%" if decision.get("vol_warning") else "✓ NORMAL"
    print(f"  Realised vol : {vol:.1f}%  {vol_flag}")

    # Signal history — computed live by sliding window
    # Build GDX price series for history lookup
    gdx_prices = prices["GDX"].dropna() if prices is not None and \
                 "GDX" in prices.columns else None
    gdx_prices_idx = gdx_prices.copy() if gdx_prices is not None else None
    if gdx_prices_idx is not None:
        gdx_prices_idx.index = pd.to_datetime(
            gdx_prices_idx.index).normalize()

    N_HISTORY_TEXT = 50   # rows shown in text table (chart shows full n_history)
    print(f"\n  ── Signal History (last {N_HISTORY_TEXT} trading days, chart covers {n_history}d) ──────────")
    print(f"  {'Date':<12} {'GDX sig':<9} {'GDX$':>6} {'XLE':>5} {'QQQ':>5}"
          f"  {'GDX':>7} {'XLE':>7} {'QQQ':>7} {'Cash':>7} {'Total':>7}"
          f"  {'GDX 10d':>9} {'XLE 10d':>9}")
    print(f"  {'─'*12} {'─'*9} {'─'*6} {'─'*5} {'─'*5}"
          f"  {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}"
          f"  {'─'*9} {'─'*9}")

    equity_df = port["equity_df"] if port is not None and \
                "equity_df" in port else None

    # Fall back to loading from disk if not available
    if equity_df is None:
        equity_path = os.path.join("aurum_backtest", "daily_equity.csv")
        if os.path.exists(equity_path):
            try:
                equity_df = pd.read_csv(equity_path, parse_dates=["date"],
                                        index_col="date")
                equity_df.index = pd.to_datetime(
                    equity_df.index).normalize().tz_localize(None)
            except Exception:
                equity_df = None

    full_path = os.path.join(OUT_DIR, "full_predictions.csv")
    actuals_s = None
    if os.path.exists(full_path):
        try:
            fp = pd.read_csv(full_path, parse_dates=["date"], index_col="date")
            fp.index = pd.to_datetime(fp.index).normalize().tz_localize(None)
            actuals_s = fp["actual"]
        except Exception:
            pass

    if recent_preds is None and fold_models is not None and \
            features_aligned is not None:
        recent_preds = generate_recent_predictions(
            fold_models, features_aligned, SCFG, n_days=n_history)

    # ── Load rotation histories from saved full_predictions ───────────────
    def load_rotation_history(ticker):
        """Load daily predictions from ticker's full_predictions.csv."""
        path = os.path.join(f"{ticker.lower()}_output", "full_predictions.csv")
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df["pred"] if "pred" in df.columns else None
        except Exception:
            return None

    xle_preds = load_rotation_history("XLE")
    qqq_preds = load_rotation_history("QQQ")

    # rot_portfolio built inside the port_series loop below
    rot_portfolio = {}
    port_series   = {}   # initialised here, populated in loop

    # Pre-build XLE and TLT 10-day forward actual return series
    xle_actuals_s = None
    tlt_actuals_s = None
    fwd = SCFG.get("forward_days", 20)
    if prices is not None:
        for ticker_act, attr_name in [("XLE", "xle_actuals_s"),
                                       ("TLT", "tlt_actuals_s")]:
            if ticker_act in prices.columns:
                p = prices[ticker_act].dropna().copy()
                p.index = pd.to_datetime(p.index).normalize().tz_localize(None)
                p = p[~p.index.duplicated(keep="last")].sort_index()
                fwd_ret = p.shift(-fwd) / p - 1
                if ticker_act == "XLE":
                    xle_actuals_s = fwd_ret
                else:
                    tlt_actuals_s = fwd_ret

    if len(recent_preds) > 0:
        preds_sorted = recent_preds.sort_index()   # oldest first
        start_date   = preds_sorted.index[0]

        # Find starting portfolio value from equity file
        start_port = INITIAL_CAPITAL
        if equity_df is not None and len(equity_df) > 0:
            avail = equity_df[equity_df.index <= start_date]
            if len(avail) > 0:
                start_port = float(avail.iloc[-1]["portfolio_$"])

        # Get GDX prices for the prediction period
        gdx_hist = None
        if gdx_prices_idx is not None:
            gdx_hist = gdx_prices_idx[gdx_prices_idx.index >= start_date - pd.DateOffset(days=5)]

        # Build portfolio value series day by day
        # Also build rotation portfolio alongside — uses same GDX base,
        # deploys idle cash into XLE when XLE is bullish conf>=40%
        SGOV_DAILY   = CASH_DAILY
        MIN_ROT_CONF = 0.40

        # Pre-compute XLE confidence per date from oof distribution
        xle_oof_path = os.path.join("xle_output", "oof_predictions.csv")
        xle_oof_preds = None
        try:
            xle_oof = pd.read_csv(xle_oof_path)
            if "pred" in xle_oof.columns:
                xle_oof_preds = xle_oof["pred"].abs().values
        except Exception:
            pass

        # Get XLE prices for rotation returns
        xle_price_s = None
        if prices is not None and "XLE" in prices.columns:
            xle_price_s = prices["XLE"].dropna().copy()
            xle_price_s.index = pd.to_datetime(
                xle_price_s.index).normalize()

        # Try to load rotation equity from rotation_backtest
        rot_equity_df = None
        rot_equity_path = os.path.join("rotation_backtest", "daily_equity.csv")
        if os.path.exists(rot_equity_path):
            try:
                rot_equity_df = pd.read_csv(
                    rot_equity_path, parse_dates=["date"], index_col="date")
                rot_equity_df.index = pd.to_datetime(
                    rot_equity_df.index).normalize().tz_localize(None)
            except Exception:
                rot_equity_df = None

        port_series       = {}
        rot_portfolio_new = {}
        running_port      = start_port
        running_rot       = start_port
        prev_gdx          = None
        prev_xle          = None

        if gdx_hist is not None:
            avail = gdx_hist[gdx_hist.index <= start_date]
            if len(avail) > 0:
                prev_gdx = float(avail.iloc[-1])
        if xle_price_s is not None:
            avail = xle_price_s[xle_price_s.index <= start_date]
            if len(avail) > 0:
                prev_xle = float(avail.iloc[-1])

        for date_idx in preds_sorted.index:
            pos = float(preds_sorted.loc[date_idx, "position"])

            # GDX price
            if gdx_hist is not None:
                avail = gdx_hist[gdx_hist.index <= date_idx]
                curr_gdx = float(avail.iloc[-1]) if len(avail) > 0 else prev_gdx
            else:
                curr_gdx = prev_gdx

            # XLE price
            curr_xle = None
            if xle_price_s is not None:
                avail = xle_price_s[xle_price_s.index <= date_idx]
                curr_xle = float(avail.iloc[-1]) if len(avail) > 0 else prev_xle

            # ── Portfolio value — read from equity_df if available ────────
            # Use the backtest equity curve directly rather than recomputing
            if equity_df is not None and date_idx in equity_df.index:
                running_port = float(equity_df.loc[date_idx, "portfolio_$"])
            elif equity_df is not None and len(equity_df) > 0:
                avail_eq = equity_df[equity_df.index <= date_idx]
                if len(avail_eq) > 0:
                    running_port = float(avail_eq.iloc[-1]["portfolio_$"])
            elif prev_gdx and curr_gdx and prev_gdx > 0 and pos > 0:
                running_port = running_port * (1 + (curr_gdx/prev_gdx - 1) * pos)

            port_series[date_idx] = {
                "portfolio_$": round(running_port, 2),
                "stock_$":     round(running_port * pos, 2),
                "cash_$":      round(running_port * (1 - pos), 2),
            }

            # ── Rotation portfolio ─────────────────────────────────────
            idle = 1.0 - pos

            # XLE position for idle cash — use live sliding window (xle_recent)
            xle_pos = 0.0
            if idle > 0.01 and xle_oof_preds is not None:
                # Try xle_recent first (live, has all dates)
                xle_src = xle_recent if xle_recent is not None and \
                          len(xle_recent) > 0 else None
                xv = None
                if xle_src is not None:
                    if date_idx in xle_src.index:
                        xv = float(xle_src.loc[date_idx, "pred"])
                    else:
                        diffs = abs(xle_src.index - date_idx)
                        ni    = diffs.argmin()
                        if diffs[ni].days <= 3:
                            xv = float(xle_src.iloc[ni]["pred"])
                # Fallback to file-based xle_preds
                if xv is None and xle_preds is not None:
                    avail_xp = xle_preds[xle_preds.index <= date_idx]
                    if len(avail_xp) > 0 and \
                            (date_idx - avail_xp.index[-1]).days <= 5:
                        xv = float(avail_xp.iloc[-1])
                if xv is not None and xv > 0.002:
                    conf = float((xle_oof_preds < abs(xv)).mean())
                    if conf >= MIN_ROT_CONF:
                        sz = 0.0
                        for (lo, hi, s) in SCFG["conf_tiers"]:
                            if lo <= conf < hi:
                                sz = s
                                break
                        xle_pos = min(sz, idle)

            cash_pos  = idle - xle_pos

            # ── Rotation portfolio — read from saved file if available ──
            if rot_equity_df is not None:
                avail_rot = rot_equity_df[rot_equity_df.index <= date_idx]
                if len(avail_rot) > 0:
                    running_rot  = float(avail_rot.iloc[-1]["portfolio_$"])
                    xle_pos_disp = float(avail_rot.iloc[-1]["xle_pos"]) \
                                   if "xle_pos" in rot_equity_df.columns \
                                   else xle_pos
                    tlt_pos_disp = float(avail_rot.iloc[-1]["tlt_pos"]) \
                                   if "tlt_pos" in rot_equity_df.columns \
                                   else 0.0
                    cash_pos_disp = float(avail_rot.iloc[-1]["cash_pos"]) \
                                    if "cash_pos" in rot_equity_df.columns \
                                    else cash_pos
                else:
                    xle_pos_disp  = xle_pos
                    tlt_pos_disp  = 0.0
                    cash_pos_disp = cash_pos
            else:
                gdx_ret   = (curr_gdx/prev_gdx - 1) * pos \
                            if prev_gdx and curr_gdx and prev_gdx > 0 and pos > 0 \
                            else 0.0
                xle_ret   = (curr_xle/prev_xle - 1) * xle_pos \
                            if prev_xle and curr_xle and prev_xle > 0 and xle_pos > 0 \
                            else 0.0
                cash_ret  = cash_pos * SGOV_DAILY
                running_rot = running_rot * (1 + gdx_ret + xle_ret) + \
                              running_rot * cash_ret
                xle_pos_disp  = xle_pos
                tlt_pos_disp  = 0.0
                cash_pos_disp = cash_pos

            rot_portfolio_new[date_idx] = {
                "portfolio_$": round(running_rot, 2),
                "gdx_$":       round(running_rot * pos, 2),
                "xle_$":       round(running_rot * xle_pos_disp, 2),
                "tlt_$":       round(running_rot * tlt_pos_disp, 2),
                "cash_$":      round(running_rot * cash_pos_disp, 2),
                "gdx_pos":     pos,
                "xle_pos":     xle_pos_disp,
            }

            if curr_gdx:
                prev_gdx = curr_gdx
            if curr_xle:
                prev_xle = curr_xle

        # ── Also build QQQ rotation portfolio ────────────────────────────
        SGOV_DAILY   = CASH_DAILY
        MIN_ROT_CONF = 0.40

        # Build both rotation portfolios in a single pass: XLE and TLT
        rot_gdx_xle   = {}
        rot_gdx_tlt   = {}
        run_xle       = start_port
        run_tlt       = start_port

        # Reset prev prices to start-of-window values
        prev_gdx2  = None
        prev_xle2  = None
        prev_tlt_p = None

        if gdx_hist is not None:
            avail = gdx_hist[gdx_hist.index <= start_date]
            if len(avail) > 0:
                prev_gdx2 = float(avail.iloc[-1])
        if xle_price_s is not None:
            avail = xle_price_s[xle_price_s.index <= start_date]
            if len(avail) > 0:
                prev_xle2 = float(avail.iloc[-1])

        tlt_price_s = None
        if prices is not None and "TLT" in prices.columns:
            tlt_price_s = prices["TLT"].dropna().copy()
            tlt_price_s.index = pd.to_datetime(
                tlt_price_s.index).normalize()
            avail = tlt_price_s[tlt_price_s.index <= start_date]
            if len(avail) > 0:
                prev_tlt_p = float(avail.iloc[-1])

        for date_idx in preds_sorted.index:
            pos   = float(preds_sorted.loc[date_idx, "position"])
            idle  = 1.0 - pos

            # GDX price and return
            avail_g  = gdx_hist[gdx_hist.index <= date_idx] \
                       if gdx_hist is not None else pd.Series()
            cg       = float(avail_g.iloc[-1]) if len(avail_g) > 0 \
                       else prev_gdx2
            gdx_ret  = (cg / prev_gdx2 - 1) * pos \
                       if prev_gdx2 and cg and prev_gdx2 > 0 and pos > 0 \
                       else 0.0

            # XLE position and return
            xle_pos2 = rot_portfolio_new.get(date_idx, {}).get("xle_pos", 0.0) \
                       if rot_portfolio_new else 0.0
            avail_xl = xle_price_s[xle_price_s.index <= date_idx] \
                       if xle_price_s is not None else pd.Series()
            cx       = float(avail_xl.iloc[-1]) if len(avail_xl) > 0 \
                       else prev_xle2
            xle_ret2 = (cx / prev_xle2 - 1) * xle_pos2 \
                       if prev_xle2 and cx and prev_xle2 > 0 and xle_pos2 > 0 \
                       else 0.0
            cash_xle = (idle - xle_pos2) * SGOV_DAILY
            run_xle  = run_xle * (1 + gdx_ret + xle_ret2) + run_xle * cash_xle
            rot_gdx_xle[date_idx] = round(run_xle, 2)

            # TLT position — rule based, only gets remaining idle after XLE
            avail_tl = tlt_price_s[tlt_price_s.index <= date_idx] \
                       if tlt_price_s is not None else pd.Series()
            ct       = float(avail_tl.iloc[-1]) if len(avail_tl) > 0 \
                       else prev_tlt_p
            tlt_pos  = 0.0
            idle_after_xle = idle - xle_pos2
            if idle_after_xle > 0.01 and ct and tlt_price_s is not None:
                window = tlt_price_s[tlt_price_s.index <= date_idx].iloc[-20:]
                if len(window) >= 20 and ct > float(window.mean()):
                    tlt_pos = idle_after_xle

            tlt_ret  = (ct / prev_tlt_p - 1) * tlt_pos \
                       if prev_tlt_p and ct and prev_tlt_p > 0 and tlt_pos > 0 \
                       else 0.0
            cash_tlt = (idle - xle_pos2 - tlt_pos) * SGOV_DAILY
            run_tlt  = run_tlt * (1 + gdx_ret + xle_ret2 + tlt_ret) + \
                       run_tlt * cash_tlt
            rot_gdx_tlt[date_idx] = round(run_tlt, 2)

            if cg:         prev_gdx2  = cg
            if cx:         prev_xle2  = cx
            if ct:         prev_tlt_p = ct

        rot_portfolio = rot_portfolio_new

        # Store for chart access
        if port is not None:
            port["port_series"]   = port_series
            port["rot_portfolio"] = rot_portfolio
            port["rot_gdx_xle"]   = rot_gdx_xle
            port["rot_gdx_tlt"]   = rot_gdx_tlt

        # Only print last N_HISTORY_TEXT days in table — newest dates
        # recent_preds is newest-first, so sort ascending then take tail
        table_preds = recent_preds.sort_index().iloc[-N_HISTORY_TEXT:].iloc[::-1]

        for date_idx, hrow in table_preds.iterrows():
            pred_val = float(hrow["pred"])
            conf_pct = float(hrow["conf_pct"])
            pos      = float(hrow["position"])
            dirn     = ("▲ BULL" if hrow["direction"] == 1 else
                        "▼ BEAR" if hrow["direction"] == -1 else "– FLAT")

            # Use combined rotation portfolio values
            if date_idx in rot_portfolio:
                rp        = rot_portfolio[date_idx]
                pv        = rp["portfolio_$"]
                cv        = rp["cash_$"]
                port_str  = f"${pv:>10,.0f}"
                cash_str  = f"${cv:>8,.0f}"
            elif date_idx in port_series:
                pv        = port_series[date_idx]["portfolio_$"]
                cv        = port_series[date_idx]["cash_$"]
                port_str  = f"${pv:>10,.0f}"
                cash_str  = f"${cv:>8,.0f}"
            else:
                port_str = cash_str = "        n/a"

            if actuals_s is not None and date_idx in actuals_s.index:
                actual = actuals_s.loc[date_idx]
                if pd.notna(actual):
                    correct = ((pred_val > 0 and actual > 0) or
                               (pred_val < 0 and actual < 0))
                    outcome = ("✓ " if correct else "✗ ") + f"{actual*100:+.1f}%"
                else:
                    outcome = "… pending"
            else:
                outcome = "… pending"

            # XLE actual 10d return helper — defined here, used after signals
            def act_str(act_series, sig_bullish):
                if act_series is None:
                    return "… pend"
                avail = act_series[act_series.index <= date_idx]
                if len(avail) == 0:
                    return "… pend"
                closest = avail.index[-1]
                if (date_idx - closest).days > 3:
                    return "… pend"
                v = float(avail.iloc[-1])
                if np.isnan(v):
                    return "… pend"
                correct = (sig_bullish and v > 0) or (not sig_bullish and v < 0)
                return ("✓" if correct else "✗") + f"{v*100:+.1f}%"

            # GDX price — fixed 6 chars: $85.02 or $100.3
            if gdx_prices_idx is not None:
                avail_gdx = gdx_prices_idx[gdx_prices_idx.index <= date_idx]
                if len(avail_gdx) > 0:
                    gp = float(avail_gdx.iloc[-1])
                    gdx_str = f"${gp:>5.1f}" if gp >= 100 else f"${gp:>5.2f}"
                else:
                    gdx_str = "  n/a"
            else:
                gdx_str = "  n/a"

            # Rotation signals — use live sliding-window if available
            def live_rot_str(live_preds, fallback_preds, min_sig=0.002):
                for p in [live_preds, fallback_preds]:
                    if p is None or len(p) == 0:
                        continue
                    if isinstance(p, pd.DataFrame):
                        # Exact match first
                        if date_idx in p.index:
                            v = float(p.loc[date_idx, "pred"])
                        else:
                            # Nearest within ±3 days
                            diffs = abs(p.index - date_idx)
                            nearest_i = diffs.argmin()
                            if diffs[nearest_i].days <= 3:
                                v = float(p.iloc[nearest_i]["pred"])
                            else:
                                continue
                    else:  # Series
                        avail = p[p.index <= date_idx]
                        if len(avail) == 0 or \
                                (date_idx - avail.index[-1]).days > 5:
                            continue
                        v = float(avail.iloc[-1])
                    if abs(v) < min_sig:
                        return "– flt "
                    return "▲ bul " if v > 0 else "▼ bea "
                return "  n/a "

            xle_dir = live_rot_str(xle_recent, xle_preds).strip()
            qqq_dir = live_rot_str(qqq_recent, qqq_preds).strip() \
                      if qqq_recent is not None else "n/a"

            # TLT disabled — all idle cash goes to SGOV
            xle_act = act_str(xle_actuals_s, xle_dir == "▲ bul")

            # GDX signal compact
            gdx_sig = f"{dirn[:6]} {conf_pct*100:>2.0f}%"

            # Get positions and portfolio value from sim_df (single source)
            gdx_pos2  = pos
            xle_pos2  = 0.0
            qqq_pos2  = 0.0
            total_pv  = None
            rot_pv    = None

            if sim_df is not None:
                dk = pd.Timestamp(date_idx).date()
                sim_dates = [pd.Timestamp(d).date() for d in sim_df.index]
                prior = [(d, v) for d, v in
                         zip(sim_dates, sim_df["portfolio_$"])
                         if d <= dk]
                if prior:
                    total_pv = prior[-1][1]
                if "xle_pos" in sim_df.columns:
                    prior_xle = [(d, v) for d, v in
                                 zip(sim_dates, sim_df["xle_pos"])
                                 if d <= dk]
                    if prior_xle:
                        xle_pos2 = prior_xle[-1][1]
                if "qqq_pos" in sim_df.columns:
                    prior_qqq = [(d, v) for d, v in
                                 zip(sim_dates, sim_df["qqq_pos"])
                                 if d <= dk]
                    if prior_qqq:
                        qqq_pos2 = prior_qqq[-1][1]
                if "rotation_$" in sim_df.columns:
                    prior_rot = [(d, v) for d, v in
                                 zip(sim_dates, sim_df["rotation_$"])
                                 if d <= dk]
                    if prior_rot:
                        rot_pv = prior_rot[-1][1]

            cash_pos2 = max(0.0, 1.0 - gdx_pos2 - xle_pos2 - qqq_pos2)

            def fmt_k(v):
                if v is None:
                    return "   n/a"
                k = v / 1000
                if k >= 1000:
                    return f"${k/1000:>4.1f}M"
                return f"${k:>5.0f}k"

            if total_pv:
                gdx_dv  = fmt_k(total_pv * gdx_pos2)
                xle_dv  = fmt_k(total_pv * xle_pos2)
                qqq_dv  = fmt_k(total_pv * qqq_pos2)
                cash_dv = fmt_k(total_pv * cash_pos2)
                tot_dv  = fmt_k(rot_pv or total_pv)
            else:
                gdx_dv = xle_dv = qqq_dv = cash_dv = tot_dv = "   n/a"

            print(f"  {date_idx.strftime('%Y-%m-%d'):<12} {gdx_sig:<9}"
                  f" {gdx_str:>6} {xle_dir:>5} {qqq_dir:>5}"
                  f"  {gdx_dv:>7} {xle_dv:>7} {qqq_dv:>7} {cash_dv:>7} {tot_dv:>7}"
                  f"  {outcome:>9} {xle_act:>9}")
    else:
        print(f"  (Insufficient feature data for history)")

    if articles is not None:
        print_news(articles, lookback_days=news_days)

    print(f"\n{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_rotation_models_and_features(ticker, prices):
    """
    Load fold models and build aligned features for a rotation instrument.
    Returns (fold_models, features_aligned, model_cfg) or (None, None, None).
    Used for sliding-window signal history generation.
    """
    out_dir  = f"{ticker.lower()}_output"
    pt_paths = sorted(glob.glob(
        os.path.join(os.path.abspath(out_dir), "fold_*_model.pt")))
    if not pt_paths:
        print(f"    {ticker}: no .pt files in {os.path.abspath(out_dir)}")
        return None, None, None

    try:
        rot_models     = []
        feature_names  = None
        # Ticker-specific lookback fallbacks for models trained before cfg was saved
        lookback_defaults = {"XLE": 40, "QQQ": 20, "SMH": 20}
        model_lookback = lookback_defaults.get(ticker, SCFG["lookback"])
        model_cfg      = dict(SCFG)

        for path in pt_paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            da   = ckpt["metrics"]["dir_acc"]
            ic   = ckpt["metrics"]["ic"]
            if da < 0.40 or ic < -0.15:
                continue
            mcfg  = ckpt["model_config"]
            model = GoldMinerLSTM(**mcfg)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            rot_models.append({"model": model,
                                "fold_num": int(os.path.basename(
                                    path).split("_")[1]),
                                "dir_acc": da, "ic": ic})
            if feature_names is None:
                feature_names = ckpt.get("feature_names")
            # Override with saved cfg if available
            if "cfg" in ckpt:
                model_lookback = int(ckpt["cfg"].get(
                    "lookback", model_lookback))

        if not rot_models or feature_names is None:
            return None, None, None

        model_cfg["lookback"] = model_lookback

        # Download extra tickers needed
        rot_tickers = ["QQQ","SPY","SMH","XLK","IWM","HYG","TLT",
                       "CL=F","BZ=F","NG=F","HO=F",
                       "XLE","XOP","OIH",
                       "UUP","^VIX","^VIX3M","^TNX","^MOVE",
                       "USDJPY=X","USDCHF=X"]
        end   = datetime.today()
        start = end - timedelta(days=max(600, model_lookback * 12))
        # Start with existing prices — strip timezone first
        frames = {}
        if prices is not None:
            for col in prices.columns:
                s = prices[col].dropna().copy()
                s.index = pd.to_datetime(s.index).normalize().tz_localize(None)
                frames[col] = s
        for t in rot_tickers:
            if t in frames:
                continue
            try:
                hist = yf.Ticker(t).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True)
                if len(hist) >= 50:
                    s = hist["Close"].rename(t)
                    s.index = pd.to_datetime(s.index).normalize().tz_localize(None)
                    frames[t] = s[s > 0].dropna()
            except Exception:
                continue

        if not frames:
            return None, None, None

        prices_rot = pd.concat(frames.values(), axis=1)
        prices_rot.index = pd.to_datetime(
            prices_rot.index).normalize().tz_localize(None)
        prices_rot = prices_rot[~prices_rot.index.duplicated(
            keep="last")].sort_index().ffill()
        cot_crude = None
        gpr_rot   = None
        trends_rot = None
        if ticker == "XLE":
            s = start.strftime("%Y-%m-%d")
            e = end.strftime("%Y-%m-%d")
            try:
                cot_crude = download_cot_crude_recent(s, e)
            except Exception:
                pass
            try:
                gpr_rot = download_gpr_recent(s, e)
            except Exception:
                pass
            try:
                trends_rot = download_google_trends_recent(
                    s, e,
                    keywords=["oil price", "energy stocks", "crude oil"],
                    tag="xle_rotation")
            except Exception:
                pass
        elif ticker == "QQQ":
            s = start.strftime("%Y-%m-%d")
            e = end.strftime("%Y-%m-%d")
            try:
                gpr_rot = download_gpr_recent(s, e)
            except Exception:
                pass
            try:
                trends_rot = download_google_trends_recent(
                    s, e,
                    keywords=["nvidia stock", "interest rates",
                              "recession", "AI stocks"],
                    tag="qqq_rotation")
            except Exception:
                pass

        features_raw = build_features_live(prices_rot, model_cfg,
                                           cot_crude=cot_crude,
                                           gpr=gpr_rot,
                                           trends=trends_rot)
        aligned      = align_features(features_raw, feature_names)
        aligned      = aligned.ffill().bfill().fillna(0)

        return rot_models, aligned, model_cfg

    except Exception as e:
        print(f"    {ticker}: failed to load — {e}")
        return None, None, None


def load_rotation_signal(ticker="QQQ"):
    """
    Load the latest ensemble signal for a rotation instrument.
    Supports QQQ (qqq_output) and SMH (smh_output).
    Returns a dict with direction, pred, conf_pct — or None if unavailable.
    """
    out_dir  = f"{ticker.lower()}_output"
    oof_path = os.path.join(out_dir, "oof_predictions.csv")
    pt_paths = sorted(glob.glob(
        os.path.join(os.path.abspath(out_dir), "fold_*_model.pt")))

    if not pt_paths or not os.path.exists(oof_path):
        return None

    try:
        rot_models    = []
        feature_names = None
        model_lookback = SCFG["lookback"]  # default, overridden from .pt

        for path in pt_paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            da   = ckpt["metrics"]["dir_acc"]
            ic   = ckpt["metrics"]["ic"]
            if da < 0.40 or ic < -0.15:
                continue
            mcfg  = ckpt["model_config"]
            model = GoldMinerLSTM(**mcfg)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            rot_models.append({"model": model, "fold_num": int(
                os.path.basename(path).split("_")[1]),
                "dir_acc": da, "ic": ic})
            if feature_names is None:
                feature_names = ckpt.get("feature_names")
            # Read lookback from model config — critical for XLE (40) vs GDX (20)
            if "cfg" in ckpt:
                model_lookback = int(ckpt["cfg"].get(
                    "lookback", model_lookback))
            elif "lookback" in mcfg:
                model_lookback = int(mcfg["lookback"])

        if not rot_models or feature_names is None:
            return None

        # Download prices — extend window for longer lookback models
        download_days = max(600, model_lookback * 10)
        end           = datetime.today()
        start         = end - timedelta(days=download_days)

        rot_tickers = ["QQQ","SPY","SMH","XLK","IWM","HYG","TLT",
                       "CL=F","BZ=F","NG=F","HO=F",
                       "XLE","XOP","OIH",
                       "UUP","^VIX","^VIX3M","^TNX","^MOVE",
                       "USDJPY=X","USDCHF=X"]
        frames = {}
        for t in rot_tickers:
            try:
                hist = yf.Ticker(t).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True)
                if len(hist) < 50:
                    continue
                s = hist["Close"].rename(t)
                frames[t] = s[s > 0].dropna()
            except Exception:
                continue

        if not frames:
            return None

        prices_rot = pd.concat(frames.values(), axis=1)
        prices_rot.index = pd.to_datetime(
            prices_rot.index).normalize().tz_localize(None)
        prices_rot = prices_rot[~prices_rot.index.duplicated(
            keep="last")].sort_index().ffill()

        # Download crude COT for XLE — key energy-specific signal
        cot_crude = None
        if ticker == "XLE":
            try:
                start_str = start.strftime("%Y-%m-%d")
                end_str   = end.strftime("%Y-%m-%d")
                cot_crude = download_cot_crude_recent(start_str, end_str)
            except Exception:
                pass

        scfg_rot = dict(SCFG)
        features_raw = build_features_live(prices_rot, scfg_rot,
                                           cot_crude=cot_crude)
        aligned      = align_features(features_raw, feature_names)
        aligned      = aligned.ffill().bfill().fillna(0)

        # Use model's own lookback — NOT SCFG lookback
        if len(aligned) < model_lookback:
            return None

        # Build a temporary cfg with correct lookback for run_ensemble
        rot_cfg = dict(SCFG)
        rot_cfg["lookback"] = model_lookback

        result = run_ensemble(rot_models, aligned, rot_cfg)
        if result is None:
            return None

        oof     = pd.read_csv(oof_path)
        conf_pct= float((oof["pred"].abs() < abs(
            result["ensemble"])).mean()) if "pred" in oof.columns else 0.5

        # Get current price
        avail = prices_rot.get(ticker, prices_rot.get("QQQ"))
        last_price = float(avail.iloc[-1]) if avail is not None else 0.0

        return {
            "ticker":     ticker,
            "ensemble":   result["ensemble"],
            "pct_ret":    result["pct_ret"],
            "direction":  result["direction"],
            "agreement":  result["agreement"],
            "n_folds":    result["n_folds"],
            "conf_pct":   conf_pct,
            "last_price": last_price,
        }
    except Exception:
        return None


def run_full_history_simulation(fold_models, feature_names, cfg,
                                prices_recent,
                                xle_models=None, xle_feature_names=None,
                                xle_cfg=None,
                                qqq_models=None, qqq_feature_names=None,
                                qqq_cfg=None):
    """
    Download full history from 2010, run GDX + XLE inference on every day,
    simulate $100k portfolio with rotation. Returns (sim_df, yearly_rows).
    Uses only .pt fold models — no external equity files.
    """
    import yfinance as yf

    INITIAL      = 100_000.0
    XLE_MIN_CONF = 0.40
    XLE_MIN_SIG  = 0.002

    print(f"\n{'═'*60}")
    print(f"  AURUM·AI — Full History Simulation  (2010 → present)")
    print(f"{'═'*60}")

    # ── 1. Download full price history ───────────────────────────────────
    print(f"\n  Downloading full price history (2010-01-01 → today)...")
    end   = datetime.today()
    start = datetime(2010, 1, 1)
    frames = {}
    for t in TICKERS:
        try:
            df = yf.Ticker(t).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = next((c for c in df.columns
                              if c.lower() == "close"), None)
            if close_col and len(df) > 50:
                s = df[close_col].rename(t)
                frames[t] = s[s > 0].dropna()
        except Exception:
            continue
    prices_full = pd.concat(frames.values(), axis=1)
    prices_full = prices_full.sort_index().ffill().dropna(how="all")
    prices_full.index = pd.to_datetime(
        prices_full.index).normalize().tz_localize(None)
    prices_full = prices_full[~prices_full.index.duplicated(keep="last")]
    print(f"  ✓ Prices: {prices_full.shape[0]} rows × {prices_full.shape[1]} tickers  "
          f"({prices_full.index[0].date()} → {prices_full.index[-1].date()})")

    # ── 2. Download auxiliary data ────────────────────────────────────────
    start_str = prices_full.index[0].strftime("%Y-%m-%d")
    end_str   = prices_full.index[-1].strftime("%Y-%m-%d")
    print(f"\n  Downloading auxiliary data...")
    gpr          = download_gpr_recent(start_str, end_str)
    cot          = download_cot_recent(start_str, end_str)
    gld_holdings = download_gld_holdings_recent(start_str, end_str)
    trends       = download_google_trends_recent(start_str, end_str)
    print(f"  ✓ Auxiliary data loaded")

    # ── 3. Build GDX features ─────────────────────────────────────────────
    print(f"\n  Building features (full history)...")
    try:
        features_raw = build_features_live(prices_full, cfg, gpr=gpr,
                                            cot=cot, gld_holdings=gld_holdings,
                                            trends=trends)
    except Exception as e:
        print(f"  ✗ GDX feature build failed: {e}")
        return None, None

    features_full = align_features(features_raw, feature_names)
    features_full[features_full.isna()] = 0.0
    features_full = features_full.ffill().bfill()
    print(f"  ✓ GDX features: {len(features_full)} rows  "
          f"({features_full.index[0].date()} → {features_full.index[-1].date()})")

    # ── 4. Run GDX inference on every possible day ────────────────────────
    lb_gdx  = cfg["lookback"]
    n_gdx   = len(features_full) - lb_gdx
    print(f"\n  Running GDX inference on {n_gdx} days...")
    gdx_preds = generate_recent_predictions(
        fold_models, features_full, cfg, n_days=n_gdx)
    gdx_preds = gdx_preds.sort_index()
    print(f"  ✓ GDX: {len(gdx_preds)} days  "
          f"({gdx_preds.index[0].date()} → {gdx_preds.index[-1].date()})")

    # ── 5. Run XLE inference if models available ──────────────────────────
    xle_preds = None
    if xle_models and xle_feature_names and xle_cfg:
        print(f"\n  Building XLE features (full history)...")
        try:
            # Build XLE-specific auxiliary data
            xle_gpr    = download_gpr_recent(start_str, end_str)
            xle_trends = download_google_trends_recent(
                start_str, end_str,
                keywords=["oil price", "energy stocks", "crude oil"],
                tag="xle_sim")
            xle_raw = build_features_live(prices_full, xle_cfg, gpr=xle_gpr,
                                           cot=None, gld_holdings=None,
                                           trends=xle_trends)
            xle_full = align_features(xle_raw, xle_feature_names)
            xle_full[xle_full.isna()] = 0.0
            xle_full = xle_full.ffill().bfill()
            lb_xle  = xle_cfg.get("lookback", cfg["lookback"])
            n_xle   = len(xle_full) - lb_xle
            print(f"  Running XLE inference on {n_xle} days...")
            xle_preds = generate_recent_predictions(
                xle_models, xle_full, xle_cfg, n_days=n_xle)
            xle_preds = xle_preds.sort_index()
            print(f"  ✓ XLE: {len(xle_preds)} days  "
                  f"({xle_preds.index[0].date()} → "
                  f"{xle_preds.index[-1].date()})")
        except Exception as e:
            print(f"  ⚠ XLE inference failed: {e} — running GDX only")
            xle_preds = None

    # ── 5b. Run QQQ inference if models available ─────────────────────────
    qqq_preds = None
    if qqq_models and qqq_feature_names and qqq_cfg:
        print(f"\n  Building QQQ features (full history)...")
        try:
            qqq_gpr    = download_gpr_recent(start_str, end_str)
            qqq_trends = download_google_trends_recent(
                start_str, end_str,
                keywords=["nvidia stock", "interest rates",
                          "recession", "AI stocks"],
                tag="qqq_sim")
            qqq_raw  = build_features_live(prices_full, qqq_cfg,
                                            gpr=qqq_gpr,
                                            cot=None,
                                            gld_holdings=None,
                                            trends=qqq_trends)
            qqq_full = align_features(qqq_raw, qqq_feature_names)
            qqq_full[qqq_full.isna()] = 0.0
            qqq_full = qqq_full.ffill().bfill()
            lb_qqq   = qqq_cfg.get("lookback", cfg["lookback"])
            n_qqq    = len(qqq_full) - lb_qqq
            print(f"  Running QQQ inference on {n_qqq} days...")
            qqq_preds = generate_recent_predictions(
                qqq_models, qqq_full, qqq_cfg, n_days=n_qqq)
            qqq_preds = qqq_preds.sort_index()
            print(f"  ✓ QQQ: {len(qqq_preds)} days  "
                  f"({qqq_preds.index[0].date()} → "
                  f"{qqq_preds.index[-1].date()})")
        except Exception as e:
            print(f"  ⚠ QQQ inference failed: {e} — skipping QQQ")
            qqq_preds = None

    # ── 6. Build price return lookups ─────────────────────────────────────
    gdx      = prices_full["GDX"].dropna()
    gdx_r    = gdx.pct_change().fillna(0)
    gdx_rlkp = {d.date(): float(v) for d, v in gdx_r.items()}

    xle_rlkp = {}
    if xle_preds is not None and "XLE" in prices_full.columns:
        xle_p    = prices_full["XLE"].dropna().pct_change().fillna(0)
        xle_rlkp = {d.date(): float(v) for d, v in xle_p.items()}

    qqq_rlkp = {}
    if qqq_preds is not None and "QQQ" in prices_full.columns:
        qqq_p    = prices_full["QQQ"].dropna().pct_change().fillna(0)
        qqq_rlkp = {d.date(): float(v) for d, v in qqq_p.items()}

    # XLE confidence lookup
    xle_pos_lkp = {}
    if xle_preds is not None:
        for d, row in xle_preds.iterrows():
            xle_pos_lkp[pd.Timestamp(d).date()] = {
                "pred": float(row["pred"]),
                "conf": float(row["conf_pct"]),
                "pos":  float(row["position"]),
            }

    # QQQ confidence lookup
    qqq_pos_lkp = {}
    if qqq_preds is not None:
        for d, row in qqq_preds.iterrows():
            qqq_pos_lkp[pd.Timestamp(d).date()] = {
                "pred": float(row["pred"]),
                "conf": float(row["conf_pct"]),
                "pos":  float(row["position"]),
            }

    # ── 7. Simulate GDX-only and GDX+XLE+QQQ portfolios ──────────────────
    cap_gdx  = INITIAL
    cap_rot  = INITIAL
    last_pos = 0.0
    rows     = []

    for date_idx, row in gdx_preds.iterrows():
        dk      = pd.Timestamp(date_idx).date()
        gdx_pos = float(row["position"])
        gdx_ret = gdx_rlkp.get(dk, 0.0)
        idle    = 1.0 - gdx_pos

        # Strength-ranked rotation cascade (patched 2026-05-28):
        # Collect XLE and QQQ candidates that pass the deployment gate
        # (min predicted return + min confidence threshold), then sort by
        # confidence percentile descending — the stronger-signal asset gets
        # first dibs on idle cash. Replaces the fixed XLE-before-QQQ priority.
        candidates = []
        if xle_preds is not None and dk in xle_pos_lkp:
            xd = xle_pos_lkp[dk]
            if xd["pred"] > XLE_MIN_SIG and xd["conf"] >= XLE_MIN_CONF:
                candidates.append(("xle", xd["pos"], xd["conf"]))
        if qqq_preds is not None and dk in qqq_pos_lkp:
            qd = qqq_pos_lkp[dk]
            if qd["pred"] > XLE_MIN_SIG and qd["conf"] >= XLE_MIN_CONF:
                candidates.append(("qqq", qd["pos"], qd["conf"]))

        candidates.sort(key=lambda c: c[2], reverse=True)
        positions = {"xle": 0.0, "qqq": 0.0}
        remaining = idle
        for name, target, _ in candidates:
            if remaining > 0.01:
                alloc = min(target, remaining)
                positions[name] = alloc
                remaining -= alloc

        xle_pos = positions["xle"]
        qqq_pos = positions["qqq"]

        cash_pos = idle - xle_pos - qqq_pos
        xle_ret  = xle_rlkp.get(dk, 0.0) if xle_pos > 0 else 0.0
        qqq_ret  = qqq_rlkp.get(dk, 0.0) if qqq_pos > 0 else 0.0

        # GDX-only portfolio
        cap_gdx = cap_gdx * (1 + gdx_ret * gdx_pos)

        # Rotation portfolio
        cap_rot = cap_rot * (1 + gdx_ret * gdx_pos +
                             xle_ret * xle_pos +
                             qqq_ret * qqq_pos +
                             CASH_DAILY * cash_pos)

        rows.append({
            "date":        date_idx,
            "gdx_pos":     gdx_pos,
            "xle_pos":     xle_pos,
            "qqq_pos":     qqq_pos,
            "cash_pos":    cash_pos,
            "gdx_ret":     gdx_ret,
            "portfolio_$": cap_gdx,
            "rotation_$":  cap_rot,
        })

    sim_df = pd.DataFrame(rows).set_index("date")

    # ── 8. Yearly summary ─────────────────────────────────────────────────
    sim_df["year"] = sim_df.index.year
    years          = sorted(sim_df["year"].unique())
    yearly_rows    = []
    prev_gdx = prev_rot = INITIAL

    for yr in years:
        yr_df    = sim_df[sim_df["year"] == yr]
        end_gdx  = float(yr_df["portfolio_$"].iloc[-1])
        end_rot  = float(yr_df["rotation_$"].iloc[-1])
        yr_gdx   = (end_gdx / prev_gdx - 1) * 100
        yr_rot   = (end_rot / prev_rot - 1) * 100
        long_pct = (yr_df["gdx_pos"] > 0).mean() * 100
        xle_pct  = (yr_df["xle_pos"] > 0).mean() * 100
        qqq_pct  = (yr_df["qqq_pos"] > 0).mean() * 100 \
                   if "qqq_pos" in yr_df.columns else 0.0
        gdx_yr   = gdx[gdx.index.year == yr]
        gdx_bh   = (float(gdx_yr.iloc[-1]) / float(gdx_yr.iloc[0]) - 1) * 100 \
                   if len(gdx_yr) > 1 else 0.0
        yearly_rows.append({
            "year": yr, "gdx_ret": yr_gdx, "rot_ret": yr_rot,
            "gdx_bh": gdx_bh, "long_pct": long_pct,
            "xle_pct": xle_pct, "qqq_pct": qqq_pct,
            "end_gdx": end_gdx, "end_rot": end_rot
        })
        prev_gdx = end_gdx
        prev_rot = end_rot

    # ── 9. Overall stats ──────────────────────────────────────────────────
    cal_years = (sim_df.index[-1] - sim_df.index[0]).days / 365.25
    final_gdx = float(sim_df["portfolio_$"].iloc[-1])
    final_rot = float(sim_df["rotation_$"].iloc[-1])
    cagr_gdx  = (np.exp(np.log(final_gdx / INITIAL) / cal_years) - 1) * 100
    cagr_rot  = (np.exp(np.log(final_rot / INITIAL) / cal_years) - 1) * 100
    vol_gdx   = sim_df["portfolio_$"].pct_change().std() * np.sqrt(252) * 100

    has_xle = xle_preds is not None
    has_qqq = qqq_preds is not None

    print(f"\n{'═'*60}")
    print(f"  FULL HISTORY RESULTS  (inference-only, $100k start, 0% cash)")
    print(f"{'═'*60}")
    print(f"  Period     : {sim_df.index[0].date()} → {sim_df.index[-1].date()}  "
          f"({cal_years:.1f} years)")
    print(f"  GDX only   : ${final_gdx:>10,.0f}  CAGR={cagr_gdx:>+5.1f}%  "
          f"Vol={vol_gdx:.1f}%")
    if has_xle or has_qqq:
        label = "GDX+XLE+QQQ" if has_xle and has_qqq else \
                "GDX+XLE" if has_xle else "GDX+QQQ"
        print(f"  {label:<11}: ${final_rot:>10,.0f}  CAGR={cagr_rot:>+5.1f}%  "
              f"Premium={cagr_rot-cagr_gdx:>+4.1f}%")
    print(f"  Note       : pre-2016 uses in-sample inference (slightly optimistic)")

    print(f"\n  ── Yearly Breakdown ─────────────────────────────────────────")
    if has_xle and has_qqq:
        print(f"  {'Year':<6} {'GDX':>8} {'Rotation':>9} {'B&H':>8} "
              f"{'Long%':>6} {'XLE%':>6} {'QQQ%':>6}  {'End (rot)':>12}")
        print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*8} {'─'*6} {'─'*6} {'─'*6}  {'─'*12}")
        for r in yearly_rows:
            print(f"  {r['year']:<6} {r['gdx_ret']:>+7.1f}% "
                  f"{r['rot_ret']:>+8.1f}% {r['gdx_bh']:>+7.1f}% "
                  f"{r['long_pct']:>5.1f}% {r['xle_pct']:>5.1f}% "
                  f"{r['qqq_pct']:>5.1f}%  ${r['end_rot']:>10,.0f}")
    elif has_xle:
        print(f"  {'Year':<6} {'GDX':>8} {'GDX+XLE':>8} {'B&H':>8} "
              f"{'Long%':>6} {'XLE%':>6}  {'End (rot)':>12}")
        print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}  {'─'*12}")
        for r in yearly_rows:
            print(f"  {r['year']:<6} {r['gdx_ret']:>+7.1f}% "
                  f"{r['rot_ret']:>+7.1f}% {r['gdx_bh']:>+7.1f}% "
                  f"{r['long_pct']:>5.1f}% {r['xle_pct']:>5.1f}%  "
                  f"${r['end_rot']:>10,.0f}")
    else:
        print(f"  {'Year':<6} {'GDX':>8} {'GDX B&H':>8} {'Long%':>7}  "
              f"{'End Value':>12}")
        print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*7}  {'─'*12}")
        for r in yearly_rows:
            print(f"  {r['year']:<6} {r['gdx_ret']:>+7.1f}% "
                  f"{r['gdx_bh']:>+7.1f}% {r['long_pct']:>6.1f}%  "
                  f"${r['end_gdx']:>10,.0f}")

    return sim_df, yearly_rows


def main():
    print(f"\n{'═'*60}")
    print(f"  AURUM·AI — Live Signal Tracker")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"{'═'*60}")

    # Auto-deploy configured seeds from candidates folders
    print(f"\n  Deploying models...")
    for ticker in ["GDX", "XLE", "QQQ"]:
        deploy_seed_models(ticker)

    N_HISTORY = 370   # Jan 2025 → present (~370 trading days)

    # 1. Load fold models and feature names
    print(f"\n  Loading fold models...")
    fold_models, feature_names = load_fold_models()
    if not fold_models:
        return
    if feature_names is None:
        print(f"  ✗ Feature names not found — retrain with gold_miner_trainer.py")
        return

    # 2. Download recent prices
    prices = download_recent_prices(SCFG)
    if prices is None or len(prices) < SCFG["lookback"] + 50:
        print(f"  ✗ Insufficient price data")
        return
    as_of_date = prices.index[-1]

    # 3. Download auxiliary data
    print(f"\n  Downloading auxiliary data...")
    start_str = prices.index[0].strftime("%Y-%m-%d")
    end_str   = prices.index[-1].strftime("%Y-%m-%d")

    gpr = download_gpr_recent(start_str, end_str)
    print(f"  {'✓ GPR loaded' if gpr is not None else '✗ GPR unavailable (features zeroed)'}")
    cot = download_cot_recent(start_str, end_str)
    print(f"  {'✓ COT loaded' if cot is not None else '✗ COT unavailable (features zeroed)'}")
    gld_holdings = download_gld_holdings_recent(start_str, end_str)
    print(f"  {'✓ GLD holdings loaded' if gld_holdings is not None else '✗ GLD holdings unavailable'}")
    trends = download_google_trends_recent(start_str, end_str)
    print(f"  {'✓ Google Trends loaded' if trends is not None else '✗ Google Trends unavailable (pytrends)'}")

    # 4. Build features
    print(f"\n  Building features...")
    try:
        features_raw = build_features_live(prices, SCFG, gpr=gpr, cot=cot,
                                            gld_holdings=gld_holdings,
                                            trends=trends)
        print(f"  ✓ Raw features: {features_raw.shape[0]} rows × {features_raw.shape[1]} cols")
    except Exception as e:
        import traceback
        print(f"  ✗ Feature build failed: {e}")
        traceback.print_exc()
        return

    # 5. Align to model's expected feature set
    features_aligned = align_features(features_raw, feature_names)
    all_nan = [c for c in features_aligned.columns if features_aligned[c].isna().all()]
    if all_nan:
        features_aligned[all_nan] = 0.0
    features_aligned = features_aligned.ffill().bfill()
    lb     = SCFG["lookback"]
    window = features_aligned.iloc[-lb:]
    if window.isna().any(axis=1).sum() > 0:
        features_aligned.iloc[-lb:] = window.fillna(0.0)
    if len(features_aligned) < lb:
        print(f"  ✗ Not enough feature rows (need {lb}, have {len(features_aligned)})")
        return
    print(f"  ✓ Features ready: {features_aligned.shape[1]} columns  {len(features_aligned)} rows")

    # 6. Run ensemble inference
    print(f"\n  Running ensemble inference...")
    result = run_ensemble(fold_models, features_aligned, SCFG)
    if result is None:
        return

    # 7. Trading decision
    decision = make_trading_decision(result, SCFG, prices)

    # 8. Generate signal history for GDX + rotation instruments
    print(f"\n  Generating signal history...")
    recent_preds = generate_recent_predictions(
        fold_models, features_aligned, SCFG, n_days=N_HISTORY)
    if len(recent_preds) > 0:
        rp_sorted = recent_preds.sort_index()
        print(f"  ✓ GDX predictions: {len(recent_preds)} days  "
              f"({rp_sorted.index[0].date()} → {rp_sorted.index[-1].date()})")

    # Generate sliding-window predictions for XLE rotation
    print(f"  Loading rotation models (XLE)...")
    xle_models, xle_aligned, xle_cfg = load_rotation_models_and_features(
        "XLE", prices)
    print(f"  XLE models: {len(xle_models) if xle_models else 0}")

    xle_recent = generate_recent_predictions(
        xle_models, xle_aligned, xle_cfg or SCFG, n_days=N_HISTORY) \
        if xle_models else pd.DataFrame()
    print(f"  ✓ XLE: {len(xle_recent)} days")

    # Generate sliding-window predictions for QQQ rotation
    print(f"  Loading rotation models (QQQ)...")
    qqq_models, qqq_aligned, qqq_cfg = load_rotation_models_and_features(
        "QQQ", prices)
    print(f"  QQQ models: {len(qqq_models) if qqq_models else 0}")

    qqq_recent = generate_recent_predictions(
        qqq_models, qqq_aligned, qqq_cfg or SCFG, n_days=N_HISTORY) \
        if qqq_models else pd.DataFrame()
    print(f"  ✓ QQQ: {len(qqq_recent)} days")

    # 9. Fetch news
    weekday   = datetime.now().weekday()
    news_days = 3 if weekday <= 1 else 1
    print(f"\n  Fetching news (last {news_days} day{'s' if news_days > 1 else ''})...")
    articles = fetch_news(lookback_days=news_days)
    print(f"  ✓ {len(articles)} relevant articles found")

    # 10. Save to log
    log = save_signal(result, decision, as_of_date)

    # 11. Full history simulation — inference-only, no external equity files
    sim_df, yearly_rows = run_full_history_simulation(
        fold_models, feature_names, SCFG, prices,
        xle_models=xle_models,
        xle_feature_names=xle_aligned.columns.tolist() if xle_aligned is not None and len(xle_aligned) > 0 else None,
        xle_cfg=xle_cfg or SCFG,
        qqq_models=qqq_models,
        qqq_feature_names=qqq_aligned.columns.tolist() if qqq_aligned is not None and len(qqq_aligned) > 0 else None,
        qqq_cfg=qqq_cfg or SCFG)

    # 12. Print live signal report
    print_signal_report(result, decision, as_of_date, log,
                        prices=prices, articles=articles,
                        news_days=news_days,
                        fold_models=fold_models,
                        features_aligned=features_aligned,
                        port=None,
                        recent_preds=recent_preds,
                        xle_recent=xle_recent,
                        qqq_recent=qqq_recent,
                        n_history=N_HISTORY,
                        sim_df=sim_df)

    # 13. Signal history chart — $100k nominal, pure from recent_preds
    print(f"\n  Generating signal history chart...")
    full_path = os.path.join(OUT_DIR, "full_predictions.csv")
    actuals_s = None
    if os.path.exists(full_path):
        try:
            fp = pd.read_csv(full_path, parse_dates=["date"], index_col="date")
            fp.index = pd.to_datetime(fp.index).normalize().tz_localize(None)
            actuals_s = fp["actual"]
        except Exception:
            pass

    # Build rotation equity dict from sim_df for chart (if available)
    rot_equity_for_chart = None
    if sim_df is not None and "rotation_$" in sim_df.columns:
        rot_equity_for_chart = {
            pd.Timestamp(d).date(): float(v)
            for d, v in sim_df["rotation_$"].items()}

    plot_signal_history(recent_preds, None, prices,
                        as_of_date, actuals_s=actuals_s,
                        port_series=None,
                        rot_portfolio=None,
                        rot_gdx_xle=rot_equity_for_chart or 100_000.0,
                        rot_gdx_tlt=None,
                        xle_recent_preds=xle_recent,
                        qqq_recent_preds=qqq_recent)

    print(f"  Signal logged → {LOG_PATH}")


if __name__ == "__main__":
    main()