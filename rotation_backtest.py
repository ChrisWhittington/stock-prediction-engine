"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AURUM·AI  —  rotation_backtest.py  (v1)                            ║
║                                                                              ║
║  Combined rotation strategy backtest: GDX → XLE → TLT → SGOV              ║
║                                                                              ║
║  STRATEGY                                                                   ║
║    Each day, allocate capital in priority order:                            ║
║    1. GDX  — ML model bullish → sized by confidence tier                   ║
║    2. XLE  — ML model bullish → deploys GDX idle cash                      ║
║    3. TLT  — above 20d MA → deploys remaining idle cash                    ║
║    4. SGOV — remainder earns ~5.2% annual (risk-free rate)                 ║
║                                                                              ║
║  INPUTS                                                                      ║
║    aurum_output/full_predictions.csv  — GDX daily predictions               ║
║    xle_output/full_predictions.csv    — XLE daily predictions               ║
║    GDX, XLE, TLT prices via yfinance                                        ║
║                                                                              ║
║  OUTPUTS                                                                    ║
║    rotation_backtest/daily_equity.csv  — daily combined portfolio           ║
║    rotation_backtest/results.png       — 5-panel chart                      ║
║    rotation_backtest/summary.txt       — key metrics                        ║
║                                                                              ║
║  RUN                                                                        ║
║    python rotation_backtest.py                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────
GDX_DIR  = "aurum_output"
XLE_DIR  = "xle_output"
OUT_DIR  = "rotation_backtest"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RCFG = {
    "initial_capital":   100_000,
    "transaction_cost":  0.001,     # 0.10% per side
    "slippage":          0.0005,    # 0.05%
    "gdx_min_signal":    0.008,   # v10: scaled for 20-day predictions (was 0.005)
    "xle_min_signal":    0.002,
    "xle_min_conf":      0.40,      # minimum confidence to deploy XLE
    "tlt_ma_days":       20,        # TLT bullish when above 20d MA
    "sgov_rate":         0.052,     # annual SGOV yield
    "hold_days":         20,        # v10: matches GDX forward_days=20 (was 10)
    "rebal_freq":        5,
    "conf_tiers": [
        (0.00, 0.25, 0.25),
        (0.25, 0.50, 0.50),
        (0.50, 0.75, 0.75),
        (0.75, 1.01, 1.00),
    ],
    "execution_lag":     1,
    # Restrict to period where both models have seen enough training data.
    # GDX model starts 2014-09-30. XLE model starts 2008 but is only
    # well-trained from mid-2015 onwards. Use 2015-01-01 to capture the
    # full GDX model history while keeping XLE predictions reasonable.
    "backtest_start":    "2015-01-01",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(out_dir, label):
    full_path = os.path.join(out_dir, "full_predictions.csv")
    oof_path  = os.path.join(out_dir, "oof_predictions.csv")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"  ✗ {label}: {full_path} not found — "
                                f"run the trainer first")

    full = pd.read_csv(full_path, parse_dates=["date"], index_col="date")
    full.index = pd.to_datetime(full.index).normalize().tz_localize(None)
    full = full[~full.index.duplicated(keep="last")].sort_index()

    # Mark OOF rows as honest
    full["is_oof"] = False
    if os.path.exists(oof_path):
        oof = pd.read_csv(oof_path, parse_dates=["date"], index_col="date")
        oof.index = pd.to_datetime(oof.index).normalize().tz_localize(None)
        full.loc[full.index.isin(oof.index), "is_oof"] = True
        oof_count = full["is_oof"].sum()
    else:
        oof_count = 0

    print(f"  ✓ {label}: {len(full)} rows  "
          f"({oof_count} OOF honest, {len(full)-oof_count} model-filled)  "
          f"{full.index[0].date()} → {full.index[-1].date()}")
    return full


# ══════════════════════════════════════════════════════════════════════════════
# 2. DOWNLOAD PRICES
# ══════════════════════════════════════════════════════════════════════════════

def download_prices(start, end):
    tickers = ["GDX", "XLE", "TLT"]
    frames  = {}
    print(f"\n  Downloading prices ({start} → {end})...")
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            hist.index = pd.to_datetime(hist.index).normalize().tz_localize(None)
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
            frames[t] = hist["Close"].rename(t)
            print(f"    {t}: {len(hist)} rows")
        except Exception as e:
            print(f"    {t}: failed — {e}")

    prices = pd.concat(frames.values(), axis=1).ffill()
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def build_rotation_signals(gdx_preds, xle_preds, prices, cfg):
    """
    Build daily allocation signals for GDX, XLE, TLT, Cash.
    Returns DataFrame with columns: gdx_pos, xle_pos, tlt_pos, cash_pos
    """
    # Align all to common date index
    all_dates = gdx_preds.index.union(xle_preds.index)
    all_dates = all_dates[all_dates >= max(gdx_preds.index[0],
                                           xle_preds.index[0])]

    # GDX signal
    gdx_pred  = gdx_preds["pred"].reindex(all_dates).ffill()
    gdx_conf  = gdx_pred.abs().rank(pct=True)

    # XLE signal
    xle_pred  = xle_preds["pred"].reindex(all_dates).ffill()
    xle_conf  = xle_pred.abs().rank(pct=True)

    # TLT MA rule
    tlt_ma_days = cfg["tlt_ma_days"]
    tlt_prices  = prices["TLT"].reindex(all_dates).ffill()
    tlt_ma      = tlt_prices.rolling(tlt_ma_days).mean()
    tlt_bull    = (tlt_prices > tlt_ma).fillna(False)

    # No execution lag shift here — handled in backtest by pct_change().shift(-1)
    # which maps today's signal to tomorrow's return

    rows = []
    for date in all_dates:
        gp   = float(gdx_pred.get(date, 0) or 0)
        gc   = float(gdx_conf.get(date, 0) or 0)
        xp   = float(xle_pred.get(date, 0) or 0)
        xc   = float(xle_conf.get(date, 0) or 0)
        tb   = bool(tlt_bull.get(date, False))

        # GDX position
        gdx_pos = 0.0
        if gp > cfg["gdx_min_signal"]:
            for (lo, hi, sz) in cfg["conf_tiers"]:
                if lo <= gc < hi:
                    gdx_pos = sz
                    break

        idle = 1.0 - gdx_pos

        # XLE position — from idle cash
        xle_pos = 0.0
        if xp > cfg["xle_min_signal"] and xc >= cfg["xle_min_conf"] and idle > 0.01:
            for (lo, hi, sz) in cfg["conf_tiers"]:
                if lo <= xc < hi:
                    xle_pos = min(sz, idle)
                    break

        idle_after_xle = idle - xle_pos

        # TLT disabled — Option B: idle cash earns SGOV (~5% annual)
        # TLT was dragging performance from 2021 onwards (rate hiking cycle)
        # Rate-filtered TLT can be re-enabled later if desired
        tlt_pos  = 0.0
        cash_pos = idle_after_xle

        rows.append({
            "date":     date,
            "gdx_pos":  round(gdx_pos,  4),
            "xle_pos":  round(xle_pos,  4),
            "tlt_pos":  round(tlt_pos,  4),
            "cash_pos": round(cash_pos, 4),
            "gdx_pred": gp,
            "xle_pred": xp,
            "tlt_bull": tb,
        })

    signals = pd.DataFrame(rows).set_index("date")
    print(f"\n  Signal summary ({len(signals)} days):")
    print(f"    GDX long days : {(signals['gdx_pos'] > 0).sum():>5}  "
          f"({(signals['gdx_pos'] > 0).mean():.1%})")
    print(f"    XLE long days : {(signals['xle_pos'] > 0).sum():>5}  "
          f"({(signals['xle_pos'] > 0).mean():.1%})")
    print(f"    TLT long days : {(signals['tlt_pos'] > 0).sum():>5}  "
          f"({(signals['tlt_pos'] > 0).mean():.1%})")
    print(f"    Cash only     : {(signals['cash_pos'] >= 0.99).sum():>5}  "
          f"({(signals['cash_pos'] >= 0.99).mean():.1%})")
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUN COMBINED BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_rotation_backtest(signals, prices, cfg, gdx_preds, xle_preds):
    """
    Daily mark-to-market simulation matching the individual backtest methodology.
    Uses actual realized returns from predictions CSV (same as aurum_backtest.py)
    spread across hold_days, rather than daily price changes.
    TLT uses daily price returns (no ML model — rule-based).
    """
    capital    = float(cfg["initial_capital"])
    tc         = cfg["transaction_cost"] + cfg["slippage"]
    sgov_daily = cfg["sgov_rate"] / 252
    hold_days  = cfg["hold_days"]

    # Align actual returns from prediction files to signal dates
    # Individual backtests shift actual by -1 (next-day execution lag)
    gdx_actual = gdx_preds["actual"].copy()
    gdx_actual.index = pd.to_datetime(gdx_actual.index).normalize().tz_localize(None)
    gdx_actual = gdx_actual.reindex(signals.index).ffill()

    xle_actual = xle_preds["actual"].copy()
    xle_actual.index = pd.to_datetime(xle_actual.index).normalize().tz_localize(None)
    xle_actual = xle_actual.reindex(signals.index).ffill()

    # TLT: compute forward 20-day return spread across hold_days
    # (matches GDX v10 hold_days=20)
    tlt_p = prices["TLT"].reindex(signals.index).ffill()
    tlt_fwd = np.log(tlt_p.shift(-20) / tlt_p).shift(1).fillna(0.0)
    tlt_actual = tlt_fwd   # already a log return over hold_days

    print(f"  Trading on {len(signals)} signal days  "
          f"({signals.index[0].date()} → {signals.index[-1].date()})")

    prev_gdx_pos = prev_xle_pos = prev_tlt_pos = 0.0
    equity_rows  = []

    for i, (date, row) in enumerate(signals.iterrows()):
        gdx_pos  = float(row["gdx_pos"])
        xle_pos  = float(row["xle_pos"])
        tlt_pos  = float(row["tlt_pos"])
        cash_pos = float(row["cash_pos"])

        # Costs only when positions change
        gdx_chg    = abs(gdx_pos - prev_gdx_pos)
        xle_chg    = abs(xle_pos - prev_xle_pos)
        tlt_chg    = abs(tlt_pos - prev_tlt_pos)
        trade_cost = (gdx_chg + xle_chg + tlt_chg) * tc * capital

        # GDX and XLE: spread 10-day actual return across each holding day
        # (matches individual backtest: daily_log = pos * actual_ret / hold_days)
        gdx_act = float(gdx_actual.iloc[i]) if i < len(gdx_actual) else 0.0
        xle_act = float(xle_actual.iloc[i]) if i < len(xle_actual) else 0.0
        tlt_act  = float(tlt_actual.iloc[i]) if i < len(tlt_actual) else 0.0

        dr_gdx  = gdx_pos  * (gdx_act  / hold_days)
        dr_xle  = xle_pos  * (xle_act  / hold_days)
        dr_tlt  = tlt_pos  * (tlt_act  / hold_days)
        dr_cash = cash_pos * sgov_daily

        # Log return update for all ML/rule instruments, linear for cash
        total_log = dr_gdx + dr_xle + dr_tlt
        daily_pnl = capital * (np.exp(total_log) - 1)
        daily_pnl += capital * dr_cash - trade_cost
        capital   += daily_pnl
        capital    = max(capital, 1.0)

        equity_rows.append({
            "date":        date,
            "portfolio_$": round(capital, 2),
            "gdx_$":       round(capital * gdx_pos, 2),
            "xle_$":       round(capital * xle_pos, 2),
            "tlt_$":       round(capital * tlt_pos, 2),
            "cash_$":      round(capital * cash_pos, 2),
            "gdx_pos":     gdx_pos,
            "xle_pos":     xle_pos,
            "tlt_pos":     tlt_pos,
            "cash_pos":    cash_pos,
            "gdx_ret":     dr_gdx,
            "xle_ret":     dr_xle,
            "tlt_ret":     dr_tlt,
        })

        prev_gdx_pos = gdx_pos
        prev_xle_pos = xle_pos
        prev_tlt_pos = tlt_pos

    return pd.DataFrame(equity_rows).set_index("date")


# ══════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity_df, label="Strategy"):
    port   = equity_df["portfolio_$"]
    rets   = port.pct_change().dropna()
    n_days = len(port)
    years  = n_days / 252

    total_ret  = (port.iloc[-1] / port.iloc[0] - 1) * 100
    ann_ret    = ((port.iloc[-1] / port.iloc[0]) ** (1 / years) - 1) * 100
    vol        = rets.std() * np.sqrt(252) * 100
    sharpe     = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    drawdowns  = (port / port.cummax() - 1) * 100
    max_dd     = float(drawdowns.min())
    long_days  = (equity_df["gdx_pos"] + equity_df["xle_pos"] +
                  equity_df["tlt_pos"]).gt(0).mean() * 100

    print(f"\n  ── {label} ────────────────────────────────────")
    print(f"  Period           : {years:.1f} calendar years")
    print(f"  Initial capital  : ${port.iloc[0]:>12,.0f}")
    print(f"  Final capital    : ${port.iloc[-1]:>12,.0f}")
    print(f"  Total return     : {total_ret:>+10.2f}%")
    print(f"  Annual return    : {ann_ret:>+10.2f}%  (CAGR)")
    print(f"  Volatility       : {vol:>10.2f}%  (annualised)")
    print(f"  Sharpe ratio     : {sharpe:>10.3f}")
    print(f"  Max drawdown     : {max_dd:>10.2f}%")
    print(f"  Days invested    : {long_days:>10.1f}%")

    return {
        "total_return_%":   total_ret,
        "annual_return_%":  ann_ret,
        "volatility_%":     vol,
        "sharpe":           sharpe,
        "max_drawdown_%":   max_dd,
        "long_days_%":      long_days,
        "final_capital_$":  port.iloc[-1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_rotation_results(equity_df, prices, gdx_only_equity, cfg):
    DARK  = "#0d1117"
    PANEL = "#161b22"
    GOLD  = "#d4af37"
    GREEN = "#3fb950"
    RED   = "#f85149"
    BLUE  = "#58a6ff"
    MUTED = "#8b949e"
    XLE_C = "#f0a500"
    TLT_C = "#58a6ff"

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color("#30363d")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(True, color="#21262d", linewidth=0.4, alpha=0.8)

    fig, axes = plt.subplots(
        4, 1, figsize=(22, 14),
        gridspec_kw={"height_ratios": [3, 3, 1.5, 1.5]},
        facecolor=DARK, sharex=True)

    fig.suptitle(
        f"AURUM·AI — Combined Rotation Backtest  "
        f"GDX → XLE → TLT → SGOV  "
        f"({equity_df.index[0].strftime('%b %Y')} → "
        f"{equity_df.index[-1].strftime('%b %Y')})",
        color=GOLD, fontsize=13, fontweight="bold", y=0.998)

    xs = np.arange(len(equity_df))
    dates = equity_df.index

    # ── Panel 1: Portfolio equity curves ──────────────────────────────────
    ax1 = axes[0]
    style_ax(ax1)

    port      = equity_df["portfolio_$"].values
    gdx_only  = gdx_only_equity.reindex(equity_df.index).ffill().values \
                if gdx_only_equity is not None else None

    # GDX B&H line
    gdx_prices = prices["GDX"].reindex(equity_df.index).ffill()
    bh_val     = float(cfg["initial_capital"])
    gdx_bh     = (gdx_prices / gdx_prices.iloc[0]) * bh_val

    ax1.plot(xs, gdx_bh.values, color=MUTED, linewidth=1.0,
             linestyle="--", alpha=0.6, label="GDX B&H")

    if gdx_only is not None:
        ax1.plot(xs, gdx_only, color=GOLD, linewidth=1.4,
                 alpha=0.8, label="GDX only")

    ax1.plot(xs, port, color=GREEN, linewidth=2.0,
             label="GDX+XLE+TLT (rotation)", zorder=5)
    ax1.fill_between(xs, gdx_bh.values, port,
                     where=port > gdx_bh.values,
                     alpha=0.12, color=GREEN)
    ax1.fill_between(xs, gdx_bh.values, port,
                     where=port < gdx_bh.values,
                     alpha=0.12, color=RED)

    ax1.set_ylabel("Portfolio ($)", color=MUTED, fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, loc="upper left")
    ax1.set_xlim(-1, len(xs))

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = axes[1]
    style_ax(ax2)

    port_s  = equity_df["portfolio_$"]
    dd      = (port_s / port_s.cummax() - 1) * 100

    ax2.fill_between(xs, dd.values, 0, alpha=0.6, color=RED)
    ax2.plot(xs, dd.values, color=RED, linewidth=0.8)

    if gdx_only is not None:
        gdx_s    = pd.Series(gdx_only, index=equity_df.index)
        gdx_dd   = (gdx_s / gdx_s.cummax() - 1) * 100
        ax2.plot(xs, gdx_dd.values, color=GOLD, linewidth=0.9,
                 linestyle="--", alpha=0.7, label="GDX only DD")
        ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED, loc="lower left")

    ax2.set_ylabel("Drawdown %", color=MUTED, fontsize=9)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.set_xlim(-1, len(xs))

    # ── Panel 3: Stacked allocation bars ──────────────────────────────────
    ax3 = axes[2]
    style_ax(ax3)

    gdx_arr  = equity_df["gdx_pos"].values * 100
    xle_arr  = equity_df["xle_pos"].values * 100
    tlt_arr  = equity_df["tlt_pos"].values * 100
    cash_arr = equity_df["cash_pos"].values * 100

    ax3.bar(xs, gdx_arr,  color=GREEN, alpha=0.85, width=0.8, label="GDX")
    ax3.bar(xs, xle_arr,  color=XLE_C, alpha=0.85, width=0.8,
            bottom=gdx_arr, label="XLE")
    ax3.bar(xs, tlt_arr,  color=TLT_C, alpha=0.85, width=0.8,
            bottom=gdx_arr + xle_arr, label="TLT")
    ax3.bar(xs, cash_arr, color=MUTED,  alpha=0.20, width=0.8,
            bottom=gdx_arr + xle_arr + tlt_arr, label="Cash/SGOV")

    ax3.axhline(0, color=MUTED, linewidth=0.5)
    ax3.set_ylabel("Allocation", color=MUTED, fontsize=8)
    ax3.set_ylim(-5, 115)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED,
               loc="upper left", ncol=4)
    ax3.set_xlim(-1, len(xs))

    # ── Panel 4: Daily contribution by instrument ──────────────────────────
    ax4 = axes[3]
    style_ax(ax4)

    gdx_contrib = equity_df["gdx_ret"].values * 100
    xle_contrib = equity_df["xle_ret"].values * 100
    tlt_contrib = equity_df["tlt_ret"].values * 100

    ax4.bar(xs, gdx_contrib, color=GREEN, alpha=0.7, width=0.8, label="GDX")
    ax4.bar(xs, xle_contrib, color=XLE_C, alpha=0.7, width=0.8,
            bottom=gdx_contrib, label="XLE")
    ax4.bar(xs, tlt_contrib, color=TLT_C, alpha=0.7, width=0.8,
            bottom=gdx_contrib + xle_contrib, label="TLT")
    ax4.axhline(0, color=MUTED, linewidth=0.5)
    ax4.set_ylabel("Daily return %", color=MUTED, fontsize=8)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED,
               loc="upper left", ncol=3)
    ax4.set_xlim(-1, len(xs))

    # ── X-axis labels ──────────────────────────────────────────────────────
    tick_pos, tick_lbl = [], []
    prev_month = None
    for i, d in enumerate(dates):
        if d.month != prev_month:
            tick_pos.append(i)
            tick_lbl.append(d.strftime("%b '%y")
                            if d.month in (1, 4, 7, 10) or i == 0
                            else d.strftime("%b"))
            prev_month = d.month
    ax4.set_xticks(tick_pos)
    ax4.set_xticklabels(tick_lbl, rotation=35, ha="right",
                        color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    out_path = os.path.join(OUT_DIR, "rotation_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"\n  ✓ Chart → {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  AURUM·AI — Combined Rotation Backtest  v1")
    print(f"  Strategy: GDX → XLE → TLT → SGOV")
    print(f"{'═'*60}")

    # Load predictions
    print(f"\n  Loading predictions...")
    try:
        gdx_preds = load_predictions(GDX_DIR, "GDX")
    except FileNotFoundError as e:
        print(e)
        return
    try:
        xle_preds = load_predictions(XLE_DIR, "XLE")
    except FileNotFoundError as e:
        print(e)
        return

    # Common date range — restrict to reliable prediction period
    backtest_start = pd.Timestamp(RCFG.get("backtest_start", "2017-01-01"))
    start = max(gdx_preds.index[0], xle_preds.index[0], backtest_start)
    end   = min(gdx_preds.index[-1], xle_preds.index[-1])
    print(f"\n  Backtest window: {start.date()} → {end.date()}  "
          f"({(end-start).days/365.25:.1f} years)")
    # Apply execution lag
    start_dl = (start - pd.DateOffset(days=RCFG["tlt_ma_days"] * 4)).strftime("%Y-%m-%d")
    end_dl   = end.strftime("%Y-%m-%d")

    # Download prices
    prices = download_prices(start_dl, end_dl)

    # Build signals
    print(f"\n  Building rotation signals...")
    gdx_trimmed = gdx_preds.loc[start:end]
    xle_trimmed = xle_preds.loc[start:end]
    signals = build_rotation_signals(gdx_trimmed, xle_trimmed, prices, RCFG)

    # Trim to common window
    signals = signals.loc[start:end]
    prices_aligned = prices.reindex(signals.index).ffill()

    # Run combined backtest
    print(f"\n  Running combined backtest...")
    equity_df = run_rotation_backtest(signals, prices_aligned, RCFG,
                                      gdx_preds.loc[start:end],
                                      xle_preds.loc[start:end])

    # Detailed diagnostics
    if len(equity_df) > 5:
        print(f"\n  Diagnostics (first 10 days):")
        cols = ["portfolio_$","gdx_pos","xle_pos","tlt_pos","gdx_ret","xle_ret","tlt_ret"]
        print(equity_df[cols].head(10).to_string())
        print(f"\n  GDX return stats: min={equity_df['gdx_ret'].min():.4f} "
              f"max={equity_df['gdx_ret'].max():.4f} "
              f"mean={equity_df['gdx_ret'].mean():.6f}")
        print(f"  XLE return stats: min={equity_df['xle_ret'].min():.4f} "
              f"max={equity_df['xle_ret'].max():.4f} "
              f"mean={equity_df['xle_ret'].mean():.6f}")
        gdx_active = equity_df[equity_df['gdx_pos'] > 0]
        print(f"  GDX active days: {len(gdx_active)}  "
              f"mean daily ret: {gdx_active['gdx_ret'].mean():.6f}")

    # Save equity
    equity_path = os.path.join(OUT_DIR, "daily_equity.csv")
    equity_df.to_csv(equity_path)
    print(f"  ✓ Daily equity → {equity_path}  ({len(equity_df)} rows)")

    # Compute metrics
    print(f"\n{'═'*60}")
    print(f"  ROTATION BACKTEST RESULTS")
    print(f"{'═'*60}")

    rot_metrics = compute_metrics(equity_df, "GDX+XLE+TLT Rotation")

    # GDX-only comparison (from aurum_backtest daily_equity if available)
    gdx_only_equity = None
    gdx_equity_path = os.path.join("aurum_backtest", "daily_equity.csv")
    if os.path.exists(gdx_equity_path):
        try:
            ge = pd.read_csv(gdx_equity_path, parse_dates=["date"],
                             index_col="date")
            ge.index = pd.to_datetime(ge.index).normalize().tz_localize(None)
            gdx_only_equity = ge["portfolio_$"].reindex(
                equity_df.index).ffill()
            print(f"\n  ── vs GDX Only ─────────────────────────────────────")
            gdx_start = float(gdx_only_equity.dropna().iloc[0])
            gdx_end   = float(gdx_only_equity.dropna().iloc[-1])
            gdx_years = len(gdx_only_equity.dropna()) / 252
            gdx_ann   = (gdx_end / gdx_start) ** (1 / gdx_years) - 1
            gdx_dd_s  = gdx_only_equity.dropna()
            gdx_dd    = (gdx_dd_s / gdx_dd_s.cummax() - 1).min() * 100
            print(f"  GDX only CAGR    : {gdx_ann*100:>+.2f}%")
            print(f"  GDX only Max DD  : {gdx_dd:>+.2f}%")
            print(f"  Rotation premium : "
                  f"{rot_metrics['annual_return_%'] - gdx_ann*100:>+.2f}% CAGR")
        except Exception:
            pass

    # GDX B&H comparison
    gdx_prices = prices_aligned["GDX"].dropna()
    bh_total   = (gdx_prices.iloc[-1] / gdx_prices.iloc[0] - 1) * 100
    years      = len(gdx_prices) / 252
    bh_ann     = ((gdx_prices.iloc[-1] / gdx_prices.iloc[0]) ** (1/years) - 1) * 100
    print(f"\n  ── vs GDX Buy & Hold ───────────────────────────────")
    print(f"  B&H total return : {bh_total:>+.2f}%")
    print(f"  B&H CAGR         : {bh_ann:>+.2f}%")
    print(f"  Rotation alpha   : {rot_metrics['annual_return_%'] - bh_ann:>+.2f}% CAGR")

    # Breakdown by instrument
    print(f"\n{'═'*60}")
    print(f"  INSTRUMENT CONTRIBUTION")
    print(f"{'═'*60}")
    total_gdx_ret  = equity_df["gdx_ret"].sum() * 100
    total_xle_ret  = equity_df["xle_ret"].sum() * 100
    total_tlt_ret  = equity_df["tlt_ret"].sum() * 100
    total_all      = total_gdx_ret + total_xle_ret + total_tlt_ret
    print(f"  GDX contribution : {total_gdx_ret:>+7.2f}%  "
          f"({total_gdx_ret/total_all*100:.0f}% of gains)")
    print(f"  XLE contribution : {total_xle_ret:>+7.2f}%  "
          f"({total_xle_ret/total_all*100:.0f}% of gains)")
    print(f"  TLT contribution : {total_tlt_ret:>+7.2f}%  "
          f"({total_tlt_ret/total_all*100:.0f}% of gains)")
    print(f"  SGOV (cash)      : (remainder)")

    # Year-by-year breakdown — use portfolio $ levels not return column sums
    print(f"\n{'═'*60}")
    print(f"  YEAR-BY-YEAR BREAKDOWN")
    print(f"{'═'*60}")
    print(f"  {'Year':<6} {'Rotation':>10} {'GDX only':>10} "
          f"{'TLT days':>9} {'XLE days':>9}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*9} {'─'*9}")

    for year in range(equity_df.index[0].year, equity_df.index[-1].year + 1):
        yr = equity_df[equity_df.index.year == year]
        if len(yr) < 5:
            continue
        rot_ret = (yr["portfolio_$"].iloc[-1] /
                   yr["portfolio_$"].iloc[0] - 1) * 100
        tlt_d = (yr["tlt_pos"] > 0).sum()
        xle_d = (yr["xle_pos"] > 0).sum()

        gdx_only_yr = ""
        if gdx_only_equity is not None:
            ge_yr = gdx_only_equity[gdx_only_equity.index.year == year].dropna()
            if len(ge_yr) > 5:
                g_ret = (ge_yr.iloc[-1] / ge_yr.iloc[0] - 1) * 100
                gdx_only_yr = f"{g_ret:>+8.1f}%"

        print(f"  {year:<6} {rot_ret:>+8.1f}%  {gdx_only_yr:>10} "
              f"{tlt_d:>7}d   {xle_d:>5}d")
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"AURUM·AI Rotation Backtest Summary\n")
        f.write(f"{'='*40}\n")
        for k, v in rot_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\n  ✓ Summary → {summary_path}")

    # Chart
    print(f"\n  Generating chart...")
    plot_rotation_results(equity_df, prices_aligned, gdx_only_equity, RCFG)

    print(f"\n{'═'*60}")
    print(f"  ✓ Done  →  {OUT_DIR}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()