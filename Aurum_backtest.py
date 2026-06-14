"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AURUM·AI  —  aurum_backtest.py  (v3)                               ║
║                                                                              ║
║  Simulates trading GDX using predictions from gold_miner_trainer.py.       ║
║                                                                              ║
║  INPUT                                                                       ║
║    oof_predictions.csv   — honest out-of-fold predictions (344 days)       ║
║    full_predictions.csv  — full history predictions (3,300+ days)          ║
║    Both saved by gold_miner_trainer.py in aurum_output/                    ║
║                                                                              ║
║  STRATEGY                                                                   ║
║    Mode: cash_long — long GDX when bullish, cash when flat/bearish.        ║
║    Position sized by confidence percentile (25/50/75/100% of capital).     ║
║    Weekly rebalance (every 5 trading days). 8% hard stop loss.             ║
║    1-day execution lag — trades at next-day open, not signal close.        ║
║                                                                              ║
║  OUTPUTS                                                                    ║
║    backtest_results.png     — 7-panel chart (equity, drawdown, positions)  ║
║    gdx_signal_chart.png     — candlestick chart with buy/sell markers      ║
║    rebalance_log.csv        — every position change with reason/cost       ║
║    sensitivity_analysis.csv — results across 8 parameter configurations   ║
║                                                                              ║
║  RUN                                                                        ║
║    python aurum_backtest.py                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR      = "aurum_output"
BACKTEST_DIR = "aurum_backtest"
os.makedirs(BACKTEST_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BCFG = {
    # ── Signal thresholds ─────────────────────────────────────────────────
    # Minimum predicted log return magnitude to take any position.
    "min_signal":           0.008,    # ~0.8% predicted move minimum (scaled for 20d)

    # ── Position sizing ───────────────────────────────────────────────────
    # Cash_long mode: when bullish, invest full capital into GDX.
    # No shorting means no margin requirement — full 100% allocation is safe.
    "max_position_frac":    1.00,

    # Confidence tiers — maps prediction percentile to position size fraction.
    "conf_tiers": [
        (0.00, 0.25, 0.25),   # bottom quartile → 25% of max position
        (0.25, 0.50, 0.50),   # second quartile → 50%
        (0.50, 0.75, 0.75),   # third quartile  → 75%
        (0.75, 1.01, 1.00),   # top quartile    → 100% of max
    ],

    # ── Direction mode ────────────────────────────────────────────────────
    # "long_short" — go long or short based on signal direction
    # "long_only"  — only go long, otherwise hold cash
    # "cash_long"  — long when bullish, zero return when bearish ← ACTIVE
    # "cash_bonds" — long GDX when bullish, long TLT when bearish
    "mode":                 "cash_long",

    # ── Rebalancing ───────────────────────────────────────────────────────
    # Signal checked every 10 trading days (fortnightly — matches forward_days=20).
    "rebal_on_change":      True,
    "rebal_threshold":      0.25,
    "hold_days":            20,       # forward prediction horizon (matches trainer)
    "rebal_freq":           10,       # rebalance every N trading days (fortnightly)

    # ── Stop loss ─────────────────────────────────────────────────────────
    # Exit if GDX falls more than this from entry price.
    "stop_loss_pct":        0.08,     # 8% hard stop loss

    # ── Execution timing ──────────────────────────────────────────────────
    # Signal generated from close data → earliest execution is next day open.
    # execution_lag_days=1 shifts actual returns by 1 day so P&L is measured
    # from next-day open rather than same-day close. More realistic.
    # 0 = trade at close (theoretical), 1 = trade at next day open (realistic)
    "execution_lag_days":   1,

    # ── Prediction source ─────────────────────────────────────────────────
    # "oof_only" — honest 344-day OOF predictions only
    # "full"     — all 3,358 days (OOF + final model fill)
    "pred_source":          "full",

    # ── Costs ─────────────────────────────────────────────────────────────
    "transaction_cost":     0.001,    # 0.10% round-trip
    "slippage":             0.0005,   # 0.05%

    # ── Capital ───────────────────────────────────────────────────────────
    "initial_capital":      100_000,

    # ── Benchmark ─────────────────────────────────────────────────────────
    "run_benchmark":        True,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD OOF PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_oof_predictions():
    oof_path = os.path.join(OUT_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        print(f"  ⚠ {oof_path} not found — run gold_miner_trainer.py first")
        return None
    df = pd.read_csv(oof_path, parse_dates=["date"], index_col="date")
    df = df.dropna()
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"  ✓ OOF predictions: {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Pred range: {df['pred'].min():.4f} → {df['pred'].max():.4f}  "
          f"mean: {df['pred'].mean():.4f}")
    return df


def load_full_predictions(oof_df):
    """
    Load full-history predictions saved by the trainer.
    Falls back to OOF-only if file not found.

    The full predictions CSV covers ALL trading days (not just fold val periods)
    using the final trained model. OOF predictions are used where available
    (they're honest — model never saw that data), and the final model fills gaps.
    """
    full_path = os.path.join(OUT_DIR, "full_predictions.csv")
    if not os.path.exists(full_path):
        print(f"  ⚠ full_predictions.csv not found")
        print(f"    Using OOF predictions only ({len(oof_df)} days)")
        print(f"    To get full coverage: retrain gold_miner_trainer.py")
        return oof_df

    df = pd.read_csv(full_path, parse_dates=["date"], index_col="date")
    df = df.dropna()
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Mark which predictions are honest OOF vs model-filled
    df["is_oof"] = df.index.isin(oof_df.index)

    oof_count  = df["is_oof"].sum()
    fill_count = (~df["is_oof"]).sum()
    print(f"  ✓ Full predictions: {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"    OOF (honest):  {oof_count} days")
    print(f"    Model-filled:  {fill_count} days")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_signals(pred_df, bcfg):
    """
    Convert predictions into daily position signals.
    Confidence = percentile rank within the full prediction distribution.
    """
    preds    = pred_df["pred"].copy()
    conf_pct = preds.abs().rank(pct=True)
    min_sig  = bcfg.get("min_signal", 0.005)

    raw_direction = np.where(preds.abs() < min_sig, 0, np.sign(preds))

    direction = raw_direction.copy()
    size_frac = np.zeros(len(preds))
    for (lo, hi, sz) in bcfg["conf_tiers"]:
        mask = (conf_pct >= lo) & (conf_pct < hi)
        size_frac[mask] = sz
    size_frac[direction == 0] = 0.0

    mode = bcfg.get("mode", "long_short")
    if mode in ("long_only", "cash_long", "cash_bonds"):
        direction = np.where(direction < 0, 0, direction)
        size_frac = np.where(direction == 0, 0, size_frac)

    position = direction * size_frac * bcfg["max_position_frac"]

    return pd.DataFrame({
        "pred":          preds,
        "actual":        pred_df["actual"],
        "direction":     direction.astype(int),
        "raw_direction": raw_direction.astype(int),   # pre-cash_long conversion
        "conf_pct":      conf_pct,
        "size_frac":     size_frac,
        "position":      position,
    }, index=pred_df.index)


# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(pred_df, bcfg, verbose=True):
    """
    Daily mark-to-market simulation.
    Signal checked every rebal_freq trading days (weekly).
    Hard stop loss exits if GDX drops stop_loss_pct from entry.
    """
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  BACKTEST SIMULATION  v3")
        print(f"  Mode: {bcfg['mode']}  |  "
              f"Max pos: {bcfg['max_position_frac']:.0%}  |  "
              f"Min signal: {bcfg['min_signal']:.3f}")
        print(f"  Rebalance: every {bcfg['rebal_freq']}d  |  "
              f"Stop loss: {bcfg.get('stop_loss_pct', None)}")
        print(f"  Cost: {bcfg['transaction_cost']:.2%}  |  "
              f"Slippage: {bcfg['slippage']:.2%}")
        print(f"  Prediction days: {len(pred_df)}")
        print(f"{'═'*60}")

    signals      = build_signals(pred_df, bcfg)
    capital      = float(bcfg["initial_capital"])
    tc           = bcfg["transaction_cost"] + bcfg["slippage"]
    hold_days    = bcfg["hold_days"]
    rebal_freq   = bcfg.get("rebal_freq", 5)
    rebal_thr    = bcfg.get("rebal_threshold", 0.25)
    stop_loss    = bcfg.get("stop_loss_pct", None)
    trend_on     = bcfg.get("trend_filter", False)
    trend_strength = bcfg.get("trend_strength", "hard")

    # ── Load GDX MA for trend filter ──────────────────────────────────────
    above_ma = np.ones(len(signals), dtype=bool)  # default: always above
    if trend_on:
        try:
            import yfinance as yf
            ma_days  = bcfg.get("trend_ma_days", 50)
            gdx_hist = yf.Ticker("GDX").history(
                start=(pred_df.index[0] - pd.DateOffset(days=ma_days * 2))
                      .strftime("%Y-%m-%d"),
                end=pred_df.index[-1].strftime("%Y-%m-%d"),
                auto_adjust=True)
            gdx_hist.index = pd.to_datetime(
                gdx_hist.index).normalize().tz_localize(None)
            gdx_close = gdx_hist["Close"]
            gdx_ma    = gdx_close.rolling(ma_days).mean()
            gdx_price = gdx_close.reindex(signals.index).ffill()
            gdx_ma    = gdx_ma.reindex(signals.index).ffill()
            above_ma  = (gdx_price > gdx_ma).fillna(False).values
            if verbose:
                print(f"  ✓ Trend filter: GDX above {ma_days}d MA "
                      f"{above_ma.mean():.0%} of days")
        except Exception as e:
            if verbose:
                print(f"  ✗ Trend filter failed: {e}")
            above_ma = np.ones(len(signals), dtype=bool)

    equity        = [capital]
    daily_rets    = []
    positions     = []
    rebalances    = []
    stop_hits     = []
    current_pos   = 0.0
    days_held     = 0
    last_size     = 0.0
    last_dir      = 0
    entry_capital = capital

    for i, (date, row) in enumerate(signals.iterrows()):
        target_pos  = float(row["position"])
        target_dir  = int(row["direction"])
        target_size = float(row["size_frac"])
        actual_ret  = float(row["actual"])
        price_above_ma = bool(above_ma[i])
        raw_dir     = int(row.get("raw_direction", target_dir))

        # ── Trend filter override ─────────────────────────────────────────
        if trend_on and price_above_ma:
            floor_pos = bcfg.get("trend_floor_pos", 0.25)
            if trend_strength == "soft":
                # Hold when bearish with low confidence
                if current_pos > 0 and raw_dir < 0 and \
                        float(row.get("conf_pct", 1.0)) < 0.40:
                    target_pos  = current_pos
                    target_dir  = last_dir
                    target_size = last_size
            elif trend_strength == "floor":
                # Keep a minimum base position when above MA
                # Full position when bullish, floor when bearish
                if raw_dir < 0 and target_pos < floor_pos:
                    target_pos  = floor_pos
                    target_dir  = 1
                    target_size = floor_pos
                # Also allow entry at floor even if currently flat
            else:  # hard — hold while above MA regardless of signal
                if current_pos > 0 and raw_dir <= 0:
                    target_pos  = current_pos
                    target_dir  = last_dir
                    target_size = last_size
        # Hard: don't enter new positions when below MA
        if trend_on and not price_above_ma and \
                trend_strength == "hard" and current_pos == 0:
            target_pos  = 0.0
            target_dir  = 0
            target_size = 0.0

        # ── Stop loss ─────────────────────────────────────────────────────
        stopped_out = False
        if stop_loss is not None and abs(current_pos) > 1e-6:
            pos_ret = (capital - entry_capital) / entry_capital
            if pos_ret < -stop_loss:
                stopped_out = True
                stop_hits.append({
                    "date":      date,
                    "position":  current_pos,
                    "loss_%":    pos_ret * 100,
                    "capital_$": capital,
                })
                target_pos = target_dir = 0
                target_size = 0.0

        # ── Rebalance decision ────────────────────────────────────────────
        weekly_rebal = (days_held >= rebal_freq)
        dir_flip     = (target_dir != last_dir
                        and target_dir != 0 and last_dir != 0)
        size_chg     = abs(target_size - last_size) >= rebal_thr
        entering     = (current_pos == 0 and target_pos != 0)
        exiting      = (current_pos != 0 and target_pos == 0)
        do_rebal     = (weekly_rebal or dir_flip or size_chg
                        or entering or exiting or stopped_out)

        # ── Apply rebalance ───────────────────────────────────────────────
        trade_cost = 0.0
        if do_rebal and abs(target_pos - current_pos) > 1e-6:
            change     = abs(target_pos - current_pos)
            trade_cost = change * tc * capital
            # Determine reason
            reason = ("stop_loss" if stopped_out  else
                      "dir_flip"  if dir_flip      else
                      "weekly"    if weekly_rebal  else
                      "entering"  if entering      else
                      "exiting"   if exiting       else
                      "size_chg")

            # Trade type
            if entering:
                trade_type = "BUY"
            elif exiting or stopped_out:
                trade_type = "SELL"
            elif target_pos > current_pos:
                trade_type = "ADD"
            elif target_pos < current_pos:
                trade_type = "TRIM"
            else:
                trade_type = "REBAL"

            # Signal that drove the trade
            pred_val   = float(row["pred"])
            conf_pct   = float(row["conf_pct"])
            pred_pct   = (pred_val) * 100   # predicted return as %

            # Why this position size — explain the tier
            if target_pos == 0:
                size_reason = "flat — signal below threshold or bearish"
            else:
                tier_pct = target_size * 100
                if conf_pct >= 0.75:
                    tier_label = "top quartile"
                elif conf_pct >= 0.50:
                    tier_label = "3rd quartile"
                elif conf_pct >= 0.25:
                    tier_label = "2nd quartile"
                else:
                    tier_label = "bottom quartile"
                size_reason = (f"{tier_label} confidence "
                               f"({conf_pct:.0%} of hist predictions)")

            # Portfolio breakdown
            stock_val = capital * target_pos
            cash_val  = capital * (1 - target_pos)

            rebalances.append({
                "date":          date,
                "trade":         trade_type,
                "old_pos_%":     round(current_pos * 100, 1),
                "new_pos_%":     round(target_pos  * 100, 1),
                "reason":        reason,
                "signal_pred_%": round(pred_pct, 2),
                "signal_conf_%": round(conf_pct * 100, 1),
                "size_reason":   size_reason,
                "stock_val_$":   round(stock_val, 2),
                "cash_val_$":    round(cash_val,  2),
                "portfolio_$":   round(capital,   2),
                "cost_$":        round(trade_cost, 2),
            })
            # Update position AFTER logging old_pos
            current_pos   = target_pos
            last_dir      = target_dir
            last_size     = target_size
            days_held     = 0
            entry_capital = capital

        # ── Daily P&L ─────────────────────────────────────────────────────
        bond_daily = np.log(1 + 0.04) / 252
        if current_pos == 0 and bcfg.get("mode") == "cash_bonds":
            daily_log = bond_daily
        else:
            daily_log = current_pos * (actual_ret / hold_days)

        daily_pnl  = capital * (np.exp(daily_log) - 1) - trade_cost
        capital   += daily_pnl
        capital    = max(capital, 1.0)
        days_held += 1

        equity.append(capital)
        daily_rets.append(daily_log)
        positions.append(current_pos)

    equity     = np.array(equity)
    daily_rets = np.array(daily_rets)
    positions  = np.array(positions)

    results_df = pd.DataFrame({
        "capital":   equity[1:],
        "daily_ret": daily_rets,
        "position":  positions,
        "signal":    signals["pred"].values,
        "conf_pct":  signals["conf_pct"].values,
    }, index=signals.index)

    rebal_df = pd.DataFrame(rebalances) if rebalances else pd.DataFrame()
    stop_df  = pd.DataFrame(stop_hits)  if stop_hits  else pd.DataFrame()

    if verbose and len(stop_df) > 0:
        print(f"  Stop losses triggered: {len(stop_df)}")

    # ── Metrics ───────────────────────────────────────────────────────────
    ann       = 252
    total_ret = (equity[-1] / equity[0] - 1) * 100
    cal_years = (signals.index[-1] - signals.index[0]).days / 365.25
    cal_years = max(cal_years, 0.01)
    ann_ret   = (np.exp(np.log(equity[-1] / equity[0]) / cal_years) - 1) * 100
    vol       = daily_rets.std() * np.sqrt(ann) * 100
    sharpe    = daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(ann)
    peak      = np.maximum.accumulate(equity)
    max_dd    = ((equity - peak) / peak).min() * 100

    long_d  = (positions > 0).sum()
    short_d = (positions < 0).sum()
    flat_d  = (positions == 0).sum()
    total_d = len(positions)

    active   = daily_rets[positions != 0]
    wins     = (active > 0).sum()
    losses   = (active <= 0).sum()
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_win  = active[active > 0].mean()  * 100 if wins   > 0 else 0
    avg_loss = active[active <= 0].mean() * 100 if losses > 0 else 0
    pf       = (active[active > 0].sum() /
                (abs(active[active <= 0].sum()) + 1e-9))

    metrics = {
        "total_return_%":   total_ret,
        "annual_return_%":  ann_ret,
        "cal_years":        cal_years,
        "volatility_%":     vol,
        "sharpe":           sharpe,
        "max_drawdown_%":   max_dd,
        "n_rebalances":     len(rebal_df),
        "n_stop_losses":    len(stop_df),
        "long_days_%":      long_d  / total_d * 100,
        "short_days_%":     short_d / total_d * 100,
        "flat_days_%":      flat_d  / total_d * 100,
        "win_rate_%":       win_rate,
        "avg_win_%":        avg_win,
        "avg_loss_%":       avg_loss,
        "profit_factor":    pf,
        "final_capital_$":  equity[-1],
        "total_days":       total_d,
    }
    return results_df, rebal_df, stop_df, metrics

def run_oof_vs_nonoof_bootstrap(pred_df, bcfg, n_boot=300, n_perm=1000, seed=42):
    """OOF blocks vs bootstrap of Non-OOF chunks — generalization test.

    Identifies the 8 OOF validation blocks, then bootstraps N random
    non-OOF chunks of equivalent size from training-only regions.
    Compares per-chunk IC, win-rate, return distributions to determine
    whether OOF performance is statistically distinguishable from
    non-OOF (= generalizes) or worse (= overfit).

    Returns dict of metrics + a 'verdict' string.
    """
    if "is_oof" not in pred_df.columns:
        return None  # OOF-only fallback path — can't compute this test
    rng = np.random.default_rng(seed)

    full = pred_df.copy().sort_index()
    oof_idx = full[full["is_oof"]].index.sort_values()
    if len(oof_idx) < 30:
        return None

    # Identify OOF blocks (contiguous runs separated by >30-day gaps)
    diffs = oof_idx.to_series().diff().dt.days.fillna(0)
    block_id = (diffs > 30).cumsum().values
    oof_blocks = []
    for b in np.unique(block_id):
        m = block_id == b
        dates = oof_idx[m]
        oof_blocks.append({"start": dates[0], "end": dates[-1], "n": len(dates)})

    if len(oof_blocks) < 4:
        return None

    median_n = int(np.median([b["n"] for b in oof_blocks]))
    hold = int(bcfg.get("hold_days", 20))
    min_sig = float(bcfg.get("min_signal", 0.008))
    non_oof = full[~full["is_oof"]]

    def _simulate(chunk):
        chunk = chunk.sort_index()
        if len(chunk) < 5:
            return None
        cap, nxt = 1.0, None
        rets = []
        for d, row in chunk.iterrows():
            if nxt is not None and d < nxt:
                continue
            p, a = float(row["pred"]), float(row["actual"])
            if abs(p) < min_sig:
                continue
            pos = 1.0 if p > 0 else -1.0
            cap *= float(np.exp(pos * a))
            rets.append(pos * a)
            nxt = d + pd.tseries.offsets.BDay(hold)
        if not rets:
            return None
        tr = np.array(rets)
        if chunk["pred"].std() > 0 and chunk["actual"].std() > 0:
            ic = float(np.corrcoef(chunk["pred"], chunk["actual"])[0, 1])
        else:
            ic = 0.0
        return {"ic": ic, "win_rate": float((tr > 0).mean()),
                "pct_return": float((cap - 1) * 100), "n_trades": len(tr)}

    # Score the OOF blocks
    oof_results = []
    for b in oof_blocks:
        mask = (full.index >= b["start"]) & (full.index <= b["end"]) & full["is_oof"]
        r = _simulate(full[mask])
        if r is not None:
            oof_results.append(r)
    if len(oof_results) < 3:
        return None
    oof_df = pd.DataFrame(oof_results)

    # Bootstrap non-OOF chunks
    non_oof_idx = non_oof.index
    if len(non_oof_idx) < median_n + 1:
        return None
    boot_results = []
    attempts = 0
    while len(boot_results) < n_boot and attempts < n_boot * 6:
        attempts += 1
        s = rng.integers(0, len(non_oof_idx) - median_n)
        cd = non_oof_idx[s:s + median_n]
        cs, ce = cd[0], cd[-1]
        if any((b["start"] <= ce) and (b["end"] >= cs) for b in oof_blocks):
            continue
        r = _simulate(non_oof.loc[cd])
        if r is not None:
            boot_results.append(r)
    if len(boot_results) < 30:
        return None
    boot_df = pd.DataFrame(boot_results)

    # Compute per-metric stats: OOF percentile, perm-test p-value
    out = {"n_oof_blocks": len(oof_df), "n_boot_chunks": len(boot_df)}
    sig_count = 0
    for col in ["ic", "win_rate", "pct_return"]:
        o_mean = oof_df[col].mean()
        n_mean = boot_df[col].mean()
        pctile = float((boot_df[col] < o_mean).mean() * 100)
        obs = o_mean - n_mean
        pool = np.concatenate([oof_df[col].values, boot_df[col].values])
        n_oof = len(oof_df)
        perm = []
        for _ in range(n_perm):
            rng.shuffle(pool)
            perm.append(pool[:n_oof].mean() - pool[n_oof:].mean())
        p = float((np.abs(np.array(perm)) >= abs(obs)).mean())
        out[f"{col}_oof_mean"] = float(o_mean)
        out[f"{col}_nonoof_mean"] = float(n_mean)
        out[f"{col}_oof_pctile"] = pctile
        out[f"{col}_perm_p"] = p
        if p < 0.05:
            sig_count += 1
    out["n_sig_flags"] = sig_count
    if sig_count == 0:
        out["verdict"] = "GENERALIZES — no significant OOF gap"
    elif sig_count == 1:
        out["verdict"] = "MARGINAL — 1 metric shows generalization gap"
    else:
        out["verdict"] = "OVERFIT — multiple metrics differ significantly"
    return out

def run_oof_only_validation(pred_df, bcfg, ticker_label="(target)"):
    """Cash-realistic backtest on OOF days only — non-overlapping trades.

    Walks the OOF predictions chronologically.  Enters a trade only when
    no previous trade is still open (free_date condition).  Each trade is
    held for hold_days and earns the realized `actual` log return.  Strategy
    is in cash between trades.  This caps time-in-market at 100% and gives
    a realistic per-trade equity curve for honest validation.

    Notes:
      - `actual` is the standard forward log return (post v9 trainer fix)
      - Long-only modes (cash_long / long_only) skip negative-pred days
      - Returns the same metric keys as before for backward-compatibility
    """
    if "is_oof" in pred_df.columns:
        oof = pred_df[pred_df["is_oof"] == True].copy()
    else:
        oof = pred_df.copy()
    oof = oof.sort_index()
    if len(oof) < 5:
        return None

    hold_days  = int(bcfg.get("hold_days", 20))
    min_signal = float(bcfg.get("min_signal", 0.008))
    capital_0  = float(bcfg["initial_capital"])
    mode       = bcfg.get("mode", "long_short")

    capital      = capital_0
    next_free    = None
    n_trades     = 0
    trade_rets   = []
    trade_dates  = []

    for date, row in oof.iterrows():
        if next_free is not None and date < next_free:
            continue
        pred   = float(row["pred"])
        actual = float(row["actual"])
        if abs(pred) < min_signal:
            continue
        if pred > 0:
            pos = 1.0
        else:
            if mode in ("cash_long", "long_only"):
                continue   # long-only: skip bearish signals (stay in cash)
            pos = -1.0

        trade_log_ret = pos * actual    # actual is forward hold_days log return
        capital *= float(np.exp(trade_log_ret))
        trade_rets.append(trade_log_ret)
        trade_dates.append(date)
        n_trades += 1
        next_free = date + pd.tseries.offsets.BDay(hold_days)

    if n_trades < 2:
        return None

    years = (oof.index[-1] - oof.index[0]).days / 365.25
    years = max(years, 0.01)
    tr    = np.array(trade_rets)

    # Time-in-market
    tim_days       = n_trades * hold_days
    total_oof_days = max((oof.index[-1] - oof.index[0]).days * 252 / 365.25, 1)
    tim_pct        = min(tim_days / total_oof_days, 1.0) * 100

    total_ret = (capital / capital_0 - 1) * 100
    ann_ret   = (np.exp(np.log(max(capital / capital_0, 1e-9)) / years) - 1) * 100

    trades_per_year = n_trades / years
    if tr.std() > 0:
        sharpe = tr.mean() / tr.std() * float(np.sqrt(max(trades_per_year, 1.0)))
    else:
        sharpe = 0.0

    # Equity path for MaxDD
    eq = np.concatenate([[capital_0],
                         capital_0 * np.exp(np.cumsum(tr))])
    peak    = np.maximum.accumulate(eq)
    max_dd  = ((eq - peak) / peak).min() * 100
    win_rate = float((tr > 0).mean())

    # IC computed on ALL OOF rows (not just traded ones)
    p_all = oof["pred"].values
    a_all = oof["actual"].values
    if p_all.std() > 0 and a_all.std() > 0:
        ic = float(np.corrcoef(p_all, a_all)[0, 1])
    else:
        ic = 0.0

    return {
        "n_predictions":    int(len(oof)),
        "n_trades":         int(n_trades),
        "total_return_%":   float(total_ret),
        "annual_return_%":  float(ann_ret),
        "sharpe":           float(sharpe),
        "max_drawdown_%":   float(max_dd),
        "win_rate":         float(win_rate),
        "trades_per_year":  float(trades_per_year),
        "ic":               float(ic),
        "final_capital_$":  float(capital),
        "time_in_market_%": float(tim_pct),
    }


def run_benchmark(pred_df, bcfg):
    """Honest Buy-and-Hold equity curve from actual GDX prices.

    Replaces the v3 implementation that compounded the trainer's sign-inverted
    forward-return column.  Now fetches yfinance OHLCV for the prediction
    date range and computes B&H equity directly from daily close prices.
    """
    import yfinance as yf
    capital_0 = float(bcfg["initial_capital"])
    start = (pred_df.index[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (pred_df.index[-1] + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker("GDX").history(start=start, end=end,
                                              auto_adjust=True)
    except Exception as e:
        print(f"  ⚠ B&H price fetch failed: {e}  — using flat baseline")
        hist = None

    if hist is None or len(hist) < 10:
        flat = pd.Series(capital_0, index=pred_df.index)
        bh_df = pd.DataFrame({"capital": flat.values}, index=pred_df.index)
        return bh_df, {
            "total_return_%":   0.0,
            "annual_return_%":  0.0,
            "sharpe":           0.0,
            "max_drawdown_%":   0.0,
            "final_capital_$":  capital_0,
            "cal_years":        max((pred_df.index[-1] - pred_df.index[0]).days/365.25, 0.01),
        }

    px = hist["Close"].copy()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    p_aligned = px.reindex(pred_df.index, method="ffill").bfill().astype(float)
    equity = (p_aligned / p_aligned.iloc[0]) * capital_0

    dr        = np.log(equity / equity.shift(1)).dropna().values
    ann       = 252
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cal_years = max((pred_df.index[-1] - pred_df.index[0]).days / 365.25, 0.01)
    ann_ret   = (np.exp(np.log(equity.iloc[-1] / equity.iloc[0]) / cal_years) - 1) * 100
    bh_sharpe = dr.mean() / (dr.std() + 1e-9) * np.sqrt(ann) if len(dr) > 1 else 0.0
    peak      = equity.cummax()
    max_dd    = ((equity - peak) / peak).min() * 100
    bh_df     = pd.DataFrame({"capital": equity.values}, index=pred_df.index)
    return bh_df, {
        "total_return_%":   total_ret,
        "annual_return_%":  ann_ret,
        "sharpe":           bh_sharpe,
        "max_drawdown_%":   max_dd,
        "final_capital_$":  float(equity.iloc[-1]),
        "cal_years":        cal_years,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_results(metrics, bh_metrics=None, bcfg=None):
    print(f"\n{'═'*60}")
    print(f"  BACKTEST RESULTS  v3")
    print(f"{'═'*60}")
    cal_years = metrics.get("cal_years", 0)
    print(f"\n  ── Portfolio Performance ─────────────────────────")
    print(f"  Period            : {cal_years:.1f} calendar years")
    print(f"  Initial capital   : ${bcfg['initial_capital']:>12,.0f}")
    print(f"  Final capital     : ${metrics['final_capital_$']:>12,.0f}")
    print(f"  Total return      : {metrics['total_return_%']:>+10.2f}%  "
          f"(over {cal_years:.1f} yrs)")
    print(f"  Annual return     : {metrics['annual_return_%']:>+10.2f}%  "
          f"(calendar-time CAGR)")
    print(f"  Volatility        : {metrics['volatility_%']:>10.2f}%  (annualised daily)")
    print(f"  Sharpe ratio      : {metrics['sharpe']:>10.3f}")
    print(f"  Max drawdown      : {metrics['max_drawdown_%']:>10.2f}%")
    if bh_metrics:
        alpha = metrics['total_return_%'] - bh_metrics['total_return_%']
        print(f"\n  ── vs Buy & Hold GDX ────────────────────────────")
        print(f"  B&H total return  : {bh_metrics['total_return_%']:>+10.2f}%")
        print(f"  B&H annual return : {bh_metrics.get('annual_return_%', 0):>+10.2f}%  (CAGR)")
        print(f"  B&H Sharpe        : {bh_metrics['sharpe']:>10.3f}")
        print(f"  B&H max drawdown  : {bh_metrics['max_drawdown_%']:>10.2f}%")
        print(f"  Alpha (excess)    : {alpha:>+10.2f}%")
    print(f"\n  ── Position Statistics ──────────────────────────")
    print(f"  Max position size : {bcfg['max_position_frac']:.0%}  of portfolio")
    print(f"  Days long         : {metrics['long_days_%']:>10.1f}%")
    print(f"  Days short        : {metrics['short_days_%']:>10.1f}%")
    print(f"  Days flat         : {metrics['flat_days_%']:>10.1f}%")
    print(f"  Rebalances        : {metrics['n_rebalances']:>10}")
    print(f"  Stop losses hit   : {metrics.get('n_stop_losses', 0):>10}")
    print(f"\n  ── Signal Quality ───────────────────────────────")
    print(f"  Win rate          : {metrics['win_rate_%']:>10.1f}%  (active days)")
    print(f"  Avg win           : {metrics['avg_win_%']:>+10.3f}%  (daily)")
    print(f"  Avg loss          : {metrics['avg_loss_%']:>+10.3f}%  (daily)")
    print(f"  Profit factor     : {metrics['profit_factor']:>10.3f}  "
          f"(>1.5 = good, >2.0 = strong)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOT
# ══════════════════════════════════════════════════════════════════════════════

DARK  = "#0a0a08"
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


def plot_backtest(results_df, bh_df, rebal_df, metrics, bh_metrics, bcfg):
    fig = plt.figure(figsize=(22, 14), facecolor=DARK)
    fig.suptitle(
        f"AURUM·AI — Backtest  "
        f"(Max pos {bcfg['max_position_frac']:.0%}  |  "
        f"{bcfg['mode']}  |  Cost {bcfg['transaction_cost']:.2%})",
        color=GOLD, fontsize=12, y=0.99, fontweight="bold")
    gs = plt.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    ax1.plot(results_df.index, results_df["capital"],
             color=GOLD, linewidth=2, label="Strategy", zorder=3)
    if bh_df is not None:
        ax1.plot(bh_df.index, bh_df["capital"],
                 color=BLUE, linewidth=1.2, linestyle="--",
                 label="Buy & Hold GDX", zorder=2)
    ax1.axhline(bcfg["initial_capital"], color=MUTED, linewidth=0.8, linestyle=":")
    ax1.fill_between(results_df.index, results_df["capital"],
                     bcfg["initial_capital"],
                     where=results_df["capital"] >= bcfg["initial_capital"],
                     alpha=0.08, color=GREEN)
    ax1.fill_between(results_df.index, results_df["capital"],
                     bcfg["initial_capital"],
                     where=results_df["capital"] < bcfg["initial_capital"],
                     alpha=0.08, color=RED)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("Portfolio Equity Curve", color=GOLD, fontsize=10)
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED)
    ax1.set_ylabel("Portfolio Value ($)", color=MUTED, fontsize=8)
    ax1.annotate(f"${results_df['capital'].iloc[-1]:,.0f}",
                 xy=(results_df.index[-1], results_df["capital"].iloc[-1]),
                 color=GOLD, fontsize=8,
                 xytext=(-65, 6), textcoords="offset points")
    if bh_df is not None:
        ax1.annotate(f"${bh_df['capital'].iloc[-1]:,.0f}",
                     xy=(bh_df.index[-1], bh_df["capital"].iloc[-1]),
                     color=BLUE, fontsize=8,
                     xytext=(-65, -14), textcoords="offset points")

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    eq   = results_df["capital"].values
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    ax2.fill_between(results_df.index, dd, 0, alpha=0.7, color=RED)
    ax2.plot(results_df.index, dd, color=RED, linewidth=0.8)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_title(f"Drawdown  (max {metrics['max_drawdown_%']:.1f}%)",
                  color=GOLD, fontsize=10)
    ax2.set_ylabel("Drawdown %", color=MUTED, fontsize=8)

    # 3. Position over time
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3)
    pos = results_df["position"]
    ax3.fill_between(results_df.index, pos * 100, 0,
                     where=pos > 0,  color=GREEN, alpha=0.6, label="Long")
    ax3.fill_between(results_df.index, pos * 100, 0,
                     where=pos < 0,  color=RED,   alpha=0.6, label="Short")
    ax3.fill_between(results_df.index, pos * 100, 0,
                     where=pos == 0, color=MUTED,  alpha=0.15, label="Flat")
    ax3.axhline(0, color=MUTED, linewidth=0.5)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.set_title("Position Size Over Time  (% of portfolio)",
                  color=GOLD, fontsize=10)
    ax3.set_ylabel("Position %", color=MUTED, fontsize=8)
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # 4. Daily return distribution
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4)
    dr   = results_df["daily_ret"].values * 100
    dr_a = dr[results_df["position"].values != 0]
    if len(dr_a) > 10:
        bins = np.linspace(np.percentile(dr_a, 1),
                           np.percentile(dr_a, 99), 40)
        ax4.hist(dr_a[dr_a > 0],  bins=bins, color=GREEN, alpha=0.7, label="Win")
        ax4.hist(dr_a[dr_a <= 0], bins=bins, color=RED,   alpha=0.7, label="Loss")
        ax4.axvline(0,            color=MUTED, linewidth=1)
        ax4.axvline(dr_a.mean(),  color=GOLD,  linewidth=1.5, linestyle="--",
                    label=f"Mean {dr_a.mean():.3f}%")
    ax4.set_title("Daily Return Distribution  (active days)",
                  color=GOLD, fontsize=10)
    ax4.set_xlabel("Daily return %", color=MUTED, fontsize=8)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # 5. Confidence vs return
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5)
    mask = results_df["position"] != 0
    if mask.sum() > 10:
        cv = results_df.loc[mask, "conf_pct"].values
        av = results_df.loc[mask, "daily_ret"].values * 100
        pv = results_df.loc[mask, "position"].values
        ax5.scatter(cv, av, c=np.sign(pv),
                    cmap="RdYlGn", alpha=0.35, s=10, vmin=-1, vmax=1)
        ax5.axhline(0, color=MUTED, linewidth=0.5)
        # Bin means
        bins = np.linspace(0, 1, 11)
        bm, bc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (cv >= lo) & (cv < hi)
            if m.sum() > 2:
                bm.append(av[m].mean())
                bc.append((lo + hi) / 2)
        ax5.plot(bc, bm, color=GOLD, linewidth=2,
                 marker="o", markersize=5, zorder=5)
    ax5.set_xlabel("Confidence percentile", color=MUTED, fontsize=8)
    ax5.set_ylabel("Daily return %", color=MUTED, fontsize=8)
    ax5.set_title("Confidence vs Return\n(gold = bin mean)",
                  color=GOLD, fontsize=9)

    # 6. Rolling Sharpe
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6)
    dr_s = results_df["daily_ret"]
    if len(dr_s) >= 63:
        rs = dr_s.rolling(63).apply(
            lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(252), raw=True)
        ax6.plot(results_df.index, rs, color=GOLD, linewidth=1.5)
        ax6.axhline(0,   color=MUTED, linewidth=0.5, linestyle="--")
        ax6.axhline(1.0, color=GREEN, linewidth=0.8, linestyle=":",  label="1.0")
        ax6.axhline(2.0, color=GREEN, linewidth=0.8, linestyle="-.", label="2.0")
        ax6.fill_between(results_df.index, rs, 0,
                         where=rs >= 0, alpha=0.1, color=GREEN)
        ax6.fill_between(results_df.index, rs, 0,
                         where=rs < 0,  alpha=0.1, color=RED)
        ax6.xaxis.set_major_locator(mdates.YearLocator())
        ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax6.set_title("Rolling Sharpe  (63-day)", color=GOLD, fontsize=10)
    ax6.set_ylabel("Sharpe", color=MUTED, fontsize=8)
    ax6.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)

    # 7. Metrics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(PANEL)
    ax7.axis("off")
    rows = [
        ("Total Return",  f"{metrics['total_return_%']:+.1f}%"),
        ("Annual Return", f"{metrics['annual_return_%']:+.1f}%"),
        ("Volatility",    f"{metrics['volatility_%']:.1f}%"),
        ("Sharpe",        f"{metrics['sharpe']:.3f}"),
        ("Max Drawdown",  f"{metrics['max_drawdown_%']:.1f}%"),
        ("Win Rate",      f"{metrics['win_rate_%']:.1f}%"),
        ("Profit Factor", f"{metrics['profit_factor']:.2f}"),
        ("Days Long",     f"{metrics['long_days_%']:.1f}%"),
        ("Days Short",    f"{metrics['short_days_%']:.1f}%"),
        ("Days Flat",     f"{metrics['flat_days_%']:.1f}%"),
        ("Rebalances",    f"{metrics['n_rebalances']}"),
    ]
    if bh_metrics:
        alpha = metrics['total_return_%'] - bh_metrics['total_return_%']
        rows.append(("Alpha vs B&H", f"{alpha:+.1f}%"))

    y = 0.96
    ax7.text(0.05, y, "Performance Summary",
             color=GOLD, fontsize=9, fontweight="bold",
             transform=ax7.transAxes)
    y -= 0.09
    for label, val in rows:
        col = GREEN if (not val.startswith("-") and
                        any(c.isdigit() for c in val)) else MUTED
        ax7.text(0.05, y, label, color=MUTED,  fontsize=8,
                 transform=ax7.transAxes)
        ax7.text(0.62, y, val,   color=col,    fontsize=8,
                 fontweight="bold", transform=ax7.transAxes)
        y -= 0.075

    path = os.path.join(BACKTEST_DIR, "backtest_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"\n  ✓ Chart → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(pred_df, base_cfg):
    print(f"\n{'═'*60}")
    print(f"  SENSITIVITY ANALYSIS  — key parameter comparisons")
    print(f"{'═'*60}")
    # pred_df already has execution lag applied from main()
    print(f"\n{'═'*60}")
    print(f"  SENSITIVITY ANALYSIS")
    print(f"{'═'*60}")

    configs = [
        # label                    max_pos  min_sig  mode          cost   stop
        # min_sig scaled for forward_days=20
        ("Base (ML only)",         1.00,    0.008,   "cash_long",  0.001, 0.08),
        ("Cash/bonds",             1.00,    0.008,   "cash_bonds", 0.001, 0.08),
        ("Conservative 20%",       0.20,    0.008,   "cash_long",  0.001, 0.08),
        ("High threshold",         1.00,    0.020,   "cash_long",  0.001, 0.08),
        ("Low threshold",          1.00,    0.004,   "cash_long",  0.001, 0.08),
        ("No stop loss",           1.00,    0.008,   "cash_long",  0.001, None),
        ("High cost 0.3%",         1.00,    0.008,   "cash_long",  0.003, 0.08),
        ("Long/short ref",         0.20,    0.008,   "long_short", 0.001, 0.08),
        # Trend filter variants — tested, rejected: degrades Sharpe vs base
        ("Trend floor 10% 50d",    1.00,    0.008,   "cash_long",  0.001, 0.08),
        ("Trend soft 50d",         1.00,    0.008,   "cash_long",  0.001, 0.08),
    ]

    trend_cfg = {
        "Trend floor 10% 50d": {"trend_filter": True, "trend_ma_days": 50, "trend_strength": "floor", "trend_floor_pos": 0.10},
        "Trend soft 50d":      {"trend_filter": True, "trend_ma_days": 50, "trend_strength": "soft",  "trend_floor_pos": 0.25},
    }

    rows = []
    for label, max_pos, min_sig, mode, tc, stop in configs:
        t = trend_cfg.get(label, {"trend_filter": False,
                                   "trend_ma_days": 50,
                                   "trend_strength": "hard"})
        cfg = {**base_cfg,
               "max_position_frac": max_pos,
               "min_signal":        min_sig,
               "mode":              mode,
               "transaction_cost":  tc,
               "stop_loss_pct":     stop,
               "min_hold_days":     base_cfg.get("rebal_freq", 5),
               **t}
        _, _, _, m = run_backtest(pred_df, cfg, verbose=False)
        bh         = run_benchmark(pred_df, cfg)[1]
        alpha      = m["total_return_%"] - bh["total_return_%"]
        rows.append({
            "Config":   label,
            "Max pos":  f"{max_pos:.0%}",
            "Mode":     mode,
            "Cost":     f"{tc:.1%}",
            "Stop":     f"{stop:.0%}" if stop else "None",
            "Trend":    (f"{t['trend_strength'][:4]}"
                         f"{int(t.get('trend_floor_pos',0)*100)}%"
                         f"{t['trend_ma_days']}d"
                         if t["trend_filter"] else "off"),
            "Tot ret":  f"{m['total_return_%']:+.1f}%",
            "Ann ret":  f"{m['annual_return_%']:+.1f}%",
            "Sharpe":   f"{m['sharpe']:.3f}",
            "Max DD":   f"{m['max_drawdown_%']:.1f}%",
            "Long%":    f"{m['long_days_%']:.0f}%",
            "Win%":     f"{m['win_rate_%']:.1f}%",
            "Alpha":    f"{alpha:+.1f}%",
        })

    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")
    path = os.path.join(BACKTEST_DIR, "sensitivity_analysis.csv")
    df.to_csv(path, index=False)
    print(f"\n  ✓ Saved → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. CANDLE CHART WITH SIGNAL OVERLAY AND BUY/SELL MARKERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_candle_signal(results_df, pred_df, lookback_years=2):
    """
    GDX candlestick chart with:
      - OHLCV candles (downloaded fresh via yfinance)
      - Model signal overlay (predicted return, colour-coded)
      - Buy markers (▲) when position goes long
      - Sell/flat markers (▼) when position exits to cash
      - Volume bars at bottom
      - Confidence shading behind candles
    """
    print(f"\n  Generating candle chart (last {lookback_years} years)...")
    import yfinance as yf

    # ── Download OHLCV ────────────────────────────────────────────────────
    end_date   = results_df.index[-1]
    start_date = end_date - pd.DateOffset(years=lookback_years)
    ticker     = yf.Ticker("GDX")
    ohlcv      = ticker.history(start=start_date, end=end_date,
                                auto_adjust=True)
    if len(ohlcv) < 10:
        print(f"  ✗ Could not download GDX OHLCV data")
        return

    ohlcv.index = pd.to_datetime(ohlcv.index).normalize().tz_localize(None)
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()

    # ── Align signals to candle period ────────────────────────────────────
    sig   = pred_df[pred_df.index >= start_date].copy()
    pos   = results_df[results_df.index >= start_date]["position"].copy()

    # ── Detect buy/sell transitions ───────────────────────────────────────
    pos_aligned = pos.reindex(ohlcv.index).ffill().fillna(0)
    prev_pos    = pos_aligned.shift(1).fillna(0)

    buy_dates  = ohlcv.index[(pos_aligned > 0) & (prev_pos == 0)]
    sell_dates = ohlcv.index[(pos_aligned == 0) & (prev_pos > 0)]
    stop_dates = ohlcv.index[(pos_aligned == 0) & (prev_pos > 0) &
                              ohlcv["Low"] < ohlcv["Low"].shift(1) * 0.95]

    # ── Layout: 3 rows — candles, signal, volume ──────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 14),
        gridspec_kw={"height_ratios": [5, 2, 1.5]},
        facecolor=DARK, sharex=True)

    fig.suptitle(
        f"GDX — Signal Chart  ({start_date.strftime('%b %Y')} → "
        f"{end_date.strftime('%b %Y')})",
        color=GOLD, fontsize=13, fontweight="bold", y=0.99)

    # ── Panel 1: Candlesticks ─────────────────────────────────────────────
    ax1 = axes[0]
    style_ax(ax1)

    # Use integer positions for x-axis — eliminates weekend/holiday gaps
    # Every trading day gets an equal-width slot regardless of calendar gaps
    n     = len(ohlcv)
    xs    = np.arange(n)
    width = 0.6

    # Build a date→integer lookup for markers and shading
    date_to_x = {d: i for i, d in enumerate(ohlcv.index)}

    # Background shading — green when long
    for i in range(n):
        d = ohlcv.index[i]
        p = float(pos_aligned.get(d, 0))
        if p > 0:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=GREEN, zorder=0)

    # Draw candles using integer x positions
    for i, row in enumerate(ohlcv.itertuples()):
        o, h, l, c = row.Open, row.High, row.Low, row.Close
        colour = GREEN if c >= o else RED
        ax1.bar(i, abs(c - o), width,
                bottom=min(o, c), color=colour, alpha=0.9, zorder=2)
        ax1.plot([i, i], [l, h],
                 color=colour, linewidth=0.8, alpha=0.8, zorder=2)

    # Buy markers
    for d in buy_dates:
        if d in date_to_x:
            xi    = date_to_x[d]
            price = float(ohlcv.loc[d, "Low"]) * 0.985
            ax1.annotate("▲", xy=(xi, price),
                         color=GREEN, fontsize=11, ha="center",
                         fontweight="bold", zorder=5)

    # Sell/exit markers
    for d in sell_dates:
        if d in date_to_x:
            xi    = date_to_x[d]
            price = float(ohlcv.loc[d, "High"]) * 1.015
            ax1.annotate("▼", xy=(xi, price),
                         color=RED, fontsize=11, ha="center",
                         fontweight="bold", zorder=5)

    ax1.set_ylabel("GDX Price ($)", color=MUTED, fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax1.set_xlim(-1, n)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GREEN,
               markersize=10, label="Buy (enter long)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=RED,
               markersize=10, label="Sell (exit to cash)"),
        plt.Rectangle((0, 0), 1, 1, fc=GREEN, alpha=0.15,
                       label="In position"),
    ]
    ax1.legend(handles=legend_els, fontsize=8,
               facecolor=PANEL, labelcolor=MUTED, loc="upper left")

    # ── Panel 2: Signal overlay ───────────────────────────────────────────
    ax2 = axes[1]
    style_ax(ax2)

    sig_aligned = sig["pred"].reindex(ohlcv.index).ffill()
    sig_vals    = sig_aligned.values

    bull_mask = sig_vals > 0
    bear_mask = sig_vals <= 0
    ax2.bar(xs[bull_mask], sig_vals[bull_mask],
            color=GREEN, alpha=0.7, width=0.8)
    ax2.bar(xs[bear_mask], sig_vals[bear_mask],
            color=RED,   alpha=0.7, width=0.8)

    ax2.axhline(0,      color=MUTED, linewidth=0.8)
    ax2.axhline( 0.005, color=GOLD,  linewidth=0.6,
                linestyle=":", label="Min signal threshold")
    ax2.axhline(-0.005, color=GOLD,  linewidth=0.6, linestyle=":")

    # Smooth signal line
    smooth_vals = pd.Series(sig_vals).rolling(5, min_periods=1).mean().values
    ax2.plot(xs, smooth_vals, color=GOLD, linewidth=1.5, alpha=0.9, zorder=4)

    ax2.set_ylabel("Model Signal\n(predicted return)", color=MUTED, fontsize=8)
    ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=MUTED)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax2.set_xlim(-1, n)

    # ── Panel 3: Volume ───────────────────────────────────────────────────
    ax3 = axes[2]
    style_ax(ax3)
    vol_colours = [GREEN if c >= o else RED
                   for o, c in zip(ohlcv["Open"], ohlcv["Close"])]
    ax3.bar(xs, ohlcv["Volume"].values / 1e6,
            color=vol_colours, alpha=0.6, width=0.8)
    ax3.set_ylabel("Volume (M)", color=MUTED, fontsize=8)
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax3.set_xlim(-1, n)

    # ── X-axis: show month labels at correct integer positions ────────────
    # Find first trading day of each month and label it
    tick_positions = []
    tick_labels    = []
    prev_month     = None
    for i, d in enumerate(ohlcv.index):
        if d.month != prev_month:
            tick_positions.append(i)
            # Every 2nd month show year too, others just month
            if d.month in (1, 3, 5, 7, 9, 11) or i == 0:
                tick_labels.append(d.strftime("%b '%y"))
            else:
                tick_labels.append(d.strftime("%b"))
            prev_month = d.month

    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=35, ha="right",
                        color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(BACKTEST_DIR, "gdx_signal_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Candle chart → {path}")
    print(f"    Buy signals  : {len(buy_dates)}")
    print(f"    Sell signals : {len(sell_dates)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  AURUM·AI — Backtest Engine  v3")
    print(f"{'═'*60}")

    # Auto-detect forward_days from saved .pt files and override hold_days
    import glob as _glob
    pt_files = sorted(_glob.glob(os.path.join(os.path.abspath(OUT_DIR),
                                               "fold_*_model.pt")))
    if pt_files:
        try:
            ckpt = torch.load(pt_files[0], map_location="cpu",
                              weights_only=True)
            fwd = int(ckpt.get("cfg", {}).get("forward_days",
                                               BCFG["hold_days"]))
            if fwd != BCFG["hold_days"]:
                print(f"  ✓ Auto-detected forward_days={fwd} from model — "
                      f"overriding hold_days ({BCFG['hold_days']} → {fwd})")
                BCFG["hold_days"]  = fwd
                BCFG["rebal_freq"] = max(1, fwd // 2)
        except Exception:
            pass

    # 1. Load predictions
    oof_df = load_oof_predictions()
    if oof_df is None:
        return

    # Use full predictions if available, otherwise fall back to OOF only
    pred_source = BCFG.get("pred_source", "full")
    if pred_source == "full":
        pred_df = load_full_predictions(oof_df)
    else:
        pred_df = oof_df
        print(f"  Using OOF predictions only ({len(pred_df)} days)")

    # Apply execution lag — shift actual returns forward by N days
    # so P&L reflects trading at next-day open, not same-day close
    lag = BCFG.get("execution_lag_days", 1)
    if lag > 0:
        pred_df = pred_df.copy()
        pred_df["actual"] = pred_df["actual"].shift(-lag)
        pred_df = pred_df.dropna(subset=["actual"])
        print(f"  ✓ Execution lag: {lag} day(s)  "
              f"(P&L from next-day open)  "
              f"{len(pred_df)} rows remaining")

    # 2. Run backtest
    results_df, rebal_df, stop_df, metrics = run_backtest(pred_df, BCFG)
    bh_df, bh_metrics                      = run_benchmark(pred_df, BCFG)

    # 3. Print results
    print_results(metrics, bh_metrics, BCFG)
    # ── OOF-only validation block ─────────────────────────────────────────
    # Honest read on the strategy using only the 264-ish out-of-fold days.
    # Big divergence here vs the headline metrics = in-sample optimism.
    oof_only = run_oof_only_validation(pred_df, BCFG)
    if oof_only is not None:
        print(f"\n  ── OOF-Only Validation ─────────────────────────")
        print(f"  Honest predictions: {oof_only['n_predictions']}  "
              f"(trades taken: {oof_only['n_trades']}, "
              f"{oof_only['trades_per_year']:.1f}/yr, "
              f"time-in-market: {oof_only.get('time_in_market_%', 0):.0f}%)")
        print(f"  IC (pred vs actual): {oof_only['ic']:>+10.4f}")
        print(f"  Total return     : {oof_only['total_return_%']:>+10.2f}%")
        print(f"  Annual return    : {oof_only['annual_return_%']:>+10.2f}%  (CAGR)")
        print(f"  Sharpe           : {oof_only['sharpe']:>10.3f}")
        print(f"  Max drawdown     : {oof_only['max_drawdown_%']:>10.2f}%")
        print(f"  Win rate         : {oof_only['win_rate']:>10.1%}")
        if metrics.get('sharpe', 0) > 0:
            sharpe_ratio = oof_only['sharpe'] / metrics['sharpe']
            print(f"  Sharpe(OOF) / Sharpe(full): {sharpe_ratio:.2f}x  "
                  f"(>0.5 = generalises; <0.3 = overfit)")

    # ── OOF-vs-NonOOF Bootstrap (generalization test) ─────────────────────
    # Compares OOF blocks against bootstrap of non-OOF chunks of same size.
    # No significant difference (p>0.05) on all 3 metrics = clean generalization.
    boot = run_oof_vs_nonoof_bootstrap(pred_df, BCFG)
    if boot is not None:
        print(f"\n  ── OOF-vs-NonOOF Bootstrap ({boot['n_boot_chunks']} chunks) ──")
        print(f"  metric          OOF mean    Non-OOF mean   OOF pctile   perm p")
        for col, label in [("ic", "IC          "), ("win_rate", "Win rate    "),
                           ("pct_return", "Pct return  ")]:
            print(f"  {label}    {boot[f'{col}_oof_mean']:+8.4f}     "
                  f"{boot[f'{col}_nonoof_mean']:+8.4f}      "
                  f"{boot[f'{col}_oof_pctile']:5.1f}%    {boot[f'{col}_perm_p']:.3f}")
        verdict_mark = {"GENERALIZES — no significant OOF gap": "✓",
                        "MARGINAL — 1 metric shows generalization gap": "⚠",
                        "OVERFIT — multiple metrics differ significantly": "✗"}.get(boot["verdict"], "?")
        print(f"  Verdict: {verdict_mark} {boot['verdict']}")



    # 4. Save logs
    if len(rebal_df) > 0:
        path = os.path.join(BACKTEST_DIR, "rebalance_log.csv")
        rebal_df.to_csv(path, index=False)
        print(f"\n  ✓ Rebalance log → {path}  ({len(rebal_df)} events)")

    if len(stop_df) > 0:
        path = os.path.join(BACKTEST_DIR, "stop_loss_log.csv")
        stop_df.to_csv(path, index=False)
        print(f"  ✓ Stop loss log → {path}  ({len(stop_df)} events)")

    # Save daily equity curve — used by aurum_signal.py to show portfolio value
    equity_path = os.path.join(BACKTEST_DIR, "daily_equity.csv")
    equity_df = results_df[["capital", "position"]].copy()
    equity_df.index.name = "date"
    equity_df.columns    = ["portfolio_$", "position"]
    equity_df["stock_$"] = equity_df["portfolio_$"] * equity_df["position"]
    equity_df["cash_$"]  = equity_df["portfolio_$"] * (1 - equity_df["position"])
    equity_df.to_csv(equity_path)
    print(f"  ✓ Daily equity → {equity_path}  ({len(equity_df)} rows)")

    # 5. Plot backtest summary
    try:
        plot_backtest(results_df, bh_df, rebal_df,
                      metrics, bh_metrics, BCFG)
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    # 6. Candle chart with signal overlay and buy/sell markers
    try:
        plot_candle_signal(results_df, pred_df, lookback_years=2)
    except Exception as e:
        print(f"  (Candle chart skipped: {e})")

    # 7. Sensitivity
    sensitivity_analysis(pred_df, BCFG)

    print(f"\n{'═'*60}")
    print(f"  ✓ Backtest complete")
    print(f"  Outputs in: ./{BACKTEST_DIR}/")
    print(f"    backtest_results.png     — 7-panel backtest chart")
    print(f"    gdx_signal_chart.png     — candle chart with buy/sell markers")
    print(f"    rebalance_log.csv        — position change events")
    print(f"    stop_loss_log.csv        — stop loss triggers")
    print(f"    daily_equity.csv         — daily portfolio/stock/cash values")
    print(f"    sensitivity_analysis.csv — parameter sweep")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()

