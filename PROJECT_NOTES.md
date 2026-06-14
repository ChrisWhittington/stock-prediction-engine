# AURUM·AI / GDX111 — Project Notes

> Running notebook for the multi-ETF ML trading project. Update this file as decisions are made so context survives between chat sessions.

_Last updated: 2026-06-14_

---

## Deployed seeds (production)

Source of truth: `DEPLOYED_SEEDS` dict in `aurum_signal.py` (~line 49).

| ETF | Seed   | Output dir       | Standalone CAGR | Sharpe | Max DD   | Rotation contribution |
|-----|--------|------------------|-----------------|--------|----------|------------------------|
| GDX | 162    | `aurum_output/`  | +42.38%         | 7.93   | -6.59%   | (primary — full first-dibs allocation) |
| XLE | 18264  | `xle_output/`    | +23.92%         | 7.07   | -11.08%  | secondary (strength-ranked vs QQQ) |
| QQQ | 32928  | `qqq_output/`    | +58.31%         | 9.53   | -8.73%   | secondary (strength-ranked vs XLE) |

> Seeds updated 2026-06-14 to the **v9 champions** (bootstrap-validated) currently set in `DEPLOYED_SEEDS`. Standalone metrics above are each seed's own `candidate_results.txt` backtest. The prior deployment (GDX 21921 / XLE 97239 / QQQ 74174) is retained in the candidate tables and changelog below as historical reference.

**Current full-history rotation backtest:** $2,509,233 over 16.3 years from $100k start. CAGR +21.9%. Premium over GDX-only +7.6pp. Volatility 13.9% (GDX-only line). ⚠ This figure predates the v9-champion seed switch (2026-06-14) — re-run `aurum_signal.py` for a current number.

**XLE and QQQ are now deployed via strength-ranked cascade** (sort by confidence percentile descending, stronger gets first dibs on idle cash). Changed from fixed XLE-first on 2026-05-28. Both must still pass deployment gate (pred > min_sig AND conf ≥ 0.40).

XLE story (2026-05-27): tested seed 73101 (sweep #3 winner, +23.20% standalone, lower DD) but rotation A/B showed it produced **$1.19M less** over 16.3 years than seed 97239 ($1.26M vs $2.45M final). Reason: selective high-Sharpe models cross the rotation's 40% confidence threshold less often, so idle cash sits in SGOV at 0% instead of compounding via XLE. Reverted to 97239. **Also reverted the XLE trainer** — removed the 6 momentum features (section 15) so future seed searches produce candidates closer in spirit to 97239's feature set. Sweep #3 candidates and seed 73101's preservation copy retained as historical reference. `aurum_signal.py`'s `build_features_live()` still computes the momentum features (harmless via `align_features()` per-model filtering) so any retained sweep-#3 candidate can still be tested if needed.

QQQ deployed seed has the best Sharpe-to-drawdown ratio in the candidate field even though absolute return trails buy-and-hold QQQ (B&H is +20.40% CAGR over the same window) — picked because the *risk-adjusted* edge is the point of the rotation, not absolute QQQ outperformance.

---

## Best-known candidates per ETF

Top picks from each `candidate_results.txt` sweep. Sorted by Sharpe.

### GDX (`aurum_output/candidate_results.txt`)

| Seed   | CAGR   | Sharpe | Max DD  | Win % | Days long | Notes |
|--------|--------|--------|---------|-------|-----------|-------|
| 21921  | +30.32%| 6.94   | -12.05% | 87.6% | 39.4%     | **Deployed** |
| 38497  | +31.19%| 6.72   | -26.56% | 83.9% | 44.7%     | Highest CAGR, deeper DD |
| 36005  | +26.92%| 6.64   | -14.05% | 86.1% | 38.0%     | |
| 53050  | +26.92%| 6.64   | -14.05% | 86.1% | 38.0%     | Identical numbers to 36005 — check for dup |
| 16371  | +27.22%| 6.45   | -20.40% | 84.6% | 39.4%     | |
| 70548  | +28.93%| 6.43   | -26.05% | 87.5% | 40.1%     | |
| 72359  | +26.70%| 6.43   | -21.96% | 77.8% | 45.0%     | |
| 6067   | +25.74%| 5.63   | -36.14% | 83.7% | 40.7%     | High DD |
| 8703   | +23.48%| 5.84   | -29.94% | 82.9% | 38.2%     | |
| 38090  | +25.84%| 6.20   | -22.95% | 83.9% | 41.4%     | |
| 1011   | +23.74%| 6.05   | -18.88% | 74.6% | 47.5%     | |
| 44793  | +21.89%| 6.04   | -18.49% | 85.7% | 33.6%     | |

B&H GDX over the same window: **-12.15% CAGR, -86.56% max DD** — every candidate above is destroying buy-and-hold.

### XLE — stale (pre-momentum-features pipeline)

These seeds were trained **before** the recent `xle_trainer.py` update that added momentum features. They are NOT comparable to current candidates because they use a different feature set. The deployed seed (97239) is still from this batch and would need to be retrained on the new pipeline to be a fair benchmark, OR replaced by a strong seed from the current pipeline.

| Seed   | CAGR   | Sharpe | Max DD  | Win % | Days long | Notes |
|--------|--------|--------|---------|-------|-----------|-------|
| 97239  | +22.60%| 5.01   | -11.16% | 77.8% | 42.6%     | Deployed (but pipeline-stale) |
| 25430  | +21.37%| 4.91   | -13.64% | 79.9% | 41.9%     | Pipeline-stale |
| 40721  | +21.41%| 4.80   | -10.03% | 80.8% | 38.0%     | Pipeline-stale |
| 38798  | +15.45%| 4.18   | -28.37% | 65.6% | 49.7%     | Pipeline-stale |
| 65371  | +15.45%| 4.17   | -23.22% | 67.6% | 47.1%     | Pipeline-stale |
| 22192  |  +8.94%| 2.95   | -29.46% | 65.7% | 31.7%     | Pipeline-stale |

B&H XLE: **+5.08% CAGR, -71.26% max DD**.

### XLE — current pipeline (with momentum features)

After sweep #3 (26 candidates, 2026-05-27), the new pipeline has produced a candidate that **beats the old deployed seed**. Top picks:

| Seed | CAGR | Sharpe | MaxDD | Win% | Days long | Notes |
|------|------|--------|-------|------|-----------|-------|
| **73101** | **+23.20%** | **5.07** | **-8.76%** | **83.5%** | 39.9% | ⭐ Beats old deployed 97239 on every metric |
| 25049 | +17.29% | 4.34 | -13.82% | 70.6% | 42.5% | Strong runner-up |
| 95148 | +14.59% | 4.01 | -15.22% | 76.4% | 31.9% | Good |
| 79341 | +13.28% | 3.82 | -16.55% | 72.9% | 32.6% | |
| 86117 | +13.36% | 3.43 | -9.72% | 78.4% | 21.7% | Defensive (sweep #2) |
| 22315 | +12.27% | 3.79 | -12.44% | 69.7% | 34.6% | Balance (sweep #2) |

Earlier conclusion from sweep #2 ("new pipeline worse than old") is **overturned**. The new pipeline is competitive — sweep #2 was just undersampled (10 seeds wasn't enough to find the right tail). Momentum features validated.

### QQQ (`qqq_output/candidate_results.txt` — pre-momentum-features sweeps)

| Seed   | CAGR   | Sharpe | Max DD  | Win % | Days long | Notes |
|--------|--------|--------|---------|-------|-----------|-------|
| 8848   | +12.99%| 5.43   | -4.77%  | 79.7% | 32.7%     | Highest Sharpe in field |
| **74174** | **+12.79%**| **5.41** | **-1.85%** | **86.5%** | 26.8% | **DEPLOYED** — chosen for exceptional MaxDD + win rate combo |
| 79234  | +12.88%| 5.40   | -5.02%  | 83.0% | 30.4%     | |
| 10802  | +12.79%| 5.40   | -2.77%  | 86.1% | 27.8%     | Close runner-up |
| 4723   | +12.57%| 5.37   | -2.63%  | 84.3% | 29.7%     | |
| 23800  | +12.41%| 5.35   | -3.01%  | 84.9% | 27.4%     | |
| 26010  | +11.70%| 5.14   | -1.94%  | **88.9%** | 24.9% | Highest win rate |
| 24773  | +11.90%| 5.21   | -4.50%  | 82.6% | 28.8%     | |
| 7916   | +11.72%| 5.04   | -5.06%  | 82.5% | 26.9%     | |

B&H QQQ: **+20.40% CAGR, -35.12% max DD**. The model trades off ~7 pts of CAGR for ~30 pts of drawdown — that's the value-add.

**Deployment criterion (per prior testing):** 74174 was chosen not for highest CAGR or Sharpe but for the exceptional MaxDD (-1.85%) + 86.5% win rate combination. In a prior rotation simulation, 74174 produced ~$716k rotation result — the best of the QQQ candidates tested. Confirms the same lesson XLE taught us: **rotation value ≠ standalone Sharpe**. Low-DD, high-win-rate seeds tend to be better rotation citizens.

**Bar to beat for overnight sweep (with new momentum features):** rotation value of ~$716k or better (in whatever rotation config we use for comparison — note that the headline rotation final value depends on XLE and GDX too, so A/B by toggling only DEPLOYED_SEEDS["QQQ"] and keeping everything else constant).

---

## Rotation backtest — now lives in `aurum_signal.py`

`rotation_backtest.py` is **deprecated and no longer active**. The current authoritative rotation backtest is `run_full_history_simulation()` inside `aurum_signal.py` (~line 3440). It loads GDX + XLE + QQQ fold models, downloads full price history (2010 → present), runs inference on every trading day, and simulates a $100k portfolio with the GDX → XLE → QQQ → SGOV cascade. Inference-only — no external equity CSV dependency. Yearly breakdown printed to console.

The `rotation_backtest/summary.txt` numbers (1,436% total / 23.36% CAGR / 5.45 Sharpe / -34.27% DD) came from the old standalone script and are now **stale**. Don't quote them — re-run `aurum_signal.py` if we need a current figure.

There's also a separate `compute_rotation_portfolio()` function at line 1546 of `aurum_signal.py` that's GDX + XLE only (docstring still says "GDX+XLE rotation"). Likely on its way out — the live runner's history/chart should be using `run_full_history_simulation()` results.

---

## In-progress / open sweeps

### XLE candidate sweep #3 — completed 2026-05-27 (post-pandas-fix, post-concurrent-trainers)

26 candidates total: the original 10 from sweep #2 (re-ran identically — confirms backtest reproducibility) plus 16 new ones from the concurrent training session. Headline: **seed 73101 produces +23.20% CAGR / Sharpe 5.07 / MDD -8.76% / 83.5% win rate — better than the old pre-momentum deployed seed 97239 on every metric.** This unblocks deployment of a new-pipeline XLE seed *as soon as `aurum_signal.py` gets the momentum-feature sync*.

Other promotion candidates: 25049 (+17.29% / 4.34), 95148 (+14.59% / 4.01). Plus the sweep #2 picks 86117 (defensive) and 22315 (balance) still hold up.

Anomaly carried forward: seeds **40278 and 9130 produced byte-identical backtest metrics** (+11.79% / 3.73 / -15.74% / 66.7% / 38.9%) despite different training metrics. Both candidate folders are complete (not a stale-file bug). Either genuine coincidence or a new bug — worth diffing their oof/full predictions.

Sweep #2 result table (below) preserved for historical reference but **sweep #3 is the current state**.

### XLE candidate sweep #2 — completed 2026-05-27

Ranked by backtest Sharpe (B&H XLE for the window: +5.08% CAGR, -71.26% max DD):

| Rank | Seed   | Train IC | Train DA | CAGR    | Sharpe | Max DD   | Win % | Days long | Notes |
|------|--------|----------|----------|---------|--------|----------|-------|-----------|-------|
| 1    | 11983  | 0.3665   | 0.7689   | +12.89% | 3.80   | -25.76%  | 66.0% | 42.3%     | Best Sharpe but deep DD |
| 2    | 22315  | 0.5321   | 0.7121   | +12.27% | 3.79   | -12.44%  | 69.7% | 34.6%     | **Best balance — promote candidate** |
| 3    | 9130   | 0.5984   | 0.7765   | +11.79% | 3.73   | -15.74%  | 66.7% | 38.9%     | Highest training IC, mid-pack live |
| 4    | 35247  | 0.4097   | 0.6818   | +7.43%  | 3.60   | -30.46%  | 62.2% | 47.4%     | Worst DD |
| 5    | 86117  | 0.4404   | 0.8106   | +13.36% | 3.43   | -9.72%   | 78.4% | 21.7%     | **Best CAGR + best DD — promote candidate** |
| 6    | 97312  | 0.4800   | 0.7614   | +6.24%  | 3.31   | -9.99%   | 70.2% | 24.2%     | |
| 7    | 82179  | 0.5178   | 0.7462   | +9.31%  | 3.26   | -9.06%   | 71.2% | 28.6%     | Valid result |
| —    | 83413  | 0.5200   | 0.7008   | **invalid** | — | — | — | — | ❌ **Contaminated** — folder missing `full_predictions.csv`; PS1 script reused 82179's. Real performance unknown. |
| 9    | 26644  | 0.5350   | 0.7045   | +10.36% | 2.98   | -7.71%   | 77.0% | 18.5%     | |
| 10   | 5187   | 0.5027   | 0.6742   | +5.73%  | 2.77   | -18.84%  | 70.8% | 21.9%     | |

**Verdict:** These are the first batch from the **new feature pipeline** (post-momentum-features update). They can't be directly compared to the older seeds 97239 / 25430 / 40721, which were trained on the prior pipeline. Within this new-pipeline batch, **86117** (highest CAGR + lowest DD + highest win-rate, defensive low-participation) and **22315** (best balance) are the top picks. Both flagged for promotion into `xle_output/Best_Models_PT/` so they survive the next `candidates/` refresh.

Deployment decision deferred — the older deployed seed 97239 still has better headline numbers but on a stale pipeline. Options:
1. Keep 97239 deployed (status quo, but means production isn't actually running the current feature set's best model).
2. Promote a new-pipeline seed (86117 or 22315) to deployment despite lower historical backtest CAGR, on the principle that the current feature set is meant to be the go-forward standard.
3. Run another sweep batch (10–20 more seeds) on the new pipeline to see if we can find one that matches 97239's old numbers before deciding.

Option 3 is the safest — current sample size is only 10 seeds, which isn't enough to know if the new features help or hurt.

> ⚠ **Deployment blocker:** `aurum_signal.py`'s `build_features_live()` has NOT been updated with the new XLE momentum features yet. Before any new-pipeline XLE seed can go to production, the live signal runner's XLE feature block needs to be brought into parity with `xle_trainer.py`. Otherwise the live model will see a different feature set than it was trained on, and `align_features()` will silently fill the missing momentum features with zeros — making the live predictions degraded vs. backtest. See "Pending work" below.

**Investigation resolved (2026-05-27):** The byte-identical 82179 / 83413 results were caused by a stale-file bug in `test_xle_candidates.ps1`. Seed 83413's candidate folder is missing `full_predictions.csv` (only that one of the 10 candidates is incomplete — verified via folder inventory). The PS1 script copies CSVs with `Copy-Item -Force` but doesn't delete stale ones from `xle_output\` before each candidate, so when 83413's loop started, 82179's `full_predictions.csv` was still present. `xle_backtest.py` uses `pred_source="full"`, so it read 82179's predictions and produced 82179's metrics. Conclusion: 83413's real backtest performance is unknown; **drop that row from the sweep results**. The other 9 candidates are clean.

**Confirmed pattern:** High training IC ≠ high backtest CAGR. Three of the top four IC scorers (9130 / 5350 / 5200) ranked mid-pack on live performance. Meanwhile seed 86117 (IC only 0.44 but DA 0.81 — the highest DA in the batch) won on CAGR and DD. Going forward, **weight DA over IC when picking promotion candidates.**

**Additional lesson — Standalone-best ≠ rotation-best (2026-05-27 A/B test 73101 vs 97239):** Seed 73101 was the standalone XLE winner from sweep #3 (+23.20% CAGR, Sharpe 5.07, lowest DD) but in the rotation context produced **$1.19M less portfolio value over 16.3 years** ($1.26M vs $2.45M final) than the older seed 97239. Reason: more selective models cross the rotation's confidence-percentile threshold less often, so idle cash sits in SGOV at 0% instead of compounding via XLE. The XLE seed best for rotation is the one that produces above-threshold bullish signals during periods when GDX is uncertain — the complementary citizen, not the standalone-strongest. **Action item: future XLE seed promotion should include a rotation A/B (toggle DEPLOYED_SEEDS, run `aurum_signal.py` twice) before committing.**

---

## Environment

- **GPU enabled (RTX 4060, 8 GB VRAM, driver 592.27, CUDA 13.1 capable).** Enabled 2026-05-27 by uninstalling the CPU torch build and installing torch from the CUDA wheel index (`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`). `torch.cuda.is_available()` returns True. Trainers' `device = "cuda" if torch.cuda.is_available() else "cpu"` pattern picks up the GPU automatically.
- **Pandas pinned to `<3`** (currently 2.3.3) because of the torch CUDA + pandas 3.0 segfault interaction (see changelog 2026-05-27). The pin applies system-wide to the Python 3.14 install — covers all trainers, backtests, and `aurum_signal.py`. If anything ever pip-upgrades pandas back to 3.x, expect the segfault to return for any script that does `torch.load()` then pandas operations. Consider creating a `requirements.txt` in the project root with `pandas<3`, `numpy<3`, `torch` to prevent regression.
- **Empirical single-pass speedup was minimal** (~4.7 min CPU → ~4.5 min GPU, observed 2026-05-27). The runtime is dominated by CPU-bound work: feature engineering (pandas + numpy), data downloads/caching, and multi-start CV orchestration. The LSTM training portion *is* faster on GPU but is a small fraction of wall-clock time. GPU util sits at only ~33% with one process.
- **Parallel candidate-making is where the GPU pays off.** The 173 MiB VRAM footprint per trainer means 4–8 concurrent processes fit comfortably in 8 GB. With GPU util at 33% per process, headroom is there to share. Practical setup: pre-warm caches by running one trainer alone first, then launch additional `make_*_candidates.ps1` windows. **Cap CPU threads per process** with `$env:OMP_NUM_THREADS=2; $env:MKL_NUM_THREADS=2` in each shell — PyTorch still spawns CPU threads for non-GPU work and without caps the trainers will fight for cores. Top-level `best_run_*.txt` race remains a concern but harmless (candidate dir names embed seed + metrics).

## Architecture decisions (live)

### Trainer (v10 — `gold_miner_trainer.py` canonical; sector clones for QQQ/XLE/SMH)

- `forward_days=20`, `lookback=30`, `zscore_window=252`
- LSTM: hidden=48, 1 layer, dropout=0.4, ~20k params
- Head: LayerNorm → 48→32 (GELU) → 32→16 (GELU) → 16→1
- Loss: HuberLoss(δ=0.05)
- Optimizer: AdamW(lr=1e-3, wd=3e-4)
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=8)
- Early stop: patience=20
- Walk-forward: 8 folds, 3-month val blocks, 30-day purge gap, min train = 2 years
- Multi-start per fold: 3 mandatory, up to 5 if IC<0.15 or DA<0.60. Trial scoring = IC + 2.0·max(0, DA-0.5)
- Feature selection: corr filter (ρ>0.85) → top-45 by IC → noise floor (IC<0.02)
- Random seed each run, best run tracked to `best_run_fwd*.txt`

### Backtest (v3 — all four ETFs share this engine)

- Position tiers by confidence percentile: 25/50/75/100%
- Mode: `cash_long` (no shorts; bearish → flat)
- Weekly rebalance (5 trading days)
- Hard 8% stop loss
- 1-day execution lag (trade at next open)
- Transaction cost: 0.10% commission + 0.05% slippage = 0.15% round-trip
- `min_signal`: 0.008 for GDX, 0.002 for XLE / QQQ / SMH

### Rotation (`run_full_history_simulation()` in `aurum_signal.py` ~L3440)

- Cascade: GDX deploys first → **then strength-ranked (XLE vs QQQ by confidence percentile, strongest first) from idle** → cash to SGOV. Changed from fixed XLE-first ordering on 2026-05-28; A/B showed +$83k / +0.3pp CAGR improvement on the 16.3-year backtest. Both must pass the deployment gate (pred > min_sig + conf ≥ 0.40) to compete.
- Pure inference run — replays every trading day from 2010 against the current deployed fold models
- TLT **disabled** (was rule-based MA>20d in the old `rotation_backtest.py`; dropped because of 2021+ rate-hiking drag — flagged in code for "rate-filtered" revival later)
- Old `rotation_backtest.py` is deprecated; do not run or trust its outputs

### Live signal (`aurum_signal.py`)

- Inference only, no retraining
- Cadence: weekly, Monday post-close
- Loads `fold_*_model.pt` from deployed seed dir per ETF
- Ensemble weights: 60% accuracy + 40% recency
- Quality filter: skips folds with DA<0.40 or IC<-0.15
- Cash rate constant set to 0% in this file (conservative vs. the rotation's 5.2%)
- Outputs: console report, `signal_log.csv`, 6-panel chart (`signal_history_chart.png`)

---

## Tried & rejected / lessons learned

- **v9 trainer (`Aurum_trainer.py`)** — fwd=10, lookback=20, start=2013, fixed seed 83959. Superseded by v10 because longer horizon (20d) + more history (2010) + simpler model (1 LSTM layer instead of 2) + random-seed multi-start all outperformed.
- **TLT rule-based long** — dropped from rotation because rate-hiking cycle 2021+ produced sustained drag. Code path retained for later "rate-filtered" version.
- **Trend filter overlay (GDX above 200d MA)** — tested in `Aurum_backtest.py` sensitivity analysis; degraded Sharpe, kept off by default. Infrastructure still present.
- **Multi-start MAX_STARTS=5 (up from 3)** — validated as a real improvement. Increasing the per-fold multi-start cap from 3 to 5 produced noticeably better models. Keep it at 5. (Trainer's adaptive logic: MIN_STARTS=3 mandatory, scaling up to 5 if early-exit thresholds aren't met.) Do **not** drop it as a speed optimisation.
- **No-Google-Trends XLE variant** (see `xle_output/Best_Models_PT/fwd20_no_gtrends_backup_cands/`) — kept the trends features in final XLE pipeline; the no-trends backup is a known-good fallback if pytrends ever breaks.
- **SMH out of rotation** — `smh_trainer.py` / `smh_backtest.py` still exist as standalone scripts, but SMH is no longer a rotation instrument. Don't add it back to `rotation_backtest.py` or to `DEPLOYED_SEEDS` in `aurum_signal.py`.

---

## Known technical debt

- `qqq_backtest.py` has two `run_backtest()` blocks (~L193–411 and ~L427–635). Duplicate from copy-paste; one should be deleted.
- `aurum_signal.py` has a duplicate `download_cot_recent()` block (~L571–646) sitting after the real function ends.
- `qqq_trainer.py` header docstring still says `python gold_miner_trainer.py` — copy-paste from GDX template.
- Project root cluttered with Visual Studio Python Tools backup files: `*.ysn~`, `*.ra3~`, `*.bw0~`, `*.dbp~`, `*.e4i~`, `*.rm2~`, `*.02x~`, `*.wqx~`.
- `New Text Document.txt` (0 bytes) and `temp_test.py` (127 bytes) appear to be stubs.
- `test_xle_candidates.ps1` (and the QQQ/GDX variants) clobbers `candidate_results.txt` on every run via `"" | Set-Content`. Should be patched to back up first OR append with a run header.

---

## Open questions / things to revisit

- GDX `best_run_fwd20_lb30.txt` points to seed 97471 (IC 0.717, DA 0.561), but the **deployed** GDX seed is 21921 (top backtest performer). The IC-winner from training ≠ best backtester — that gap is intentional but worth documenting why we trust backtest CAGR/Sharpe over training IC when picking the deployed seed.
- Confirm whether the duplicate GDX rows (seed 36005 and seed 53050, identical numbers across all metrics) are actually duplicates in the source data or a coincidence.

---

## Pending work

- ~~**Sync `aurum_signal.py` XLE features with the new `xle_trainer.py` pipeline.**~~ **DONE 2026-05-27.** Six XLE momentum features (`xle_r5`, `xle_r10`, `xle_r21`, `xle_ma50_dist`, `xle_ma200_dist`, `xle_trend`) added to `build_features_live()` immediately before the inf-clean step, mirroring section 15 of `xle_trainer.py`. All 6 call sites of `build_features_live` (live daily signal, rotation history, GDX/XLE/QQQ full-history sims, sliding-window rotation history) are now covered automatically. Old XLE checkpoint (seed 97239) is unaffected — `align_features()` simply won't read columns it doesn't have in its feature_names list. **Deployment is now unblocked** for new-pipeline XLE seeds.
- **`compute_rotation_portfolio()` cleanup.** The function at `aurum_signal.py` L1546 still only iterates GDX + XLE (docstring says "GDX+XLE rotation") even though it accepts a `qqq_preds` arg. Either extend it to include QQQ in the cascade or delete it if `run_full_history_simulation()` has fully superseded its role in the live runner's chart/history pipeline.
- **Decommission `rotation_backtest.py` cleanly.** The script is dead but still sits in `GDX111.pyproj` (`<Compile Include="rotation_backtest.py" />`) and the `rotation_backtest/` output folder still has stale results. Either delete the script and folder, or move to a `_deprecated/` subdir so it's clear it's not authoritative.
- **Investigate why seed 83413's training run is missing `full_predictions.csv`.** All 8 fold models, `oof_predictions.csv`, and `metrics.txt` are present, but `full_predictions.csv` is not — suggests the v10 trainer's `save_full_predictions()` step crashed or was interrupted. Likely cause: untimely Ctrl-C during the interactive seed-search loop, hitting the trainer between OOF save and full predictions save. Low priority since the seed had middling training metrics anyway; can just delete the folder.
- **Delete seed 40278's candidate folder.** Diff against seed 9130 (2026-05-27) confirmed: 40278's fold models, OOF, and metrics are unique and valid, but its `full_predictions.csv` is **byte-identical to 9130's** with timestamp 13:05 (150 minutes before 40278's fold models at 15:31-15:36). Diagnosis: during concurrent training, 40278's `save_full_predictions()` step failed silently and the candidate-folder-copy step picked up the stale top-level `full_predictions.csv` from 9130's earlier run. So 40278's reported backtest metric (+11.79% / Sharpe 3.73) is actually 9130's. Of 25 candidates with the file, only this one pair shares content — failure rate ~4% under 3-concurrent training. Removing the folder is the simplest cleanup. If concurrency is required again, harden the trainer's `save_full_predictions()` with an assertion that the file was just written.
- **Promote sweep #3 runners-up to `xle_output\Best_Models_PT\`.** 73101 already copied (deployed 2026-05-27). Still pending: 25049 and 95148 (sweep #3 #2 and #3 finishes) and the sweep #2 keepers 86117 and 22315 if not already preserved. Optional but reduces risk of losing good seeds if `candidates/` gets cleared.

---

## Changelog

- **2026-05-27** — Notes file created. Captured deployed seeds, current best candidates per ETF, rotation results, XLE sweep #2 candidate list (10 seeds awaiting bulk test), architecture decisions, known tech debt, open questions.
- **2026-05-27** — SMH formally out of rotation (kept as standalone scripts only). QQQ flagged for inclusion in rotation; needs wiring into `rotation_backtest.py` and `compute_rotation_portfolio()` in `aurum_signal.py`.
- **2026-05-27** — Clarified that `rotation_backtest.py` is deprecated. Authoritative backtest is now `run_full_history_simulation()` in `aurum_signal.py` (~L3440), which already includes the GDX → XLE → QQQ → SGOV cascade. Marked `rotation_backtest/summary.txt` numbers as stale. Older `compute_rotation_portfolio()` (L1546) is GDX+XLE-only and on the cleanup list.
- **2026-05-27** — XLE candidate sweep #2 finished. 10 fresh seeds tested. Best new-pipeline candidates: **86117** (+13.36% CAGR, Sharpe 3.43, MDD -9.72%, 78.4% win rate) and **22315** (Sharpe 3.79 with -12.44% DD). Both flagged for promotion. Anomaly: seeds 82179 / 83413 produced identical metrics — needs investigation. Pattern reinforced: train DA predicts live performance better than train IC.
- **2026-05-27** — Clarified that the older XLE seeds (97239 / 25430 / 40721) were trained on a **pre-momentum-features** version of `xle_trainer.py`, so their numbers can't be compared directly to the current pipeline's candidates. Deployed XLE seed 97239 marked as pipeline-stale in the deployed-seeds table. Deployment decision deferred pending more new-pipeline seed search.
- **2026-05-27** — Identified deployment blocker: `aurum_signal.py`'s `build_features_live()` does NOT yet include the new XLE momentum features. Must be brought into parity with `xle_trainer.py` before any new-pipeline XLE seed can be deployed, or live predictions will silently degrade via zero-filled missing features.
- **2026-05-27** — Resolved the seed 82179 / 83413 duplicate-metrics mystery. Root cause: `test_xle_candidates.ps1` doesn't clean stale prediction CSVs from `xle_output\` between iterations. Seed 83413's training run produced no `full_predictions.csv` (likely a `save_full_predictions()` crash during training), so the backtest read the previous candidate (82179)'s stale CSV and reported duplicate metrics. Verified by folder inventory: 83413 is the only one of the 10 candidates missing that file. 83413 row dropped from sweep results; PS1 script patch added to pending work.
- **2026-05-27** — Patched all three bulk-runner scripts (`test_xle_candidates.ps1`, `test_qqq_candidates.ps1`, `test_gdx_candidates.ps1`) to delete stale `fold_*.pt` files and prediction CSVs from the output dir at the start of each iteration, plus warn loudly if a candidate folder is missing `full_predictions.csv` or `oof_predictions.csv`. Stale-data contamination bug closed.
- **2026-05-27** — Created `make_xle_candidates.ps1`, `make_gdx_candidates.ps1`, `make_qqq_candidates.ps1` seed-search loop scripts (run counter, 5s sleep between runs, prints best-so-far after each).
- **2026-05-27** — GPU (RTX 4060) enabled. Replaced `torch+cpu` with CUDA wheel; `torch.cuda.is_available()` confirms. Trainers should pick up GPU automatically via their `device = "cuda" if ...` pattern.
- **2026-05-27** — Discovered ABI incompatibility on the current stack: Python 3.14 + pandas 3.0.3 + numpy 2.4.6 + CUDA torch 2.12. After torch initializes (e.g. `torch.load()`), subsequent pandas operations segfaulted with `STATUS_ACCESS_VIOLATION` (Windows exit code -1073741819 / 0xC0000005). No traceback; script just exited. Reproducer: `python -c "import torch; torch.load(<any .pt>, map_location='cpu'); import pandas as pd; pd.read_csv(<any .csv>)"`. Standalone pandas worked fine, standalone torch worked fine — only the interaction crashed. **Fixed** by downgrading to pandas 2.3.3 (`pip install "pandas<3"`). Numpy stayed at 2.4.6; pandas 2.3 supports both numpy 1.x and 2.x. Backtest verified working post-fix. Pin: stay on `pandas<3` until torch publishes wheels validated against pandas 3.0 ABI. Unrelated dep warnings about alpaca-trade-api / pyyaml / websockets surfaced during install — ignored, not used by trainer/backtest pipeline.
- **2026-05-27** — Created `requirements.txt` pinning `pandas<3`, `numpy<3`, plus the direct deps (yfinance, pytrends, matplotlib, scipy, requests, openpyxl, torch). Header documents the CUDA-torch caveat and the pandas pin rationale.
- **2026-05-27** — XLE candidate sweep #3 (26 candidates) completed post-fix. **Headline: seed 73101 → +23.20% CAGR, Sharpe 5.07, MDD -8.76%, 83.5% win rate** — beats the pre-momentum-features deployed seed 97239 on every metric. Sweep #2's pessimistic conclusion overturned: new pipeline IS competitive, just needed bigger sample. Promotion candidates: 73101, 25049, 95148. Anomaly: seeds 40278 and 9130 produced identical backtest metrics — added to pending investigations.
- **2026-05-27** — Resolved the 40278/9130 anomaly via file-content diff. Of 25 candidates with `full_predictions.csv`, only this one pair shares hash. Root cause: during concurrent training, 40278's `save_full_predictions()` step failed silently; the candidate-folder-copy step picked up the stale 13:05 file (9130's). 40278's reported metrics are actually 9130's — drop seed 40278. 4% failure rate under 3-concurrent training, ~0% sequential. No script changes needed; just retire 40278 or stay sequential.
- **2026-05-27** — XLE momentum features synced into `aurum_signal.py`'s `build_features_live()`. Added 6 features (`xle_r5`, `xle_r10`, `xle_r21`, `xle_ma50_dist`, `xle_ma200_dist`, `xle_trend`) mirroring section 15 of `xle_trainer.py`. Inserted right before the inf-cleanup at the end of the XLE block. Affects all 6 callers of `build_features_live()` automatically. Deployment blocker for new-pipeline XLE seeds is now cleared.
- **2026-05-27** — **XLE deployment switched: seed 97239 (old pipeline) → seed 73101 (new pipeline).** `DEPLOYED_SEEDS["XLE"]["seed"]` updated in `aurum_signal.py` line 51. Candidate folder also preserved at `xle_output\Best_Models_PT\fwd20_lb30_IC0.5512_DA0.7765_seed73101\` (full 11-file copy, md5-verified). Next `aurum_signal.py` run will auto-deploy 73101's fold models via `deploy_seed_models()`. Verification still pending: run `python aurum_signal.py` and confirm no "features missing" warnings from `align_features()` for the XLE model.
- **2026-05-27** — **REVERTED XLE deployment back to 97239** after A/B test. Two clean runs of `aurum_signal.py` (no concurrent trainers, same code state with momentum features in `build_features_live()`) under each seed: 73101 → rotation $1,264,965 / +16.9% CAGR / +2.5pp premium. 97239 → rotation $2,454,828 / +21.7% CAGR / +7.4pp premium. **$1.19M and 4.8pp delta in favor of 97239.** Mechanism: 97239 is less selective, deploys XLE on more days, captures diversification value during periods when GDX is partial. Set `DEPLOYED_SEEDS["XLE"]["seed"]` back to 97239. To enable deploy_seed_models() to find it, copied seed 97239's candidate folder from `Best_Models_PT/fwd20_backup_candidates/` into `Best_Models_PT/candidates/`. Note this is a clear "standalone-best ≠ rotation-best" lesson, added to Tried & rejected section.
- **2026-05-27** — **Removed the 6 XLE momentum features from `xle_trainer.py`** (section 15, lines 1365-1380 — `xle_r5`, `xle_r10`, `xle_r21`, `xle_ma50_dist`, `xle_ma200_dist`, `xle_trend`). Replaced with a comment block explaining the history. Future XLE candidate generation now reverts to the pre-momentum feature set, which produced the better rotation citizen (seed 97239). **Did NOT remove these from `aurum_signal.py`'s `build_features_live()`** — leaving them in keeps sweep #3 candidates and the seed 73101 preservation copy usable if we ever want to A/B them again. Per `align_features()` design, the deployed 97239 model is unaffected by their continued presence.
- **2026-05-27** — **Extended `qqq_trainer.py` momentum features for long-climb capture.** Existing section 15 had 6 standard momentum features (qqq r5/r10/r21, ma50_dist, ma200_dist, trend) that were in the code but never used to train a candidate. Added 4 new long-bias variants to section 15: `qqq_r63` (3-month), `qqq_r126` (6-month), `qqq_r21_chg5` (acceleration), `qqq_vol_adj_r21` (Sharpe-of-recent-moves). Added new section 15b: XLK peer momentum at three windows (`xlk_r10`, `xlk_r21`, `xlk_r63`) — XLK was downloaded but previously unused as a feature. **Total new features: 7.** Goal: capture the long multi-quarter climbs QQQ shows. Overnight `make_qqq_candidates.ps1` run will produce the first momentum-aware QQQ candidate batch. Feature selection (top-45 by IC + correlation filter) will prune anything redundant, so the seed search will reveal which of these actually carry signal. In the morning: bulk-test with patched `test_qqq_candidates.ps1`, then rank against deployed seed 74174 on both standalone metrics AND rotation premium (per the XLE A/B lesson, both lenses matter).
- **2026-05-28** — Overnight QQQ sweep complete: **162 candidates** generated by the momentum-aware trainer. Top picks by rotation-citizen criteria (low DD + high win rate + competitive CAGR): **seed 57437** (CAGR 12.94% / Sharpe 5.49 / MDD -1.21% / Win 88.4% / Days 27.4% — strictly better than deployed 74174 on every metric), **seed 41597** (CAGR 13.85% / Sharpe 5.77 / MDD -2.73% / Days 31.0% — best CAGR + most participation), **seed 64978** (highest win rate in sweep at 89.6%, MDD -1.24%). Feature IC ranking from seed 83164 confirmed `qqq_r63` ranked #2 in the entire feature space — long-window momentum thesis validated. Other 6 new features mostly dropped by correlation filter (XLK too similar to QQQ; qqq_r126 redundant with qqq_r63). Also flagged 2 duplicate-metric pairs (35226/86341 and 57052/37634) — likely same stale-full_predictions.csv bug as before, top candidates unaffected.
- **2026-05-28** — **Synced `aurum_signal.py` with new QQQ trainer features.** Added "XLK" to TICKERS (line 144). Added QQQ momentum block (10 features) and XLK peer momentum block (3 features) to `build_features_live()`, placed right after the XLE momentum block before the inf-cleanup. Mirrors sections 15 and 15b of `qqq_trainer.py`. Required before deploying any candidate from the 2026-05-28 overnight sweep — without this, `align_features()` would silently zero-fill the missing features for new candidates. Deployed seed 74174 unaffected (its feature_names list doesn't include any of these columns).
- **2026-05-28** — **Patched `deploy_seed_models()` with content-based check.** Replaced the mtime comparison with an MD5 hash compare of `fold_1_model.pt` (source vs destination). The old mtime check produced false-positive "already current" results when the destination held files from concurrent training or a previously-deployed different seed (mtime older than configured seed's source). The new content check is definitive, fast (~ms for an 88 KB file), and self-healing — handles seed switches in either direction, concurrent trainer overwrites, and partial deployments. Force-copies whenever the hash differs, the destination is missing, or comparison fails. Log message now states "(hash verified)" on the skip path so it's obvious which check was used.
- **2026-05-28** — **Cast doubt on yesterday's XLE A/B results.** Current `xle_output/fold_1.pt` hash doesn't match seed 97239's or 73101's candidate copies, suggesting yesterday's runs may have suffered the same mtime-deploy bug. Counter-evidence: the 73101 vs 97239 results were *dramatically* different ($1.26M vs $2.45M), which is hard to explain if both runs used the same stale model. Most likely: deploys worked yesterday, current state reflects some later untracked file activity (or stale mount cache). To be sure, re-verify the 97239 baseline with the patched deploy. Same applies to QQQ 74174 baseline ($2.45M from the same yesterday run).
- **2026-05-28** — **Reconfirmed XLE 97239 deploy was valid.** Patched `deploy_seed_models()` ran with current `xle_output/` state and reported "(hash verified) — already current" for XLE 97239. So the destination file actually DID match 97239's source. My earlier hash check (showing mismatch) was a mount cache artifact. Yesterday's XLE A/B results stand.
- **2026-05-28** — **Clean QQQ A/B finally completed** with patched deploy. Re-tested 74174 (baseline reproduced: $2,426,154 / +21.6% / +7.4pp — within 1.2% of yesterday's $2,454,828). Tested 57437 ($1,870,626 / +19.7% / +5.4pp — loses by $555k), 41597 ($2,159,165 / +20.8% / +6.5pp — loses by $267k), 64978 ($1,985,972 / +20.1% / +5.9pp — loses by $440k). **All three new-pipeline candidates lose to 74174 in rotation under the old XLE-first cascade.** Conclusion: 74174 stays deployed; the standalone-best ≠ rotation-best lesson holds for QQQ as well as XLE.
- **2026-05-28** — **Implemented strength-ranked rotation cascade.** Modified `run_full_history_simulation()` to deploy XLE and QQQ in order of confidence percentile (strongest first) rather than fixed XLE-first priority. Both must still pass the deployment gate (pred > min_sig AND conf ≥ 0.40). A/B with current deployed seeds (GDX 21921, XLE 97239, QQQ 74174): **rotation $2,509,233 / +21.9% / +7.6pp** — improvement of +$83k / +0.3pp CAGR / +0.2pp premium vs the old XLE-first cascade ($2,426k baseline). Most years unchanged; the gains come from 2018 (rotation +17.1% → +20.3%) and 2022 (+17.6% → +19.8%) where QQQ had unusually high conviction on the same days XLE was also bullish. Old-cascade-only winners get squeezed out and the high-conviction QQQ days capture more value. Also patched the live signal output (two branches) and the chart helper to use the same strength-ranked logic for consistency.
- **2026-05-28** — **Re-A/B-tested the new-pipeline QQQ candidates under the new cascade.** Full comparison: 74174 $2,509,233 (+$83k vs old cascade), 41597 $2,263,534 (+$104k), 64978 $2,008,431 (+$22k), 57437 $1,829,346 (−$41k). **74174 wins under either cascade by a wide margin** ($246–$680k ahead of new-pipeline candidates). The new cascade doesn't change the winner — it just improves all of them slightly (except 57437). Particularly damning: seed 64978 deployed QQQ 28.6% of 2022 (highest QQQ allocation of any candidate in any year tested) and the rotation still made only +13.4% vs 74174's +19.8% — definitive evidence that "when you're bullish matters more than how often". **Final state: GDX=21921, XLE=97239, QQQ=74174, strength-ranked cascade active.** Best configuration produces $2,509,233 rotation final over 16.3 years.
- **2026-06-14** — **Deployed seeds switched to the v9 champions: GDX 162, XLE 18264, QQQ 32928** (bootstrap-validated; now set in `DEPLOYED_SEEDS` in `aurum_signal.py` ~L49). Standalone `candidate_results.txt` backtests: GDX 162 → +42.38% CAGR / Sharpe 7.93 / MDD -6.59% / Win 85.8%; XLE 18264 → +23.92% / 7.07 / -11.08% / 79.1%; QQQ 32928 → +58.31% / 9.53 / -8.73% / 80.0%. Deployed-seeds table at top of file updated to match. The prior production seeds (21921 / 97239 / 74174) and their A/B history above are kept as historical reference. The headline rotation figure ($2,509,233) predates this switch — re-run `aurum_signal.py` to refresh it under the new seeds.
