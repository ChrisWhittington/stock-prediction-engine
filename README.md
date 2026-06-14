# AURUM·AI — Multi-ETF ML Trading Engine

A machine-learning research pipeline that trains LSTM models to forecast forward
returns for a set of ETFs/stocks, backtests them with realistic costs, and runs a
**confidence-ranked rotation strategy** that allocates capital across the strongest
signals (falling back to cash/SGOV when nothing is convincing).

The flagship instrument is **GDX** (gold miners); **XLE** (energy) and **QQQ**
(Nasdaq-100) act as diversifying rotation members. Several other tickers
(AEM, SHEL, TSM, SMH, plus the `aurum` GDX line) have standalone trainer/backtest
pipelines for research.

> ⚠️ **Disclaimer:** This is a personal research project, not investment advice.
> Backtested results are historical simulations and do not guarantee future
> performance. Use at your own risk.

---

## How it works

The pipeline has four stages:

```
  trainer  ──►  backtest  ──►  rotation sim  ──►  live signal
 (*_trainer)   (*_backtest)  (aurum_signal.py)  (aurum_signal.py)
   train          score          combine            deploy
   models       standalone      instruments         weekly
```

1. **Train** — `*_trainer.py` runs a walk-forward, multi-start cross-validation
   (8 folds, 3-month validation blocks, 30-day purge gap) and saves the per-fold
   LSTM checkpoints (`fold_1..8_model.pt` + `final_model.pt`).
2. **Backtest** — `*_backtest.py` replays a single instrument's signal with
   confidence-tiered position sizing, weekly rebalance, an 8% stop, 1-day
   execution lag, and 0.15% round-trip costs.
3. **Rotate** — `run_full_history_simulation()` in `aurum_signal.py` loads the
   deployed fold models for GDX/XLE/QQQ, replays every trading day since 2010,
   and simulates a portfolio using the **GDX → (strength-ranked XLE/QQQ) → SGOV**
   cascade.
4. **Deploy / live** — `aurum_signal.py` is inference-only. Run weekly
   (Monday post-close) to get the current signal, position size, and a
   refreshed history chart.

A key lesson baked into the project: **standalone-best ≠ rotation-best**. The
seed that wins on standalone Sharpe is often *not* the best rotation citizen —
what matters is producing high-conviction signals on the days the primary
instrument is uncertain. See `PROJECT_NOTES.md` for the full A/B history.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `aurum_signal.py` | **Live engine** + authoritative rotation backtest (`run_full_history_simulation()`). Inference only. |
| `gold_miner_trainer.py` | Canonical v10 GDX trainer; the sector trainers are clones of it. |
| `*_trainer.py` | Per-instrument trainers (`aem`, `qqq`, `shel`, `smh`, `tsm`, `xle`). |
| `*_backtest.py` | Per-instrument standalone backtesters. |
| `make_*_candidates.ps1` | Seed-search loops — train many random-seed candidates overnight. |
| `test_*_candidates.ps1` | Bulk-backtest a folder of candidates and rank them. |
| `find_gdx_diversifiers*.py`, `check_substitute_basis_risk.py` | Diversifier / basis-risk research utilities. |
| `*_output/` | Generated artifacts (models, predictions, charts). **Gitignored** except the deployed models below. |
| `PROJECT_NOTES.md` | Running research log — deployed seeds, sweep results, architecture decisions, changelog. **Read this for context.** |
| `requirements.txt` | Python dependencies (note the `pandas<3` pin). |

> `aurum_trainer.py` / `Aurum_backtest.py` (v9) and `rotation_backtest.py` are
> **deprecated** — kept for reference only. The authoritative rotation lives in
> `aurum_signal.py`.

---

## Setup

Requires **Python 3.14** (3.10+ should work).

```bash
pip install -r requirements.txt
```

**GPU (optional, recommended for seed searches).** `requirements.txt` installs
the CPU-only PyTorch. For CUDA, install the GPU wheel separately *first*:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **`pandas<3` is a hard pin.** On the current stack (Python 3.14 + CUDA
> torch + numpy 2.x), calling `torch.load()` followed by any pandas operation
> segfaults with `STATUS_ACCESS_VIOLATION`. Do not lift the pin until torch
> ships wheels validated against the pandas 3.x ABI. (See the 2026-05-27
> changelog entry in `PROJECT_NOTES.md`.)

### Models & Git LFS

The deployed models (`final_model.pt` + `fold_1..8_model.pt` for each ticker) are
versioned with **Git LFS**. After cloning, pull the weights with:

```bash
git lfs install
git lfs pull
```

Without LFS you'll get small pointer files instead of the actual checkpoints.
New models live inside the gitignored `*_output/` folders — to version a freshly
trained model, add it explicitly: `git add -f <ticker>_output/final_model.pt`.

---

## Usage

```bash
# Train a single instrument (saves fold models into *_output/)
python gold_miner_trainer.py        # GDX (canonical)
python qqq_trainer.py               # QQQ
python xle_trainer.py               # XLE

# Backtest a single instrument
python qqq_backtest.py

# Run the live signal + full-history rotation simulation
python aurum_signal.py
```

**Overnight seed search** (Windows PowerShell): run a `make_*_candidates.ps1`
loop to generate many random-seed candidates, then `test_*_candidates.ps1` to
bulk-backtest and rank them. Cap CPU threads per process when running several
concurrently:

```powershell
$env:OMP_NUM_THREADS=2; $env:MKL_NUM_THREADS=2
.\make_qqq_candidates.ps1
```

---

## Model architecture (v10)

- **Horizon / lookback:** `forward_days=20`, `lookback=30`, `zscore_window=252`
- **Network:** 1-layer LSTM (hidden=48, dropout=0.4, ~20k params) → LayerNorm →
  48→32 (GELU) → 32→16 (GELU) → 16→1
- **Loss / optim:** HuberLoss(δ=0.05), AdamW(lr=1e-3, wd=3e-4),
  ReduceLROnPlateau, early stop patience 20
- **Validation:** 8-fold walk-forward, 3-month val blocks, 30-day purge gap,
  ≥2-year minimum train window
- **Multi-start:** 3 mandatory restarts per fold, up to 5 if IC<0.15 or DA<0.60;
  trial score = `IC + 2.0·max(0, DA−0.5)`
- **Features:** correlation filter (ρ>0.85) → top-45 by IC → noise floor (IC<0.02)

**Backtest engine:** confidence-percentile position tiers (25/50/75/100%),
long/cash only (bearish → flat), weekly rebalance, 8% hard stop, 1-day execution
lag, 0.15% round-trip cost.

The currently deployed seeds are defined in the `DEPLOYED_SEEDS` dict near the
top of `aurum_signal.py` — that dict is the source of truth, not this README.

---

## Notes

- Data is downloaded live (yfinance, Google Trends via pytrends, COT, GPR) — no
  bundled price CSVs or API keys.
- `PROJECT_NOTES.md` is the running log of every decision, sweep, and bug; start
  there if you're picking the project back up.
