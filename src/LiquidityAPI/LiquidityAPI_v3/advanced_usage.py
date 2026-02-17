"""
Advanced Usage — Global Liquidity Factor (Block-PCA v3)
========================================================
Requires: quick_start.py has been run at least once so that results/ contains:
  results/block_pca_scores.csv
  results/block_pca_loadings.csv
  results/global_liquidity_data.csv

Functions
---------
1.  regime_table()            Classify every month per block (Expansionary / Neutral / Contractionary)
2.  block_correlation()       Rolling cross-block correlations (validates block independence)
3.  composite_score()         Equal-weight composite across all PC1s + optional chart
4.  signal_backtest()         Long/cash strategy on one block score vs synthetic assets
5.  leading_indicators()      Cross-correlogram — which blocks lead / lag others
6.  block_divergence()        Flag months where blocks disagree (useful regime-change signal)
7.  loadings_report()         Print/plot top indicator loadings per block
8.  run_live()                Fetch fresh data, rerun block-PCA, print current readings
9.  weekly_crypto_backtest()  Two-stage weekly signal: level filter + WALCL momentum

Run all examples:
  python advanced_usage.py

Run one function from another script:
  from advanced_usage import composite_score, signal_backtest
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── File paths ─────────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
SCORES_FILE   = os.path.join(RESULTS_DIR, "block_pca_scores.csv")
LOADINGS_FILE = os.path.join(RESULTS_DIR, "block_pca_loadings.csv")
DATA_FILE     = os.path.join(RESULTS_DIR, "global_liquidity_data.csv")

# ── Expected block columns (PC2 may not exist if data coverage was thin) ───────
BLOCK_COLS = [
    "Monetary_Aggregates_PC1",
    "Monetary_Aggregates_PC2",
    "CB_Balance_Sheets_PC1",
    "Credit_Spreads_PC1",
    "FX_Commodities_PC1",
    "Macro_Flows_PC1",
]

# One PC1 per block (used for composite + lead-lag)
PC1_COLS = [c for c in BLOCK_COLS if c.endswith("_PC1")]

PALETTE = {
    "Monetary_Aggregates_PC1": "#2E86AB",
    "Monetary_Aggregates_PC2": "#7AB8D0",
    "CB_Balance_Sheets_PC1":   "#A23B72",
    "Credit_Spreads_PC1":      "#F18F01",
    "FX_Commodities_PC1":      "#3BB273",
    "Macro_Flows_PC1":         "#7B2D8B",
}

LABELS = {
    "Monetary_Aggregates_PC1": "Monetary Aggregates (PC1)",
    "Monetary_Aggregates_PC2": "Monetary Aggregates (PC2)",
    "CB_Balance_Sheets_PC1":   "CB Balance Sheets (PC1)",
    "Credit_Spreads_PC1":      "Credit Spreads (PC1)",
    "FX_Commodities_PC1":      "FX / Commodities (PC1)",
    "Macro_Flows_PC1":         "Macro Flows (PC1)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_scores() -> pd.DataFrame:
    if not os.path.exists(SCORES_FILE):
        raise FileNotFoundError(
            f"{SCORES_FILE} not found.\n"
            "Run quick_start.py (or glf.construct_block_factors()) first."
        )
    df = pd.read_csv(SCORES_FILE, index_col=0, parse_dates=True)
    cols = [c for c in BLOCK_COLS if c in df.columns]
    return df[cols].sort_index()


def _load_loadings() -> pd.DataFrame:
    if not os.path.exists(LOADINGS_FILE):
        raise FileNotFoundError(f"{LOADINGS_FILE} not found.")
    return pd.read_csv(LOADINGS_FILE)


def _savefig(name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGIME TABLE
# ══════════════════════════════════════════════════════════════════════════════

def regime_table(threshold: float = 0.5) -> pd.DataFrame:
    """
    Classify every month into a liquidity regime for each block.

    Score >  threshold  ->  Expansionary
    Score < -threshold  ->  Contractionary
    Otherwise           ->  Neutral

    Saves: results/block_regimes.csv
           results/regime_heatmap.png
    """
    print("\n" + "=" * 55)
    print("1. REGIME CLASSIFICATION")
    print("=" * 55)
    scores = _load_scores()

    def _classify(s):
        conditions = [s > threshold, s < -threshold]
        choices    = ["Expansionary", "Contractionary"]
        return pd.Series(
            np.select(conditions, choices, default="Neutral"),
            index=s.index,
        )

    regimes = pd.DataFrame(
        {col: _classify(scores[col]) for col in scores.columns},
        index=scores.index,
    )

    print(f"\nRegime frequency  (threshold = +/-{threshold}):\n")
    for col in regimes.columns:
        freq = regimes[col].value_counts(normalize=True) * 100
        print(f"  {LABELS.get(col, col)}")
        for r in ["Expansionary", "Neutral", "Contractionary"]:
            print(f"    {r:<16s}  {freq.get(r, 0):5.1f}%")

    print(f"\nCurrent regime  ({regimes.index[-1].strftime('%Y-%m')}):")
    for col in regimes.columns:
        print(f"  {LABELS.get(col, col):<40s}  {regimes[col].iloc[-1]}")

    regimes.to_csv(os.path.join(RESULTS_DIR, "block_regimes.csv"))
    print("\n  Saved -> results/block_regimes.csv")

    # Heatmap (last 5 years)
    recent = regimes.tail(60)
    cmap_vals = {"Expansionary": 1, "Neutral": 0, "Contractionary": -1}
    # Convert categorical → int explicitly so imshow receives float-compatible data
    heat = recent.apply(lambda col: col.astype(str).map(cmap_vals)).astype(float)

    fig, ax = plt.subplots(figsize=(16, max(3, len(heat.columns) * 0.7)))
    im = ax.imshow(heat.T.values, aspect="auto", cmap="RdYlGn",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(heat.columns)))
    ax.set_yticklabels(
        [LABELS.get(c, c).replace(" (PC1)", "").replace(" (PC2)", " PC2")
         for c in heat.columns], fontsize=9)
    step = max(1, len(heat) // 12)
    ax.set_xticks(range(0, len(heat), step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in heat.index[::step]],
        rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, ticks=[-1, 0, 1],
                 label="Contractionary  /  Neutral  /  Expansionary")
    ax.set_title(f"Liquidity Regime Heatmap - last 5 years  (threshold +/-{threshold})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig("regime_heatmap.png")
    return regimes


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROLLING CROSS-BLOCK CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def block_correlation(window: int = 24) -> None:
    """
    Plot rolling correlation between every PC1 pair.
    Low cross-block correlation confirms each block captures a distinct dimension.

    Saves: results/rolling_block_correlation.png
    """
    print("\n" + "=" * 55)
    print(f"2. ROLLING {window}-MONTH CROSS-BLOCK CORRELATION")
    print("=" * 55)
    scores = _load_scores()
    pc1 = [c for c in scores.columns if c.endswith("_PC1")]
    pairs = [(pc1[i], pc1[j]) for i in range(len(pc1)) for j in range(i + 1, len(pc1))]

    fig, axes = plt.subplots(len(pairs), 1,
                             figsize=(14, 2.8 * len(pairs)), sharex=True)
    if len(pairs) == 1:
        axes = [axes]

    for ax, (a, b) in zip(axes, pairs):
        roll = scores[a].rolling(window).corr(scores[b])
        la = LABELS.get(a, a).replace(" (PC1)", "")
        lb = LABELS.get(b, b).replace(" (PC1)", "")
        ax.plot(roll.index, roll.values, linewidth=1.6, color=PALETTE.get(a, "steelblue"))
        ax.axhline(0,    color="black",  linewidth=0.6, linestyle="--", alpha=0.4)
        ax.axhline(0.5,  color="orange", linewidth=0.6, linestyle=":",  alpha=0.6)
        ax.axhline(-0.5, color="orange", linewidth=0.6, linestyle=":",  alpha=0.6)
        ax.fill_between(roll.index, roll.values, 0,
                        where=(roll.values > 0), alpha=0.08, color="green")
        ax.fill_between(roll.index, roll.values, 0,
                        where=(roll.values < 0), alpha=0.08, color="red")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("r", fontsize=9)
        ax.set_title(f"{la}  x  {lb}", fontsize=9, loc="left", fontweight="bold")
        ax.grid(True, alpha=0.18)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(f"Rolling {window}-Month Cross-Block Correlation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("rolling_block_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPOSITE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def composite_score(weights=None, plot=True):
    """
    Build a single composite liquidity score from all PC1s.
    Default: equal-weight average. Pass a dict to override per-block weights.

    Example custom weights:
      composite_score(weights={'CB_Balance_Sheets_PC1': 0.30,
                                'Credit_Spreads_PC1':    0.25})

    Saves: results/composite_score.csv
           results/composite_score.png  (if plot=True)
    """
    print("\n" + "=" * 55)
    print("3. COMPOSITE LIQUIDITY SCORE")
    print("=" * 55)
    scores = _load_scores()
    pc1 = [c for c in scores.columns if c.endswith("_PC1")]

    if weights is None:
        w = {c: 1.0 / len(pc1) for c in pc1}
    else:
        explicit_sum = sum(weights.get(c, 0) for c in pc1)
        missing = [c for c in pc1 if c not in weights]
        remainder = max(0.0, 1.0 - explicit_sum)
        per_missing = (remainder / len(missing)) if missing else 0.0
        w = {c: weights.get(c, per_missing) for c in pc1}

    print("\nWeights applied:")
    for col, wt in w.items():
        print(f"  {LABELS.get(col, col):<40s}  {wt:.3f}")

    composite = sum(scores[c] * wt for c, wt in w.items())
    composite.name = "Composite_Liquidity"

    print(f"\nLatest composite score  ({composite.index[-1].strftime('%Y-%m-%d')}):  "
          f"{composite.iloc[-1]:+.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    composite.to_csv(os.path.join(RESULTS_DIR, "composite_score.csv"), header=True)
    print("  Saved -> results/composite_score.csv")

    if not plot:
        return composite

    roll = composite.rolling(12).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(composite.index, 0, composite.values,
                    where=(composite.values >= 0), alpha=0.2, color="green",
                    label="Expansionary")
    ax.fill_between(composite.index, 0, composite.values,
                    where=(composite.values < 0),  alpha=0.2, color="red",
                    label="Contractionary")
    ax.plot(composite.index, composite.values,
            color="#2E86AB", linewidth=1.0, alpha=0.6)
    ax.plot(roll.index, roll.values,
            color="#2E86AB", linewidth=2.2, label="12M average")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_title("Composite Global Liquidity Score  (equal-weight PC1 average)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (z-score units)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    _savefig("composite_score.png")
    return composite


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL BACKTEST  (real OHLCV + DCA + Monte Carlo windows)
# ══════════════════════════════════════════════════════════════════════════════

def _load_ohlcv(csv_path: str, asset_name: str | None = None) -> pd.Series:
    """
    Load a headerless OHLCV CSV and return a daily Close price Series.

    Expected column order (no header row):
        timestamp, open, high, low, close, volume, trades

    Timestamps are auto-parsed; Unix milliseconds and ISO strings both work.

    Parameters
    ----------
    csv_path   : str   Path to the CSV file.
    asset_name : str   Optional name for the returned Series (defaults to
                       the filename stem).

    Returns
    -------
    pd.Series  of daily Close prices, DatetimeIndex, sorted ascending.
    """
    name = asset_name or os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
    )

    # ── Parse timestamp ────────────────────────────────────────────────────
    # Try numeric first (Unix ms / s), fall back to string parsing
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts.notna().mean() > 0.9:
        # Detect milliseconds vs seconds by magnitude
        if ts.median() > 1e11:
            df.index = pd.to_datetime(ts, unit="ms", utc=True).dt.tz_localize(None)
        else:
            df.index = pd.to_datetime(ts, unit="s",  utc=True).dt.tz_localize(None)
    else:
        df.index = pd.to_datetime(df["timestamp"], infer_datetime_format=True)

    close = df["close"].astype(float).sort_index()
    close.name = name
    return close


def _daily_to_monthly_returns(daily_prices: pd.Series) -> pd.Series:
    """
    Resample a daily Close price Series to month-end returns.
    Uses the last available close of each calendar month.
    """
    monthly = daily_prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.name = daily_prices.name
    return returns


def _compute_stats(returns: pd.Series, label: str = "") -> dict:
    """
    Compute full performance statistics for a monthly return Series.

    Metrics
    -------
    ann_return   Annualised arithmetic return
    ann_vol      Annualised volatility
    sharpe       Sharpe ratio (risk-free = 0)
    max_dd       Maximum drawdown (peak-to-trough)
    win_rate     % of months with positive return
    calmar       Annualised return / abs(max drawdown)
    n_months     Number of months in sample
    """
    r = returns.dropna()
    if len(r) < 3:
        return {}

    ann_ret = r.mean() * 12
    ann_vol = r.std()  * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum     = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    max_dd   = dd.min()

    win_rate = (r > 0).mean()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "label":      label,
        "ann_return": ann_ret,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "win_rate":   win_rate,
        "calmar":     calmar,
        "n_months":   len(r),
    }


def _build_signal(
    score: pd.Series,
    mode: str,
    threshold: float,
    lag: int,
) -> pd.Series:
    """
    Build a position-sizing signal from a block score.

    mode = 'long_only'   : +1 when score > threshold, else 0
    mode = 'long_short'  : +1 when score > threshold,
                           -1 when score < -threshold, else 0
    mode = 'continuous'  : position = clipped z-score in [-1, +1],
                           threshold acts as a dead-band (|score| < threshold → 0)

    lag : months to lag the signal before applying to returns (default 1).
    """
    s = score.shift(lag)

    if mode == "long_only":
        return (s > threshold).astype(float)

    elif mode == "long_short":
        sig = pd.Series(0.0, index=s.index)
        sig[s >  threshold] =  1.0
        sig[s < -threshold] = -1.0
        return sig

    elif mode == "continuous":
        # Clip to [-1, +1]; zero out the dead-band around zero
        sig = s.clip(-1, 1)
        sig[s.abs() < threshold] = 0.0
        return sig

    else:
        raise ValueError(f"mode must be 'long_only', 'long_short', or 'continuous'. Got '{mode}'.")


def signal_backtest(
    csv_path: str,
    asset_name: str | None = None,
    block: str = "CB_Balance_Sheets_PC1",
    mode: str = "long_only",
    threshold: float = 0.0,
    lag: int = 1,
) -> dict:
    """
    Backtest a liquidity-driven signal against a single real asset.

    Parameters
    ----------
    csv_path    : str
        Path to a headerless OHLCV CSV file.
        Column order: timestamp, open, high, low, close, volume, trades
        Timestamps can be Unix milliseconds, Unix seconds, or ISO strings.

    asset_name  : str, optional
        Display name for the asset (defaults to the CSV filename stem).

    block       : str
        Block score column to use as the signal driver.
        Options: 'Monetary_Aggregates_PC1', 'CB_Balance_Sheets_PC1',
                 'Credit_Spreads_PC1', 'FX_Commodities_PC1',
                 'Macro_Flows_PC1', 'Composite'
        Pass 'Composite' to use an equal-weight average of all PC1s.

    mode        : str
        Signal construction mode:
          'long_only'   — long when score > threshold, flat otherwise
          'long_short'  — long above +threshold, short below -threshold
          'continuous'  — position scales linearly with score (clipped ±1),
                          dead-band applied inside ±threshold

    threshold   : float
        Entry/exit threshold in z-score units (default 0.0).
        For 'continuous' mode, acts as a dead-band: positions inside
        ±threshold are zeroed out.

    lag         : int
        Months to lag the signal before applying to returns (default 1).
        lag=1 means signal is formed at end of month t, applied to return
        of month t+1 — avoids look-ahead bias.

    Returns
    -------
    dict with keys:
        'returns'    pd.DataFrame  monthly returns (Strategy + BuyHold columns)
        'cumulative' pd.DataFrame  cumulative return series
        'stats'      pd.DataFrame  performance stats side-by-side
        'signal'     pd.Series     the raw position signal
        'chart_path' str           path to saved PNG

    Example
    -------
    result = signal_backtest(
        csv_path   = "data/BTC_daily.csv",
        asset_name = "BTC",
        block      = "CB_Balance_Sheets_PC1",
        mode       = "long_short",
        threshold  = 0.5,
    )
    print(result['stats'])
    """
    print("\n" + "=" * 55)
    print(f"4. SIGNAL BACKTEST")
    print(f"   Asset    : {asset_name or os.path.splitext(os.path.basename(csv_path))[0]}")
    print(f"   Block    : {block}")
    print(f"   Mode     : {mode}  |  threshold={threshold}  |  lag={lag}m")
    print("=" * 55)

    # ── 1. Load block scores ───────────────────────────────────────────────
    scores = _load_scores()

    if block == "Composite":
        pc1_cols = [c for c in scores.columns if c.endswith("_PC1")]
        score_series = scores[pc1_cols].mean(axis=1)
        score_series.name = "Composite"
    elif block not in scores.columns:
        raise ValueError(
            f"'{block}' not found in block scores.\n"
            f"Available: {list(scores.columns)} + 'Composite'"
        )
    else:
        score_series = scores[block]

    # ── 2. Load & resample price data ─────────────────────────────────────
    print("\nLoading price data...")
    daily_close   = _load_ohlcv(csv_path, asset_name)
    asset_label   = daily_close.name
    monthly_rets  = _daily_to_monthly_returns(daily_close)

    print(f"  Price range : {daily_close.index[0].date()} → {daily_close.index[-1].date()}")
    print(f"  Monthly obs : {len(monthly_rets)}")

    # ── 3. Build signal ───────────────────────────────────────────────────
    signal = _build_signal(score_series, mode=mode, threshold=threshold, lag=lag)

    # ── 4. Align on common date range ─────────────────────────────────────
    common = monthly_rets.index.intersection(signal.index)
    if len(common) < 12:
        raise ValueError(
            f"Only {len(common)} overlapping months between price data and "
            f"block scores. Check date ranges:\n"
            f"  Prices : {monthly_rets.index[0].date()} → {monthly_rets.index[-1].date()}\n"
            f"  Scores : {signal.index[0].date()} → {signal.index[-1].date()}"
        )

    ret    = monthly_rets.loc[common]
    sig    = signal.loc[common]

    strat_ret = ret * sig
    strat_ret.name = f"{asset_label}_Strategy"
    bh_ret        = ret.copy()
    bh_ret.name   = f"{asset_label}_BuyHold"

    returns_df = pd.concat([strat_ret, bh_ret], axis=1)
    cum_df     = (1 + returns_df).cumprod()

    print(f"  Overlapping : {common[0].date()} → {common[-1].date()}  ({len(common)} months)")
    print(f"  Signal ON   : {(sig != 0).mean():.0%} of months")

    # ── 5. Performance stats ──────────────────────────────────────────────
    stats_strat = _compute_stats(strat_ret, label="Strategy")
    stats_bh    = _compute_stats(bh_ret,    label="Buy & Hold")
    stats_df    = pd.DataFrame([stats_strat, stats_bh]).set_index("label")

    _print_stats_table(stats_df, asset_label, block, mode, threshold)

    # ── 6. Chart ──────────────────────────────────────────────────────────
    chart_path = _plot_backtest(
        cum_df, sig, ret, stats_df, asset_label, block, mode, threshold
    )

    # ── 7. Save returns CSV ───────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = asset_label.replace("/", "_")
    csv_out = os.path.join(RESULTS_DIR, f"backtest_{safe}_{block[:8]}.csv")
    returns_df.to_csv(csv_out)
    print(f"  Saved returns -> {csv_out}")

    return {
        "returns":    returns_df,
        "cumulative": cum_df,
        "stats":      stats_df,
        "signal":     sig,
        "chart_path": chart_path,
    }


def _print_stats_table(
    stats_df: pd.DataFrame,
    asset: str,
    block: str,
    mode: str,
    threshold: float,
) -> None:
    block_label = LABELS.get(block, block)
    print(f"\nPerformance  |  {asset}  |  {block_label}  [{mode}, thr={threshold}]")
    print(f"  {'Metric':<20}  {'Strategy':>12}  {'Buy & Hold':>12}  {'DCA':>12}  {'Sig Edge':>10}")
    print("  " + "─" * 72)

    rows = [
        ("Ann. Return",  "ann_return", ".1%"),
        ("Ann. Vol",     "ann_vol",    ".1%"),
        ("Sharpe",       "sharpe",     ".3f"),
        ("Max Drawdown", "max_dd",     ".1%"),
        ("Win Rate",     "win_rate",   ".1%"),
        ("Calmar",       "calmar",     ".3f"),
    ]
    for display, key, fmt in rows:
        def _fmt(label):
            try:
                v = stats_df.loc[label, key]
                return format(v, fmt) if isinstance(v, float) else str(v)
            except KeyError:
                return "N/A"
        s_str  = _fmt("Strategy")
        bh_str = _fmt("Buy & Hold")
        dc_str = _fmt("DCA")
        try:
            edge = stats_df.loc["Strategy", key] - stats_df.loc["Buy & Hold", key]
            edge_str = format(edge, f"+{fmt}") if isinstance(edge, float) else ""
        except (KeyError, TypeError):
            edge_str = ""
        print(f"  {display:<20}  {s_str:>12}  {bh_str:>12}  {dc_str:>12}  {edge_str:>10}")


def _plot_backtest(
    cum_df:     pd.DataFrame,
    signal:     pd.Series,
    returns:    pd.Series,
    stats_df:   pd.DataFrame,
    asset:      str,
    block:      str,
    mode:       str,
    threshold:  float,
) -> str:
    block_label = LABELS.get(block, block)
    colour = PALETTE.get(block, "#2E86AB")

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )

    # ── Panel 1: Cumulative returns ────────────────────────────────────────
    ax = axes[0]
    strat_col = f"{asset}_Strategy"
    bh_col    = f"{asset}_BuyHold"

    ax.plot(cum_df.index, cum_df[strat_col],
            linewidth=2.2, color=colour, label="Liquidity-Timed")
    ax.plot(cum_df.index, cum_df[bh_col],
            linewidth=1.5, linestyle="--", alpha=0.6, color="grey",
            label="Buy & Hold")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)

    # Shade signal-on periods
    long_on  = signal >  0
    short_on = signal < -0 if mode == "long_short" else pd.Series(False, index=signal.index)
    ax.fill_between(signal.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=long_on,  alpha=0.07, color="green",  label="Long")
    if mode == "long_short":
        ax.fill_between(signal.index, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=(signal < 0), alpha=0.07, color="red", label="Short")

    # Annotate key stats in corner
    s  = stats_df.loc["Strategy"]
    bh = stats_df.loc["Buy & Hold"]
    txt = (f"Strategy   Sharpe {s['sharpe']:.2f}  |  Ann. Ret {s['ann_return']:+.1%}"
           f"  |  Max DD {s['max_dd']:.1%}\n"
           f"Buy & Hold  Sharpe {bh['sharpe']:.2f}  |  Ann. Ret {bh['ann_return']:+.1%}"
           f"  |  Max DD {bh['max_dd']:.1%}")
    ax.text(0.01, 0.03, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

    ax.set_title(f"{asset}  —  Liquidity Signal Backtest",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Cumulative Return (× 1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 2: Signal ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.step(signal.index, signal.values,
             linewidth=1.5, color=colour, where="post")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.fill_between(signal.index, 0, signal.values,
                     where=(signal > 0), alpha=0.25, color="green", step="post")
    ax2.fill_between(signal.index, 0, signal.values,
                     where=(signal < 0), alpha=0.25, color="red",   step="post")
    ax2.set_ylabel("Position", fontsize=9)
    ax2.set_title(f"Signal  [{block_label}  |  mode={mode}  |  threshold={threshold}]",
                  fontsize=9, loc="left", fontweight="bold")
    ax2.set_ylim(-1.4, 1.4)
    ax2.grid(True, alpha=0.15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 3: Monthly return bars (strategy vs B&H) ────────────────────
    ax3 = axes[2]
    bar_w = 20
    strat_r = cum_df[strat_col].pct_change().fillna(0)
    ax3.bar(strat_r.index, strat_r.values * 100,
            width=bar_w, color=np.where(strat_r >= 0, colour, "#E63946"),
            alpha=0.75, label="Strategy monthly")
    ax3.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax3.set_ylabel("Monthly %", fontsize=9)
    ax3.set_title("Monthly Returns (Strategy)", fontsize=9, loc="left",
                  fontweight="bold")
    ax3.grid(True, alpha=0.15, axis="y")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset.replace("/", "_")
    bname = block[:15].replace("/", "_")
    out   = os.path.join(RESULTS_DIR, f"backtest_{safe}_{bname}_{mode}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved chart  -> {out}")
    plt.close()
    return out


def _dca_returns(prices: pd.Series, monthly_contribution: float = 1000.0) -> pd.Series:
    """
    Simulate a fixed-dollar monthly DCA strategy against a price series.

    Each month a fixed dollar amount is invested at that month's closing price.
    Returns the month-over-month percentage change in total portfolio value,
    matching the index of `prices` (month-end).

    Parameters
    ----------
    prices               : pd.Series  Month-end close prices.
    monthly_contribution : float      Fixed dollar amount invested each month.

    Returns
    -------
    pd.Series  of monthly portfolio returns, same index as prices.
    """
    units_held      = 0.0
    portfolio_value = 0.0
    prev_value      = None
    returns         = {}

    for date, price in prices.items():
        if price <= 0 or np.isnan(price):
            continue
        # Buy this month's contribution at close
        units_held      += monthly_contribution / price
        portfolio_value  = units_held * price

        if prev_value is not None and prev_value > 0:
            returns[date] = (portfolio_value - prev_value - monthly_contribution) / prev_value
        else:
            returns[date] = 0.0          # first month: no prior base

        prev_value = portfolio_value

    s = pd.Series(returns, name="DCA")
    s.index = pd.to_datetime(s.index)
    return s


def _window_stats(
    ret_strat : pd.Series,
    ret_bh    : pd.Series,
    ret_dca   : pd.Series,
) -> dict:
    """
    Compute all performance metrics for one window across three strategies.
    Returns a flat dict keyed as  {strategy}_{metric}.
    """
    result = {}
    for label, r in [("Strategy", ret_strat),
                     ("BuyHold",  ret_bh),
                     ("DCA",      ret_dca)]:
        r = r.dropna()
        if len(r) < 3:
            for m in ("ann_return","ann_vol","sharpe","max_dd","win_rate","calmar"):
                result[f"{label}_{m}"] = np.nan
            continue
        ann_ret  = r.mean() * 12
        ann_vol  = r.std()  * np.sqrt(12)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum      = (1 + r).cumprod()
        dd       = (cum - cum.cummax()) / cum.cummax()
        max_dd   = dd.min()
        win_rate = (r > 0).mean()
        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        result.update({
            f"{label}_ann_return" : ann_ret,
            f"{label}_ann_vol"    : ann_vol,
            f"{label}_sharpe"     : sharpe,
            f"{label}_max_dd"     : max_dd,
            f"{label}_win_rate"   : win_rate,
            f"{label}_calmar"     : calmar,
        })
    return result


def _run_monte_carlo(
    monthly_ret  : pd.Series,   # full strategy monthly returns
    monthly_bh   : pd.Series,   # full buy-and-hold monthly returns
    monthly_dca  : pd.Series,   # full DCA monthly returns
    window_months: int,
    n_windows    : int,
    seed         : int = 42,
) -> pd.DataFrame:
    """
    Draw `n_windows` random start dates, slice a window of `window_months`
    from each series, compute stats, and return a DataFrame of all results.
    """
    rng = np.random.default_rng(seed)

    # Common valid index across all three series
    common = (monthly_ret.dropna().index
              .intersection(monthly_bh.dropna().index)
              .intersection(monthly_dca.dropna().index))
    common = common.sort_values()

    if len(common) < window_months + 1:
        # Diagnostic so the user can see exactly what date ranges are misaligned
        print(f"\n  ⚠  Date overlap diagnostic:")
        print(f"     Strategy returns : {monthly_ret.dropna().index[0].date()} → "
              f"{monthly_ret.dropna().index[-1].date()}  "
              f"({monthly_ret.dropna().shape[0]} months)")
        print(f"     Buy & Hold       : {monthly_bh.dropna().index[0].date()} → "
              f"{monthly_bh.dropna().index[-1].date()}  "
              f"({monthly_bh.dropna().shape[0]} months)")
        print(f"     DCA              : {monthly_dca.dropna().index[0].date()} → "
              f"{monthly_dca.dropna().index[-1].date()}  "
              f"({monthly_dca.dropna().shape[0]} months)")
        print(f"     Common overlap   : {len(common)} months  "
              f"(need {window_months + 1} for a {window_months}-month window)")

        max_window = len(common) - 1
        if max_window < 6:
            raise ValueError(
                f"Only {len(common)} overlapping months — too few for any meaningful analysis.\n"
                f"Ensure your CSV date range overlaps with the block score history "
                f"in results/block_pca_scores.csv."
            )
        window_months = max_window
        print(f"\n  ↳  Auto-shrinking window to {window_months} months "
              f"({window_months / 12:.1f} years) to fit available overlap.")

    max_start = len(common) - window_months
    rows = []
    for _ in range(n_windows):
        i   = int(rng.integers(0, max_start))
        idx = common[i : i + window_months]
        rows.append(_window_stats(
            monthly_ret.reindex(idx),
            monthly_bh.reindex(idx),
            monthly_dca.reindex(idx),
        ))

    return pd.DataFrame(rows)


def _summarise_mc(mc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse Monte Carlo results to  median | p5 | p95  per metric per strategy.
    Returns a tidy DataFrame with MultiIndex columns (strategy, stat).
    """
    strategies = ["Strategy", "BuyHold", "DCA"]
    metrics    = ["ann_return", "ann_vol", "sharpe", "max_dd", "win_rate", "calmar"]
    records    = []
    for strat in strategies:
        for metric in metrics:
            col = f"{strat}_{metric}"
            if col not in mc_df.columns:
                continue
            s = mc_df[col].dropna()
            records.append({
                "strategy" : strat,
                "metric"   : metric,
                "median"   : s.median(),
                "p5"       : s.quantile(0.05),
                "p95"      : s.quantile(0.95),
            })
    return pd.DataFrame(records).set_index(["strategy", "metric"])


def _plot_mc_distributions(
    mc_df      : pd.DataFrame,
    asset      : str,
    block      : str,
    mode       : str,
    window_yrs : float,
    n_windows  : int,
) -> str:
    """
    3×2 grid of histograms — one per metric.
    Each histogram overlays Strategy / Buy & Hold / DCA distributions.
    """
    metrics = [
        ("ann_return", "Ann. Return",  ".0%"),
        ("sharpe",     "Sharpe",       ".2f"),
        ("max_dd",     "Max Drawdown", ".0%"),
        ("ann_vol",    "Ann. Vol",     ".0%"),
        ("win_rate",   "Win Rate",     ".0%"),
        ("calmar",     "Calmar",       ".2f"),
    ]
    colours  = {"Strategy": PALETTE.get(block, "#2E86AB"),
                "BuyHold":  "grey",
                "DCA":      "#F18F01"}
    labels   = {"Strategy": "Liquidity Signal",
                "BuyHold":  "Buy & Hold",
                "DCA":      "DCA"}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (col, title, fmt) in zip(axes, metrics):
        for strat in ["Strategy", "BuyHold", "DCA"]:
            data = mc_df[f"{strat}_{col}"].dropna()
            ax.hist(data, bins=40, alpha=0.45,
                    color=colours[strat], label=labels[strat],
                    edgecolor="none")
            med = data.median()
            ax.axvline(med, color=colours[strat], linewidth=1.8, linestyle="--")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(title)
        ax.set_ylabel("Windows")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    block_label = LABELS.get(block, block)
    fig.suptitle(
        f"Monte Carlo Window Analysis  |  {asset}  |  {n_windows}× {window_yrs:.0f}-year windows\n"
        f"Signal: {block_label}  [{mode}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset.replace("/", "_")
    bname = block[:15].replace("/", "_")
    out   = os.path.join(RESULTS_DIR,
                         f"mc_distributions_{safe}_{bname}_{mode}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved MC chart   -> {out}")
    plt.close()
    return out


def _print_mc_summary(summary: pd.DataFrame, asset: str, block: str) -> None:
    """Pretty-print the median / p5 / p95 table."""
    block_label = LABELS.get(block, block)
    print(f"\nMonte Carlo Summary  |  {asset}  |  {block_label}")
    print(f"  {'Metric':<14}  "
          f"{'Strategy med':>13} {'[p5–p95]':>16}  "
          f"{'Buy&Hold med':>13} {'[p5–p95]':>16}  "
          f"{'DCA med':>10} {'[p5–p95]':>16}")
    print("  " + "─" * 98)

    fmt_map = {
        "ann_return": ".1%", "ann_vol": ".1%", "max_dd": ".1%",
        "win_rate": ".1%",   "sharpe": ".3f",  "calmar": ".3f",
    }
    display = {
        "ann_return": "Ann. Return", "ann_vol": "Ann. Vol",
        "sharpe": "Sharpe",         "max_dd": "Max DD",
        "win_rate": "Win Rate",      "calmar": "Calmar",
    }

    for metric, label in display.items():
        row = []
        for strat in ["Strategy", "BuyHold", "DCA"]:
            try:
                r   = summary.loc[(strat, metric)]
                fmt = fmt_map[metric]
                med = format(r["median"], fmt)
                p5  = format(r["p5"],     fmt)
                p95 = format(r["p95"],    fmt)
                row.append(f"{med:>13}  [{p5} – {p95}]")
            except KeyError:
                row.append(f"{'N/A':>13}  {'':>16}")
        print(f"  {label:<14}  {'  '.join(row)}")


def signal_backtest(
    csv_path             : str,
    asset_name           : str | None = None,
    block                : str  = "CB_Balance_Sheets_PC1",
    mode                 : str  = "long_only",
    threshold            : float = 0.0,
    lag                  : int  = 1,
    monthly_contribution : float = 1000.0,
    n_windows            : int  = 500,
    window_years         : float = 3.0,
    mc_seed              : int  = 42,
) -> dict:
    """
    Backtest a liquidity-driven signal against a real OHLCV asset, benchmarked
    against both Buy & Hold and a monthly DCA strategy.

    A Monte Carlo analysis over random sub-windows removes start/end date bias
    and shows the distribution of outcomes across all plausible entry periods.

    Parameters
    ----------
    csv_path             : str
        Headerless CSV.  Column order: timestamp, open, high, low, close, volume, trades.
        Timestamps: Unix ms, Unix s, or ISO strings — all auto-detected.

    asset_name           : str, optional
        Display label.  Defaults to the CSV filename stem.

    block                : str
        Block score column driving the signal.
        Options: any BLOCK_COLS entry, or 'Composite' (equal-weight PC1 average).

    mode                 : str
        'long_only'   — +1 when score > threshold, 0 otherwise.
        'long_short'  — +1 above +threshold, -1 below -threshold, 0 inside.
        'continuous'  — position ∈ [-1,+1] scaled by score; dead-band = threshold.

    threshold            : float   Entry/exit threshold in z-score units.
    lag                  : int     Signal lag in months (default 1 = no look-ahead).

    monthly_contribution : float
        Fixed dollar amount invested at month-end close for the DCA benchmark.

    n_windows            : int     Number of random windows for MC analysis (default 500).
    window_years         : float   Length of each random window in years (default 3).
    mc_seed              : int     Random seed for reproducibility.

    Returns
    -------
    dict
        'returns'      pd.DataFrame  Full-period monthly returns (3 columns).
        'cumulative'   pd.DataFrame  Full-period cumulative returns.
        'stats_full'   pd.DataFrame  Full-period stats (Strategy / BuyHold / DCA).
        'mc_raw'       pd.DataFrame  All MC window stats (n_windows rows).
        'mc_summary'   pd.DataFrame  Median / p5 / p95 per metric per strategy.
        'signal'       pd.Series     Monthly position signal.
        'chart_path'   str           Full-period backtest chart path.
        'mc_chart'     str           MC distribution chart path.

    Examples
    --------
    # Minimal call
    result = signal_backtest("data/BTC_daily.csv")
    print(result["mc_summary"])

    # Long/short with custom DCA amount
    result = signal_backtest(
        csv_path             = "data/ETH_daily.csv",
        asset_name           = "ETH",
        block                = "Composite",
        mode                 = "long_short",
        threshold            = 0.5,
        monthly_contribution = 500.0,
        n_windows            = 1000,
        window_years         = 2.0,
    )
    """
    print("\n" + "=" * 60)
    name = asset_name or os.path.splitext(os.path.basename(csv_path))[0]
    print(f"4. SIGNAL BACKTEST  |  {name}")
    print(f"   Block    : {block}  |  mode={mode}  |  thr={threshold}  |  lag={lag}m")
    print(f"   DCA      : ${monthly_contribution:,.0f}/month")
    print(f"   MC       : {n_windows} windows × {window_years:.1f} years  (seed={mc_seed})")
    print("=" * 60)

    # ── 1. Load block scores ───────────────────────────────────────────────
    scores = _load_scores()
    if block == "Composite":
        pc1_cols = [c for c in scores.columns if c.endswith("_PC1")]
        score_series = scores[pc1_cols].mean(axis=1)
        score_series.name = "Composite"
    elif block not in scores.columns:
        raise ValueError(
            f"'{block}' not found.\nAvailable: {list(scores.columns)} + 'Composite'"
        )
    else:
        score_series = scores[block]

    # ── 2. Load prices → monthly returns ──────────────────────────────────
    print("\nLoading price data...")
    daily_close   = _load_ohlcv(csv_path, name)
    asset_label   = daily_close.name
    monthly_price = daily_close.resample("ME").last()
    monthly_bh    = monthly_price.pct_change().dropna()
    monthly_bh.name = "BuyHold"

    print(f"  Price range  : {daily_close.index[0].date()} → {daily_close.index[-1].date()}")
    print(f"  Monthly obs  : {len(monthly_bh)}")

    # ── 3. Build signal & strategy returns ────────────────────────────────
    signal     = _build_signal(score_series, mode=mode,
                               threshold=threshold, lag=lag)
    common     = monthly_bh.index.intersection(signal.index)
    if len(common) < 12:
        raise ValueError(
            f"Only {len(common)} overlapping months. "
            f"Prices: {monthly_bh.index[0].date()} → {monthly_bh.index[-1].date()}  |  "
            f"Scores: {signal.index[0].date()} → {signal.index[-1].date()}"
        )

    ret_bh    = monthly_bh.loc[common]
    sig       = signal.loc[common]
    ret_strat = (ret_bh * sig).rename("Strategy")

    # ── 4. DCA returns ─────────────────────────────────────────────────────
    # DCA is run on the same common-index monthly prices (no signal dependency)
    ret_dca = _dca_returns(
        monthly_price.loc[common], monthly_contribution
    ).reindex(common).fillna(0.0)

    print(f"  Overlap      : {common[0].date()} → {common[-1].date()}  "
          f"({len(common)} months)")
    print(f"  Signal ON    : {(sig != 0).mean():.0%} of months")

    window_months_req = int(round(window_years * 12))
    if len(common) < window_months_req + 1:
        print(f"\n  ⚠  Overlap ({len(common)} months) is shorter than requested window "
              f"({window_months_req} months).")
        print(f"     Block scores span: {score_series.dropna().index[0].date()} → "
              f"{score_series.dropna().index[-1].date()}")
        print(f"     Price data spans : {monthly_bh.index[0].date()} → "
              f"{monthly_bh.index[-1].date()}")
        print(f"     Window will be auto-shrunk to {len(common) - 1} months.")

    # ── 5. Full-period stats ───────────────────────────────────────────────
    full_stats_dict = _window_stats(ret_strat, ret_bh, ret_dca)
    stats_full = pd.DataFrame([
        {"label": "Strategy",   **{k.replace("Strategy_", ""): v
                                   for k, v in full_stats_dict.items()
                                   if k.startswith("Strategy_")}},
        {"label": "Buy & Hold", **{k.replace("BuyHold_", ""): v
                                   for k, v in full_stats_dict.items()
                                   if k.startswith("BuyHold_")}},
        {"label": "DCA",        **{k.replace("DCA_", ""): v
                                   for k, v in full_stats_dict.items()
                                   if k.startswith("DCA_")}},
    ]).set_index("label")

    _print_stats_table(stats_full, asset_label, block, mode, threshold)

    # ── 6. Full-period chart ───────────────────────────────────────────────
    returns_df = pd.concat([ret_strat, ret_bh, ret_dca], axis=1)
    cum_df     = (1 + returns_df).cumprod()
    chart_path = _plot_backtest_full(
        cum_df, sig, ret_strat, stats_full, asset_label, block, mode, threshold
    )

    # ── 7. Monte Carlo ─────────────────────────────────────────────────────
    window_months = int(round(window_years * 12))
    print(f"\nRunning Monte Carlo  ({n_windows} windows × {window_months} months)...")
    mc_raw     = _run_monte_carlo(
        ret_strat, ret_bh, ret_dca,
        window_months=window_months,
        n_windows=n_windows,
        seed=mc_seed,
    )
    mc_summary = _summarise_mc(mc_raw)
    _print_mc_summary(mc_summary, asset_label, block)

    mc_chart = _plot_mc_distributions(
        mc_raw, asset_label, block, mode, window_years, n_windows
    )

    # ── 8. Save ────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset_label.replace("/", "_")
    bname = block[:15].replace("/", "_")
    returns_df.to_csv(
        os.path.join(RESULTS_DIR, f"backtest_{safe}_{bname}_{mode}_returns.csv")
    )
    mc_summary.to_csv(
        os.path.join(RESULTS_DIR, f"backtest_{safe}_{bname}_{mode}_mc_summary.csv")
    )
    print(f"  Saved returns    -> results/backtest_{safe}_{bname}_{mode}_returns.csv")
    print(f"  Saved MC summary -> results/backtest_{safe}_{bname}_{mode}_mc_summary.csv")

    return {
        "returns":    returns_df,
        "cumulative": cum_df,
        "stats_full": stats_full,
        "mc_raw":     mc_raw,
        "mc_summary": mc_summary,
        "signal":     sig,
        "chart_path": chart_path,
        "mc_chart":   mc_chart,
    }


def _plot_backtest_full(
    cum_df    : pd.DataFrame,
    signal    : pd.Series,
    ret_strat : pd.Series,
    stats_df  : pd.DataFrame,
    asset     : str,
    block     : str,
    mode      : str,
    threshold : float,
) -> str:
    """
    4-panel chart:
      1. Cumulative returns — Strategy vs Buy & Hold vs DCA
      2. Position signal
      3. Monthly return bars (strategy)
      4. Drawdown comparison
    """
    colour_s   = PALETTE.get(block, "#2E86AB")
    colour_bh  = "grey"
    colour_dca = "#F18F01"
    block_label = LABELS.get(block, block)

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 16),
        gridspec_kw={"height_ratios": [3, 1, 1, 1.5]},
        sharex=True,
    )

    # ── Panel 1: Cumulative returns ────────────────────────────────────────
    ax = axes[0]
    for col, colour, lbl, lw, ls in [
        ("Strategy",  colour_s,   "Liquidity Signal", 2.2, "-"),
        ("BuyHold",   colour_bh,  "Buy & Hold",       1.5, "--"),
        ("DCA",       colour_dca, "DCA",               1.5, "-."),
    ]:
        if col in cum_df.columns:
            ax.plot(cum_df.index, cum_df[col],
                    linewidth=lw, linestyle=ls, color=colour, label=lbl)

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    # Signal shading
    ax.fill_between(signal.index,
                    ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(signal > 0), alpha=0.05, color="green")
    if mode == "long_short":
        ax.fill_between(signal.index,
                        ax.get_ylim()[0], ax.get_ylim()[1],
                        where=(signal < 0), alpha=0.05, color="red")

    # Stats annotation
    rows_txt = []
    for lbl, idx_key in [("Signal", "Strategy"), ("B&H", "Buy & Hold"), ("DCA", "DCA")]:
        if idx_key in stats_df.index:
            s = stats_df.loc[idx_key]
            rows_txt.append(
                f"{lbl:<6} Ret {s['ann_return']:+.1%}  "
                f"Sharpe {s['sharpe']:.2f}  "
                f"DD {s['max_dd']:.1%}"
            )
    ax.text(0.01, 0.03, "\n".join(rows_txt), transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

    ax.set_title(f"{asset}  —  Full Period Backtest",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Cumulative Return (× 1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 2: Signal ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.step(signal.index, signal.values,
             linewidth=1.5, color=colour_s, where="post")
    ax2.fill_between(signal.index, 0, signal.values,
                     where=(signal > 0), alpha=0.25, color="green", step="post")
    ax2.fill_between(signal.index, 0, signal.values,
                     where=(signal < 0), alpha=0.25, color="red", step="post")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.set_ylabel("Position", fontsize=9)
    ax2.set_title(f"Signal  [{block_label}  |  {mode}  |  thr={threshold}]",
                  fontsize=9, loc="left", fontweight="bold")
    ax2.set_ylim(-1.4, 1.4)
    ax2.grid(True, alpha=0.15)

    # ── Panel 3: Monthly return bars ──────────────────────────────────────
    ax3 = axes[2]
    bar_w = 20
    r = ret_strat.reindex(cum_df.index).fillna(0)
    ax3.bar(r.index, r.values * 100,
            width=bar_w,
            color=np.where(r >= 0, colour_s, "#E63946"),
            alpha=0.75)
    ax3.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax3.set_ylabel("Monthly %", fontsize=9)
    ax3.set_title("Monthly Returns (Strategy)", fontsize=9,
                  loc="left", fontweight="bold")
    ax3.grid(True, alpha=0.15, axis="y")

    # ── Panel 4: Drawdown comparison ──────────────────────────────────────
    ax4 = axes[3]
    for col, colour, lbl, ls in [
        ("Strategy",  colour_s,   "Strategy",  "-"),
        ("BuyHold",   colour_bh,  "Buy & Hold","--"),
        ("DCA",       colour_dca, "DCA",        "-."),
    ]:
        if col in cum_df.columns:
            c  = cum_df[col]
            dd = (c - c.cummax()) / c.cummax() * 100
            ax4.fill_between(dd.index, dd.values, 0,
                             alpha=0.2, color=colour)
            ax4.plot(dd.index, dd.values,
                     linewidth=1.2, linestyle=ls,
                     color=colour, label=lbl)

    ax4.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax4.set_ylabel("Drawdown %", fontsize=9)
    ax4.set_title("Drawdown Comparison", fontsize=9,
                  loc="left", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.15)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset.replace("/", "_")
    bname = block[:15].replace("/", "_")
    out   = os.path.join(RESULTS_DIR,
                         f"backtest_{safe}_{bname}_{mode}_full.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved full chart -> {out}")
    plt.close()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEAD / LAG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def leading_indicators(target="Credit_Spreads_PC1", max_lag=6):
    """
    Cross-correlogram: correlation of each predictor with `target` at
    lags -max_lag ... +max_lag months.

    Positive lag = predictor leads target (actionable for timing entries).

    Saves: results/leadlag_{target}.png
    """
    print("\n" + "=" * 55)
    print(f"5. LEAD / LAG ANALYSIS  [target: {target}]")
    print("=" * 55)
    scores = _load_scores()

    if target not in scores.columns:
        raise ValueError(f"'{target}' not in scores. Options: {list(scores.columns)}")

    predictors = [c for c in scores.columns if c != target]
    lags       = range(-max_lag, max_lag + 1)
    records    = []

    for pred in predictors:
        for lag in lags:
            corr = scores[target].corr(scores[pred].shift(lag))
            records.append({"predictor": pred, "lag": lag, "correlation": corr})

    df = pd.DataFrame(records)
    best = (
        df.groupby("predictor")
          .apply(lambda g: g.loc[g["correlation"].abs().idxmax()],
                 include_groups=False)
          .reset_index()                        # predictor comes back as a column
          .sort_values("correlation", ascending=False)
    )

    target_label = LABELS.get(target, target)
    print(f"\nBest lead/lag per predictor  ->  {target_label}:\n")
    print(f"  {'Predictor':<38}  {'Lag':>6}  {'Corr':>7}  Direction")
    print("  " + "-" * 62)
    for _, row in best.iterrows():
        direction = "leads" if row["lag"] > 0 else ("lags" if row["lag"] < 0 else "same")
        print(f"  {LABELS.get(row['predictor'], row['predictor']):<38}  "
              f"{int(row['lag']):>+6}  {row['correlation']:>+6.3f}  ({direction})")

    fig, ax = plt.subplots(figsize=(12, 5))
    for pred in predictors:
        sub = df[df["predictor"] == pred]
        ax.plot(sub["lag"], sub["correlation"],
                marker="o", markersize=3.5, linewidth=1.5,
                color=PALETTE.get(pred, "grey"),
                label=LABELS.get(pred, pred).replace(" (PC1)", "").replace(" (PC2)", " PC2"))

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Lag (months) - positive = predictor leads target", fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title(f"Lead / Lag Analysis  ->  {target_label}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlim(-max_lag - 0.5, max_lag + 0.5)
    ax.set_xticks(list(lags))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig(f"leadlag_{target.replace('/', '_')}.png")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. BLOCK DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

def block_divergence(threshold=0.5, window=3):
    """
    Flag months where blocks are in disagreement.
    Divergence = std dev across PC1 scores.

    High divergence = mixed signals = regime-change risk.

    Saves: results/block_divergence.csv
           results/block_divergence.png
    """
    print("\n" + "=" * 55)
    print("6. BLOCK DIVERGENCE ANALYSIS")
    print("=" * 55)
    scores = _load_scores()
    pc1 = scores[[c for c in scores.columns if c.endswith("_PC1")]]

    div       = pc1.std(axis=1).rename("divergence")
    smoothed  = div.rolling(window).mean().rename("smoothed")
    flagged   = (div > threshold).rename("flagged")
    n_expand  = (pc1 > threshold).sum(axis=1).rename("n_expansionary")
    n_contract = (pc1 < -threshold).sum(axis=1).rename("n_contractionary")

    out = pd.concat([div, smoothed, flagged, n_expand, n_contract], axis=1)

    print(f"\nDivergence > {threshold}: {flagged.mean() * 100:.1f}% of months")
    print(f"Current divergence  ({out.index[-1].strftime('%Y-%m-%d')}):  "
          f"{div.iloc[-1]:.3f}  {'*** FLAGGED ***' if flagged.iloc[-1] else 'OK'}")
    print(f"Blocks expansionary:   {int(n_expand.iloc[-1])}/{len(pc1.columns)}")
    print(f"Blocks contractionary: {int(n_contract.iloc[-1])}/{len(pc1.columns)}")

    out.to_csv(os.path.join(RESULTS_DIR, "block_divergence.csv"))
    print("  Saved -> results/block_divergence.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for col in pc1.columns:
        ax1.plot(pc1.index, pc1[col], linewidth=1.2, alpha=0.75,
                 color=PALETTE.get(col, "grey"),
                 label=LABELS.get(col, col).replace(" (PC1)", ""))
    ax1.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax1.axhline(threshold,  color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax1.axhline(-threshold, color="red",   linewidth=0.5, linestyle=":", alpha=0.5)
    ax1.set_title("Block PC1 Scores", fontsize=11, fontweight="bold", loc="left")
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax1.grid(True, alpha=0.18)

    ax2.plot(div.index, div.values, color="grey", linewidth=1.0, alpha=0.5)
    ax2.plot(smoothed.index, smoothed.values, color="#E63946", linewidth=2.0,
             label=f"{window}M smoothed")
    ax2.axhline(threshold, color="orange", linewidth=1, linestyle="--",
                label=f"Threshold {threshold}")
    ax2.fill_between(div.index, div.values, threshold,
                     where=(div.values > threshold), alpha=0.2, color="orange",
                     label="High divergence")
    ax2.set_title("Cross-Block Divergence (std dev of PC1 scores)",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.set_ylabel("Std Dev")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.18)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    _savefig("block_divergence.png")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. LOADINGS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def loadings_report(top_n=8):
    """
    Print and chart the top/bottom indicator loadings for every block PC.
    Tells you *why* a block score is high or low on any given date.

    Saves: results/loadings_report.png
    """
    print("\n" + "=" * 55)
    print("7. LOADINGS REPORT")
    print("=" * 55)
    ld = _load_loadings()
    components = ld["component"].unique()

    fig, axes = plt.subplots(len(components), 1,
                             figsize=(12, 3.5 * len(components)))
    if len(components) == 1:
        axes = [axes]

    for ax, comp in zip(axes, components):
        sub = ld[ld["component"] == comp].sort_values("loading", ascending=False)
        ev  = sub["explained_var"].iloc[0]
        top = sub.head(top_n)
        bot = sub.tail(top_n).sort_values("loading")
        plot_df = pd.concat([top, bot]).drop_duplicates("indicator").sort_values("loading")

        colours = ["#E63946" if v < 0 else "#2E86AB" for v in plot_df["loading"]]
        ax.barh(plot_df["indicator"], plot_df["loading"], color=colours, height=0.7)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(f"{comp}  (explains {ev:.1%} variance)",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Loading")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="x", alpha=0.2)

        print(f"\n{comp}  (explained variance: {ev:.1%})")
        print(f"  Top {top_n} positive loadings:")
        for _, r in top.iterrows():
            print(f"    {r['indicator']:<40s}  {r['loading']:+.3f}")
        print(f"  Top {top_n} negative loadings:")
        for _, r in bot.sort_values("loading").iterrows():
            print(f"    {r['indicator']:<40s}  {r['loading']:+.3f}")

    plt.suptitle("Block PCA - Indicator Loadings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("loadings_report.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. LIVE REFRESH
# ══════════════════════════════════════════════════════════════════════════════

def run_live(fred_api_key):
    """
    Fetch the latest data, recompute block scores, print current readings.
    Use this for scheduled runs (cron / Windows Task Scheduler).

    Returns pd.DataFrame of block scores.
    """
    print("\n" + "=" * 55)
    print("8. LIVE REFRESH")
    print("=" * 55)
    from global_liquidity_factor import GlobalLiquidityFactor, Config

    Config.FRED_API_KEY = fred_api_key
    glf = GlobalLiquidityFactor(fred_api_key)

    data = glf.fetch_all_data()
    if data.empty:
        print("No data fetched - check API key and network.")
        return pd.DataFrame()

    glf.save_data()
    scores = glf.construct_block_factors()
    glf.plot_block_factors()
    glf.plot_block_correlation()

    print(f"\nCurrent readings  ({scores.index[-1].strftime('%Y-%m-%d')}):\n")
    print(f"  {'Block':<40}  {'Score':>7}  Regime")
    print("  " + "-" * 60)
    for col, val in scores.iloc[-1].items():
        arrow  = "^" if val > 0 else "v"
        regime = ("Expansionary"   if val >  0.5
                  else "Contractionary" if val < -0.5
                  else "Neutral")
        print(f"  {arrow} {LABELS.get(col, col):<39}  {val:>+6.3f}  {regime}")

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 9. TWO-STAGE WEEKLY CRYPTO BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_weekly_walcl(fred_api_key: str) -> pd.Series:
    """
    Fetch WALCL (Fed total assets) at weekly frequency directly from FRED.
    Returns a Series of asset levels indexed by Friday week-end dates.
    """
    import requests
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":     "WALCL",
        "api_key":       fred_api_key,
        "file_type":     "json",
        "observation_start": "2010-01-01",
        "frequency":     "w",        # weekly
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"])
         for o in obs if o["value"] != "."},
        name="WALCL",
    )
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _interpolate_monthly_to_weekly(monthly: pd.Series) -> pd.Series:
    """
    Upsample a monthly series to weekly using cubic-spline interpolation.
    Weekly index uses W-WED (Wednesday week-end) to align with FRED's WALCL.
    Steps:
      1. Resample monthly to W-WED — produces NaNs between month-ends
      2. Interpolate with cubic spline (smooth, no step artifacts)
      3. Clip to original date range to avoid spline overshoot at edges
    """
    weekly_idx = pd.date_range(
        start=monthly.index[0],
        end=monthly.index[-1],
        freq="W-WED",
    )
    # Reindex onto weekly grid (most values NaN), then spline
    upsampled = monthly.reindex(monthly.index.union(weekly_idx)).sort_index()
    interpolated = upsampled.interpolate(method="cubicspline")
    return interpolated.reindex(weekly_idx)


def _build_weekly_signal(
    walcl        : pd.Series,    # raw weekly WALCL levels
    monthly_scores: pd.DataFrame, # block_pca_scores (monthly)
    block        : str,
    level_threshold : float,
    momentum_window : int,        # weeks for WALCL rolling growth (default 4)
    lag_weeks    : int,           # weeks to lag before applying (default 1)
) -> pd.Series:
    """
    Construct the two-stage weekly signal.

    Stage 1 — Liquidity Level (monthly → weekly via spline):
        Use the named block score (or Composite).
        Signal-1 = score_weekly > level_threshold

    Stage 2 — Fed Balance Sheet Momentum (genuinely weekly):
        WALCL_4w_growth = (WALCL / WALCL.shift(momentum_window)) - 1
        Signal-2 = rolling_growth > 0

    Final signal = Signal-1 AND Signal-2, lagged by lag_weeks.
    Returns a weekly Series of {0, 1}.
    """
    # ── Stage 1: monthly level → weekly ───────────────────────────────────
    if block == "Composite":
        pc1_cols = [c for c in monthly_scores.columns if c.endswith("_PC1")]
        monthly_level = monthly_scores[pc1_cols].mean(axis=1)
    elif block not in monthly_scores.columns:
        raise ValueError(
            f"'{block}' not in block scores. "
            f"Options: {list(monthly_scores.columns)} + 'Composite'"
        )
    else:
        monthly_level = monthly_scores[block]

    print(f"     Monthly level    : {len(monthly_level)} months, "
          f"{monthly_level.index[0].date()} → {monthly_level.index[-1].date()}")

    weekly_level = _interpolate_monthly_to_weekly(monthly_level)
    print(f"     After spline     : {len(weekly_level)} weeks, "
          f"{weekly_level.index[0].date() if len(weekly_level) else 'EMPTY'} → "
          f"{weekly_level.index[-1].date() if len(weekly_level) else 'EMPTY'}")

    stage1 = (weekly_level > level_threshold).astype(float)
    print(f"     Stage 1 series   : {len(stage1)} weeks")

    # ── Stage 2: WALCL 4-week rolling growth ──────────────────────────────
    walcl_growth  = walcl.pct_change(periods=momentum_window)
    stage2        = (walcl_growth > 0).astype(float)
    print(f"     Stage 2 series   : {len(stage2)} weeks")

    # ── Combine on common weekly index ────────────────────────────────────
    common = stage1.index.intersection(stage2.index)
    print(f"     Common index     : {len(common)} weeks")
    if len(common) == 0:
        print(f"       Stage 1 index sample: {stage1.index[:3].tolist() if len(stage1) else 'empty'}")
        print(f"       Stage 2 index sample: {stage2.index[:3].tolist() if len(stage2) else 'empty'}")

    combined = (stage1.reindex(common) * stage2.reindex(common))  # AND

    # Lag by lag_weeks to avoid look-ahead
    return combined.shift(lag_weeks).rename("signal")


def _load_ohlcv_weekly(csv_path: str, asset_name: str | None = None) -> pd.Series:
    """
    Load a headerless OHLCV CSV and resample to weekly (Wednesday) close prices.
    Uses W-WED to align with FRED's WALCL weekly frequency.
    Column order: timestamp, open, high, low, close, volume, trades
    Timestamps: Unix seconds, Unix ms, or ISO strings — all auto-detected.
    """
    daily = _load_ohlcv(csv_path, asset_name)
    weekly = daily.resample("W-WED").last().dropna()
    return weekly


def _compute_weekly_stats(returns: pd.Series, label: str = "",
                          periods_per_year: int = 52) -> dict:
    """Performance stats for a weekly return series."""
    r = returns.dropna()
    if len(r) < 10:
        return {"label": label}
    ann_ret  = r.mean()  * periods_per_year
    ann_vol  = r.std()   * np.sqrt(periods_per_year)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = (1 + r).cumprod()
    dd       = (cum - cum.cummax()) / cum.cummax()
    max_dd   = dd.min()
    win_rate = (r > 0).mean()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return dict(label=label, ann_return=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, win_rate=win_rate, calmar=calmar)


def weekly_crypto_backtest(
    csv_path          : str,
    fred_api_key      : str,
    asset_name        : str | None = None,
    block             : str   = "CB_Balance_Sheets_PC1",
    level_threshold   : float = 0.0,
    momentum_window   : int   = 4,
    lag_weeks         : int   = 1,
    n_windows         : int   = 500,
    window_years      : float = 3.0,
    mc_seed           : int   = 42,
) -> dict:
    """
    Two-stage weekly liquidity signal backtest for crypto assets.

    Signal architecture
    -------------------
    Stage 1 — Liquidity Level  (monthly block score → spline-interpolated weekly)
        Long when block score (or Composite) > level_threshold.
        Captures the broad regime: is global liquidity expansionary?

    Stage 2 — Fed Balance Sheet Momentum  (WALCL, natively weekly on FRED)
        Long when WALCL 4-week rolling growth > 0.
        Captures the near-term impulse: is the Fed actively adding reserves?

    Entry  = Stage 1 AND Stage 2  (both conditions true)
    Exit   = either condition fails
    Signal lagged 1 week to avoid look-ahead bias.

    Parameters
    ----------
    csv_path        : str    Headerless OHLCV CSV (timestamp,o,h,l,c,v,trades).
    fred_api_key    : str    FRED API key — needed to fetch live WALCL weekly data.
    asset_name      : str    Display label (defaults to CSV filename stem).
    block           : str    Block score for Stage 1. Use 'Composite' for equal-weight PC1.
    level_threshold : float  Stage 1 z-score threshold (default 0.0).
    momentum_window : int    Weeks for WALCL rolling growth window (default 4).
    lag_weeks       : int    Signal lag in weeks (default 1 = no look-ahead).
    n_windows       : int    Monte Carlo windows (default 500).
    window_years    : float  MC window length in years (default 3).
    mc_seed         : int    RNG seed.

    Returns
    -------
    dict with keys:
        'returns'      pd.DataFrame  Weekly returns (Strategy / BuyHold columns)
        'cumulative'   pd.DataFrame  Cumulative returns
        'stats'        pd.DataFrame  Full-period performance stats
        'signal'       pd.Series     Weekly {0,1} signal
        'stage1'       pd.Series     Weekly Stage-1 level signal (before AND)
        'stage2'       pd.Series     Weekly Stage-2 momentum signal (before AND)
        'walcl'        pd.Series     Raw WALCL weekly series
        'mc_summary'   pd.DataFrame  Monte Carlo median/p5/p95
        'chart_path'   str
        'mc_chart'     str

    Example
    -------
    result = weekly_crypto_backtest(
        csv_path       = "data/BTC_daily.csv",
        fred_api_key   = "YOUR_KEY",
        asset_name     = "BTC",
        block          = "CB_Balance_Sheets_PC1",
        level_threshold = 0.0,
    )
    print(result["stats"])
    """
    name = asset_name or os.path.splitext(os.path.basename(csv_path))[0]

    print("\n" + "=" * 62)
    print(f"9. TWO-STAGE WEEKLY CRYPTO BACKTEST  |  {name}")
    print(f"   Stage 1 : {block} > {level_threshold}  (monthly → weekly spline)")
    print(f"   Stage 2 : WALCL {momentum_window}-week growth > 0  (natively weekly)")
    print(f"   Lag     : {lag_weeks} week(s)  |  MC: {n_windows}× {window_years:.1f}yr windows")
    print("=" * 62)

    # ── 1. Load monthly block scores ──────────────────────────────────────
    monthly_scores = _load_scores()

    # ── 2. Fetch WALCL weekly from FRED ───────────────────────────────────
    print("\nFetching WALCL (weekly Fed assets) from FRED...")
    walcl = _fetch_weekly_walcl(fred_api_key)
    print(f"  WALCL range : {walcl.index[0].date()} → {walcl.index[-1].date()}"
          f"  ({len(walcl)} weeks)")

    # ── 3. Load asset prices → weekly returns ─────────────────────────────
    print("Loading price data...")
    weekly_price = _load_ohlcv_weekly(csv_path, name)
    asset_label  = weekly_price.name
    weekly_ret   = weekly_price.pct_change().dropna()
    print(f"  Price range : {weekly_price.index[0].date()} → {weekly_price.index[-1].date()}"
          f"  ({len(weekly_ret)} weeks)")

    # ── 4. Build two-stage signal ─────────────────────────────────────────
    print("\nBuilding two-stage signal...")
    signal = _build_weekly_signal(
        walcl=walcl,
        monthly_scores=monthly_scores,
        block=block,
        level_threshold=level_threshold,
        momentum_window=momentum_window,
        lag_weeks=lag_weeks,
    )

    if signal.empty or signal.isna().all():
        print(f"\n  ⚠  Signal construction failed — no overlap between components.")
        print(f"     Monthly scores : {monthly_scores.index[0].date()} → {monthly_scores.index[-1].date()}")
        print(f"     WALCL (weekly) : {walcl.index[0].date()} → {walcl.index[-1].date()}")
        print(f"     After spline interpolation and Stage-1/Stage-2 intersection, "
              f"no common dates remain.")
        print(f"\n  Most likely cause: block_pca_scores.csv has too short a history.")
        print(f"  Rerun construct_block_factors() after verifying _prepare_block thresh fix is applied.")
        raise ValueError("Signal series is empty — check date range overlap above.")

    # Expose individual stages for diagnostics
    if block == "Composite":
        pc1_cols = [c for c in monthly_scores.columns if c.endswith("_PC1")]
        monthly_level = monthly_scores[pc1_cols].mean(axis=1)
    else:
        monthly_level = monthly_scores[block]
    stage1_weekly = _interpolate_monthly_to_weekly(monthly_level)
    stage1_series = (stage1_weekly > level_threshold).astype(float)
    walcl_growth  = walcl.pct_change(periods=momentum_window)
    stage2_series = (walcl_growth > 0).astype(float)

    # ── 5. Align to common weekly index ───────────────────────────────────
    common = weekly_ret.index.intersection(signal.index)
    if len(common) < 52:
        raise ValueError(
            f"Only {len(common)} overlapping weeks.\n"
            f"  Prices : {weekly_ret.index[0].date()} → {weekly_ret.index[-1].date()}\n"
            f"  Signal : {signal.index[0].date()} → {signal.index[-1].date()}\n"
            f"Check that block_pca_scores.csv date range overlaps with your price data."
        )

    ret   = weekly_ret.loc[common]
    sig   = signal.loc[common].fillna(0)
    strat = (ret * sig).rename("Strategy")
    bh    = ret.rename("BuyHold")

    on_pct = (sig > 0).mean()
    print(f"\n  Overlap     : {common[0].date()} → {common[-1].date()}"
          f"  ({len(common)} weeks)")
    print(f"  Signal ON   : {on_pct:.1%} of weeks")
    print(f"  Stage 1 ON  : {stage1_series.reindex(common).fillna(0).mean():.1%}"
          f"  (level filter)")
    print(f"  Stage 2 ON  : {stage2_series.reindex(common).fillna(0).mean():.1%}"
          f"  (momentum filter)")

    returns_df = pd.concat([strat, bh], axis=1)
    cum_df     = (1 + returns_df).cumprod()

    # ── 6. Full-period stats ───────────────────────────────────────────────
    stats_list = [
        _compute_weekly_stats(strat, "Strategy"),
        _compute_weekly_stats(bh,    "Buy & Hold"),
    ]
    stats_df = pd.DataFrame(stats_list).set_index("label")
    _print_weekly_stats(stats_df, asset_label, block, level_threshold, momentum_window)

    # ── 7. Chart ───────────────────────────────────────────────────────────
    chart_path = _plot_weekly_backtest(
        cum_df, sig, stage1_series, stage2_series, walcl,
        strat, stats_df, asset_label, block, level_threshold, momentum_window
    )

    # ── 8. Monte Carlo ─────────────────────────────────────────────────────
    window_weeks = int(round(window_years * 52))
    print(f"\nRunning Monte Carlo  ({n_windows} windows × {window_weeks} weeks)...")

    rng    = np.random.default_rng(mc_seed)
    common_w = strat.index.intersection(bh.index)
    max_start = len(common_w) - window_weeks
    if max_start < 1:
        window_weeks = len(common_w) - 1
        max_start    = 1
        print(f"  ↳  Auto-shrunk window to {window_weeks} weeks to fit overlap.")

    mc_rows = []
    for _ in range(n_windows):
        i   = int(rng.integers(0, max_start)) if max_start > 1 else 0
        idx = common_w[i : i + window_weeks]
        rs  = strat.reindex(idx)
        rb  = bh.reindex(idx)
        mc_rows.append({
            "Strategy_ann_return" : rs.mean() * 52,
            "Strategy_sharpe"     : (rs.mean() * 52) / (rs.std() * np.sqrt(52))
                                     if rs.std() > 0 else np.nan,
            "Strategy_max_dd"     : ((1+rs).cumprod() /
                                      (1+rs).cumprod().cummax() - 1).min(),
            "Strategy_win_rate"   : (rs > 0).mean(),
            "BuyHold_ann_return"  : rb.mean() * 52,
            "BuyHold_sharpe"      : (rb.mean() * 52) / (rb.std() * np.sqrt(52))
                                     if rb.std() > 0 else np.nan,
            "BuyHold_max_dd"      : ((1+rb).cumprod() /
                                      (1+rb).cumprod().cummax() - 1).min(),
            "BuyHold_win_rate"    : (rb > 0).mean(),
        })

    mc_df = pd.DataFrame(mc_rows)
    mc_summary = _summarise_weekly_mc(mc_df)
    _print_weekly_mc_summary(mc_summary, asset_label, block)

    mc_chart = _plot_weekly_mc(
        mc_df, asset_label, block, level_threshold, momentum_window,
        window_years, n_windows
    )

    # ── 9. Save ────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset_label.replace("/", "_")
    bname = block[:15].replace("/", "_")
    stem  = f"weekly_{safe}_{bname}"
    returns_df.to_csv(os.path.join(RESULTS_DIR, f"{stem}_returns.csv"))
    mc_summary.to_csv(os.path.join(RESULTS_DIR, f"{stem}_mc_summary.csv"))
    print(f"  Saved returns    -> results/{stem}_returns.csv")
    print(f"  Saved MC summary -> results/{stem}_mc_summary.csv")

    return {
        "returns":    returns_df,
        "cumulative": cum_df,
        "stats":      stats_df,
        "signal":     sig,
        "stage1":     stage1_series.reindex(common),
        "stage2":     stage2_series.reindex(common),
        "walcl":      walcl,
        "mc_summary": mc_summary,
        "chart_path": chart_path,
        "mc_chart":   mc_chart,
    }


def _print_weekly_stats(stats_df, asset, block, threshold, momentum_window):
    block_label = LABELS.get(block, block)
    print(f"\nFull-Period Performance  |  {asset}  |  weekly  |  {block_label}")
    print(f"  Stage 1: score > {threshold}   Stage 2: WALCL {momentum_window}-week growth > 0")
    print(f"  {'Metric':<20}  {'Strategy':>12}  {'Buy & Hold':>12}  {'Edge':>10}")
    print("  " + "─" * 58)
    rows = [("Ann. Return","ann_return",".1%"), ("Ann. Vol","ann_vol",".1%"),
            ("Sharpe","sharpe",".3f"), ("Max Drawdown","max_dd",".1%"),
            ("Win Rate","win_rate",".1%"), ("Calmar","calmar",".3f")]
    for display, key, fmt in rows:
        def _f(lbl):
            try:
                v = stats_df.loc[lbl, key]
                return format(v, fmt) if isinstance(v, float) and not np.isnan(v) else "N/A"
            except KeyError:
                return "N/A"
        try:
            edge = stats_df.loc["Strategy", key] - stats_df.loc["Buy & Hold", key]
            edge_str = format(edge, f"+{fmt}") if isinstance(edge, float) else ""
        except (KeyError, TypeError):
            edge_str = ""
        print(f"  {display:<20}  {_f('Strategy'):>12}  {_f('Buy & Hold'):>12}  {edge_str:>10}")


def _summarise_weekly_mc(mc_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ann_return", "sharpe", "max_dd", "win_rate"]
    records = []
    for strat in ["Strategy", "BuyHold"]:
        for metric in metrics:
            col = f"{strat}_{metric}"
            if col not in mc_df.columns:
                continue
            s = mc_df[col].dropna()
            records.append({"strategy": strat, "metric": metric,
                            "median": s.median(), "p5": s.quantile(0.05),
                            "p95": s.quantile(0.95)})
    return pd.DataFrame(records).set_index(["strategy", "metric"])


def _print_weekly_mc_summary(summary: pd.DataFrame, asset: str, block: str) -> None:
    block_label = LABELS.get(block, block)
    print(f"\nMonte Carlo Summary  |  {asset}  |  weekly  |  {block_label}")
    fmt_map = {"ann_return":".1%", "sharpe":".3f", "max_dd":".1%", "win_rate":".1%"}
    display = {"ann_return":"Ann. Return", "sharpe":"Sharpe",
               "max_dd":"Max DD", "win_rate":"Win Rate"}
    print(f"  {'Metric':<14}  {'Strategy med':>13} {'[p5–p95]':>18}  "
          f"{'BuyHold med':>13} {'[p5–p95]':>18}")
    print("  " + "─" * 82)
    for metric, label in display.items():
        row = []
        for strat in ["Strategy", "BuyHold"]:
            try:
                r   = summary.loc[(strat, metric)]
                fmt = fmt_map[metric]
                row.append(f"{format(r['median'], fmt):>13}  "
                           f"[{format(r['p5'], fmt)} – {format(r['p95'], fmt)}]")
            except KeyError:
                row.append(f"{'N/A':>13}  {'':>18}")
        print(f"  {label:<14}  {'  '.join(row)}")


def _plot_weekly_backtest(
    cum_df, signal, stage1, stage2, walcl,
    ret_strat, stats_df, asset, block, threshold, momentum_window
) -> str:
    colour    = PALETTE.get(block, "#2E86AB")
    bh_colour = "grey"
    block_label = LABELS.get(block, block)

    fig, axes = plt.subplots(
        5, 1, figsize=(16, 18),
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 1.5]},
        sharex=True,
    )

    # ── Panel 1: Cumulative returns ────────────────────────────────────────
    ax = axes[0]
    ax.plot(cum_df.index, cum_df["Strategy"],
            linewidth=2.0, color=colour, label="Two-Stage Signal")
    ax.plot(cum_df.index, cum_df["BuyHold"],
            linewidth=1.4, linestyle="--", alpha=0.6,
            color=bh_colour, label="Buy & Hold")
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.fill_between(signal.index,
                    ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(signal > 0), alpha=0.05, color="green")

    if "Strategy" in stats_df.index and "Buy & Hold" in stats_df.index:
        s  = stats_df.loc["Strategy"]
        bh = stats_df.loc["Buy & Hold"]
        txt = (f"Signal  Ret {s.get('ann_return', float('nan')):+.1%}  "
               f"Sharpe {s.get('sharpe', float('nan')):.2f}  "
               f"DD {s.get('max_dd', float('nan')):.1%}\n"
               f"B&H     Ret {bh.get('ann_return', float('nan')):+.1%}  "
               f"Sharpe {bh.get('sharpe', float('nan')):.2f}  "
               f"DD {bh.get('max_dd', float('nan')):.1%}")
        ax.text(0.01, 0.03, txt, transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

    ax.set_title(f"{asset}  —  Two-Stage Weekly Signal Backtest",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Cumulative Return")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 2: Combined signal ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.step(signal.index, signal.values,
             linewidth=1.4, color=colour, where="post")
    ax2.fill_between(signal.index, 0, signal.values,
                     alpha=0.3, color="green", step="post")
    ax2.set_ylim(-0.2, 1.4)
    ax2.set_ylabel("Signal", fontsize=9)
    ax2.set_title("Combined Signal (Stage 1 AND Stage 2)", fontsize=9,
                  loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.15)

    # ── Panel 3: Stage 1 — level ───────────────────────────────────────────
    ax3 = axes[2]
    common_idx = stage1.index.intersection(signal.index)
    s1 = stage1.reindex(common_idx)
    ax3.step(s1.index, s1.values,
             linewidth=1.2, color="#A23B72", where="post", alpha=0.8)
    ax3.fill_between(s1.index, 0, s1.values,
                     alpha=0.2, color="#A23B72", step="post")
    ax3.set_ylim(-0.2, 1.4)
    ax3.set_ylabel("ON/OFF", fontsize=9)
    ax3.set_title(f"Stage 1  —  {block_label} > {threshold}  (monthly → weekly spline)",
                  fontsize=9, loc="left", fontweight="bold")
    ax3.grid(True, alpha=0.15)

    # ── Panel 4: Stage 2 — WALCL momentum ─────────────────────────────────
    ax4 = axes[3]
    s2 = stage2.reindex(common_idx)
    walcl_growth = walcl.pct_change(periods=momentum_window).reindex(common_idx) * 100
    ax4_r = ax4.twinx()
    ax4_r.plot(walcl_growth.index, walcl_growth.values,
               linewidth=0.8, color="lightgrey", alpha=0.7, label="WALCL 4w growth %")
    ax4.step(s2.index, s2.values,
             linewidth=1.2, color="#F18F01", where="post", alpha=0.9)
    ax4.fill_between(s2.index, 0, s2.values,
                     alpha=0.2, color="#F18F01", step="post")
    ax4.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
    ax4.set_ylim(-0.2, 1.4)
    ax4.set_ylabel("ON/OFF", fontsize=9)
    ax4_r.set_ylabel("WALCL growth %", fontsize=8, color="grey")
    ax4.set_title(f"Stage 2  —  WALCL {momentum_window}-week growth > 0  (natively weekly)",
                  fontsize=9, loc="left", fontweight="bold")
    ax4.grid(True, alpha=0.15)

    # ── Panel 5: Drawdown ──────────────────────────────────────────────────
    ax5 = axes[4]
    for col, clr, lbl, ls in [
        ("Strategy",  colour,    "Strategy",   "-"),
        ("BuyHold",   bh_colour, "Buy & Hold", "--"),
    ]:
        if col in cum_df.columns:
            c  = cum_df[col]
            dd = (c - c.cummax()) / c.cummax() * 100
            ax5.fill_between(dd.index, dd.values, 0, alpha=0.18, color=clr)
            ax5.plot(dd.index, dd.values, linewidth=1.2,
                     linestyle=ls, color=clr, label=lbl)
    ax5.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax5.set_ylabel("Drawdown %", fontsize=9)
    ax5.set_title("Drawdown", fontsize=9, loc="left", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.15)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset.replace("/", "_")
    bname = block[:15].replace("/", "_")
    out   = os.path.join(RESULTS_DIR, f"weekly_backtest_{safe}_{bname}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved chart      -> {out}")
    plt.close()
    return out


def _plot_weekly_mc(mc_df, asset, block, threshold, momentum_window,
                    window_years, n_windows) -> str:
    metrics  = [("ann_return","Ann. Return"), ("sharpe","Sharpe"),
                ("max_dd","Max Drawdown"),    ("win_rate","Win Rate")]
    colours  = {"Strategy": PALETTE.get(block, "#2E86AB"), "BuyHold": "grey"}
    labels   = {"Strategy": "Two-Stage Signal", "BuyHold": "Buy & Hold"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for strat in ["Strategy", "BuyHold"]:
            data = mc_df.get(f"{strat}_{col}", pd.Series(dtype=float)).dropna()
            if data.empty:
                continue
            ax.hist(data, bins=40, alpha=0.45,
                    color=colours[strat], label=labels[strat], edgecolor="none")
            ax.axvline(data.median(), color=colours[strat],
                       linewidth=1.8, linestyle="--")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(title)
        ax.set_ylabel("Windows")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    block_label = LABELS.get(block, block)
    fig.suptitle(
        f"Weekly Two-Stage MC  |  {asset}  |  {n_windows}× {window_years:.0f}-year windows\n"
        f"Stage 1: {block_label} > {threshold}   "
        f"Stage 2: WALCL {momentum_window}-week growth > 0",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe  = asset.replace("/", "_")
    bname = block[:15].replace("/", "_")
    out   = os.path.join(RESULTS_DIR, f"weekly_mc_{safe}_{bname}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved MC chart   -> {out}")
    plt.close()
    return out


if __name__ == "__main__":
    result = weekly_crypto_backtest(
        csv_path        = r"C:\Users\st_ha\OneDrive\Documents\github\automated_trading_test_environment.jl\src\data\Kraken_OHLCVT\XBTUSD_1440.csv",
        fred_api_key    = "6dd5d65c0b6778e9433a7934ee82eb94",
        asset_name      = "BTC",
        block           = "Composite",
        level_threshold = 0.0,
        momentum_window = 1,      # weeks for WALCL rolling growth
        lag_weeks       = 1,
    )
    print(result["stats"])
