import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")



"""
compute_market_neutral_pnl_adaptive

Two-regime trading strategy:
  HIGH-VOL REGIME : XGBoost residual signal (crisis alpha)
  CALM REGIME     : VOL_SPREAD mean reversion (pairs trading)

Regime detection uses rolling z-score of log(realized_vol), computed
from past data (no lookahead), identical to the main CV implementation.

COST MODEL (EEX-eyeballed, one-way):
  spread:  15 bps   bid-ask, liquid front month
  fees:     4 bps   exchange + clearing
  impact:   6 bps   market impact
  total:  ~25 bps   (very) rough baseline
  use cost_bps to override with a single number
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_zscore_series(series: pd.Series, window: int, min_periods: int = 30) -> pd.Series:
    """Rolling z-score, past-only (shift(1) applied). No lookahead."""
    mu  = series.rolling(window, min_periods=min_periods).mean().shift(1)
    sig = series.rolling(window, min_periods=min_periods).std().shift(1).clip(lower=1e-8)
    return (series - mu) / sig


def _reneutralise(df: pd.DataFrame, signal_col: str) -> pd.Series:
    """Demean signal cross-sectionally each day, then normalise abs-sum to 1."""
    demeaned = df[signal_col] - df.groupby('DATE')[signal_col].transform('mean')
    abs_sum  = demeaned.groupby(df['DATE']).transform(lambda x: x.abs().sum()).replace(0, np.nan)
    return (demeaned / abs_sum).fillna(0.0)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_market_neutral_pnl_adaptive(
    bt: pd.DataFrame,
    # ----------------------------------------------------------------
    # Regime detection — based on log(realized_vol), same as main CV
    # ----------------------------------------------------------------
    vol_regime_window: int = 252,       # rolling window for zscore baseline
    vol_regime_threshold: float = 0.5,  # zscore above this = high-vol regime
    # ----------------------------------------------------------------
    # HIGH-VOL regime: XGBoost residual signal
    # ----------------------------------------------------------------
    threshold: float = 0.5,             # signal entry threshold (z of pred)
    lookback: int = 30,                 # rolling window for pred z-score
    clip_signal: float = 2.0,
    z_fillna: float = 0.0,
    adaptive_threshold: str = "static", # {"static","quantile","vol"}
    quantile: float = 0.85,
    vol_k: float = 1.0,
    min_threshold: float = 0.1,
    ewma_alpha: float = None,
    # ----------------------------------------------------------------
    # CALM regime: VOL_SPREAD mean reversion
    # Enters long DE / short FR when spread is unusually low (DE cheap)
    # and vice versa. Exits when spread reverts toward mean.
    # Requires DE_realized_vol and FR_realized_vol in bt.
    # ----------------------------------------------------------------
    use_calm_strategy: bool = True,
    calm_spread_window: int = 60,       # rolling window for spread z-score
    calm_entry_zscore: float = 1.0,     # enter when |spread_z| > this
    calm_exit_zscore: float = 0.25,     # exit when |spread_z| < this
    calm_position_scale: float = 0.3,   # max position size in calm regime
                                        # (fraction of full position)
    # ----------------------------------------------------------------
    # HIGH-VOL position scale (1.0 = full, lower = more conservative)
    # ----------------------------------------------------------------
    high_vol_position_scale: float = 1.0,
    # ----------------------------------------------------------------
    # Vol targeting (optional, applied on top of regime sizing)
    # ----------------------------------------------------------------
    enable_vol_target: bool = False,
    target_ann_vol: float = 0.10,
    vol_window: int = 126,
    vol_cap: float = 2.0,
    vol_floor: float = 0.1,
    # ----------------------------------------------------------------
    # Rebalancing
    # ----------------------------------------------------------------
    rebalance_thr: float = None,
    # ----------------------------------------------------------------
    # Costs
    # ----------------------------------------------------------------
    cost_bps_spread: float = 0.0015,
    cost_bps_fees: float = 0.0004,
    cost_bps_impact: float = 0.0006,
    cost_bps: float = None,             # single override (legacy)
    # ----------------------------------------------------------------
    # Legacy Markov regime (backward compat, ignored if use_vol_regime)
    # ----------------------------------------------------------------
    use_regime_prob: dict = None,
    regime_shrink_to: float = 0.3,
    # ----------------------------------------------------------------
    verbose: bool = False,
) -> pd.DataFrame:

    df = bt.copy().sort_values(['DATE', 'COUNTRY']).reset_index(drop=True)

    total_cost_bps = cost_bps if cost_bps is not None else (
        cost_bps_spread + cost_bps_fees + cost_bps_impact
    )

    # ----------------------------------------------------------------
    # STEP 1: Regime detection from log(realized_vol)
    # ----------------------------------------------------------------
    # Requires DE_realized_vol and FR_realized_vol columns.
    # If not present, fall back to abs(true) zscore with a warning.
    df['vol_regime_zscore'] = 0.0

    rv_found = False
    for pref in ['DE', 'FR']:
        rv_col = f'{pref}_realized_vol'
        if rv_col not in df.columns:
            continue
        rv_found = True
        mask = df['COUNTRY'] == pref
        log_rv = np.log(df.loc[mask, rv_col].astype(float).clip(lower=1e-12))
        zs = _safe_zscore_series(log_rv, vol_regime_window)
        df.loc[mask, 'vol_regime_zscore'] = zs.values

    if not rv_found:
        if verbose:
            print("WARNING: DE/FR realized_vol columns not found. "
                  "Falling back to abs(true) zscore for regime detection. "
                  "Merge panel rv columns into bt before calling this function.")
        abs_true = df.groupby('COUNTRY')['true'].transform(lambda x: x.abs())
        df['vol_regime_zscore'] = df.groupby('COUNTRY')['true'].transform(
            lambda x: _safe_zscore_series(x.abs(), vol_regime_window)
        ).fillna(0.0)

    df['vol_regime_flag'] = (df['vol_regime_zscore'] > vol_regime_threshold).astype(int)
    # 1 = high-vol, 0 = calm

    # ----------------------------------------------------------------
    # STEP 2: High-vol signal — XGBoost residual z-score
    # ----------------------------------------------------------------
    grp = df.groupby('COUNTRY')['pred']
    df['pred_mean_past'] = grp.transform(
        lambda x: x.rolling(lookback, min_periods=5).mean().shift(1)
    )
    df['pred_std_past'] = grp.transform(
        lambda x: x.rolling(lookback, min_periods=5).std().shift(1)
    ).fillna(
        df.groupby('COUNTRY')['pred'].transform(
            lambda x: x.rolling(lookback, min_periods=5).std().shift(1)
        ).groupby(df['COUNTRY']).transform('median')
    ).replace(0, 1e-6)

    df['z_pred'] = ((df['pred'] - df['pred_mean_past']) / df['pred_std_past']).fillna(z_fillna)

    # Adaptive threshold
    if adaptive_threshold == 'static':
        df['threshold_used'] = threshold
    elif adaptive_threshold == 'quantile':
        df['threshold_used'] = df.groupby('COUNTRY')['z_pred'].transform(
            lambda x: x.abs().rolling(lookback, min_periods=5).quantile(quantile).shift(1)
        ).fillna(threshold)
    elif adaptive_threshold == 'vol':
        df['threshold_used'] = (vol_k * df.groupby('COUNTRY')['z_pred'].transform(
            lambda x: x.abs().rolling(lookback, min_periods=5).std().shift(1)
        )).fillna(threshold)
    else:
        df['threshold_used'] = threshold
    df['threshold_used'] = df['threshold_used'].fillna(threshold).clip(lower=min_threshold)

    df['signal_hv_raw'] = np.where(df['z_pred'].abs() > df['threshold_used'], df['z_pred'], 0.0)
    df['signal_hv'] = df['signal_hv_raw'].clip(-clip_signal, clip_signal)

    if ewma_alpha is not None and 0.0 < ewma_alpha <= 1.0:
        df['signal_hv'] = df.groupby('COUNTRY')['signal_hv'].transform(
            lambda s: s.fillna(0).ewm(alpha=ewma_alpha, adjust=False).mean()
        )

    # Cross-sectional neutralise high-vol signal
    df['signal_hv_cs'] = df['signal_hv'] - df.groupby('DATE')['signal_hv'].transform('mean')

    # ----------------------------------------------------------------
    # STEP 3: Calm-regime signal — VOL_SPREAD mean reversion
    # ----------------------------------------------------------------
    df['signal_calm'] = 0.0
    df['spread_zscore'] = 0.0

    if use_calm_strategy:
        # VOL_SPREAD = DE_realized_vol - FR_realized_vol
        # Use log spread for stationarity: log(DE_rv) - log(FR_rv)
        de_rv_col = 'DE_realized_vol'
        fr_rv_col = 'FR_realized_vol'

        if de_rv_col in df.columns and fr_rv_col in df.columns:
            # Build a date-indexed spread series from the DE rows
            # (spread is the same for both countries on a given date)
            date_df = (df[df['COUNTRY'] == 'DE']
                       .set_index('DATE')[[de_rv_col, fr_rv_col]]
                       .sort_index())
            log_spread = (
                np.log(date_df[de_rv_col].clip(lower=1e-12))
                - np.log(date_df[fr_rv_col].clip(lower=1e-12))
            )
            spread_z = _safe_zscore_series(log_spread, calm_spread_window)

            # Map back to full df
            spread_z_map = spread_z.to_dict()
            df['spread_zscore'] = df['DATE'].map(spread_z_map).fillna(0.0)

            # Mean reversion signal:
            # When spread_z > entry: DE vol relatively high → short DE, long FR
            # When spread_z < -entry: FR vol relatively high → long DE, short FR
            # Exit when |spread_z| < exit threshold
            def _calm_signal_for_country(row):
                sz = row['spread_zscore']
                c  = row['COUNTRY']
                if abs(sz) < calm_exit_zscore:
                    return 0.0
                if abs(sz) < calm_entry_zscore:
                    return 0.0   # between exit and entry: no new position
                # spread_z > 0 means DE vol high relative to FR
                # → short DE (expect reversion down), long FR
                direction = np.sign(sz)
                return -direction if c == 'DE' else direction

            df['signal_calm_raw'] = df.apply(_calm_signal_for_country, axis=1)

            # Scale calm signal
            df['signal_calm'] = df['signal_calm_raw'] * calm_position_scale

        else:
            if verbose:
                print("WARNING: DE/FR realized_vol not found — calm strategy disabled.")

    # ----------------------------------------------------------------
    # STEP 4: Blend strategies by regime
    # ----------------------------------------------------------------
    # High-vol regime: use XGBoost signal at full scale
    # Calm regime: use calm mean-reversion signal
    # Both are already cross-sectionally neutral; blend by regime flag

    df['signal_final'] = np.where(
        df['vol_regime_flag'] == 1,
        df['signal_hv_cs'] * high_vol_position_scale,
        df['signal_calm'],          # calm regime: mean reversion
    )

    # Re-neutralise after blending (regime flag can differ by country on same date,
    # breaking neutrality — this corrects it)
    df['signal_final'] = df['signal_final'] - df.groupby('DATE')['signal_final'].transform('mean')
    abs_sum_final = df.groupby('DATE')['signal_final'].transform(
        lambda x: x.abs().sum()
    ).replace(0, np.nan)
    df['position_desired'] = (df['signal_final'] / abs_sum_final).fillna(0.0)

    # ----------------------------------------------------------------
    # STEP 5: Vol targeting (optional)
    # ----------------------------------------------------------------
    df['scale_vol_target'] = 1.0
    if enable_vol_target:
        df['realized_vol_true'] = df.groupby('COUNTRY')['true'].transform(
            lambda x: x.rolling(vol_window, min_periods=10).std().shift(1)
        ).replace(0, np.nan)
        df['realized_ann_vol'] = df['realized_vol_true'] * np.sqrt(252)
        df['scale_vol_target'] = (
            target_ann_vol / df['realized_ann_vol'].replace(0, np.nan)
        ).fillna(1.0).clip(lower=vol_floor, upper=vol_cap)
        df['position_desired'] = df['position_desired'] * df['scale_vol_target']
        # Re-normalise after vol scaling
        abs_sum_vt = df.groupby('DATE')['position_desired'].transform(
            lambda x: x.abs().sum()
        ).replace(0, np.nan)
        df['position_desired'] = (df['position_desired'] / abs_sum_vt).fillna(0.0)

    # ----------------------------------------------------------------
    # STEP 6: Legacy Markov regime shrinkage (backward compat)
    # ----------------------------------------------------------------
    df['scale_regime'] = 1.0
    if use_regime_prob is not None:
        df['regime_prob_used'] = 0.0
        for c in df['COUNTRY'].unique():
            colname = use_regime_prob.get(c) if isinstance(use_regime_prob, dict) else None
            if colname and colname in df.columns:
                mask = df['COUNTRY'] == c
                df.loc[mask, 'regime_prob_used'] = (
                    df.loc[mask, colname].fillna(0.0).clip(0.0, 1.0)
                )
        df['scale_regime'] = (
            1.0 - df['regime_prob_used'] * (1.0 - regime_shrink_to)
        ).clip(lower=regime_shrink_to, upper=1.0)
        df['position_desired'] = df['position_desired'] * df['scale_regime']

    # ----------------------------------------------------------------
    # STEP 7: Rebalance threshold
    # ----------------------------------------------------------------
    df['prev_executed_position'] = (
        df.groupby('COUNTRY')['position_desired'].shift(1).fillna(0.0)
    )

    if rebalance_thr is None:
        df['executed_position'] = df['position_desired']
    else:
        executed_list = []
        for dt in pd.to_datetime(df['DATE']).unique():
            mask_dt = df['DATE'] == dt
            rows    = df.loc[mask_dt]
            prev    = rows['prev_executed_position'].values.astype(float)
            desired = rows['position_desired'].values.astype(float)
            delta   = desired - prev
            keep    = np.abs(delta) >= rebalance_thr
            executed = prev.copy()
            executed[keep] = desired[keep]
            abs_sum = np.abs(executed).sum()
            if abs_sum > 0:
                executed = executed / abs_sum
                executed = executed - np.mean(executed)
            executed_list.append(pd.Series(executed, index=rows.index))
        df['executed_position'] = (
            pd.concat(executed_list).sort_index().reindex(df.index).fillna(0.0)
        )

    # ----------------------------------------------------------------
    # STEP 8: Transaction costs
    # ----------------------------------------------------------------
    df['prev_position_used'] = (
        df.groupby('COUNTRY')['executed_position'].shift(1).fillna(0.0)
    )
    df['turnover'] = (df['executed_position'] - df['prev_position_used']).abs()

    if cost_bps is not None:
        df['tcost'] = df['turnover'] * cost_bps
    else:
        df['tcost'] = (
            df['turnover'] * (cost_bps_spread + cost_bps_fees)
            + cost_bps_impact * (df['turnover'] ** 1.5)
        )

    # ----------------------------------------------------------------
    # STEP 9: P&L
    # ----------------------------------------------------------------
    df['pnl_before_cost'] = df['executed_position'] * df['true']
    df['pnl']             = df['pnl_before_cost'] - df['tcost']
    df['position']        = df['executed_position']

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    if verbose:
        daily_pos_sum = df.groupby('DATE')['position'].sum()
        max_dev = daily_pos_sum.abs().max()
        if not np.isfinite(max_dev):
            print("Warning: position sums contain non-finite values.")
        elif max_dev > 1e-4:
            print(f"Warning: max daily position sum deviation = {max_dev:.3e}")
        else:
            print(f"Market neutrality OK: max deviation = {max_dev:.2e}")

        n_high = int(df['vol_regime_flag'].sum())
        n_total = len(df)
        print(f"High-vol regime: {n_high}/{n_total} rows ({n_high/n_total:.1%})")
        print(f"Calm regime:     {n_total-n_high}/{n_total} rows ({(n_total-n_high)/n_total:.1%})")
        print(f"Total cost rate: {total_cost_bps*1e4:.1f} bps per unit turnover")
        if use_calm_strategy and 'spread_zscore' in df.columns:
            calm_active = (df.loc[df['vol_regime_flag']==0, 'signal_calm'].abs() > 1e-6).mean()
            print(f"Calm strategy active: {calm_active:.1%} of calm-regime rows")

    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_trading_performance(
    bt: pd.DataFrame,
    cost_bps_label: str = None,
) -> tuple:
    """
    Performance evaluation with regime breakdown and cost sensitivity.

    WIN RATE NOTE:
    Win rate < 50% is because the payoff is right-skewed: many small losses, fewer larger wins.
    Win_rate_active is when position != 0 and might make more sense to look at.
    """
    daily_pnl    = bt.groupby('DATE')['pnl'].sum()
    daily_pnl_bc = bt.groupby('DATE')['pnl_before_cost'].sum()

    sharpe    = (daily_pnl.mean()    / (daily_pnl.std()    + 1e-9)) * np.sqrt(252)
    sharpe_bc = (daily_pnl_bc.mean() / (daily_pnl_bc.std() + 1e-9)) * np.sqrt(252)

    downside = daily_pnl[daily_pnl < 0].std() + 1e-9
    sortino  = (daily_pnl.mean() / downside) * np.sqrt(252)

    cum_pnl  = daily_pnl.cumsum()
    max_dd   = (cum_pnl - cum_pnl.cummax()).min()
    ann_pnl  = daily_pnl.mean() * 252
    calmar   = ann_pnl / abs(max_dd) if max_dd != 0 else np.nan

    avg_daily_turnover = bt.groupby('DATE')['turnover'].sum().mean()
    total_pnl  = daily_pnl.sum()
    total_cost = bt['tcost'].sum()
    pnl_bc     = bt['pnl_before_cost'].sum()
    cost_coverage = pnl_bc / (total_cost + 1e-9)

    win_rate_daily = (daily_pnl > 0).mean()

    active_days = bt.groupby('DATE').apply(
        lambda x: (x['position'].abs() > 1e-6).any()
    )
    win_rate_active = (daily_pnl[active_days] > 0).mean()

    # Regime breakdown
    regime_stats = {}
    if 'vol_regime_flag' in bt.columns:
        for regime_val, label in [(0, 'calm'), (1, 'high_vol')]:
            sub = bt[bt['vol_regime_flag'] == regime_val]
            if len(sub) == 0:
                continue
            sub_pnl = sub.groupby('DATE')['pnl'].sum()
            sub_pnl_bc = sub.groupby('DATE')['pnl_before_cost'].sum()
            sub_sharpe = (sub_pnl.mean() / (sub_pnl.std() + 1e-9)) * np.sqrt(252)
            sub_wr = (sub_pnl > 0).mean()
            regime_stats[label] = {
                'sharpe':        sub_sharpe,
                'mean_daily_pnl': sub_pnl.mean(),
                'win_rate':      sub_wr,
                'total_pnl':     sub_pnl.sum(),
                'n_days':        len(sub_pnl),
                'pct_days':      len(sub_pnl) / len(daily_pnl),
            }

    metrics = {
        'sharpe':             sharpe,
        'sharpe_before_cost': sharpe_bc,
        'sortino':            sortino,
        'calmar':             calmar,
        'max_drawdown':       max_dd,
        'avg_daily_turnover': avg_daily_turnover,
        'win_rate_daily':     win_rate_daily,
        'win_rate_active':    win_rate_active,
        'total_pnl':          total_pnl,
        'total_cost':         total_cost,
        'pnl_before_cost':    pnl_bc,
        'cost_coverage':      cost_coverage,
        'n_days':             len(daily_pnl),
        'regime_stats':       regime_stats,
    }

    label_str = f" [{cost_bps_label}]" if cost_bps_label else ""
    print(f"=== Trading Performance{label_str} ===")
    print(f"Sharpe:             {sharpe:.4f}  (before cost: {sharpe_bc:.4f})")
    print(f"Sortino:            {sortino:.4f}")
    print(f"Calmar:             {calmar:.4f}")
    print(f"Max Drawdown:       {max_dd:.4f}")
    print(f"Win Rate (all):     {win_rate_daily:.2%}")
    print(f"Win Rate (active):  {win_rate_active:.2%}")
    print(f"Avg Daily Turnover: {avg_daily_turnover:.4f}")
    print(f"Total PnL:          {total_pnl:.4f}")
    print(f"Total Cost:         {total_cost:.4f}")
    print(f"PnL Before Cost:    {pnl_bc:.4f}")
    print(f"Cost Coverage:      {cost_coverage:.2f}x")

    if regime_stats:
        print(f"\nRegime breakdown:")
        for label, rs in regime_stats.items():
            print(f"  {label:10s}  Sharpe={rs['sharpe']:.3f}  "
                  f"PnL={rs['total_pnl']:.3f}  "
                  f"WinRate={rs['win_rate']:.2%}  "
                  f"days={rs['n_days']} ({rs['pct_days']:.1%})")

    print(f"\nCost sensitivity:")
    daily_turn = bt.groupby('DATE')['turnover'].sum()
    for test_bps in [0.0005, 0.0015, 0.0025, 0.0050, 0.0100]:
        hyp_cost_daily = daily_turn * test_bps
        hyp_pnl = daily_pnl_bc - hyp_cost_daily
        s = (hyp_pnl.mean() / (hyp_pnl.std() + 1e-9)) * np.sqrt(252)
        cov = pnl_bc / (hyp_cost_daily.sum() + 1e-9)
        print(f"  {test_bps*1e4:5.1f} bps:  Sharpe={s:.3f}  coverage={cov:.1f}x")

    return metrics, daily_pnl