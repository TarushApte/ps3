"""
Rule-Based Trading Signal Generator
=====================================
Implements a Dual Moving Average Crossover (DMAC) strategy on top of the
preprocessed per-asset DataFrames produced by portfolio_analysis.py.

Strategy logic
--------------
  - Compute a fast SMA (default 50-day) and a slow SMA (default 200-day).
  - BUY  (+1) : fast SMA crosses ABOVE the slow SMA  ("golden cross")
  - SHORT (-1) : fast SMA crosses BELOW the slow SMA  ("death cross")
  - CASH  ( 0) : position is held until the next crossover signal fires

Output columns added to each asset's DataFrame
-----------------------------------------------
  sma_fast        – rolling fast simple moving average
  sma_slow        – rolling slow simple moving average
  raw_signal      – raw crossover flag (+1 / -1) on every bar
  signal          – position carried forward until the next crossover
  target_weight   – same as signal, but expressed as a portfolio weight
                    (useful when combining multiple assets)

Usage
-----
    from portfolio_analysis import main as fetch_and_preprocess
    from signal_generator import run_strategy, print_signal_summary

    portfolio = fetch_and_preprocess()          # dict[ticker, DataFrame]
    signals   = run_strategy(portfolio)         # dict[ticker, DataFrame]
    print_signal_summary(signals)

Dependencies:
    pip install pandas yfinance  (yfinance only needed via portfolio_analysis)
"""

import numpy as np
import pandas as pd
from typing import Literal


# ---------------------------------------------------------------------------
# Strategy configuration — tweak these without touching any logic
# ---------------------------------------------------------------------------

# Fast and slow SMA windows (in trading days)
FAST_WINDOW: int = 50
SLOW_WINDOW: int = 200

# Whether to allow short positions.
# If False, the signal is clamped to {0, 1}  (long-or-cash only).
ALLOW_SHORT: bool = True


# ---------------------------------------------------------------------------
# Core building blocks  (each function does exactly one thing)
# ---------------------------------------------------------------------------

def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute a simple moving average over a rolling window.

    Parameters
    ----------
    prices : pd.Series
        Adjusted closing prices indexed by date.
    window : int
        Look-back period in trading days.

    Returns
    -------
    pd.Series
        Rolling SMA; the first (window - 1) values are NaN.
    """
    return prices.rolling(window=window, min_periods=window).mean()


def compute_raw_crossover_signal(
    sma_fast: pd.Series,
    sma_slow: pd.Series,
) -> pd.Series:
    """
    Emit +1 when fast SMA > slow SMA and -1 when fast SMA < slow SMA.
    Bars where either SMA is still NaN (warm-up period) emit 0.

    This is the *instantaneous* regime flag — it fires on every bar, not
    only on the day of the actual crossover.  The position signal (below)
    is derived from this by forward-filling crossover transitions.

    Parameters
    ----------
    sma_fast : pd.Series   – fast moving average
    sma_slow : pd.Series   – slow moving average

    Returns
    -------
    pd.Series[int]   values in {-1, 0, +1}
    """
    regime = pd.Series(0, index=sma_fast.index, dtype=int)
    regime[sma_fast > sma_slow] =  1
    regime[sma_fast < sma_slow] = -1
    # Zero out the warm-up period where either SMA is undefined
    regime[sma_fast.isna() | sma_slow.isna()] = 0
    return regime


def compute_position_signal(
    raw_signal: pd.Series,
    allow_short: bool = ALLOW_SHORT,
) -> pd.Series:
    """
    Convert the raw instantaneous regime into a *tradeable* position signal.

    The raw signal already represents a held position (+1 long, -1 short,
    0 in warm-up), so we simply clamp it when shorting is disabled.

    Key design choice
    -----------------
    The crossover strategy naturally holds a position until the next flip,
    so there is no need to forward-fill: raw_signal already encodes that
    persistence because it equals +1 for every bar the fast SMA is above
    the slow SMA, not just on the day of the cross.

    Parameters
    ----------
    raw_signal  : pd.Series   raw crossover flag {-1, 0, +1}
    allow_short : bool        if False, clamp short signals to 0 (cash)

    Returns
    -------
    pd.Series[int]   tradeable position in {-1, 0, +1} (or {0, 1} if no shorts)
    """
    position = raw_signal.copy()
    if not allow_short:
        position = position.clip(lower=0)   # replace -1 with 0
    return position


def compute_target_weights(
    signal: pd.Series,
    position_size: float = 1.0,
) -> pd.Series:
    """
    Scale the discrete signal into a continuous portfolio weight.

    For a single-asset strategy this is trivial (signal × size), but the
    function exists so that multi-asset aggregation can normalise across
    assets without changing the signal logic.

    Parameters
    ----------
    signal        : pd.Series   position signal {-1, 0, +1}
    position_size : float       fraction of capital per position (default 1.0)

    Returns
    -------
    pd.Series[float]   target weight in [-position_size, +position_size]
    """
    return signal * position_size


# ---------------------------------------------------------------------------
# Per-asset strategy runner
# ---------------------------------------------------------------------------

def apply_dmac_strategy(
    df: pd.DataFrame,
    fast_window: int = FAST_WINDOW,
    slow_window: int = SLOW_WINDOW,
    allow_short: bool = ALLOW_SHORT,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """
    Apply the Dual Moving Average Crossover strategy to one asset's DataFrame.

    This function is the single point of truth for strategy logic. To swap
    out the strategy entirely, replace only this function.

    Parameters
    ----------
    df            : pd.DataFrame   output of portfolio_analysis.preprocess()
                                   must contain a 'close' column
    fast_window   : int            fast SMA window in trading days
    slow_window   : int            slow SMA window in trading days
    allow_short   : bool           whether short (-1) positions are permitted
    position_size : float          capital fraction per position

    Returns
    -------
    pd.DataFrame
        The input DataFrame with five new columns appended:
            sma_fast, sma_slow, raw_signal, signal, target_weight
    """
    out = df.copy()

    # Step 1 – moving averages
    out["sma_fast"] = compute_sma(out["close"], fast_window)
    out["sma_slow"] = compute_sma(out["close"], slow_window)

    # Step 2 – instantaneous crossover regime
    out["raw_signal"] = compute_raw_crossover_signal(out["sma_fast"], out["sma_slow"])

    # Step 3 – tradeable position (respects allow_short flag)
    out["signal"] = compute_position_signal(out["raw_signal"], allow_short)

    # Step 4 – target portfolio weight
    out["target_weight"] = compute_target_weights(out["signal"], position_size)

    return out


# ---------------------------------------------------------------------------
# Multi-asset runner
# ---------------------------------------------------------------------------

def run_strategy(
    portfolio: dict[str, pd.DataFrame],
    fast_window: int = FAST_WINDOW,
    slow_window: int = SLOW_WINDOW,
    allow_short: bool = ALLOW_SHORT,
) -> dict[str, pd.DataFrame]:
    """
    Apply the DMAC strategy to every asset in the portfolio dict.

    Each asset receives an independent signal; weights are not normalised
    across assets here — do that in a portfolio construction layer if needed.

    Parameters
    ----------
    portfolio   : dict[str, pd.DataFrame]   output of portfolio_analysis.main()
    fast_window : int
    slow_window : int
    allow_short : bool

    Returns
    -------
    dict[str, pd.DataFrame]
        Same keys as input; each DataFrame now includes signal columns.
    """
    results: dict[str, pd.DataFrame] = {}

    for ticker, df in portfolio.items():
        print(f"Generating signals for {ticker} "
              f"(SMA{fast_window} / SMA{slow_window}, "
              f"{'long/short' if allow_short else 'long-only'}) …")

        results[ticker] = apply_dmac_strategy(
            df,
            fast_window=fast_window,
            slow_window=slow_window,
            allow_short=allow_short,
        )

    print()
    return results


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------

def print_signal_summary(signals: dict[str, pd.DataFrame]) -> None:
    """
    Print a human-readable summary of crossover events and current position.

    Parameters
    ----------
    signals : dict[str, pd.DataFrame]   output of run_strategy()
    """
    for ticker, df in signals.items():
        # Isolate rows where a crossover actually occurred (signal changes)
        crossovers = df[df["signal"].diff().fillna(0) != 0][
            ["close", "sma_fast", "sma_slow", "signal", "target_weight"]
        ]

        print(f"{'=' * 60}")
        print(f"  {ticker}  —  Dual Moving Average Crossover Summary")
        print(f"{'=' * 60}")
        print(f"  Total bars            : {len(df):,}")
        print(f"  Warm-up bars (signal=0): {(df['signal'] == 0).sum():,}")
        print(f"  Bars long  (+1)       : {(df['signal'] == 1).sum():,}")
        print(f"  Bars short (-1)       : {(df['signal'] == -1).sum():,}")
        print(f"  Total crossover events: {len(crossovers)}")
        print()
        print("  Latest 5 crossover events:")
        if crossovers.empty:
            print("    (no crossovers in the available history)")
        else:
            print(crossovers.tail(5).to_string(
                float_format=lambda x: f"{x:.2f}",
            ))
        print()

        latest = df.iloc[-1]
        pos_label = {1: "LONG", -1: "SHORT", 0: "CASH"}.get(
            int(latest["signal"]), "UNKNOWN"
        )
        print(f"  Current position      : {pos_label}  (weight={latest['target_weight']:.2f})")
        print(f"  SMA{FAST_WINDOW} (fast)          : {latest['sma_fast']:.2f}")
        print(f"  SMA{SLOW_WINDOW} (slow)         : {latest['sma_slow']:.2f}")
        print(f"  Last close            : {latest['close']:.2f}")
        print()


# ---------------------------------------------------------------------------
# Optional: export helper
# ---------------------------------------------------------------------------

def export_signals(
    signals: dict[str, pd.DataFrame],
    suffix: str = "_signals",
) -> None:
    """
    Write each asset's enriched DataFrame (including signal columns) to CSV.

    Parameters
    ----------
    signals : dict[str, pd.DataFrame]
    suffix  : str   appended to ticker name before '.csv'
    """
    for ticker, df in signals.items():
        filename = f"{ticker}{suffix}.csv"
        df.to_csv(filename)
        print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Main — demonstrates the full pipeline end-to-end
# ---------------------------------------------------------------------------

def main() -> dict[str, pd.DataFrame]:
    """
    Fetch data → preprocess → generate signals → summarise → export.

    Returns the enriched signal DataFrames for downstream use.
    """
    # Import here to keep this module usable without portfolio_analysis.py
    # being on the Python path (e.g. during unit testing with mocked data).
    from portfolio_analysis import main as fetch_and_preprocess

    # Step 1: get preprocessed per-asset DataFrames
    portfolio = fetch_and_preprocess()

    # Step 2: apply the DMAC strategy to each asset
    signals = run_strategy(
        portfolio,
        fast_window=FAST_WINDOW,
        slow_window=SLOW_WINDOW,
        allow_short=ALLOW_SHORT,
    )

    # Step 3: print a readable summary
    print_signal_summary(signals)

    # Step 4: persist to CSV
    export_signals(signals)

    return signals


if __name__ == "__main__":
    main()
