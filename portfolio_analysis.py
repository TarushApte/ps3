"""
Multi-Asset Portfolio Data Fetcher & Preprocessor
===================================================
Fetches 3 years of daily OHLCV data for a set of assets using yfinance,
cleans missing values, computes daily percentage returns, and computes
a 20-day rolling annualised volatility for each asset.

Dependencies:
    pip install pandas yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS: list[str] = [
    "SPY",  # Equities  – SPDR S&P 500 ETF
    "TLT",  # Bonds     – iShares 20+ Year Treasury Bond ETF
    "GLD",  # Commodities – SPDR Gold Shares ETF
]

YEARS_OF_HISTORY: int = 3

# Number of trading days used for the rolling volatility window
ROLLING_WINDOW: int = 20

# Annualisation factor – square-root-of-time rule for daily vol → annual vol
TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: list[str],
    years: int = YEARS_OF_HISTORY,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for every ticker via yfinance.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance ticker symbols.
    years : int
        How many calendar years of history to request.

    Returns
    -------
    pd.DataFrame
        Multi-level column DataFrame (price_field, ticker) indexed by date.
        Example columns: ('Close', 'SPY'), ('Volume', 'TLT'), …
    """
    end_date   = date.today()
    start_date = end_date - timedelta(days=years * 365)

    print(f"Fetching data for {tickers} from {start_date} to {end_date} …")

    # auto_adjust=True adjusts for dividends and splits automatically
    raw = yf.download(
        tickers,
        start=str(start_date),
        end=str(end_date),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("yfinance returned no data. Check tickers and internet connection.")

    print(f"  → Downloaded {len(raw)} trading days for {len(tickers)} assets.\n")
    return raw


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    raw: pd.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
) -> dict[str, pd.DataFrame]:
    """
    Clean the raw data and engineer features for each asset.

    Steps
    -----
    1. Extract the 'Close' and 'Volume' price series for all tickers.
    2. Forward-fill isolated missing close prices (e.g. one-off data gaps),
       then back-fill any remaining leading NaNs so no asset starts empty.
    3. Drop any date where *every* ticker is still NaN after filling
       (e.g. genuine market-wide holidays with no data at all).
    4. Compute daily log returns: ln(P_t / P_{t-1}).
       Log returns are additive across time and more statistically tractable
       than simple percentage returns, but we also include simple % returns
       for direct interpretability.
    5. Compute a rolling 20-day realised volatility (annualised) from log
       returns using a standard deviation over the rolling window, scaled
       by sqrt(252).

    Parameters
    ----------
    raw : pd.DataFrame
        Multi-level column DataFrame as returned by `fetch_price_data`.
    rolling_window : int
        Look-back window (in trading days) for the rolling volatility.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker symbol → per-asset DataFrame with columns:
            close          – adjusted closing price
            volume         – daily traded volume
            simple_return  – daily simple percentage return  (%)
            log_return     – daily log return
            rolling_vol_20d – 20-day rolling annualised volatility (%)
    """

    # --- 1. Extract Close and Volume ------------------------------------------
    close  = raw["Close"].copy()   # shape: (days, n_tickers)
    volume = raw["Volume"].copy()

    # --- 2. Handle missing values ---------------------------------------------
    # Forward-fill first: propagates the last known price into gaps
    # (handles weekends included by accident, data-provider dropouts, etc.)
    close  = close.ffill()
    volume = volume.ffill()

    # Back-fill remaining NaNs at the very start of the series
    close  = close.bfill()
    volume = volume.bfill()

    # --- 3. Drop rows where all tickers are still NaN -------------------------
    all_nan_rows = close.isna().all(axis=1)
    if all_nan_rows.any():
        n_dropped = all_nan_rows.sum()
        print(f"  [Warning] Dropping {n_dropped} date(s) where all assets lack data.")
        close  = close.loc[~all_nan_rows]
        volume = volume.loc[~all_nan_rows]

    # --- 4 & 5. Compute returns and rolling volatility per ticker -------------
    results: dict[str, pd.DataFrame] = {}

    for ticker in close.columns:
        c = close[ticker]
        v = volume[ticker]

        # Simple percentage return: (P_t - P_{t-1}) / P_{t-1}
        simple_ret = c.pct_change() * 100  # expressed in %

        # Log return: ln(P_t / P_{t-1})  ← preferred for vol calculations
        log_ret = (c / c.shift(1)).apply(pd.np.log if hasattr(pd, "np") else __import__("numpy").log)

        # Rolling 20-day realised volatility (annualised, in %)
        # std of log returns × sqrt(252) × 100
        rolling_vol = (
            log_ret
            .rolling(window=rolling_window, min_periods=rolling_window)
            .std()
            * (TRADING_DAYS_PER_YEAR ** 0.5)
            * 100
        )

        results[ticker] = pd.DataFrame(
            {
                "close":           c,
                "volume":          v,
                "simple_return":   simple_ret,
                "log_return":      log_ret,
                f"rolling_vol_{rolling_window}d": rolling_vol,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Summary Reporter
# ---------------------------------------------------------------------------

def print_summary(asset_data: dict[str, pd.DataFrame]) -> None:
    """
    Print a concise summary table for each asset.

    Parameters
    ----------
    asset_data : dict[str, pd.DataFrame]
        Output from `preprocess`.
    """
    for ticker, df in asset_data.items():
        vol_col = [c for c in df.columns if "rolling_vol" in c][0]

        print(f"{'=' * 55}")
        print(f"  {ticker}")
        print(f"{'=' * 55}")
        print(f"  Date range : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Rows       : {len(df):,}")
        print(f"  Missing close prices after fill : {df['close'].isna().sum()}")
        print()
        print("  Price & Volume")
        print(f"    Latest close  : {df['close'].iloc[-1]:.2f}")
        print(f"    52-week high  : {df['close'].tail(252).max():.2f}")
        print(f"    52-week low   : {df['close'].tail(252).min():.2f}")
        print()
        print("  Returns (full period)")
        print(f"    Mean daily simple return : {df['simple_return'].mean():.4f} %")
        print(f"    Std  daily simple return : {df['simple_return'].std():.4f} %")
        print()
        print(f"  Rolling {vol_col.split('_')[-1]} Volatility (annualised)")
        print(f"    Latest  : {df[vol_col].iloc[-1]:.2f} %")
        print(f"    Mean    : {df[vol_col].mean():.2f} %")
        print(f"    Maximum : {df[vol_col].max():.2f} %")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, pd.DataFrame]:
    """
    Orchestrate data fetch, preprocessing, and summary output.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned and feature-enriched per-asset DataFrames, keyed by ticker.
        Suitable for further analysis, visualisation, or export.
    """
    # Fetch raw multi-level OHLCV data
    raw = fetch_price_data(TICKERS, years=YEARS_OF_HISTORY)

    # Preprocess and engineer features
    asset_data = preprocess(raw, rolling_window=ROLLING_WINDOW)

    # Print a human-readable summary to stdout
    print_summary(asset_data)

    # --- Optional: export each asset to CSV -----------------------------------
    for ticker, df in asset_data.items():
        filename = f"{ticker}_processed.csv"
        df.to_csv(filename)
        print(f"  Saved {filename}")

    return asset_data


if __name__ == "__main__":
    portfolio = main()
