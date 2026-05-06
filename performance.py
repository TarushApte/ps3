"""
Performance Evaluation Module
==============================
Takes a simulated daily equity curve (from backtester.py) and computes a
comprehensive set of portfolio performance metrics, including Alpha and Beta
relative to a benchmark (defaults to SPY).

All metrics are returned as a clean, flat dictionary so they can be printed,
serialised to JSON, logged, or fed into downstream reporting tools.

Metrics computed
----------------
  Core
    cumulative_return_pct     – total growth over the full period (%)
    annualised_return_pct     – CAGR over the full period (%)
    annualised_volatility_pct – annualised std-dev of daily returns (%)
    max_drawdown_pct          – largest peak-to-trough decline (%)
    sharpe_ratio              – excess return per unit of total risk
                                (risk-free rate = 3 %, configurable)
  Drawdown detail
    max_drawdown_start        – date the drawdown peak was set
    max_drawdown_end          – date the trough was reached
    max_drawdown_duration_days– calendar days from peak to trough

  Market-relative (vs SPY or any benchmark)
    beta                      – sensitivity to benchmark daily moves
    alpha_annualised_pct      – Jensen's Alpha, annualised (%)
    correlation               – Pearson correlation with benchmark returns
    r_squared                 – proportion of variance explained by benchmark
    treynor_ratio             – excess return per unit of systematic risk
    information_ratio         – alpha per unit of tracking error

Usage
-----
    from backtester import run_backtest
    from performance import evaluate

    ledger, trade_log, _ = run_backtest(signals)
    metrics = evaluate(ledger["total_equity"])
    print_metrics(metrics)

Dependencies:
    pip install pandas numpy yfinance
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import timedelta


# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

RISK_FREE_RATE:        float = 0.03    # 3 % annual risk-free rate
TRADING_DAYS_PER_YEAR: int   = 252
BENCHMARK_TICKER:      str   = "SPY"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _daily_returns(equity: pd.Series) -> pd.Series:
    """Compute daily simple returns from an equity curve."""
    return equity.pct_change().dropna()


def _annualise_return(total_return: float, n_years: float) -> float:
    """Convert a total return to a CAGR given the number of years."""
    if n_years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / n_years) - 1


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the rolling drawdown series (negative values, fraction)."""
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def _fetch_benchmark_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    ticker: str = BENCHMARK_TICKER,
) -> pd.Series:
    """
    Download benchmark daily returns from yfinance for the strategy period.

    Parameters
    ----------
    start  : pd.Timestamp  first date of the strategy equity curve
    end    : pd.Timestamp  last date of the strategy equity curve
    ticker : str           benchmark ticker symbol

    Returns
    -------
    pd.Series  daily simple returns indexed by date; empty if download fails
    """
    try:
        raw = yf.download(
            ticker,
            start=str(start.date()),
            end=str((end + timedelta(days=1)).date()),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.Series(dtype=float)
        return raw["Close"].squeeze().pct_change().dropna()
    except Exception as exc:
        print(f"  [Warning] Could not fetch benchmark data for {ticker}: {exc}")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_core_metrics(
    equity: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute stand-alone performance metrics that need only the equity curve.

    Parameters
    ----------
    equity          : pd.Series  daily total portfolio value, date-indexed
    risk_free_rate  : float      annual risk-free rate (decimal, e.g. 0.03)

    Returns
    -------
    dict with keys described in the module docstring (core group)
    """
    if equity.empty or len(equity) < 2:
        raise ValueError("equity series must have at least 2 data points.")

    equity = equity.sort_index().copy()
    returns = _daily_returns(equity)

    # --- Return ------------------------------------------------------------------
    cumulative_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    n_days  = len(equity)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    ann_return = _annualise_return(cumulative_return, n_years)

    # --- Volatility --------------------------------------------------------------
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # --- Sharpe ratio ------------------------------------------------------------
    daily_rf       = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_rf
    sharpe = (
        excess_returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std() > 0 else 0.0
    )

    # --- Maximum drawdown --------------------------------------------------------
    dd_series  = _drawdown_series(equity)
    max_dd     = dd_series.min()                # most negative value

    # Locate the trough
    trough_date = dd_series.idxmin()
    # The preceding peak is the last date the drawdown was 0 before the trough
    pre_trough  = dd_series.loc[:trough_date]
    peak_date   = pre_trough[pre_trough == 0].index[-1] \
                  if (pre_trough == 0).any() else pre_trough.index[0]
    dd_duration = (trough_date - peak_date).days

    return {
        # Returns
        "cumulative_return_pct":      round(cumulative_return * 100, 4),
        "annualised_return_pct":      round(ann_return * 100, 4),
        # Volatility
        "annualised_volatility_pct":  round(ann_vol * 100, 4),
        # Risk-adjusted
        "sharpe_ratio":               round(sharpe, 4),
        # Drawdown
        "max_drawdown_pct":           round(max_dd * 100, 4),
        "max_drawdown_start":         str(peak_date.date()),
        "max_drawdown_end":           str(trough_date.date()),
        "max_drawdown_duration_days": dd_duration,
        # Metadata
        "n_trading_days":             n_days,
        "n_years":                    round(n_years, 4),
        "period_start":               str(equity.index[0].date()),
        "period_end":                 str(equity.index[-1].date()),
        "risk_free_rate_pct":         risk_free_rate * 100,
    }


# ---------------------------------------------------------------------------
# Market-relative metrics  (Alpha, Beta, correlation, etc.)
# ---------------------------------------------------------------------------

def compute_market_metrics(
    equity: pd.Series,
    risk_free_rate:   float = RISK_FREE_RATE,
    benchmark_ticker: str   = BENCHMARK_TICKER,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict:
    """
    Compute Alpha, Beta, and related metrics vs. a market benchmark.

    Alpha and Beta are derived via OLS regression of strategy excess returns
    on benchmark excess returns (Jensen's Alpha framework):

        r_strategy - rf  =  alpha  +  beta × (r_benchmark - rf)  +  ε

    Parameters
    ----------
    equity            : pd.Series          daily portfolio equity curve
    risk_free_rate    : float              annual risk-free rate (decimal)
    benchmark_ticker  : str                Yahoo Finance ticker for benchmark
    benchmark_returns : pd.Series | None   pre-fetched returns; if None,
                                           they are downloaded automatically

    Returns
    -------
    dict with keys described in the module docstring (market-relative group).
    Returns a dict of NaN values with a warning key if benchmark is unavailable.
    """
    equity  = equity.sort_index().copy()
    strat_r = _daily_returns(equity)

    # --- Fetch or accept benchmark returns ------------------------------------
    if benchmark_returns is None:
        bm_r = _fetch_benchmark_returns(equity.index[0], equity.index[-1], benchmark_ticker)
    else:
        bm_r = benchmark_returns.copy()

    if bm_r.empty:
        return {
            "benchmark_ticker":         benchmark_ticker,
            "beta":                     float("nan"),
            "alpha_annualised_pct":     float("nan"),
            "correlation":              float("nan"),
            "r_squared":                float("nan"),
            "treynor_ratio":            float("nan"),
            "information_ratio":        float("nan"),
            "warning": "Benchmark data unavailable; market metrics could not be computed.",
        }

    # --- Align strategy and benchmark on common dates -------------------------
    common = strat_r.index.intersection(bm_r.index)
    strat_r = strat_r.loc[common]
    bm_r    = bm_r.loc[common]

    # --- Daily risk-free rate -------------------------------------------------
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    strat_excess = strat_r - daily_rf
    bm_excess    = bm_r    - daily_rf

    # --- OLS regression: excess_strategy = alpha_daily + beta * excess_bm ----
    bm_arr  = bm_excess.values
    st_arr  = strat_excess.values
    # Design matrix [1, bm_excess]
    X       = np.column_stack([np.ones(len(bm_arr)), bm_arr])
    # Least-squares solution: [alpha_daily, beta]
    coeffs, _, _, _ = np.linalg.lstsq(X, st_arr, rcond=None)
    alpha_daily, beta = float(coeffs[0]), float(coeffs[1])

    # Annualise alpha
    alpha_annualised = (1 + alpha_daily) ** TRADING_DAYS_PER_YEAR - 1

    # --- Correlation and R² ---------------------------------------------------
    correlation = float(np.corrcoef(strat_r.values, bm_r.values)[0, 1])
    r_squared   = correlation ** 2

    # --- Treynor ratio: annualised excess return / beta -----------------------
    ann_excess = strat_excess.mean() * TRADING_DAYS_PER_YEAR
    treynor    = ann_excess / beta if beta != 0 else float("nan")

    # --- Information ratio: alpha / tracking error ---------------------------
    tracking_error = (strat_r - bm_r).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    info_ratio     = (alpha_annualised / tracking_error
                      if tracking_error > 0 else float("nan"))

    return {
        "benchmark_ticker":         benchmark_ticker,
        "beta":                     round(beta, 4),
        "alpha_annualised_pct":     round(alpha_annualised * 100, 4),
        "correlation":              round(correlation, 4),
        "r_squared":                round(r_squared, 4),
        "treynor_ratio":            round(treynor, 4),
        "information_ratio":        round(info_ratio, 4),
        "n_overlapping_days":       len(common),
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def evaluate(
    equity: pd.Series,
    risk_free_rate:   float = RISK_FREE_RATE,
    benchmark_ticker: str   = BENCHMARK_TICKER,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict:
    """
    Compute the full set of performance metrics and return them as one dict.

    This is the primary public API of this module. Pass the 'total_equity'
    column from the backtester ledger and get back a single flat dictionary
    ready for display, serialisation, or downstream comparison.

    Parameters
    ----------
    equity            : pd.Series          date-indexed portfolio equity curve
    risk_free_rate    : float              annual risk-free rate (default 3 %)
    benchmark_ticker  : str                benchmark symbol (default 'SPY')
    benchmark_returns : pd.Series | None   optional pre-fetched benchmark returns

    Returns
    -------
    dict  flat dictionary of all metrics (core + market-relative)
    """
    core    = compute_core_metrics(equity, risk_free_rate)
    market  = compute_market_metrics(
        equity,
        risk_free_rate=risk_free_rate,
        benchmark_ticker=benchmark_ticker,
        benchmark_returns=benchmark_returns,
    )
    # Merge into one flat dict; market keys are prefixed to avoid collisions
    return {**core, **market}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """
    Render the metrics dictionary as a formatted tearsheet to stdout.

    Parameters
    ----------
    metrics : dict  output of evaluate()
    """
    def _fmt(val) -> str:
        if isinstance(val, float):
            return f"{val:.4f}" if not val != val else "N/A"   # NaN guard
        return str(val)

    sections = {
        "Period": [
            "period_start", "period_end", "n_trading_days", "n_years",
            "risk_free_rate_pct",
        ],
        "Returns": [
            "cumulative_return_pct", "annualised_return_pct",
        ],
        "Risk": [
            "annualised_volatility_pct",
            "max_drawdown_pct", "max_drawdown_start",
            "max_drawdown_end", "max_drawdown_duration_days",
        ],
        "Risk-Adjusted": [
            "sharpe_ratio", "treynor_ratio", "calmar_ratio",
        ],
        f"Market-Relative (vs {metrics.get('benchmark_ticker', 'SPY')})": [
            "beta", "alpha_annualised_pct",
            "correlation", "r_squared", "information_ratio",
            "n_overlapping_days",
        ],
    }

    label_map = {
        "period_start":               "Start date",
        "period_end":                 "End date",
        "n_trading_days":             "Trading days",
        "n_years":                    "Years",
        "risk_free_rate_pct":         "Risk-free rate (%)",
        "cumulative_return_pct":      "Cumulative return (%)",
        "annualised_return_pct":      "Annualised return / CAGR (%)",
        "annualised_volatility_pct":  "Annualised volatility (%)",
        "max_drawdown_pct":           "Max drawdown (%)",
        "max_drawdown_start":         "Drawdown peak date",
        "max_drawdown_end":           "Drawdown trough date",
        "max_drawdown_duration_days": "Drawdown duration (days)",
        "sharpe_ratio":               "Sharpe ratio",
        "treynor_ratio":              "Treynor ratio",
        "calmar_ratio":               "Calmar ratio",
        "beta":                       "Beta",
        "alpha_annualised_pct":       "Jensen's Alpha annualised (%)",
        "correlation":                "Correlation with benchmark",
        "r_squared":                  "R-squared",
        "information_ratio":          "Information ratio",
        "n_overlapping_days":         "Overlapping days with benchmark",
    }

    print(f"\n{'=' * 56}")
    print("  PERFORMANCE EVALUATION REPORT")
    print(f"{'=' * 56}")
    for section, keys in sections.items():
        print(f"\n  {section}")
        print(f"  {'-' * 52}")
        for key in keys:
            if key not in metrics:
                continue
            label = label_map.get(key, key)
            val   = _fmt(metrics[key])
            print(f"    {label:<38}: {val:>10}")
    if "warning" in metrics:
        print(f"\n  [!] {metrics['warning']}")
    print()


# ---------------------------------------------------------------------------
# Main — full pipeline demo
# ---------------------------------------------------------------------------

def main() -> dict:
    """
    Run the complete pipeline and evaluate performance.
    Returns the final metrics dictionary.
    """
    from portfolio_analysis import main as fetch_and_preprocess
    from signal_generator   import run_strategy
    from backtester         import run_backtest

    # Step 1 – 3: data → signals → backtest
    portfolio             = fetch_and_preprocess()
    signals               = run_strategy(portfolio)
    ledger, trade_log, _  = run_backtest(signals, export=False)

    # Step 4: evaluate the equity curve
    print("\nEvaluating performance metrics …")
    metrics = evaluate(
        ledger["total_equity"],
        risk_free_rate=RISK_FREE_RATE,
        benchmark_ticker=BENCHMARK_TICKER,
    )

    # Step 5: pretty-print and return
    print_metrics(metrics)

    return metrics


if __name__ == "__main__":
    metrics = main()
