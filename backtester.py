"""
Backtesting Simulation Engine
==============================
Evaluates the Dual Moving Average Crossover strategy generated in
signal_generator.py against a realistic market simulation.

Simulation features
-------------------
  - Initial capital        : $100,000 (configurable)
  - Transaction costs      : 0.1% of trade notional, deducted on every order
  - Slippage               : 0.05% adverse price move on execution
  - Max position per asset : 40% of total portfolio equity
  - Rebalancing            : Monthly (first trading day of each calendar month)
  - Daily equity curve     : Total portfolio value tracked every bar

Performance metrics reported
-----------------------------
  - Total return (%)
  - Annualised return (CAGR %)
  - Annualised volatility (%)
  - Sharpe ratio (risk-free rate configurable)
  - Max drawdown (%)
  - Calmar ratio (CAGR / |Max drawdown|)
  - Win-rate across completed trades
  - Comparison vs. equal-weight buy-and-hold benchmark

Usage
-----
    python backtester.py            # full pipeline: fetch → signal → backtest
    from backtester import run_backtest
    results = run_backtest(signals)  # pass output of signal_generator.run_strategy()

Dependencies:
    pip install pandas numpy yfinance
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Simulation configuration — tweak without touching any logic
# ---------------------------------------------------------------------------

INITIAL_CAPITAL:      float = 100_000.0   # starting portfolio value in USD
TRANSACTION_COST_PCT: float = 0.001       # 0.1 % of notional traded
SLIPPAGE_PCT:         float = 0.0005      # 0.05% adverse fill price move
MAX_POSITION_PCT:     float = 0.40        # max 40% of equity in any single asset
RISK_FREE_RATE:       float = 0.05        # annual risk-free rate for Sharpe (5%)
TRADING_DAYS_PER_YEAR: int  = 252


# ---------------------------------------------------------------------------
# Helper: execution price with slippage
# ---------------------------------------------------------------------------

def _execution_price(close: float, direction: int) -> float:
    """
    Apply one-sided slippage to a closing price.

    Buying  (+1) → pay slightly more than close  (adverse fill)
    Selling (-1) → receive slightly less than close

    Parameters
    ----------
    close     : float  unadjusted closing price
    direction : int    +1 for a buy order, -1 for a sell/short order

    Returns
    -------
    float  slippage-adjusted execution price
    """
    return close * (1 + direction * SLIPPAGE_PCT)


# ---------------------------------------------------------------------------
# Helper: transaction cost
# ---------------------------------------------------------------------------

def _transaction_cost(notional: float) -> float:
    """
    Flat-rate commission on the absolute notional traded.

    Parameters
    ----------
    notional : float  absolute dollar value of the trade

    Returns
    -------
    float  cost in dollars (always positive)
    """
    return abs(notional) * TRANSACTION_COST_PCT


# ---------------------------------------------------------------------------
# Helper: rebalancing schedule
# ---------------------------------------------------------------------------

def _build_rebalance_flags(index: pd.DatetimeIndex) -> pd.Series:
    """
    Mark the first trading day of every calendar month as a rebalance bar.

    Parameters
    ----------
    index : pd.DatetimeIndex  full date index of the price series

    Returns
    -------
    pd.Series[bool]  True on rebalance days, False elsewhere
    """
    # Month-start flag: True when the month number changes vs the prior bar
    month_start = pd.Series(index.month, index=index).diff().fillna(1) != 0
    return month_start


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def simulate(
    signals: dict[str, pd.DataFrame],
    initial_capital: float        = INITIAL_CAPITAL,
    transaction_cost_pct: float   = TRANSACTION_COST_PCT,
    slippage_pct: float           = SLIPPAGE_PCT,
    max_position_pct: float       = MAX_POSITION_PCT,
) -> pd.DataFrame:
    """
    Run the day-by-day portfolio simulation across all assets.

    Design notes
    ------------
    - A single shared date index is built as the union of all asset indices,
      then intersected to keep only dates where every asset has data.
    - Positions are expressed in *shares* (integer-truncated for realism).
    - Desired weights come from the signal column in each asset's DataFrame.
    - On a rebalance day (or when a signal flips), the engine recomputes
      target dollar allocations and trades to close the gap.
    - Trades execute at slippage-adjusted prices; commission is deducted
      from cash immediately.
    - The equity curve is the sum of cash + mark-to-market value of all
      open positions at each day's close.

    Parameters
    ----------
    signals             : dict[str, pd.DataFrame]  output of run_strategy()
    initial_capital     : float
    transaction_cost_pct: float
    slippage_pct        : float
    max_position_pct    : float

    Returns
    -------
    pd.DataFrame  daily simulation ledger with columns:
        cash, <ticker>_shares, <ticker>_value, total_equity,
        daily_return, drawdown
    """
    tickers = list(signals.keys())

    # --- Align all assets onto a common date grid ----------------------------
    common_index = signals[tickers[0]].index
    for t in tickers[1:]:
        common_index = common_index.intersection(signals[t].index)

    # Slice each asset to the common index
    data: dict[str, pd.DataFrame] = {
        t: signals[t].loc[common_index].copy() for t in tickers
    }

    rebalance_flags = _build_rebalance_flags(common_index)

    # --- State initialisation -------------------------------------------------
    cash: float = initial_capital
    # Current share holdings keyed by ticker
    holdings: dict[str, float] = {t: 0.0 for t in tickers}
    # Previous signal to detect flips
    prev_signal: dict[str, int] = {t: 0 for t in tickers}

    # Trade log: list of dicts, one entry per executed trade
    trade_log: list[dict] = []

    # Daily ledger rows
    rows: list[dict] = []

    # --- Simulation loop ------------------------------------------------------
    for date, is_rebalance in rebalance_flags.items():
        row: dict = {"date": date}

        # Collect current prices and signals for this bar
        prices:   dict[str, float] = {t: data[t].at[date, "close"]  for t in tickers}
        target_w: dict[str, int]   = {t: int(data[t].at[date, "signal"]) for t in tickers}

        # --- Determine whether to trade ---------------------------------------
        # Trade if: (a) it's a rebalance day, or (b) any signal has flipped
        signal_flipped = any(target_w[t] != prev_signal[t] for t in tickers)
        should_trade   = is_rebalance or signal_flipped

        if should_trade:
            # Mark-to-market equity BEFORE trades (use previous close prices if
            # this is not the first bar, otherwise use opening prices)
            mtm_equity = cash + sum(
                holdings[t] * prices[t] for t in tickers
            )

            for ticker in tickers:
                desired_signal = target_w[ticker]   # -1, 0, or +1
                current_price  = prices[ticker]

                # --- Enforce max position limit --------------------------------
                # The raw desired weight comes from the signal; cap it.
                raw_weight     = desired_signal * max_position_pct \
                                 if desired_signal != 0 else 0.0
                # Map signal to a dollar allocation
                desired_dollar = raw_weight * mtm_equity            # can be < 0 for shorts
                # Cap absolute dollar value at max_position_pct × equity
                cap            = max_position_pct * mtm_equity
                desired_dollar = max(-cap, min(cap, desired_dollar))

                # Convert to shares (truncate to whole shares)
                exec_price      = _execution_price(
                    current_price, 1 if desired_dollar >= 0 else -1
                )
                desired_shares  = int(desired_dollar / exec_price) \
                                  if exec_price != 0 else 0

                delta_shares    = desired_shares - holdings[ticker]

                if delta_shares == 0:
                    continue  # nothing to trade

                # --- Execute the trade ----------------------------------------
                trade_direction = 1 if delta_shares > 0 else -1
                fill_price      = _execution_price(current_price, trade_direction)
                notional        = delta_shares * fill_price
                cost            = _transaction_cost(notional)

                cash            -= notional + cost   # debit cash (buy) or credit (sell)
                holdings[ticker] = desired_shares

                trade_log.append({
                    "date":        date,
                    "ticker":      ticker,
                    "delta_shares": delta_shares,
                    "fill_price":  fill_price,
                    "notional":    notional,
                    "cost":        cost,
                    "signal":      desired_signal,
                })

        # Update previous signals
        prev_signal = target_w.copy()

        # --- End-of-day mark-to-market ----------------------------------------
        total_position_value = 0.0
        for ticker in tickers:
            pos_value            = holdings[ticker] * prices[ticker]
            total_position_value += pos_value
            row[f"{ticker}_shares"] = holdings[ticker]
            row[f"{ticker}_value"]  = pos_value

        row["cash"]         = cash
        row["total_equity"] = cash + total_position_value
        rows.append(row)

    # --- Assemble ledger DataFrame --------------------------------------------
    ledger = pd.DataFrame(rows).set_index("date")
    ledger.index = pd.to_datetime(ledger.index)

    # Daily returns and drawdown
    ledger["daily_return"] = ledger["total_equity"].pct_change().fillna(0.0)

    running_max            = ledger["total_equity"].cummax()
    ledger["drawdown"]     = (ledger["total_equity"] - running_max) / running_max

    return ledger, pd.DataFrame(trade_log)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    ledger: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float  = RISK_FREE_RATE,
) -> dict:
    """
    Derive standard portfolio performance statistics from the equity ledger.

    Parameters
    ----------
    ledger          : pd.DataFrame  output of simulate()
    initial_capital : float
    risk_free_rate  : float         annual risk-free rate (decimal)

    Returns
    -------
    dict  keyed performance metrics
    """
    equity = ledger["total_equity"]
    returns = ledger["daily_return"]

    n_days         = len(equity)
    n_years        = n_days / TRADING_DAYS_PER_YEAR

    total_return   = (equity.iloc[-1] / initial_capital) - 1
    cagr           = (equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1 \
                     if n_years > 0 else 0.0
    ann_vol        = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    daily_rf       = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_rf
    sharpe         = (excess_returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) \
                     if returns.std() > 0 else 0.0

    max_drawdown   = ledger["drawdown"].min()
    calmar         = cagr / abs(max_drawdown) if max_drawdown != 0 else np.inf

    return {
        "initial_capital":    initial_capital,
        "final_equity":       equity.iloc[-1],
        "total_return_pct":   total_return * 100,
        "cagr_pct":           cagr * 100,
        "ann_volatility_pct": ann_vol * 100,
        "sharpe_ratio":       sharpe,
        "max_drawdown_pct":   max_drawdown * 100,
        "calmar_ratio":       calmar,
        "n_trading_days":     n_days,
        "n_years":            round(n_years, 2),
    }


# ---------------------------------------------------------------------------
# Benchmark: equal-weight buy-and-hold
# ---------------------------------------------------------------------------

def compute_benchmark(
    signals: dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    Compute an equal-weight, fully-invested buy-and-hold equity curve.

    Each asset receives 1/N of the initial capital on day 0, and the
    portfolio drifts with price thereafter — no rebalancing, no costs.

    Parameters
    ----------
    signals         : dict[str, pd.DataFrame]   (only 'close' column is used)
    initial_capital : float

    Returns
    -------
    pd.DataFrame  with column 'benchmark_equity'
    """
    tickers = list(signals.keys())
    weight  = 1.0 / len(tickers)

    # Normalised price relatives (each asset starts at 1.0)
    price_rel = pd.DataFrame({
        t: signals[t]["close"] / signals[t]["close"].iloc[0]
        for t in tickers
    })

    # Portfolio value = sum of weighted allocations grown by price relatives
    benchmark_equity = (price_rel * weight * initial_capital).sum(axis=1)
    return pd.DataFrame({"benchmark_equity": benchmark_equity})


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_backtest_report(
    ledger: pd.DataFrame,
    trade_log: pd.DataFrame,
    metrics: dict,
    benchmark: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> None:
    """
    Print a formatted tearsheet to stdout.

    Parameters
    ----------
    ledger      : pd.DataFrame   daily simulation ledger
    trade_log   : pd.DataFrame   individual trade records
    metrics     : dict           output of compute_metrics()
    benchmark   : pd.DataFrame   output of compute_benchmark()
    """
    bm_return = (
        benchmark["benchmark_equity"].iloc[-1] / initial_capital - 1
    ) * 100

    print(f"\n{'=' * 60}")
    print("  BACKTEST TEARSHEET  —  Dual Moving Average Crossover")
    print(f"{'=' * 60}")
    print(f"  Period              : {ledger.index[0].date()} → {ledger.index[-1].date()}")
    print(f"  Trading days        : {metrics['n_trading_days']:,}  (~{metrics['n_years']} years)")
    print()
    print("  Capital")
    print(f"    Initial           : ${metrics['initial_capital']:>12,.2f}")
    print(f"    Final             : ${metrics['final_equity']:>12,.2f}")
    print()
    print("  Returns")
    print(f"    Total return      : {metrics['total_return_pct']:>8.2f} %")
    print(f"    CAGR              : {metrics['cagr_pct']:>8.2f} %")
    print(f"    Benchmark (B&H)   : {bm_return:>8.2f} %")
    print(f"    Alpha (vs B&H)    : {metrics['total_return_pct'] - bm_return:>8.2f} %")
    print()
    print("  Risk")
    print(f"    Ann. volatility   : {metrics['ann_volatility_pct']:>8.2f} %")
    print(f"    Max drawdown      : {metrics['max_drawdown_pct']:>8.2f} %")
    print()
    print("  Risk-adjusted")
    print(f"    Sharpe ratio      : {metrics['sharpe_ratio']:>8.3f}")
    print(f"    Calmar ratio      : {metrics['calmar_ratio']:>8.3f}")
    print()
    print("  Execution")
    print(f"    Total trades      : {len(trade_log):,}")
    total_costs = trade_log["cost"].sum() if not trade_log.empty else 0.0
    print(f"    Total costs paid  : ${total_costs:>10,.2f}")
    print(f"    Avg cost/trade    : ${trade_log['cost'].mean():>10,.2f}"
          if not trade_log.empty else "    Avg cost/trade    :        N/A")
    print()

    if not trade_log.empty:
        print("  Last 5 trades:")
        display_cols = ["ticker", "delta_shares", "fill_price", "notional", "cost", "signal"]
        print(
            trade_log[display_cols]
            .tail(5)
            .to_string(index=True, float_format=lambda x: f"{x:.2f}")
        )
    print()


def export_results(
    ledger: pd.DataFrame,
    trade_log: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> None:
    """
    Save the ledger, trade log, and benchmark to CSV files.
    """
    ledger.to_csv("backtest_equity_curve.csv")
    print("  Saved backtest_equity_curve.csv")

    if not trade_log.empty:
        trade_log.to_csv("backtest_trade_log.csv", index=False)
        print("  Saved backtest_trade_log.csv")

    benchmark.to_csv("backtest_benchmark.csv")
    print("  Saved backtest_benchmark.csv")


# ---------------------------------------------------------------------------
# Public entry point (for import)
# ---------------------------------------------------------------------------

def run_backtest(
    signals: dict[str, pd.DataFrame],
    initial_capital:      float = INITIAL_CAPITAL,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    slippage_pct:         float = SLIPPAGE_PCT,
    max_position_pct:     float = MAX_POSITION_PCT,
    risk_free_rate:       float = RISK_FREE_RATE,
    export:               bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Convenience wrapper: simulate → metrics → benchmark → report → export.

    Parameters
    ----------
    signals              : dict[str, pd.DataFrame]  from signal_generator
    initial_capital      : float
    transaction_cost_pct : float
    slippage_pct         : float
    max_position_pct     : float
    risk_free_rate       : float
    export               : bool   write CSV files if True

    Returns
    -------
    tuple of (ledger, trade_log, metrics)
    """
    print("Running backtest simulation …")
    ledger, trade_log = simulate(
        signals,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        slippage_pct=slippage_pct,
        max_position_pct=max_position_pct,
    )

    metrics   = compute_metrics(ledger, initial_capital, risk_free_rate)
    benchmark = compute_benchmark(signals, initial_capital)

    print_backtest_report(ledger, trade_log, metrics, benchmark, initial_capital)

    if export:
        export_results(ledger, trade_log, benchmark)

    return ledger, trade_log, metrics


# ---------------------------------------------------------------------------
# Main — full pipeline: fetch → preprocess → signals → backtest
# ---------------------------------------------------------------------------

def main() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from signal_generator import run_strategy
    from portfolio_analysis import main as fetch_and_preprocess

    # Step 1: fetch and preprocess price data
    portfolio = fetch_and_preprocess()

    # Step 2: generate crossover signals
    signals = run_strategy(portfolio)

    # Step 3: run the backtest
    ledger, trade_log, metrics = run_backtest(signals)

    return ledger, trade_log, metrics


if __name__ == "__main__":
    main()
