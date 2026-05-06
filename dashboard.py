"""
Multi-Asset Portfolio Dashboard
=================================
Interactive Streamlit dashboard that wires together the full pipeline:
  portfolio_analysis → signal_generator → backtester → performance

Panels
------
  Sidebar   – tune every model parameter live
  Panel 1   – Equity curve vs buy-and-hold benchmark
  Panel 2   – Key metrics summary table
  Panel 3   – Drawdown over time
  Panel 4   – Per-asset allocation weights and trade log
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── local pipeline modules ────────────────────────────────────────────────────
from portfolio_analysis import fetch_price_data, preprocess
from signal_generator   import run_strategy
from backtester         import simulate, compute_benchmark
from performance        import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – all tuneable parameters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Strategy Settings")
    st.caption("Adjust parameters and the dashboard updates automatically.")

    st.subheader("Universe")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="SPY, TLT, GLD",
        help="Any Yahoo Finance tickers. Use 3–5 assets for best results.",
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    years = st.slider("Years of history", min_value=1, max_value=10, value=3)

    st.divider()
    st.subheader("Signal — Moving Averages")
    fast_window = st.slider("Fast SMA window (days)", 10, 100, 50, step=5)
    slow_window = st.slider("Slow SMA window (days)", 50, 300, 200, step=10)
    allow_short = st.toggle("Allow short positions", value=True)

    if fast_window >= slow_window:
        st.warning("Fast window must be shorter than the slow window.")

    st.divider()
    st.subheader("Risk Management")
    initial_capital   = st.number_input("Initial capital ($)", 10_000, 10_000_000, 100_000, step=10_000)
    max_position_pct  = st.slider("Max position per asset (%)", 10, 100, 40, step=5) / 100
    transaction_cost  = st.slider("Transaction cost (%)", 0.0, 1.0, 0.1, step=0.05) / 100
    slippage          = st.slider("Slippage (%)", 0.0, 0.5, 0.05, step=0.01) / 100

    st.divider()
    st.subheader("Performance")
    risk_free_rate    = st.slider("Risk-free rate (%)", 0.0, 10.0, 3.0, step=0.25) / 100
    benchmark_ticker  = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading – cached so it only re-runs when parameters change
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_pipeline(
    tickers, years, fast_window, slow_window, allow_short,
    initial_capital, max_position_pct, transaction_cost, slippage,
    risk_free_rate, benchmark_ticker,
):
    """Run the full pipeline and return all result objects."""
    # 1. Fetch + preprocess
    raw       = fetch_price_data(tickers, years=years)
    portfolio = preprocess(raw)

    # 2. Generate crossover signals
    signals   = run_strategy(portfolio, fast_window=fast_window,
                             slow_window=slow_window, allow_short=allow_short)

    # 3. Backtest simulation
    ledger, trade_log = simulate(
        signals,
        initial_capital      = initial_capital,
        transaction_cost_pct = transaction_cost,
        slippage_pct         = slippage,
        max_position_pct     = max_position_pct,
    )

    # 4. Benchmark
    benchmark = compute_benchmark(signals, initial_capital)

    # 5. Performance metrics
    metrics = evaluate(
        ledger["total_equity"],
        risk_free_rate   = risk_free_rate,
        benchmark_ticker = benchmark_ticker,
    )

    return signals, ledger, trade_log, benchmark, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if fast_window >= slow_window:
    st.error("Please set the fast SMA window to be shorter than the slow SMA window.")
    st.stop()

with st.spinner("Running pipeline — fetching data, generating signals, simulating…"):
    try:
        signals, ledger, trade_log, benchmark, metrics = load_pipeline(
            tuple(tickers), years, fast_window, slow_window, allow_short,
            initial_capital, max_position_pct, transaction_cost, slippage,
            risk_free_rate, benchmark_ticker,
        )
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 Multi-Asset Portfolio Dashboard")
st.caption(
    f"Strategy: Dual Moving Average Crossover  •  "
    f"SMA{fast_window} / SMA{slow_window}  •  "
    f"{'Long/Short' if allow_short else 'Long-only'}  •  "
    f"{', '.join(tickers)}  •  "
    f"{metrics.get('period_start', '')} → {metrics.get('period_end', '')}"
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 – KPI cards
# ─────────────────────────────────────────────────────────────────────────────

def delta_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"

col1, col2, col3, col4, col5 = st.columns(5)

total_ret   = metrics.get("cumulative_return_pct", 0)
ann_ret     = metrics.get("annualised_return_pct", 0)
sharpe      = metrics.get("sharpe_ratio", 0)
max_dd      = metrics.get("max_drawdown_pct", 0)
alpha       = metrics.get("alpha_annualised_pct", float("nan"))

col1.metric("Total Return",       f"{total_ret:.2f}%",  delta=f"{total_ret:.2f}%")
col2.metric("CAGR",               f"{ann_ret:.2f}%",    delta=f"{ann_ret:.2f}%")
col3.metric("Sharpe Ratio",       f"{sharpe:.3f}")
col4.metric("Max Drawdown",       f"{max_dd:.2f}%")
col5.metric("Alpha vs Benchmark", f"{alpha:.2f}%" if not np.isnan(alpha) else "N/A")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 – Equity curve vs benchmark
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Portfolio Equity Curve vs. Buy-and-Hold Benchmark")

fig_equity = go.Figure()

# Strategy equity curve
fig_equity.add_trace(go.Scatter(
    x    = ledger.index,
    y    = ledger["total_equity"],
    name = "Strategy (DMAC)",
    line = dict(color="#2563EB", width=2.5),
    hovertemplate="<b>Strategy</b><br>Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>",
))

# Benchmark
aligned_bm = benchmark.reindex(ledger.index).ffill()
fig_equity.add_trace(go.Scatter(
    x    = aligned_bm.index,
    y    = aligned_bm["benchmark_equity"],
    name = f"Buy & Hold ({', '.join(tickers)})",
    line = dict(color="#9CA3AF", width=1.8, dash="dash"),
    hovertemplate="<b>Benchmark</b><br>Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>",
))

# Initial capital reference line
fig_equity.add_hline(
    y=initial_capital, line_dash="dot",
    line_color="#6B7280", line_width=1,
    annotation_text="Initial capital", annotation_position="bottom right",
)

fig_equity.update_layout(
    height       = 420,
    hovermode    = "x unified",
    legend       = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title  = "Date",
    yaxis_title  = "Portfolio Value ($)",
    yaxis_tickformat = "$,.0f",
    margin       = dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_equity, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 – Drawdown chart + metrics table (side by side)
# ─────────────────────────────────────────────────────────────────────────────

col_dd, col_table = st.columns([3, 2], gap="large")

with col_dd:
    st.subheader("Drawdown Over Time")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x    = ledger.index,
        y    = ledger["drawdown"] * 100,
        name = "Drawdown",
        fill = "tozeroy",
        line = dict(color="#EF4444", width=1.5),
        fillcolor = "rgba(239,68,68,0.15)",
        hovertemplate="<b>Drawdown</b><br>Date: %{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))
    fig_dd.update_layout(
        height       = 320,
        hovermode    = "x unified",
        xaxis_title  = "Date",
        yaxis_title  = "Drawdown (%)",
        yaxis_ticksuffix = "%",
        margin       = dict(l=0, r=0, t=10, b=0),
        showlegend   = False,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with col_table:
    st.subheader("Performance Summary")

    rows = [
        ("Cumulative Return",        f"{metrics.get('cumulative_return_pct', 0):.2f}%"),
        ("Annualised Return (CAGR)", f"{metrics.get('annualised_return_pct', 0):.2f}%"),
        ("Annualised Volatility",    f"{metrics.get('annualised_volatility_pct', 0):.2f}%"),
        ("Sharpe Ratio",             f"{metrics.get('sharpe_ratio', 0):.3f}"),
        ("Max Drawdown",             f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("Drawdown Duration",        f"{metrics.get('max_drawdown_duration_days', 0)} days"),
        ("Calmar Ratio",             f"{metrics.get('calmar_ratio', float('nan')):.3f}"
                                      if not np.isnan(metrics.get('calmar_ratio', float('nan'))) else "N/A"),
        ("Beta",                     f"{metrics.get('beta', float('nan')):.3f}"
                                      if not np.isnan(metrics.get('beta', float('nan'))) else "N/A"),
        ("Alpha (annualised)",       f"{metrics.get('alpha_annualised_pct', float('nan')):.2f}%"
                                      if not np.isnan(metrics.get('alpha_annualised_pct', float('nan'))) else "N/A"),
        ("Information Ratio",        f"{metrics.get('information_ratio', float('nan')):.3f}"
                                      if not np.isnan(metrics.get('information_ratio', float('nan'))) else "N/A"),
        ("Risk-Free Rate",           f"{risk_free_rate*100:.2f}%"),
        ("Benchmark",                benchmark_ticker),
        ("Trading Days",             f"{metrics.get('n_trading_days', 0):,}"),
    ]

    table_df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(
        table_df,
        hide_index = True,
        use_container_width = True,
        height = 490,
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 – Per-asset signals and prices (SMA overlay)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Signal & Moving Average Overlay — Per Asset")

for ticker in tickers:
    df = signals[ticker]

    with st.expander(f"**{ticker}** — price chart with SMA crossover signals", expanded=(ticker == tickers[0])):
        fig_sig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )

        # Close price
        fig_sig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            name="Close", line=dict(color="#1F2937", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}: $%{y:.2f}<extra>Close</extra>",
        ), row=1, col=1)

        # Fast SMA
        fig_sig.add_trace(go.Scatter(
            x=df.index, y=df["sma_fast"],
            name=f"SMA{fast_window}", line=dict(color="#F59E0B", width=1.6, dash="dot"),
            hovertemplate=f"SMA{fast_window}: $%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

        # Slow SMA
        fig_sig.add_trace(go.Scatter(
            x=df.index, y=df["sma_slow"],
            name=f"SMA{slow_window}", line=dict(color="#8B5CF6", width=1.6, dash="dot"),
            hovertemplate=f"SMA{slow_window}: $%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

        # Buy/sell markers from trade log
        if not trade_log.empty:
            asset_trades = trade_log[trade_log["ticker"] == ticker]
            buys  = asset_trades[asset_trades["delta_shares"] > 0]
            sells = asset_trades[asset_trades["delta_shares"] < 0]

            if not buys.empty:
                buy_prices = df.loc[df.index.isin(buys["date"]), "close"]
                fig_sig.add_trace(go.Scatter(
                    x=buy_prices.index, y=buy_prices.values,
                    mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=11, color="#10B981"),
                    hovertemplate="BUY  %{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
                ), row=1, col=1)

            if not sells.empty:
                sell_prices = df.loc[df.index.isin(sells["date"]), "close"]
                fig_sig.add_trace(go.Scatter(
                    x=sell_prices.index, y=sell_prices.values,
                    mode="markers", name="Sell / Short",
                    marker=dict(symbol="triangle-down", size=11, color="#EF4444"),
                    hovertemplate="SELL  %{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
                ), row=1, col=1)

        # Signal bar (bottom sub-panel)
        signal_colors = df["signal"].map({1: "#10B981", 0: "#9CA3AF", -1: "#EF4444"}).fillna("#9CA3AF")
        fig_sig.add_trace(go.Bar(
            x=df.index, y=df["signal"],
            name="Position",
            marker_color=signal_colors,
            hovertemplate="Position: %{y}<extra></extra>",
        ), row=2, col=1)

        fig_sig.update_layout(
            height      = 480,
            hovermode   = "x unified",
            margin      = dict(l=0, r=0, t=10, b=0),
            legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_sig.update_yaxes(title_text="Price ($)", row=1, col=1, tickprefix="$")
        fig_sig.update_yaxes(
            title_text="Signal", row=2, col=1,
            tickvals=[-1, 0, 1], ticktext=["Short", "Cash", "Long"],
        )
        st.plotly_chart(fig_sig, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 – Trade log table
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Trade Log")

if trade_log.empty:
    st.info("No trades were executed in this backtest window.")
else:
    display_trades = trade_log.copy()
    display_trades["date"]       = pd.to_datetime(display_trades["date"]).dt.date
    display_trades["notional"]   = display_trades["notional"].map("${:,.2f}".format)
    display_trades["fill_price"] = display_trades["fill_price"].map("${:.2f}".format)
    display_trades["cost"]       = display_trades["cost"].map("${:.2f}".format)
    display_trades["signal"]     = display_trades["signal"].map({1: "Long", -1: "Short", 0: "Cash"})
    display_trades = display_trades.rename(columns={
        "date":         "Date",
        "ticker":       "Ticker",
        "delta_shares": "Shares Δ",
        "fill_price":   "Fill Price",
        "notional":     "Notional",
        "cost":         "Cost",
        "signal":       "Position",
    })

    st.dataframe(
        display_trades,
        hide_index         = True,
        use_container_width= True,
        height             = 340,
    )

    total_cost = trade_log["cost"].sum()
    st.caption(
        f"**{len(trade_log)} trades** executed  •  "
        f"Total transaction costs: **${total_cost:,.2f}**  •  "
        f"Avg cost per trade: **${trade_log['cost'].mean():,.2f}**"
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 6 – Asset allocation value over time
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Asset Allocation Over Time")

value_cols = [f"{t}_value" for t in tickers if f"{t}_value" in ledger.columns]
if value_cols:
    alloc_df = ledger[value_cols + ["cash"]].copy()
    alloc_df.columns = [c.replace("_value", "") for c in value_cols] + ["Cash"]

    fig_alloc = go.Figure()
    palette   = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]

    for i, col in enumerate(alloc_df.columns):
        fig_alloc.add_trace(go.Scatter(
            x         = alloc_df.index,
            y         = alloc_df[col],
            name      = col,
            stackgroup= "one",
            mode      = "lines",
            line      = dict(width=0.5, color=palette[i % len(palette)]),
            fillcolor = palette[i % len(palette)],
            hovertemplate=f"<b>{col}</b>: $%{{y:,.0f}}<extra></extra>",
        ))

    fig_alloc.update_layout(
        height       = 380,
        hovermode    = "x unified",
        xaxis_title  = "Date",
        yaxis_title  = "Dollar Allocation ($)",
        yaxis_tickformat = "$,.0f",
        legend       = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin       = dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

st.caption("Built with Streamlit · Data via yfinance · Strategy: Dual Moving Average Crossover")
