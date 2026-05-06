# Hedge Fund Risk Modeling & Semi-Automated Trading System

## Team Information
- **Team Name**: SAGE
- **Year**: 1ST
- **All-Female Team**: NO

## Architecture Overview

How does your system ingest and preprocess the varying data sources (market, macro, sentiment)?
Market data is fetched via APIs and cleaned using forward/backward filling to remove gaps. Key features like returns and volatility are computed. The modular pipeline allows easy addition of macro and sentiment data with similar preprocessing.

What risk modeling techniques were selected, and how are they integrated into the trading decision pipeline?
Risk is modeled using rolling volatility and portfolio constraints like position caps. Metrics such as Sharpe ratio and drawdown evaluate risk-adjusted performance, with Alpha and Beta estimated via regression.

How does your semi-automated strategy generate signals while respecting portfolio constraints and handling realistic conditions like slippage?
Signals are generated using a Dual Moving Average Crossover strategy. The backtester simulates trades with transaction costs and slippage while enforcing position limits and periodic rebalancing.

How is the dashboard designed to provide explainable insights and key metrics (Sharpe, drawdown) to stakeholders?
An interactive dashboard displays key metrics like returns, Sharpe ratio, and drawdown. Users can adjust parameters and instantly see results, ensuring transparency and easy interpretation.
