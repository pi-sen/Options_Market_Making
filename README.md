# Delta-Based Options Market Making Strategy

## 🎯 Project Overview
This project implements a framework for options market making and backtesting using real and historical data. The system fetches SPY options data from Yahoo Finance or loads historical options data from CSV files, calculates theoretical prices and Greeks, and simulates market making strategies with inventory and risk management.

## 🔍 Core Concepts
**Market Making**: Acting as a liquidity provider by continuously quoting bid/ask prices for options contracts, aiming to profit from the bid-ask spread while managing inventory risk.

**Delta Hedging**: Maintaining a delta-neutral portfolio by dynamically adjusting positions in the underlying asset to offset directional risk from options exposure.

## 📊 Strategy Components
- **Data Import**: Fetches live SPY options data or loads historical data from CSV for backtesting.
- **Options Pricing Engine**: Black-Scholes pricing and Greeks (Delta, Gamma, Theta, Vega) calculations. Calculate IV from market prices. Use market IV to fit a smooth model
- **Build vol surface**: Use IV model to compute theoretical prices
- **Market Making Logic**: Sets bid/ask quotes around theoretical fair value, simulates fills, and manages inventory.
- **Risk Management**: Delta hedging, position limits, and PnL tracking.
- **Backtesting Framework**: Evaluates strategy performance on historical data.

## 🧮 Mathematical Framework
- Black-Scholes Model for European option pricing
- Greeks calculations (Delta, Gamma, Theta, Vega)
- Implied volatility estimation
- Inventory and risk-adjusted quoting

## 🏗️ Project Structure
- `src/data_handler.py`: Data import from Yahoo Finance or CSV
- `src/options_pricing.py`: Black-Scholes pricing and Greeks
- `src/market_maker.py`: Market making logic and simulation
- `src/risk_manager.py`: Risk and inventory management
- `src/backtester.py`: Backtesting engine

