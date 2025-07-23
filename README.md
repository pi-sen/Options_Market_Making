# Delta-Based Options Market Making Strategy

## 🎯 Project Overview
This project implements a framework for options market making and backtesting using historical data. The system fetches SPY options data from Yahoo Finance or loads historical options data from CSV files, calculates theoretical prices and Greeks, and simulates market making strategies with inventory and risk management.

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
- `src/vol_surface.py`: Fitting a vol surface using SABR model
- `src/risk_manager.py`: Risk and inventory management
- `src/backtester.py`: Backtesting engine

## SABR Volatility Surface Fitting Workflow

This section describes the steps for fitting a volatility surface using the SABR model in the `src/vol_surface.py` module.

### 1. Import Required Libraries
- Use `numpy`, `pandas`, `scipy.optimize`, and `matplotlib` for calculations and visualization.

### 2. Implement the SABR Implied Volatility Formula
- Write a function to compute SABR implied volatility given parameters (alpha, beta, rho, nu), strike, forward, and expiry.

### 3. Define the SABR Calibration Objective
- Create a function that fits SABR parameters to market IVs for a given expiry by minimizing the error between model and market IVs.

### 4. Fit SABR Parameters for Each Expiry
- For each unique expiry (or expiry/tenor group), fit SABR parameters to the observed IVs across strikes.

### 5. Store and Interpolate the Fitted Surface
- Store the fitted SABR parameters for each expiry.
- Optionally, interpolate parameters across expiries for a smooth surface.

### 6. Create a Class or Functions for Surface Evaluation
- Implement a class or functions that return the SABR IV for a given strike and expiry using the fitted parameters.

### 7. Visualization and Diagnostics
- Plot the fitted SABR surface versus market IVs to check fit quality.

### 8. (Optional) Save/Load Fitted Surface
- Save the fitted parameters to disk for reuse in future analysis.

### 9. Enhanced Parameter Initialization
The calibration process has been improved with smarter initial parameter guesses:

1. **Alpha (α)**: Initialized using ATM implied volatility
   - Uses moneyness (K/F) to find closest ATM option
   - Provides better starting point for optimization

2. **Beta (β)**: Maturity-dependent initialization
   - β = 0.5 for short-term options (T < 1 year)
   - β = 0.7 for longer-dated options
   - Typical range for equity options: [0.3, 0.7]

3. **Rho (ρ)**: Data-driven skew estimation
   - Calculated from observed volatility skew
   - Uses ITM (moneyness < 0.95) and OTM (moneyness > 1.05) IV differences
   - Sign aligned with market skew direction
   - Bounded within [-0.5, 0.5]

4. **Nu (ν)**: Term structure aware
   - ν = 0.3 for short-term options
   - ν = 0.5 for longer-dated options
   - Reflects higher vol-of-vol in longer expiries

### Validation and Error Handling
- Input data validation (required columns, minimum data points)
- Parameter bounds validation pre and post optimization
- RMSE quality checks for fit assessment
- Comprehensive error handling and messaging

### Visualization
- Volatility smile plots now use moneyness (K/F) instead of strike
- Added reference line for ATM point (moneyness = 1.0)
- Display of calibrated parameters and fit quality metrics

These improvements provide more robust and reliable SABR parameter calibration, particularly for equity options markets where the volatility surface exhibits typical characteristics like negative skew and term structure effects.

---

**Summary:**
- Implement SABR formula and calibration
- Fit parameters for each expiry
- Evaluate and visualize the surface
- Save/load fitted results as needed

