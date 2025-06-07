"""
Options Pricing Engine for Delta-Based Market Making Strategy
Implements Black-Scholes pricing, Greeks calculations, and implied volatility solving
"""
import numpy as np
from scipy.stats import norm # Importing norm for cumulative distribution function
from scipy.optimize import brentq # Brent's method for root finding
import math # For mathematical operations

# Stock parameters
S = 100.0  # Current stock price
K = 100.0  # Strike price  
T = 1.5    # Time to expiration in years
r = 0.05   # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset

def blackscholes (S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes price of a European option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    option_type (str): 'call' for call option, 'put' for put option
    
    Returns:
    float: Price of the option
    """
    if T == 0: # edge case for immediate expiration
        if option_type == 'call':
            return max(S - K,0)
        else: return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K  *np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return price

def delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Delta of a European option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    option_type (str): 'call' for call option, 'put' for put option
    
    Returns:
    float: Delta of the option
    """
    if T == 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else: return 1.0 if K> S else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1

def implied_volatility(S, K, T, r, market_price, option_type='call'): # calculate implied volatility using Brent's method
    """
    Calculate the implied volatility of a European option using Brent's method.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    market_price (float): Market price of the option
    option_type (str): 'call' for call option, 'put' for put option
    
    Returns:
    float: Implied volatility of the option
    """
    # Sanity check on market price
    intrinsic_value = max(S - K,0) if option_type == 'call' else max(K - S,0)
    if market_price < intrinsic_value:
        return None
    def objective_function(sigma):
        return blackscholes(S, K, T, r, sigma, option_type) - market_price

    # Use Brent's method to find the root of the objective function
    try:
        implied_vol = brentq(objective_function, 1e-6, 5.0)
        return implied_vol
    except ValueError:
        return None  # Return None if no solution is found


price = blackscholes(S, K, T, r, sigma, option_type='call')
print(f"Call Option Price: {price:.2f}")

delta_value = delta(S, K, T, r, sigma, option_type='call')
print(f"Call Option Delta: {delta_value:.2f}")

market_price = 10.0  # Example market price for implied volatility calculation
implied_vol = implied_volatility(S, K, T, r, market_price, option_type='call')
print (f"Implied Volatility: {implied_vol:.2f}")

