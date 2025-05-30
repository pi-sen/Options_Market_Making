# Test more features with this code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def plot_option_payoff(S, K, option_type='call'):
    """Plot basic option payoff"""
    stock_prices = np.linspace(S*0.7, S*1.3, 100)
    
    if option_type == 'call':
        payoff = np.maximum(stock_prices - K, 0)
    else:  # put
        payoff = np.maximum(K - stock_prices, 0)
        
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, payoff, label=f'{option_type.capitalize()} Option Payoff')
    plt.axvline(x=K, color='r', linestyle='--', label='Strike Price')
    plt.grid(True)
    plt.legend()
    plt.title(f'{option_type.capitalize()} Option Payoff Diagram')
    plt.xlabel('Stock Price')
    plt.ylabel('Payoff')
    plt.show()

# Test with SPY data
spy = yf.Ticker('SPY')
current_price = spy.history(period='1d')['Close'].iloc[-1]
strike_price = round(current_price, -1)  # Round to nearest 10

# Plot both call and put payoffs
plot_option_payoff(current_price, strike_price, 'call')
plot_option_payoff(current_price, strike_price, 'put')