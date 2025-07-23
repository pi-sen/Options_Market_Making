import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import time
from vol_surface import sabr_implied_vol, calibrate_sabr_surface
from options_pricing import blackscholes

class QuoteGenerator:
    def __init__(self, params_file='data/processed/sabr_parameters.csv', risk_free_rate = 0.04):
        """
        Initialize the QuoteGenerator with SABR parameters DataFrame.
        
        Parameters:
            sabr_params_df: DataFrame containing SABR parameters for each expiry
        """
        self.sabr_params_df = pd.read_csv(params_file)
        self.risk_free_rate = risk_free_rate
    
    def get_sabr_params(self, dte):
        """
        Get SABR parameters for a given expiry.
        """
        try:
            closest_dte = self.sabr_params_df['dte'].iloc[(self.sabr_params_df['dte'] - dte).abs().argsort()[0]]
            params = self.sabr_params_df[self.sabr_params_df['dte'] == closest_dte].iloc[0]
            return {
                'alpha': params['alpha'],
                'beta': params['beta'],
                'rho': params['rho'],
                'nu': params['nu']
            }
        except KeyError:
            raise KeyError(f"No SABR parameters found for DTE: {dte}")

    def get_theoretical_price(self,spot, expiry, strike, option_type='call'):
        """
        Get SABR implied volatility and then calculate the theoretical price of an option.
        """
        T = expiry / 365.0  # Convert days to years
        params = self.get_sabr_params(expiry) # Extract SABR parameters for the given expiry
        implied_vol = sabr_implied_vol(spot, strike, T, params['alpha'], params['beta'], params['rho'], params['nu']) # Calculate implied volatility
        # Use Black-Scholes formula to calculate the theoretical price
        price = blackscholes(spot, strike, T, self.risk_free_rate, implied_vol, option_type='call')
        return {'price': price, 'implied_vol': implied_vol}
    
    def calculate_spread(self, theo_price, spot, strike, dte, params):
        """
        Calculate the spread based on theoretical price, spot, strike, and DTE.
        
        Parameters:
            theo_price: Theoretical price of the option
            spot: Current spot price of the underlying asset
            strike: Strike price of the option
            dte: Days to expiration
            params: Additional parameters for spread calculation
            
        Returns:
            float: Calculated spread
        """
        # Example spread calculation (this can be customized)
        base_spread = theo_price * 0.05 # 5% of the theoretical price

        # Adjust for moneyness
        moneyness = strike / spot
        moneyness_factor = abs(1 - moneyness) * 0.5

        # Adjust for time to expiration
        time_factor = np.sqrt(dte / 365.0)  # Scale by square root of time

        # Adjust for volatility of volatility
        vol_factor = params['nu'] * 0.5

        total_spread = base_spread * (1 + moneyness_factor + time_factor + vol_factor)
        return total_spread
    
    def generate_quotes(self, spot, strike, dte):
        # Get theoretical price
        price_info = self.get_theoretical_price(spot, dte, strike, option_type = 'call')
        theo_price = price_info['price']
        implied_vol = price_info['implied_vol']

        # Get SABR parameters for the given expiry
        params = self.get_sabr_params(dte)
        spread = self.calculate_spread(theo_price, spot, strike, dte, params)

        # Generate bid  and ask prices
        bid = theo_price - spread / 2
        ask = theo_price + spread / 2

        return {
            'theoretical_price': round(theo_price,4),
            'bid': round(bid,4),
            'ask': round(ask,4),
            'spread': round(spread,4)
        }  

class MarketMaker:
    def __init__(self, quote_generator, initial_capital = 100_000):
        self.quote_generator = quote_generator
        self.capital = initial_capital
        # Position tracking
        self.position = {} # Dictionary to track positions by {(strike, expiry): quantity}
        self.trades = []  # List to track executed trades
        # Risk management parameters
        self.max_position_size = 10  # Maximum position size per strike
        self.min_capital = initial_capital * 0.9  # Minimum capital to maintain

    def process_market_data(self, market_data):
        """
        Core trading logic to get quotes and compare with market prices
        Make trading decisions based on quotes and market prices.
        market_data: DataFrame with columns date, spot_price, strike, dte, market_bid, market_ask
        """
        # First check if we have enough capital
        if self.capital < self.min_capital:
            return "NO TRADE - LOW CAPITAL"
        
        # Get theoretical quotes from quote generator
        our_quotes = self.quote_generator.generate_quotes(
            spot = market_data['spot_price'],
            strike = market_data['strike'],
            dte = market_data['dte'])
        
        # Compare with market quotes
        sell_signal = market_data['market_bid'] > our_quotes['ask'] # Sell if market bid is higher than our ask
        buy_signal = market_data['market_ask'] < our_quotes['bid'] # Buy if market ask is lower than our bid

        # Check current position and capital
        position_key = (market_data['strike'], market_data['dte'])
        current_position = self.position.get(position_key, 0)

        # Calculate potential trade value before executing
        potential_trade_value = market_data['market_ask'] * 100  # Worst case scenario cost

        # Make trading decisions
        # Make trading decisions with capital checks
        if sell_signal and current_position < self.max_position_size:
            # Check if we have enough capital buffer after the trade
            if (self.capital + potential_trade_value) >= self.min_capital:
                self.execute_trade(
                    strike=market_data['strike'],
                    dte=market_data['dte'],
                    price=our_quotes['ask'],
                    quantity=-1,  # Sell is negative quantity
                    timestamp=market_data['date'])
                return "SOLD"
        
        elif buy_signal and current_position > -self.max_position_size:
            # Check if we have enough capital for the purchase
            if (self.capital - potential_trade_value) >= self.min_capital:
                self.execute_trade(
                    strike=market_data['strike'],
                    dte=market_data['dte'],
                    price=our_quotes['bid'],
                    quantity=1,
                    timestamp=market_data['date'])
                return "BOUGHT"
        
        return "NO TRADE"
    
    def execute_trade(self, strike, dte, price, quantity, timestamp):
        """
        Execute a trade and update position and capital
        """
        position_key = (strike, dte)
        # Update position
        self.position[position_key] = self.position.get(position_key, 0) + quantity

        # Update capital (assuming 100 shares per contract)
        trade_value = price * quantity * 100
        self.capital -= trade_value if quantity > 0 else -trade_value

        # Record the trade
        trade_record = {
            'timestamp': timestamp,
            'strike': strike,
            'dte': dte,
            'price': price,
            'quantity': quantity,
            'trade_value': trade_value,
            'remaining_capital': self.capital
        }
        self.trades.append(trade_record)



if __name__ == "__main__":
    # First calibrate SABR parameters
    df = pd.read_parquet('data/processed/option_data_final.parquet').head(1000)
    calibrate_sabr_surface(df)  # This saves parameters to CSV
    
    # Create quote generator using saved parameters
    quote_gen = QuoteGenerator()  # This loads the calibrated parameters
    market_maker = MarketMaker(quote_generator=quote_gen, initial_capital=100_000)
    
    # Testing the strategy
    print("\nTesting trading strategy...")
    results = []

    for _, row in df.iterrows():
        market_data = {
            'date': row['QUOTE_DATE'],
            'spot_price': row['UNDERLYING_LAST'],
            'strike': row['STRIKE'],
            'dte': row['DTE'],
            'market_bid': row['C_BID'],
            'market_ask': row['C_ASK']
        }
        
        market_maker.process_market_data(market_data)
    
    # Print only the trades and final summary
    print("\nTrade History:")
    print("=" * 80)
    print(f"{'Date':<12} {'Action':<8} {'Strike':>8} {'DTE':>6} {'Price':>10} {'Quantity':>8} {'Trade Value':>12}")
    print("-" * 80)
    
    for trade in market_maker.trades:
        print(f"{str(trade['timestamp']):<12} "
              f"{'SELL' if trade['quantity'] < 0 else 'BUY':<8} "
              f"{trade['strike']:>8.1f} "
              f"{trade['dte']:>6} "
              f"{trade['price']:>10.2f} "
              f"{abs(trade['quantity']):>8} "
              f"${abs(trade['trade_value']):>11,.2f}")
    
    print("\nFinal Summary:")
    print("=" * 40)
    print(f"Initial Capital: ${100_000:,.2f}")
    print(f"Final Capital:   ${market_maker.capital:,.2f}")
    print(f"Total P&L:      ${market_maker.capital - 100_000:,.2f}")
    print(f"Total Trades:    {len(market_maker.trades)}")