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
        price = blackscholes(spot, strike, T, self.risk_free_rate, implied_vol, option_type=option_type)
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
    
    def generate_quotes(self, spot, strike, dte, option_type='call'):
        # Get theoretical price
        price_info = self.get_theoretical_price(spot, dte, strike, option_type=option_type)
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
        self.cash_capital = initial_capital
        # Position tracking
        self.call_position = {} # Dictionary to track call positions by {(strike, expiry): quantity}
        self.put_position = {}  # Dictionary to track put positions by {(strike, expiry): quantity}
        self.trades = []  # List to track executed trades
        # Risk management parameters
        self.max_position_size = 10  # Maximum position size per strike
        self.min_capital = initial_capital * 0.9  # Minimum capital to maintain
        self.position_market_prices = {} # Track market prices for positions

    def update_position_market_prices(self, market_data):
        """
        Update the market prices for current positions based on latest market data.
        """
        position_key = (market_data['strike'], market_data['dte'])

        # print(f"\nUpdating prices for {position_key}")

        # Update price for this strike/expiry if we have positions
        if position_key in self.call_position:
            self.position_market_prices[(position_key, 'call')] = {
                'bid': market_data['C_BID'],
                'ask': market_data['C_ASK'],
                'mid': (market_data['C_BID'] + market_data['C_ASK']) / 2
            }
            
        if position_key in self.put_position:
            self.position_market_prices[(position_key, 'put')] = {
                'bid': market_data['P_BID'],
                'ask': market_data['P_ASK'],
                'mid': (market_data['P_BID'] + market_data['P_ASK']) / 2
            }
    
    def calculate_portfolio_value(self):
        """Calculate total portfolio value including cash and market value of positions"""
        portfolio_value = self.cash_capital # Start with cash capital
        position_value = 0  # Initialize position value

        # Track entry prices for positions
        for trade in self.trades: # self.trades contains executed trades
            position_key = (trade['strike'], trade['dte']) # Create a unique key for the position
            if trade['option_type'] == 'call':
                # For short positions, profit is entry price minus current marekt price
                # For long positions, profit is current market price minus entry price
                if (position_key, 'call') in self.position_market_prices: # self.position_market_prices contains current market prices
                    current_price = self.position_market_prices[(position_key, 'call')]['mid'] # current price
                    entry_price = trade['price']
                    quantity = trade['quantity']
                    pnl = (current_price - entry_price) * quantity * 100
                    position_value += pnl
            else:
                if (position_key, 'put') in self.position_market_prices:
                    current_price = self.position_market_prices[(position_key, 'put')]['mid']
                    entry_price = trade['price']
                    quantity = trade['quantity']
                    pnl = (current_price - entry_price) * quantity * 100
                    position_value += pnl
                

        # Total portfolio value is cash + market value of positions
        total_value = self.cash_capital + position_value

        return total_value

    
    def process_market_data(self, market_data, option_type='call'):
        """
        Core trading logic to get quotes and compare with market prices
        Make trading decisions based on quotes and market prices.
        market_data: DataFrame with columns date, spot_price, strike, dte, market_bid, market_ask
        """

        # Update market prices for positions first
        self.update_position_market_prices(market_data)

        # First check if we have enough capital
        if self.cash_capital < self.min_capital:
            return "NO TRADE - LOW CAPITAL"
        
        # Use correct option type to get quotes
        market_bid = market_data['C_BID'] if option_type == 'call' else market_data['P_BID']
        market_ask = market_data['C_ASK'] if option_type == 'call' else market_data['P_ASK']


        # Get theoretical quotes from quote generator
        our_quotes = self.quote_generator.generate_quotes(
            spot = market_data['spot_price'],
            strike = market_data['strike'],
            dte = market_data['dte'],
            option_type=option_type)
        
        # Compare with market quotes
        sell_signal = market_bid > our_quotes['ask'] # Sell if market bid is higher than our ask
        buy_signal = market_ask < our_quotes['bid'] # Buy if market ask is lower than our bid

        # Check current position and capital
        position_key = (market_data['strike'], market_data['dte'])
        if option_type == 'call':
            current_position = self.call_position.get(position_key, 0)
        else: 
            current_position = self.put_position.get(position_key, 0)
       
        # Calculate potential trade value before executing
        potential_trade_value = market_ask * 100  # Worst case scenario cost

         # Make trading decisions with capital checks
        if sell_signal and current_position < self.max_position_size:
            # Check if we have enough capital buffer after the trade
            if (self.cash_capital + potential_trade_value) >= self.min_capital:
                self.execute_trade(
                    strike=market_data['strike'],
                    dte=market_data['dte'],
                    price=our_quotes['ask'],
                    quantity=-1,  # Sell is negative quantity
                    timestamp=market_data['date'],
                    option_type=option_type)
                return f"SOLD {option_type.upper()}"
        
        elif buy_signal and current_position > -self.max_position_size:
            # Check if we have enough capital for the purchase
            if (self.cash_capital - potential_trade_value) >= self.min_capital:
                self.execute_trade(
                    strike=market_data['strike'],
                    dte=market_data['dte'],
                    price=our_quotes['bid'],
                    quantity=1,
                    timestamp=market_data['date'],
                    option_type=option_type)
                return f"BOUGHT {option_type.upper()}"
        
        return "NO TRADE"
    
    def execute_trade(self, strike, dte, price, quantity, timestamp, option_type):
        """
        Execute a trade and update position and capital
        """
        position_key = (strike, dte)
        # Update position
        if option_type == 'call':
            self.call_position[position_key] = self.call_position.get(position_key, 0) + quantity
        else:
            self.put_position[position_key] = self.put_position.get(position_key, 0) + quantity
        
        # Update capital (assuming 100 shares per contract)
        trade_value = price * quantity * 100
        self.cash_capital -= trade_value if quantity > 0 else -trade_value

        # Record the trade
        trade_record = {
            'timestamp': timestamp,
            'strike': strike,
            'dte': dte,
            'price': price,
            'quantity': quantity,
            'option_type': option_type,
            'trade_value': trade_value,
            'cash capital': self.cash_capital,
            'portfolio_value': self.calculate_portfolio_value()
        }
        self.trades.append(trade_record)


if __name__ == "__main__":
    # First calibrate SABR parameters
    df = pd.read_parquet('data/processed/option_data_final.parquet').head(50000)
    calibrate_sabr_surface(df)  # This saves parameters to CSV
    
    # Create quote generator using saved parameters
    quote_gen = QuoteGenerator()  # This loads the calibrated parameters
    market_maker = MarketMaker(quote_generator=quote_gen, initial_capital=100_000)
    
    print("\nRunning trading strategy...")
    
    # Process market data
    for _, row in df.iterrows():
        market_data = {
            'date': row['QUOTE_DATE'],
            'spot_price': row['UNDERLYING_LAST'],
            'strike': row['STRIKE'],
            'dte': row['DTE'],
            'C_BID': row['C_BID'],
            'C_ASK': row['C_ASK'],
            'P_BID': row['P_BID'],
            'P_ASK': row['P_ASK']
        }
        
        # Process both calls and puts
        call_result = market_maker.process_market_data(market_data, 'call')
        put_result = market_maker.process_market_data(market_data, 'put')

    # Print final summary
    print("\nTrading Summary")
    print("=" * 60)
    print(f"Initial Capital:    ${100_000:,.2f}")
    print(f"Final Cash:         ${market_maker.cash_capital:,.2f}")
    
    portfolio_value = market_maker.calculate_portfolio_value()
    position_value = portfolio_value - market_maker.cash_capital
    print(f"Position P&L:       ${position_value:,.2f}")
    print(f"Portfolio Value:    ${portfolio_value:,.2f}")
    print(f"Total P&L:         ${portfolio_value - 100_000:,.2f}")
    print(f"Total Trades:       {len(market_maker.trades)}")

    # Print open positions
    print("\nOpen Positions")
    print("=" * 100)
    print(f"{'Type':<6} {'Strike':>8} {'DTE':>6} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>12}")
    print("-" * 100)
    
    # Print non-zero call positions
    for (strike, dte), qty in market_maker.call_position.items():
        if qty != 0:  # Only show non-zero positions
            price_key = ((strike, dte), 'call')
            if price_key in market_maker.position_market_prices:
                current_price = market_maker.position_market_prices[price_key]['mid']
                # Find the most recent trade for this position
                entry_price = next((t['price'] for t in reversed(market_maker.trades) 
                                 if t['strike'] == strike and t['dte'] == dte 
                                 and t['option_type'] == 'call'), 0)
                pnl = (current_price - entry_price) * qty * 100
                print(f"{'CALL':<6} {strike:>8.1f} {dte:>6} {qty:>6} "
                      f"{entry_price:>10.2f} {current_price:>10.2f} "
                      f"${pnl:>11,.2f}")

    # Print non-zero put positions
    for (strike, dte), qty in market_maker.put_position.items():
        if qty != 0:  # Only show non-zero positions
            price_key = ((strike, dte), 'put')
            if price_key in market_maker.position_market_prices:
                current_price = market_maker.position_market_prices[price_key]['mid']
                # Find the most recent trade for this position
                entry_price = next((t['price'] for t in reversed(market_maker.trades) 
                                 if t['strike'] == strike and t['dte'] == dte 
                                 and t['option_type'] == 'put'), 0)
                pnl = (current_price - entry_price) * qty * 100
                print(f"{'PUT':<6} {strike:>8.1f} {dte:>6} {qty:>6} "
                      f"{entry_price:>10.2f} {current_price:>10.2f} "
                      f"${pnl:>11,.2f}")

    # Print position statistics
    print("\nPosition Statistics")
    print("=" * 60)
    total_long_calls = sum(1 for q in market_maker.call_position.values() if q > 0)
    total_short_calls = sum(1 for q in market_maker.call_position.values() if q < 0)
    total_long_puts = sum(1 for q in market_maker.put_position.values() if q > 0)
    total_short_puts = sum(1 for q in market_maker.put_position.values() if q < 0)
    
    print(f"Long Calls:  {total_long_calls:3d}    Short Calls: {total_short_calls:3d}")
    print(f"Long Puts:   {total_long_puts:3d}    Short Puts:  {total_short_puts:3d}")
    print(f"Total Long:  {total_long_calls + total_long_puts:3d}    "
          f"Total Short: {total_short_calls + total_short_puts:3d}")