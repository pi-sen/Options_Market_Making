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
            'theoretical_price': theo_price,
            'bid': bid,
            'ask': ask,
            'spread': spread
        }  


if __name__ == "__main__":
    # First calibrate SABR parameters
    df = pd.read_parquet('data/processed/option_data_final.parquet')
    calibrate_sabr_surface(df)  # This saves parameters to CSV
    
    # Create quote generator using saved parameters
    quote_gen = QuoteGenerator()
    
    # Generate some quotes
    quotes = quote_gen.generate_quotes(
        spot=100.0,
        strike=105.0,
        dte=30
    )
    print(f"Generated quotes: {quotes}")
        


