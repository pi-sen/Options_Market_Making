import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import time

start_time = time.time()

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    SABR model Hagan 2002 lognormal implied volatility approximation.
    Parameters:
        F: Forward price (can use spot price for equity options)
        K: Strike price
        T: Time to expiry (in years)
        alpha, beta, rho, nu: SABR parameters
    Returns:
        Black-Scholes implied volatility (float)
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan
    
    # Handle ATM case separately
    if abs(F - K) < 1e-10:
        numer = alpha
        denom = F**(1-beta)
        term1 = 1 + (((1-beta)**2)/24)*(alpha**2/(F**(2-2*beta))) \
                + (rho*beta*nu*alpha)/(4*F**(1-beta)) \
                + ((2-3*rho**2)/24)*(nu**2)
        return numer/denom * term1
    
    # Non-ATM case
    FK = F*K
    logFK = np.log(F/K)
    z = (nu/alpha) * (FK)**((1-beta)/2) * logFK
    x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho)/(1 - rho))
    numer = alpha * (1 + (((1-beta)**2)/24)*(logFK**2) + ((1-beta)**4/1920)*(logFK**4))
    denom = (FK)**((1-beta)/2) * (1 + (((1-beta)**2)/24)*(np.log(F/K))**2 + ((1-beta)**4/1920)*(np.log(F/K))**4)
    iv = numer/denom * (z/x_z)
    
    # Correction terms
    term1 = 1 + (((1-beta)**2)/24)*(alpha**2/(FK**(1-beta))) \
            + (rho*beta*nu*alpha)/(4*FK**((1-beta)/2)) \
            + ((2-3*rho**2)/24)*(nu**2)
    return iv * term1

def fit_sabr_slice(df_slice):
    """
    Fit SABR parameters to a slice of options data with the same expiry.
    
    Parameters:
        df_slice: DataFrame slice containing options with same expiry
                 Must have columns: UNDERLYING_LAST, STRIKE, DTE, C_IV_CALC
    Returns:
        dict: Fitted SABR parameters and metrics
    """
    required_columns = ['UNDERLYING_LAST', 'STRIKE', 'DTE', 'C_IV_CALC']
    
    if len(df_slice) < 5:  # Need minimum points for reliable fit
        return {
            'success': False,
            'message': 'Insufficient data points'
        }


    F = df_slice['UNDERLYING_LAST'].iloc[0]  # Use first spot price
    T = df_slice['DTE'].iloc[0] / 365.0  # Convert DTE to years

    if F<=0 or T <= 0: # Validate forward price and time to expiry
        return {
            'success': False,
            'message': 'Invalid forward price or time to expiry'
        }
    
    moneyness = df_slice['STRIKE'] / F # Calculate moneyness

    try:
        # Trying better initial parameters based on empirical values
        # 1. Alpha is approximated to be ATM volatility IV
        atm_idx = np.abs(moneyness - 1).argmin() # Retrieve index of ATM option
        atm_iv = df_slice['C_IV_CALC'].iloc[atm_idx] # Get ATM implied volatility

        # 2. Beta is typically around 0.5 for equity options
        beta_int = 0.5 if T < 1 else 0.7

        # 3. Rho (correlation from observed skewness)
        # Skew is estimated as the difference between OTM and ATM IVs
        itm_iv = df_slice[moneyness < 0.95]['C_IV_CALC'].mean()  # In-the-money IV
        otm_iv = df_slice[moneyness > 1.05]['C_IV_CALC'].mean()  # Out-of-the-money IV

        if np.isnan(itm_iv) or np.isnan(otm_iv):
            rho_init = 0.0  # Default to zero if no ITM/OTM IVs
        else:
            skew = otm_iv - atm_iv
            rho_init = -np.sign(skew) * min(0.5, abs(skew)) # Limit rho to [-0.5, 0.5]

        # 4. Nu (volatility of volatility)
        # Higher volatility of volatility for longer expiries
        nu_init = 0.3 if T < 1 else 0.5

        initial_guess = [atm_iv, beta_int, rho_init, nu_init]  # [alpha, beta, rho, nu]

        # Bounds with validation
        bounds = [(1e-6, 2.0),  # alpha > 0
                 (0.0, 1.0),  # 0 < beta < 1
                 (-1.0, 1.0), # -1 < rho < 1
                 (0.01, 1.0)]   # nu > 0
        

    
        def objective(params):
            """
            Objective function to minimize the sum of squared errors
            between model and market implied volatilities.
            Parameters:
                params: List of SABR parameters [alpha, beta, rho, nu]
            Returns:
                float: Sum of squared errors
            """
            alpha, beta, rho, nu = params

            # Early validation of parameters
            if not (0 < alpha < 2 and 0 < beta < 1 and -1 < rho < 1 and nu > 0):
                return 1e10  # Large penalty for invalid parameters
            
            model_ivs = []
            for K in df_slice['STRIKE']:
                try:
                    iv = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
                    if np.isnan(iv) or iv <= 0 or iv > 2:  # Invalid IV
                        return 1e10
                    model_ivs.append(iv)
                except Exception:
                    return 1e10
            
            model_ivs = np.array(model_ivs)
            error = np.sum((model_ivs - df_slice['C_IV_CALC'])**2)
            return error

    
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
        if result.success:
            alpha, beta, rho, nu = result.x
            fitted_ivs = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) 
                     for K in df_slice['STRIKE']]
            rmse = np.sqrt(np.mean((np.array(fitted_ivs) - df_slice['C_IV_CALC'])**2))
        
            return {
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'nu': nu,
                'rmse': rmse,
                'success': True,
                'message': result.message
            }
        else:
            return {
                'success': False,
                'message': result.message
            }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error during calibration: {str(e)}'
        }

def calibrate_sabr_surface(df):
    """
    Calibrate SABR parameters for each expiry in the dataset.
    
    Parameters:
        df: DataFrame with options data
           Must have columns: UNDERLYING_LAST, STRIKE, DTE, C_IV_CALC
    Returns:
        DataFrame: SABR parameters for each expiry
    """
    # Group by expiry (DTE)
    results = []
    for dte, group in df.groupby('DTE'):
        result = fit_sabr_slice(group)
        if result['success']:
            result['dte'] = dte
            results.append(result)
    
    # Convert results to DataFrame
    params_df = pd.DataFrame(results)
    return params_df

def plot_sabr_fit(df_slice, params):
    """
    Plot fitted SABR volatility curve against market data for a single expiry.
    
    Parameters:
        df_slice: DataFrame slice for one expiry
        params: Dict of SABR parameters (alpha, beta, rho, nu)
    """
    F = df_slice['UNDERLYING_LAST'].iloc[0]
    T = df_slice['DTE'].iloc[0] / 365.0
    
    # Calculate the moneyness
    df_slice['MONEYNESS'] = df_slice['STRIKE'] / F
    
    # Sort by strike for plotting
    df_sorted = df_slice.sort_values('MONEYNESS')
    
    # Calculate fitted IVs
    fitted_ivs = [sabr_implied_vol(F, K, T, params['alpha'], params['beta'], 
                                 params['rho'], params['nu']) 
                 for K in df_sorted['STRIKE']]
    
    plt.figure(figsize=(12, 7))
    plt.plot(df_sorted['MONEYNESS'], df_sorted['C_IV_CALC'], 'o', label='Market IV')
    plt.plot(df_sorted['MONEYNESS'], fitted_ivs, '-', label='SABR Fit')
    plt.title(f'SABR Fit vs Market IV (DTE={df_slice["DTE"].iloc[0]:.0f})')
    plt.xlabel('MONEYNESS')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/sabr_fit_dte_{df_slice["DTE"].iloc[0]:.0f}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory

# Example usage:
if __name__ == "__main__":
    # Load your data
    parquet_path = 'data/processed/option_data_final.parquet'
    df = pd.read_parquet(parquet_path).head(20000)  # Load the parquet file
    
    # Filter out rows with NaN implied volatility
    df.dropna(subset=['C_IV_CALC'])

    # Filtering high IV values
    df = df[df['C_IV_CALC'] < 1.25]


    # Calibrate SABR surface
    sabr_params = calibrate_sabr_surface(df)
    print("\nSABR Parameters by Expiry:")
    print(sabr_params)
    
    # Plot fits for all expiries
    if not sabr_params.empty:
        print(f"\nGenerating plots for {len(sabr_params)} expiries...")
        for dte in sabr_params['dte']:
            matching_params = sabr_params[sabr_params['dte'] == dte]
            expiry_data = df[df['DTE'] == dte]
            plot_sabr_fit(expiry_data, matching_params.iloc[0])
            print(f"Saved plot for DTE = {dte:.0f}")
        
        print("All plots have been saved in the results directory.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


