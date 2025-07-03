import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


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
    F = df_slice['UNDERLYING_LAST'].iloc[0]  # Use first spot price
    T = df_slice['DTE'].iloc[0] / 365.0  # Convert DTE to years
    
    def objective(params):
        alpha, beta, rho, nu = params
        model_ivs = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) 
                    for K in df_slice['STRIKE']]
        error = np.sum((np.array(model_ivs) - df_slice['C_IV_CALC'])**2)
        return error
    
    # Initial guess and bounds
    initial_guess = [0.2, 0.5, 0.0, 0.2]  # [alpha, beta, rho, nu]
    bounds = [(1e-6, 2.0),  # alpha > 0
             (0.01, 0.99),  # 0 < beta < 1
             (-0.99, 0.99), # -1 < rho < 1
             (1e-6, 2.0)]   # nu > 0
    
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
    
    # Sort by strike for plotting
    df_sorted = df_slice.sort_values('STRIKE')
    
    # Calculate fitted IVs
    fitted_ivs = [sabr_implied_vol(F, K, T, params['alpha'], params['beta'], 
                                 params['rho'], params['nu']) 
                 for K in df_sorted['STRIKE']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['STRIKE'], df_sorted['C_IV_CALC'], 'o', label='Market IV')
    plt.plot(df_sorted['STRIKE'], fitted_ivs, '-', label='SABR Fit')
    plt.title(f'SABR Fit vs Market IV (DTE={df_slice["DTE"].iloc[0]:.0f})')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load your data
    parquet_path = 'data/processed/option_data_final.parquet'
    df = pd.read_parquet(parquet_path).head(2000)  # Load a subset for testing
    
    # Filter out rows with NaN implied volatility
    df.dropna(subset=['C_IV_CALC'])

    # Filtering high IV values
    df = df[df['C_IV_CALC'] < 1.25]


    # Calibrate SABR surface
    sabr_params = calibrate_sabr_surface(df)
    print("\nSABR Parameters by Expiry:")
    print(sabr_params)
    
    # Plot fit for a specific expiry
    first_dte = df['DTE'].iloc[0]
    df_slice = df[df['DTE'] == first_dte]
    first_params = sabr_params[sabr_params['dte'] == first_dte].iloc[0].to_dict()
    plot_sabr_fit(df_slice, first_params)


