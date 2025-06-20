import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.optimize._minimize as opt

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    SABR model Hagan 2002 lognormal implied volatility approximation.
    Parameters:
        F: Forward price (can use spot for equity options)
        K: Strike price
        T: Time to expiry (in years)
        alpha, beta, rho, nu: SABR parameters
    Returns:
        Black-Scholes implied volatility (float)
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan
    if F == K:
        # ATM formula
        numer = alpha
        denom = F**(1-beta)
        term1 = 1 + (((1-beta)**2)/24)*(alpha**2/(F**(2-2*beta))) \
                + (rho*beta*nu*alpha)/(4*F**(1-beta)) \
                + ((2-3*rho**2)/24)*(nu**2)
        return numer/denom * term1
    else:
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
    
def fit_sabr_model(df, F_col='F', K_col='K', T_col='T', iv_col='IV'):
    def objective_function(params):
        alpha, beta, rho, nu = params
        ivs = df.apply(lambda row: sabr_implied_vol(
            F_col, K_col, T_col, alpha, beta, rho, nu
        )
        , axis=1)
        return np.sum((ivs - df[iv_col])**2)
    intial_guess = [0.2, 0.5, 0.0, 0.2]  # Initial guess for alpha, beta, rho, nu
    bounds = [(1e-6, None), (0.01, 1), (-1, 1), (1e-6, None)]  # Bounds for parameters
    result = opt.minimize(objective_function, intial_guess, bounds=bounds)
    if result.success:
        return result.x  # Returns the fitted parameters [alpha, beta, rho, nu]
    else:
        raise ValueError("SABR model fitting failed: " + result.message)

print (fit_sabr_model(df,)
