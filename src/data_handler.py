import os
import dask.dataframe as dd
import pandas as pd
from options_pricing import blackscholes, delta, implied_volatility
import winsound

# Function to load all option text files from a folder
def load_all_option_txt_files(folder_path, sep=','):
    all_files = f"{folder_path}/*.txt"
    ddf = dd.read_csv(
        all_files,
        sep=sep,
        assume_missing=True,
        na_values=['', ' ', '<NA>'],  # Treat empty strings and <NA> as NaN
        dtype=str  # Read all columns as string initially
    )
    
    # Clean up column names
    ddf.columns = [col.replace('[', '').replace(']', '').replace(' ', '').strip() for col in ddf.columns]
    
    # Replace <NA> with '0' (if any remain)
    ddf = ddf.replace('<NA>', '0')
    
    # Convert columns to numeric (except QUOTE_DATE)
    for col in ddf.columns:
        if col != 'QUOTE_DATE':
            ddf[col] = ddf[col].str.strip().astype(float)
    # Convert QUOTE_DATE to datetime format if it exists
    if 'QUOTE_DATE' in ddf.columns:
        ddf['QUOTE_DATE'] = dd.to_datetime(ddf['QUOTE_DATE'], errors='coerce')
    return ddf

def calc_iv_partition(df):
    # Calculate IV for call options in a DataFrame partition
    df['C_IV_CALC'] = df.apply(
        lambda row: implied_volatility(
            S=row['UNDERLYING_LAST'],
            K=row['STRIKE'],
            T=row['DTE'] / 365,
            r=0.05,
            market_price=row['C_LAST'],
            option_type='call'
        ) if pd.notnull(row['C_LAST']) and row['C_LAST'] > 0 else 0, axis=1
    )
    return df

if __name__== "__main__":
    parquet_path = 'data/processed/option_data.parquet'
    data_folder = 'data/raw'
    
    if os.path.exists(parquet_path): # Check if Parquet file exists
        df = pd.read_parquet(parquet_path)
        print("Loaded data from Parquet file.")
        ddf = dd.from_pandas(df, npartitions=4)
    else:
        print("Parquet file not found, loading from text files.")
        ddf = load_all_option_txt_files(data_folder, sep=',')
    # Add a column for moneyness
    if 'UNDERLYING_LAST' in ddf.columns and 'STRIKE' in ddf.columns:
        ddf['MONEYNESS'] = ddf['UNDERLYING_LAST'] / ddf['STRIKE']
        print("\nMoneyness column added.")
    
    # Filter: moneyness between 0.5 and 1.2, and (C_VOLUME > 1000 or P_VOLUME > 1000)
    if 'MONEYNESS' in ddf.columns:
        ddf = ddf[(ddf['MONEYNESS'] >= 0.2) & (ddf['MONEYNESS'] <= 1.5)]
    if 'C_VOLUME' in ddf.columns and 'P_VOLUME' in ddf.columns:
        ddf = ddf[(ddf['C_VOLUME'] > 500) | (ddf['P_VOLUME'] > 500)]
    print(f"\nShape of Dataframe before dropping: {len(ddf)} x {len(ddf.columns)}")
    
    # Dropping unwanted columns
    remove_columns = [
        "QUOTE_UNIXTIME", "QUOTE_READTIME", "QUOTE_TIME_HOURS", "EXPIRE_UNIX",
        "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV", "C_SIZE",
        "P_SIZE", "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV",
        "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT"
    ]
    ddf = ddf.drop(columns=remove_columns, errors='ignore')
    print(f"\nShape of Dataframe after dropping: {len(ddf)} x {len(ddf.columns)}")
    # Print column names and first row values together
    first_row = ddf.head(1, npartitions=-1)
    print("Columns and first row values:")
    if not first_row.empty:
        for col in first_row.columns:
            print(f"{col}: {first_row.iloc[0][col]}")
    else:
        print("No rows available in DataFrame after filtering.")
    # Repartition for efficiency
    ddf = ddf.repartition(npartitions=4)
    # Calculate IV for call options in parallel using Dask
    ddf = ddf.map_partitions(calc_iv_partition)
    
    # Compute the result and convert to pandas DataFrame
    df = ddf.compute()
    print("\nIV calculation complete. First 10 calculated IV values:")
    print(df['C_IV_CALC'].head(10))
    print("\nDataFrame description for C_IV_CALC:")
    print(df['C_IV_CALC'].describe())

    # Count number of NaN values in C_IV_CALC
    nan_count = df['C_IV_CALC'].isna().sum()
    print (f"\nNumber of NaN values in C_IV_CALC: {nan_count}")
    
    # Remove rows where C_IV_CALC is NaN
    df = df.dropna(subset=['C_IV_CALC'])
    print(f"\nShape after dropping NaN IVs: {df.shape}")
    
    # Play a beep sound when script finishes
    winsound.Beep(1000, 250)

