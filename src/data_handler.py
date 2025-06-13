import os
import dask.dataframe as dd
import pandas as pd
from options_pricing import blackscholes, delta, implied_volatility
import swifter
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
    df = ddf.compute()
    
    # Clean up column names
    df.columns = [col.replace('[', '').replace(']', '').replace(' ', '').strip() for col in df.columns]
    # Replace <NA> with '0' (if any remain)
    df = df.replace('<NA>', '0')
    # Convert columns to numeric and fill missing with 0
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()  # Remove leading/trailing spaces
            df[col] = pd.to_numeric(df[col])
            df[col] = df[col].fillna(0)
        except Exception:
            pass
    
    # Convert QUOTE_DATE to datetime format if it exists
    if 'QUOTE_DATE' in df.columns:
        df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
    return df

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
        ddf = dd.from_pandas(df, npartitions=8)
    else:
        print("Parquet file not found, loading from text files.")
        ddf = dd.read_csv(
            f"{data_folder}/*.txt",
            sep=',',
            assume_missing=True,
            na_values=['', ' ', '<NA>'],
            dtype=str
        )
        # Clean up column names
        ddf.columns = [col.replace('[', '').replace(']', '').replace(' ', '').strip() for col in ddf.columns]
        # Replace <NA> with '0' (if any remain)
        ddf = ddf.replace('<NA>', '0')
        # Convert columns to numeric and fill missing with 0
        for col in ddf.columns:
            try:
                ddf[col] = ddf[col].str.strip()
                ddf[col] = ddf[col].astype(float)
                ddf[col] = ddf[col].fillna(0)
            except Exception:
                pass
        # Convert QUOTE_DATE to datetime format if it exists
        if 'QUOTE_DATE' in ddf.columns:
            ddf['QUOTE_DATE'] = dd.to_datetime(ddf['QUOTE_DATE'], errors='coerce')
    
    # Add a column for moneyness
    if 'UNDERLYING_LAST' in ddf.columns and 'STRIKE' in ddf.columns:
        ddf['MONEYNESS'] = ddf['UNDERLYING_LAST'] / ddf['STRIKE']
        print("\nMoneyness column added.")
    
    # Filter: moneyness between 0.5 and 1.2, and (C_VOLUME > 1000 or P_VOLUME > 1000)
    if 'MONEYNESS' in ddf.columns:
        ddf = ddf[(ddf['MONEYNESS'] >= 0.5) & (ddf['MONEYNESS'] <= 1.2)]
    if 'C_VOLUME' in ddf.columns and 'P_VOLUME' in ddf.columns:
        ddf = ddf[(ddf['C_VOLUME'] > 1000) | (ddf['P_VOLUME'] > 1000)]
    print(f"\nFiltered data shape (Dask): {ddf.shape}")
    
    # Calculate IV for call options in parallel using Dask
    ddf = ddf.map_partitions(calc_iv_partition)
    
    # Compute the result and convert to pandas DataFrame
    df = ddf.compute()
    print("\nIV calculation complete. First 10 calculated IV values:")
    print(df['C_IV_CALC'].head(10))
    
    print ("\nDataFrame description for C_IV_CALC:")
    print(df['C_IV_CALC'].describe())

    # Save to Parquet
    # df.to_parquet(parquet_path, index=False)
    # print("Data saved to Parquet file.")

    # Play a beep sound when script finishes
    winsound.Beep(1000, 200)

