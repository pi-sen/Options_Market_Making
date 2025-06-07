import os
import dask.dataframe as dd
import pandas as pd

def load_all_option_txt_files(folder_path, sep=','):
    all_files = f"{folder_path}/*.txt"
    # Use Dask to read all text files in the folder
    ddf = dd.read_csv(
        all_files,
        sep=sep,
        assume_missing=True,
        na_values=['', ' '],  # Treat empty strings as NaN
        dtype=str  # Read all columns as string initially
    )
    df = ddf.compute()
    # Clean up column names
    df.columns = [col.replace('[', '').replace(']', '').replace(' ', '').strip() for col in df.columns]
    for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df

if __name__== "__main__":
    data_folder = 'data/raw'
    # Load all option data files
    df = load_all_option_txt_files(data_folder, sep=',')
    print(df.head())