import os
import re
import pandas as pd
import numpy as np

def clean_column_name(col):
    """Standardize column headers into clean alphanumeric identifiers."""
    c = re.sub(r'\(.*?\)', '', str(col)).strip()
    c = c.replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_')
    c = re.sub(r'_+', '_', c)
    return c.rstrip('_')

def load_and_clean_data(csv_path="combined_air_quality_2017_2024.csv", drop_cols=None):
    """
    Load raw air quality CAAQMS observations with strict deduplication,
    mixed-format timestamp parsing, and daily frequency alignment.
    
    Returns:
        df (pd.DataFrame): Cleaned chronological daily dataframe.
        report (dict): Data quality audit metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot locate dataset at: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    initial_shape = df_raw.shape

    # 1. Deduplicate identical rows
    df = df_raw.drop_duplicates().copy()
    dedup_rows = len(df_raw) - len(df)

    # Clean and standardize raw column headers immediately
    df.columns = [clean_column_name(c) for c in df.columns]

    # 2. Parse timestamps robustly with multi-layered fallback
    time_col = None
    # Case-insensitive exact check
    for col in df.columns:
        c_clean = str(col).lower().strip('_').strip()
        if c_clean in ['timestamp', 'date', 'datetime', 'time']:
            time_col = col
            break
            
    # Substring check
    if not time_col:
        for col in df.columns:
            c_low = str(col).lower()
            if 'timestamp' in c_low or 'datetime' in c_low or 'date' in c_low:
                time_col = col
                break

    # First column fallback if it contains date strings
    if not time_col and len(df.columns) > 0:
        first_col = df.columns[0]
        try:
            sample_dates = pd.to_datetime(df[first_col].dropna().head(30), format='mixed', errors='coerce')
            if sample_dates.notnull().mean() > 0.5:
                time_col = first_col
        except Exception:
            pass

    if not time_col:
        if all(c in df.columns for c in ['Year', 'Month', 'Day']):
            df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
        else:
            raise ValueError(f"No timestamp or Year/Month/Day columns found. Available columns: {list(df.columns)}")
    else:
        df['Date'] = pd.to_datetime(df[time_col], format='mixed', errors='coerce')
        if time_col != 'Date':
            df = df.drop(columns=[time_col])

    # Check for invalid timestamps
    nat_dates = df['Date'].isnull().sum()
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # 3. Clean string artifacts and convert all non-Date columns to numeric
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("'", "").str.strip(), errors='coerce')

    # 4. Check for any duplicate dates after parsing
    duplicate_dates_count = df['Date'].duplicated().sum()
    if duplicate_dates_count > 0:
        # Group by Date and average observations if multi-station or duplicate
        num_cols = [c for c in df.columns if c != 'Date']
        df = df.groupby('Date')[num_cols].mean().reset_index()

    # 5. Standardize headers
    df.columns = [clean_column_name(c) if c != 'Date' else 'Date' for c in df.columns]

    # 6. Drop 100% missing or unwanted columns
    if drop_cols:
        cleaned_drop = [clean_column_name(c) for c in drop_cols]
        existing_drop = [c for c in cleaned_drop if c in df.columns]
        if existing_drop:
            df = df.drop(columns=existing_drop)

    # Also automatically drop columns with >= 90% missingness (e.g. O_Xylene, Eth_Benzene, MP_Xylene)
    high_missing = [c for c in df.columns if c != 'Date' and df[c].isnull().mean() >= 0.90]
    if high_missing:
        df = df.drop(columns=high_missing)

    # 7. Ensure continuous daily calendar frequency
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    missing_calendar_days = len(full_date_range) - len(df)

    if missing_calendar_days > 0:
        df = df.set_index('Date').reindex(full_date_range).rename_axis('Date').reset_index()

    report = {
        'initial_rows': initial_shape[0],
        'initial_cols': initial_shape[1],
        'deduplicated_dropped_rows': dedup_rows,
        'final_clean_rows': len(df),
        'final_clean_cols': len(df.columns),
        'min_date': df['Date'].min().strftime('%Y-%m-%d'),
        'max_date': df['Date'].max().strftime('%Y-%m-%d'),
        'missing_calendar_days': missing_calendar_days,
        'nat_dates_dropped': nat_dates,
        'duplicate_dates_handled': duplicate_dates_count,
        'dropped_sparse_cols': high_missing
    }

    return df, report
