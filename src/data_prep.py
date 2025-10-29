# src/data_prep.py
import pandas as pd
from typing import Tuple
import numpy as np

# Helper function to convert decimal year to datetime
def decimal_year_to_date(year_decimal: float) -> pd.Timestamp:
    """Converts decimal year (e.g., 2023.2027) to a pandas Timestamp."""
    # This logic assumes the input is a clean float
    year = int(year_decimal)
    rem = year_decimal - year
    date = pd.to_datetime(f'{year}-01-01') + pd.to_timedelta(rem * 365.25, unit='D')
    return date.round('D')

def load_and_prep_data(file_path: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Loads, cleans, and prepares the CO2 time series data by explicitly
    selecting required columns by their position and ensuring numeric types.
    Returns the time series (CO2) and the full DataFrame.
    """
    # Load data, skipping the first row (header=1 uses the second row as header)
    df_raw = pd.read_csv(file_path, header=1)
    
    # 1. Select Columns by Integer Position (iloc) to avoid duplicate name ambiguity
    # Index 3: Decimal Year (e.g., 1960.041)
    # Index 4: Raw CO2 (ppm)
    df = df_raw.iloc[:, [3, 4]].copy()
    
    # 2. Assign unambiguous column names
    df.columns = ['Date_Decimal', 'CO2_Raw_ppm']

    # --- CRITICAL FIX for ValueError ---
    # 3. Force conversion to numeric (float), coercing any invalid string values (like 'NaN' or headers) to NaN
    df['Date_Decimal'] = pd.to_numeric(df['Date_Decimal'], errors='coerce')
    df['CO2_Raw_ppm'] = pd.to_numeric(df['CO2_Raw_ppm'], errors='coerce')

    # 4. Handle Missing Values in CO2 data
    co2_raw = df['CO2_Raw_ppm']
    df['CO2_ppm'] = co2_raw.ffill().bfill()
    
    # 5. Remove any rows that failed the numeric conversion (Date_Decimal will be NaN if invalid)
    # We must have a valid date and a filled CO2 value.
    df.dropna(subset=['Date_Decimal', 'CO2_ppm'], inplace=True)
    
    # 6. Convert decimal year to datetime and set as index
    # Now that the column is guaranteed to be numeric (float), this apply function will work.
    df['Date'] = df['Date_Decimal'].apply(decimal_year_to_date)
    
    # Create the final time series (TS) with Month Start frequency ('MS')
    ts = df.set_index('Date')['CO2_ppm'].asfreq('MS')
    
    print(f"Data loaded, cleaned, and indexed from {ts.index.min().strftime('%Y-%m-%d')} to {ts.index.max().strftime('%Y-%m-%d')}.")
    return ts, df

if __name__ == '__main__':
    # Test/Run the data preparation - adjust path if necessary
    TS, _ = load_and_prep_data("../monthly_flask_co2_mlo.csv")
    print("\nPrepared Time Series head:")
    print(TS.head())