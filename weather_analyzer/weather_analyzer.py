# weather_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # ensure date column exists; adapt column names to your dataset
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise ValueError("No date column found.")
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Example numeric columns: 'temp', 'rain', 'humidity'
    # Rename or map according to your CSV
    # Fill numeric NaNs with interpolation then mean fallback
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate().fillna(df[numeric_cols].mean())
    # Add month/year
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

def compute_stats(df: pd.DataFrame):
    # Example: daily mean temp (if multiple entries per day)
    df_daily = df.set_index('date').resample('D').mean().reset_index()
    # monthly aggregation
    monthly = df.set_index('date').resample('M').agg({'temp':'mean', 'rain':'sum', 'humidity':'mean'})
    return df_daily, monthly

def plot_daily_temp(df_daily: pd.DataFrame):
    plt.figure()
    plt.plot(df_daily['date'], df_daily['temp'])
    plt.title("Daily Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "daily_temp.png")
    plt.close()

def plot_monthly_rain(monthly: pd.DataFrame):
    plt.figure()
    plt.bar(monthly.index.strftime('%Y-%m'), monthly['rain'])
    plt.xticks(rotation=45)
    plt.title("Monthly Rainfall")
    plt.xlabel("Month")
    plt.ylabel("Total Rainfall")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "monthly_rainfall.png")
    plt.close()

def plot_humidity_vs_temp(df):
    plt.figure()
    plt.scatter(df['temp'], df['humidity'])
    plt.title("Humidity vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "humidity_vs_temp.png")
    plt.close()

def main():
    raw_path = DATA_DIR / "raw_weather.csv"
    df = load_data(raw_path)
    df_clean = clean_data(df)
    df_clean.to_csv(DATA_DIR / "cleaned_weather.csv", index=False)

    df_daily, monthly = compute_stats(df_clean)
    plot_daily_temp(df_daily)
    plot_monthly_rain(monthly)
    plot_humidity_vs_temp(df_clean)

if __name__ == "__main__":
    main()