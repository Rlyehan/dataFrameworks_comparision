import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
import time
import os

def convert_csv_to_parquet(csv_file_path, parquet_file_path):
    data = pd.read_csv(csv_file_path)
    data.to_parquet(parquet_file_path, index=False)

def load_data(file_path):
    data = pd.read_parquet(file_path)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    data.dropna(subset=['Timestamp'], inplace=True)
    return data

def aggregations(data):
    numeric_cols = ['Measurement', 'Humidity', 'WindSpeed']
    grouped_numeric = data.groupby('StationName')[numeric_cols]

    stats = grouped_numeric.agg(['mean', 'median', 'std'])
    percentiles_5 = grouped_numeric.quantile(0.05).rename(columns=lambda x: x + '_5th')
    percentiles_95 = grouped_numeric.quantile(0.95).rename(columns=lambda x: x + '_95th')
    percentiles = pd.merge(percentiles_5, percentiles_95, left_index=True, right_index=True)

    #print("Basic Statistics:")
    #print(stats)
    #print("5th and 95th Percentiles:")
    #print(percentiles)

def correlation_and_outliers(data):
    correlation_matrix = data[['Measurement', 'Humidity', 'WindSpeed']].corr()
    #print("Correlation Matrix:")
    #print(correlation_matrix)

    data['Measurement_zscore'] = zscore(data['Measurement'])
    outliers = data[abs(data['Measurement_zscore']) > 2]
    #print("Outliers based on Z-score:")
    #print(outliers[['StationName', 'Measurement', 'Measurement_zscore']])

def ts_analysis(data):
    features = ['Measurement', 'Humidity', 'WindSpeed']
    summary = {}

    # Set 'Timestamp' as the index if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.set_index(pd.to_datetime(data['Timestamp']), inplace=True)

    # ensure all feature columns are numeric
    data[features] = data[features].apply(pd.to_numeric, errors='coerce')

    monthly_means = data.groupby([pd.grouper(freq='m'), 'stationname'])[features].mean()
    overall_monthly_avg = monthly_means.groupby(level=0).mean()
    monthly_volatility = monthly_means.groupby(level=0).std()

    # use apply to safely calculate idxmax and idxmin for each feature
    highest_values = data[features].max()
    lowest_values = data[features].min()
    highest_value_dates = {feature: data[feature].idxmax() for feature in features}
    lowest_value_dates = {feature: data[feature].idxmin() for feature in features}

    summary['overall monthly average'] = overall_monthly_avg.head()
    summary['monthly volatility'] = monthly_volatility.head()
    summary['highest values'] = highest_values
    summary['lowest values'] = lowest_values
    summary['dates of highest values'] = highest_value_dates
    summary['dates of lowest values'] = lowest_value_dates

    # Debug: Print summary to check output
    for key, value in summary.items():
        print(f"{key}:\n{value}\n")


def main():
    csv_file_path = '../datagenerator/measurements.csv'
    parquet_file_path = '../datagenerator/measurements.parquet'


    times = {}
    start_time = time.time()
    convert_csv_to_parquet(csv_file_path, parquet_file_path)
    times['csv_to_parquet'] = time.time() - start_time

    load_time = time.time()
    data = load_data(parquet_file_path)
    times['load_data'] = time.time() - load_time

    agg_time = time.time()
    aggregations(data)
    times['aggregations'] = time.time() - agg_time

    corr_time = time.time()
    correlation_and_outliers(data)
    times['correlation_and_outliers'] = time.time() - corr_time

    ts_time = time.time()
    ts_analysis(data)
    times['ts_analysis'] = time.time() - ts_time

    total_time = time.time() - start_time
    times['total'] = total_time

    # Prepare header and rows for the output table
    header = f"{'Component':<25}{'Time (seconds)':<20}{'Fraction of Total':<20}"
    rows = [header]
    for key, value in times.items():
        if key != 'total':  # Skip total for fraction calculation
            fraction = value / total_time
            rows.append(f"{key:<25}{value:<20.4f}{fraction:<20.4f}")
    # Add total time at the end
    rows.append(f"{'total':<25}{total_time:<20.4f}{'1.0000':<20}")

    # Write to a file with the script name
    with open('timing_results.txt', 'a') as f:
        f.write(f"Results from: {os.path.basename(__file__)}\n")
        for row in rows:
            f.write(row + '\n')
        f.write('\n')  # Add an extra newline for spacing between entries

if __name__ == "__main__":
    main()
