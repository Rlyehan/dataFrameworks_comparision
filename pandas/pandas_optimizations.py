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
    dtype_optimizations = {'Measurement': 'float32', 'Humidity': 'float32', 'WindSpeed': 'float32', 'StationName': 'category'}
    data = pd.read_parquet(file_path, columns=list(dtype_optimizations.keys()) + ['Timestamp'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    data.dropna(subset=['Timestamp'], inplace=True)
    data['Timestamp'] = data['Timestamp'].dt.normalize()

    for column, dtype in dtype_optimizations.items():
        data[column] = data[column].astype(dtype)

    if 'Timestamp' not in data.index.names:
        data.set_index(pd.DatetimeIndex(data['Timestamp']), inplace=True)

    return data

def aggregations(data):
    numeric_cols = ['Measurement', 'Humidity', 'WindSpeed']
    return data.groupby('StationName', observed=True)[numeric_cols].agg(['mean', 'median', 'std']).quantile([0.05, 0.95]).unstack(level=-1)


def correlation_and_outliers(data):
    correlation_matrix = data[['Measurement', 'Humidity', 'WindSpeed']].corr()
    data['Measurement_zscore'] = zscore(data['Measurement'])
    outliers = data[abs(data['Measurement_zscore']) > 2]

def ts_analysis(data):
    summary = {}
    features = ['Measurement', 'Humidity', 'WindSpeed']

    for feature in features:
        grouped = data.groupby([pd.Grouper(freq='D'), 'StationName'], observed=False)[feature]
        overall_avg = grouped.mean().groupby(level=0).mean()
        volatility = grouped.std().groupby(level=0).std()
        max_val = grouped.max()
        min_val = grouped.min()

        summary[feature] = {
            'Overall Daily Average': overall_avg.head(),
            'Daily Volatility': volatility.head(),
            'Max Value': max_val.head(),
            'Min Value': min_val.head()
        }

    return summary

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
