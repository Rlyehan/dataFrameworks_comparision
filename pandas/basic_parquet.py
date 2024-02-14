import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
import time
import os

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
    summary = {}

    features = ['Measurement', 'Humidity', 'WindSpeed']

    for feature in features:
        station_monthly_avg = data.groupby('StationName').resample('ME', on='Timestamp')[feature].mean().reset_index()
        overall_monthly_avg = station_monthly_avg.groupby('Timestamp')[feature].mean()
        monthly_volatility = station_monthly_avg.groupby('Timestamp')[feature].std()

        highest_value = data[feature].max()
        lowest_value = data[feature].min()
        highest_value_date = data[data[feature] == highest_value]['Timestamp'].iloc[0]
        lowest_value_date = data[data[feature] == lowest_value]['Timestamp'].iloc[0]

        summary[f"{feature} - Overall Monthly Average"] = overall_monthly_avg.head()
        summary[f"{feature} - Monthly Volatility"] = monthly_volatility.head()
        summary[f"{feature} - Highest Value"] = highest_value
        summary[f"{feature} - Lowest Value"] = lowest_value
        summary[f"{feature} - Date of Highest Value"] = highest_value_date
        summary[f"{feature} - Date of Lowest Value"] = lowest_value_date

    #print(summary)

def main():
    file_path = '../datagenerator/measurements.parquet'

    times = {}
    start_time = time.time()
    data = load_data(file_path)
    times['load_data'] = time.time() - start_time

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
