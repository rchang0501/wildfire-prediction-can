import pandas as pd
import numpy as np

def process_weather_data(input_file, output_file):
    """
    Processes weather data by:
    1. Dropping 'Network Name' and 'Native ID' columns
    2. Rounding latitude and longitude to the nearest tenth
    3. Aggregating data by averaging values for entries with the same date and rounded coordinates
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the processed CSV file
    """
    # Read the data
    df = pd.read_csv(input_file)
    
    # Drop specified columns
    df = df.drop(['Network Name', 'Native ID'], axis=1)
    
    # Round latitude and longitude to the nearest tenth
    df['Longitude'] = np.round(df['Longitude'], 1)
    df['Latitude'] = np.round(df['Latitude'], 1)
    
    # Convert numeric columns to float (handling potential non-numeric values)
    numeric_cols = df.columns.difference(['Date', 'Longitude', 'Latitude'])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by date and rounded coordinates, then average the values
    grouped_df = df.groupby(['Date', 'Longitude', 'Latitude'], as_index=False).agg({
        col: 'mean' for col in numeric_cols
    })
    
    # Save the processed data
    grouped_df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
    print(f"Original row count: {len(df)}")
    print(f"After processing row count: {len(grouped_df)}")
    
    return grouped_df

if __name__ == "__main__":
    input_file = "./data/weather/raw/2017_to_2023.csv"  # Change this to your input file path
    output_file = "./data/weather/processing/1_rounded_2017_to_2023.csv"  # Change this to your desired output file path
    
    processed_data = process_weather_data(input_file, output_file)
    
    # Display the first few rows of processed data
    print("\nSample of processed data:")
    print(processed_data.head())