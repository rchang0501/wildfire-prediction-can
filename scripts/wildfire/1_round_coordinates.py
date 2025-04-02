import pandas as pd
import numpy as np

def process_wildfire_data(input_file, output_file):
    """
    Processes wildfire data by:
    1. Rounding latitude and longitude to the nearest tenth
    2. Aggregating fire sizes for entries with the same date and rounded coordinates
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the processed CSV file
    """
    # Read the data
    df = pd.read_csv(input_file)
    
    # Round latitude and longitude to the nearest tenth
    df['LATITUDE'] = np.round(df['LATITUDE'], 1)
    df['LONGITUDE'] = np.round(df['LONGITUDE'], 1)
    
    # Group by date and rounded coordinates, then sum the fire sizes
    grouped_df = df.groupby(['IGNITION_DATE', 'LATITUDE', 'LONGITUDE'], as_index=False).agg({
        'FIRE_SIZE_HA': 'sum'
    })
    
    # Save the processed data
    grouped_df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
    print(f"Original row count: {len(df)}")
    print(f"After processing row count: {len(grouped_df)}")
    
    return grouped_df

if __name__ == "__main__":
    input_file = "./data/wildfire/raw/2017_to_2022.csv"  # Change this to your input file path
    output_file = "./data/wildfire/processing/1_round_coordinates.csv"  # Change this to your desired output file path
    
    processed_data = process_wildfire_data(input_file, output_file)
    
    # Display the first few rows of processed data
    print("\nSample of processed data:")
    print(processed_data.head())