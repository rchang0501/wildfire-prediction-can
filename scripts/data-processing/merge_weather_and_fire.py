import pandas as pd

def merge_wildfire_weather_data(wildfire_file, weather_file, output_file):
    """
    Merges processed wildfire and weather data based on matching date and coordinates.
    
    Args:
        wildfire_file (str): Path to the processed wildfire CSV file
        weather_file (str): Path to the processed weather CSV file
        output_file (str): Path to save the merged CSV file
    """
    # Read the processed data files
    wildfire_df = pd.read_csv(wildfire_file)
    weather_df = pd.read_csv(weather_file)
    
    # Ensure date columns are in the same format
    # Convert IGNITION_DATE to match the Date format in weather data
    wildfire_df['Date'] = pd.to_datetime(wildfire_df['IGNITION_DATE']).dt.strftime('%Y-%m-%d')
    wildfire_df = wildfire_df.drop('IGNITION_DATE', axis=1)
    
    # Rename columns to ensure consistency for merging
    wildfire_df = wildfire_df.rename(columns={
        'LATITUDE': 'Latitude',
        'LONGITUDE': 'Longitude'
    })
    
    # Merge the dataframes on Date, Longitude, and Latitude
    merged_df = pd.merge(
        weather_df, 
        wildfire_df,
        on=['Date', 'Longitude', 'Latitude'],
        how='inner'  # Only keep rows that match in both datasets
    )
    
    # Save the merged data
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged data saved to {output_file}")
    print(f"Wildfire data row count: {len(wildfire_df)}")
    print(f"Weather data row count: {len(weather_df)}")
    print(f"Merged data row count: {len(merged_df)}")
    print(f"Number of matching events: {len(merged_df)}")
    
    return merged_df

if __name__ == "__main__":
    wildfire_file = "./data/wildfire/processing/1_round_coordinates.csv"  # Path to your processed wildfire data
    weather_file = "./data/weather/processing/interpolation_steps/5_final_interpolated.csv"    # Path to your processed weather data
    output_file = "./data/wildfire_weather_interpolated_merged.csv"    # Output file path
    
    merged_data = merge_wildfire_weather_data(wildfire_file, weather_file, output_file)
    
    # Display the first few rows and columns of merged data
    print("\nSample of merged data (first 5 rows, selected columns):")
    columns_to_show = ['Date', 'Longitude', 'Latitude', 'FIRE_SIZE_HA']
    # Add a few weather columns if they exist
    for col in ['AirTemp', 'HUMIDITY', 'WIND_SPEED', 'PRECIPITATION_NEW']:
        if col in merged_data.columns:
            columns_to_show.append(col)
    
    sample_cols = [col for col in columns_to_show if col in merged_data.columns]
    print(merged_data[sample_cols].head())