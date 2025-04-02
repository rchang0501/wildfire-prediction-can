import pandas as pd

def drop_empty_columns(input_csv, output_csv=None):
    """
    Drops all columns from a CSV file that are completely empty.
    
    Parameters:
    input_csv (str): Path to the input CSV file.
    output_csv (str, optional): Path to save the output CSV file. 
                               If None, returns the DataFrame without saving.
    
    Returns:
    pandas.DataFrame: DataFrame with empty columns removed.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Calculate the number of non-null values in each column
    non_null_counts = df.count()
    
    # Find columns that have zero non-null values (i.e., empty columns)
    empty_columns = non_null_counts[non_null_counts == 0].index.tolist()
    
    # Drop the empty columns
    df_cleaned = df.drop(columns=empty_columns)
    
    # If any columns were dropped, print information
    if empty_columns:
        print(f"Dropped {len(empty_columns)} empty columns: {empty_columns}")
    else:
        print("No empty columns found.")
    
    # Save to a new CSV file if output path is provided
    if output_csv:
        df_cleaned.to_csv(output_csv, index=False)
        print(f"Saved cleaned CSV to {output_csv}")
    
    return df_cleaned

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "./data/wildfire_weather_interpolated_merged.csv"
    output_file = "./data/wildfire_weather_interpolated_merged_cleaned.csv"
    
    drop_empty_columns(input_file, output_file)