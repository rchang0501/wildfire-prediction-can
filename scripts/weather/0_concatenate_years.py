import os
import pandas as pd
import glob

def concatenate_csvs(input_folder, output_file):
    """
    Concatenate all CSV files in the input folder into a single CSV file.
    
    Parameters:
    input_folder (str): Path to the folder containing CSV files
    output_file (str): Path to save the concatenated CSV file
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Get a list of all CSV files in the folder
        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {input_folder}")
            return False
        
        print(f"Found {len(csv_files)} CSV files to concatenate.")
        
        # Create an empty list to store individual dataframes
        dfs = []
        
        # Read each CSV file and append to the list
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Read {file} with {len(df)} rows")
        
        # Concatenate all dataframes into one
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save the combined dataframe to a new CSV file
        combined_df.to_csv(output_file, index=False)

        combined_df = combined_df.drop(['Network Name', 'Native ID'])
        
        print(f"Successfully concatenated {len(csv_files)} files into {output_file}")
        print(f"Total rows in combined file: {len(combined_df)}")
        
        return True
    
    except Exception as e:
        print(f"Error concatenating CSV files: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Replace these with your actual folder and output file paths
    input_folder = "./data/weather/raw"
    output_file = "./data/weather/raw/2017_to_2023.csv"
    
    concatenate_csvs(input_folder, output_file)