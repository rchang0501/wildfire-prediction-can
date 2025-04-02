import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
import time
import os
from datetime import datetime
warnings.filterwarnings('ignore')

def interpolate_weather_data(input_file, output_folder):
    """
    Interpolate missing values in a weather dataset using multiple strategies.
    Optimized for large datasets with intermediate file outputs and detailed debugging.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_folder : str
        Folder to save the intermediate and final interpolated data files
    """
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_folder, f"interpolation_log_{timestamp}.txt")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a log function to write to both console and log file
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    # Start timing
    start_time = time.time()
    log_message(f"[{timestamp}] Starting interpolation process")
    log_message(f"Reading data from {input_file}...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        log_message(f"Successfully read the CSV file with shape: {df.shape}")
    except Exception as e:
        log_message(f"ERROR: Failed to read the CSV file: {str(e)}")
        return None
    
    # Convert Date to datetime for proper time-based interpolation
    try:
        log_message("Converting 'Date' column to datetime...")
        df['Date'] = pd.to_datetime(df['Date'])
        log_message("Date conversion successful")
    except Exception as e:
        log_message(f"ERROR: Failed to convert 'Date' column: {str(e)}")
        return None
    
    # Basic information about the dataset
    log_message(f"Dataset shape before interpolation: {df.shape}")
    missing_values = df.isna().sum().sum()
    log_message(f"Missing values before interpolation: {missing_values}")
    
    # Save memory statistics
    import psutil
    memory_info = psutil.Process().memory_info()
    log_message(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    # Identify coordinate columns that should not be interpolated
    coord_columns = ['Date', 'Longitude', 'Latitude']
    
    # Get columns to interpolate (all except coordinate columns)
    columns_to_interpolate = [col for col in df.columns if col not in coord_columns]
    log_message(f"Number of columns to interpolate: {len(columns_to_interpolate)}")
    
    # Output list of columns with missing values
    columns_with_missing = [col for col in columns_to_interpolate if df[col].isna().any()]
    log_message(f"Columns with missing values: {len(columns_with_missing)}")
    log_message(f"Top 10 columns with most missing values: {df[columns_with_missing].isna().sum().nlargest(10).to_dict()}")
    
    # Sort by date to ensure time-based interpolation works correctly
    log_message("Sorting data by date...")
    df = df.sort_values(by='Date')
    log_message("Sorting complete")
    
    # Save the original dataset
    original_output_file = os.path.join(output_folder, "0_original.csv")
    log_message(f"Saving original dataset to {original_output_file}")
    df.to_csv(original_output_file, index=False)
    
    # Create a copy of the original dataframe to track changes
    null_count_original = df[columns_to_interpolate].isna().sum().sum()
    
    # Step 1: Linear interpolation for time series data
    log_message("\n" + "="*80)
    log_message("STEP 1: Performing time-based linear interpolation...")
    time_start = time.time()
    
    # Process in chunks to reduce memory usage
    chunk_size = 10  # Process 10 columns at a time
    for i in range(0, len(columns_to_interpolate), chunk_size):
        chunk_cols = columns_to_interpolate[i:i+chunk_size]
        log_message(f"  Processing columns {i+1}-{min(i+chunk_size, len(columns_to_interpolate))} of {len(columns_to_interpolate)}")
        
        for col in chunk_cols:
            # Count missing values before
            missing_before = df[col].isna().sum()
            
            # Only interpolate if the column has some non-null values and has missing values
            if df[col].notna().sum() > 0 and missing_before > 0:
                log_message(f"    Interpolating column: {col} (missing: {missing_before})")
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                
                # Count missing values after
                missing_after = df[col].isna().sum()
                log_message(f"    Column {col}: Filled {missing_before - missing_after} values, {missing_after} still missing")
            else:
                if missing_before == 0:
                    log_message(f"    Skipping column {col}: No missing values")
                else:
                    log_message(f"    Skipping column {col}: No non-null values available for interpolation")
        
        if (i + chunk_size) % 30 == 0 or i + chunk_size >= len(columns_to_interpolate):
            log_message(f"  Progress: {min(i + chunk_size, len(columns_to_interpolate))}/{len(columns_to_interpolate)} columns...")
            # Update memory usage
            memory_info = psutil.Process().memory_info()
            log_message(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    time_end = time.time()
    log_message(f"Time-based interpolation completed in {time_end - time_start:.2f} seconds")
    
    # Save data after time-based interpolation
    temporal_output_file = os.path.join(output_folder, "1_after_temporal.csv")
    log_message(f"Saving dataset after time-based interpolation to {temporal_output_file}")
    df.to_csv(temporal_output_file, index=False)
    
    # Track progress after temporal interpolation
    null_count_after_temporal = df[columns_to_interpolate].isna().sum().sum()
    filled_temporal = null_count_original - null_count_after_temporal
    log_message(f"Missing values filled by temporal interpolation: {filled_temporal} ({filled_temporal/null_count_original*100:.2f}%)")
    
    # Step 2: For values still missing, use optimized spatial interpolation
    log_message("\n" + "="*80)
    log_message("STEP 2: Performing spatial interpolation...")
    time_start = time.time()
    
    # Instead of processing each date individually, we'll use a more efficient approach
    # Group data by date and process dates with enough data points
    log_message("Counting data points per date...")
    date_counts = df['Date'].value_counts()
    dates_with_enough_points = date_counts[date_counts >= 3].index
    log_message(f"Found {len(dates_with_enough_points)} dates with at least 3 data points")
    
    # Track progress
    date_count = len(dates_with_enough_points)
    progress_step = max(1, date_count // 20)  # Show progress every 5%
    
    # Track columns with spatial interpolation
    columns_with_spatial = set()
    
    # Only process dates with enough data
    for i, date in enumerate(dates_with_enough_points):
        if i % progress_step == 0:
            log_message(f"  Processing date {i+1}/{date_count} ({i/date_count*100:.1f}%)...")
            # Update memory usage
            memory_info = psutil.Process().memory_info()
            log_message(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        
        date_df = df[df['Date'] == date]
        
        # Extract coordinates for this date
        coords = date_df[['Longitude', 'Latitude']].values
        
        # Process multiple columns at once for efficiency
        for col in columns_to_interpolate:
            # Skip if not enough non-null values for this column on this date
            col_notna_count = date_df[col].notna().sum()
            col_na_count = date_df[col].isna().sum()
            
            if col_notna_count < 2 or col_na_count == 0:
                continue
            
            # Get values for this column
            values = date_df[col].values.reshape(-1, 1)
            
            # Only proceed if there are missing values to fill
            if pd.isna(values).any():
                try:
                    # Use KNN imputation
                    n_neighbors = min(3, col_notna_count)
                    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                    imputed_values = imputer.fit_transform(values)
                    
                    # Update original dataframe
                    df.loc[date_df.index, col] = imputed_values
                    
                    # Track which columns had spatial interpolation
                    columns_with_spatial.add(col)
                    
                except Exception as e:
                    if i < 3:  # Only log the first few errors to avoid log file bloat
                        log_message(f"    Error imputing {col} for date {date}: {str(e)}")
    
    time_end = time.time()
    log_message(f"Spatial interpolation completed in {time_end - time_start:.2f} seconds")
    log_message(f"Columns processed with spatial interpolation: {len(columns_with_spatial)}")
    if columns_with_spatial:
        log_message(f"Sample columns: {list(columns_with_spatial)[:5]}")
    
    # Save data after spatial interpolation
    spatial_output_file = os.path.join(output_folder, "2_after_spatial.csv")
    log_message(f"Saving dataset after spatial interpolation to {spatial_output_file}")
    df.to_csv(spatial_output_file, index=False)
    
    # Track progress after spatial interpolation
    null_count_after_spatial = df[columns_to_interpolate].isna().sum().sum()
    filled_spatial = null_count_after_temporal - null_count_after_spatial
    log_message(f"Missing values filled by spatial interpolation: {filled_spatial} ({filled_spatial/null_count_original*100:.2f}%)")
    
    # Step 3: Vectorized correlation-based imputation
    log_message("\n" + "="*80)
    log_message("STEP 3: Performing optimized correlation-based imputation...")
    time_start = time.time()
    
    # Define groups of related weather variables
    related_columns = {
        'temperature': [col for col in columns_to_interpolate if any(term in col.upper() for term in ['TEMP', 'AIR_TEMP'])],
        'wind': [col for col in columns_to_interpolate if any(term in col.upper() for term in ['WIND', 'WSPD', 'WDIR'])],
        'precipitation': [col for col in columns_to_interpolate if any(term in col.upper() for term in 
                                                               ['PRECIPITATION', 'RAIN', 'SNOW', 'PCP', 'PCPN'])],
        'pressure': [col for col in columns_to_interpolate if any(term in col.upper() for term in ['PRESS', 'BAR', 'MSLP'])],
        'humidity': [col for col in columns_to_interpolate if any(term in col.upper() for term in ['HUMIDITY', 'RH'])]
    }
    
    # Log information about the groups
    for group, cols in related_columns.items():
        log_message(f"  {group} group has {len(cols)} variables")
        if cols:
            log_message(f"    Sample variables: {cols[:5]}")
    
    # Track columns processed
    columns_with_correlation = set()
    
    # Process each group of related variables using vectorized operations
    for group_name, group_cols in related_columns.items():
        log_message(f"  Processing {group_name} variables...")
        
        # Only keep columns that exist in the dataframe
        group_cols = [col for col in group_cols if col in df.columns]
        
        if len(group_cols) <= 1:
            log_message(f"    Skipping {group_name} group: Not enough columns for correlation")
            continue
        
        log_message(f"    Working with {len(group_cols)} columns in {group_name} group")
        
        # Create a sub-dataframe with just these columns
        sub_df = df[group_cols].copy()
        
        # Calculate column means for scaling
        col_means = sub_df.mean()
        log_message(f"    Calculated means for scaling")
        
        # Process each column
        for col in group_cols:
            # Skip if no missing values
            missing_count = pd.isna(df[col]).sum()
            if missing_count == 0:
                log_message(f"    Skipping {col}: No missing values")
                continue
                
            log_message(f"    Processing {col} ({missing_count} missing values)")
                
            # Find rows with missing values in this column
            missing_mask = pd.isna(df[col])
            
            # For each row with missing values
            missing_indices = df[missing_mask].index
            log_message(f"      Found {len(missing_indices)} rows with missing values")
            
            # Process in chunks to avoid memory issues
            chunk_size = 10000
            chunks_total = (len(missing_indices) - 1) // chunk_size + 1
            
            values_filled = 0
            
            for chunk_idx in range(chunks_total):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, len(missing_indices))
                chunk_indices = missing_indices[chunk_start:chunk_end]
                
                if chunk_idx % max(1, chunks_total // 5) == 0 or chunk_idx == chunks_total - 1:
                    log_message(f"      Processing chunk {chunk_idx+1}/{chunks_total} ({len(chunk_indices)} rows)")
                
                # For each row in the chunk, use related columns to estimate the value
                chunk_filled = 0
                
                for idx in chunk_indices:
                    # Find which related columns have values for this row
                    row_data = df.loc[idx, group_cols]
                    related_with_values = [c for c in group_cols if c != col and not pd.isna(row_data[c])]
                    
                    if related_with_values:
                        # Calculate the value based on related columns
                        est_values = []
                        for rel_col in related_with_values:
                            # Only use if means are available
                            if not pd.isna(col_means[col]) and not pd.isna(col_means[rel_col]) and col_means[rel_col] != 0:
                                scale_factor = col_means[col] / col_means[rel_col]
                                est_values.append(df.loc[idx, rel_col] * scale_factor)
                        
                        if est_values:
                            df.loc[idx, col] = np.mean(est_values)
                            chunk_filled += 1
                
                values_filled += chunk_filled
                
                if chunk_idx % max(1, chunks_total // 5) == 0 or chunk_idx == chunks_total - 1:
                    log_message(f"      Filled {chunk_filled} values in this chunk")
            
            log_message(f"    Total values filled for {col}: {values_filled}")
            if values_filled > 0:
                columns_with_correlation.add(col)
    
    time_end = time.time()
    log_message(f"Correlation-based imputation completed in {time_end - time_start:.2f} seconds")
    log_message(f"Columns processed with correlation-based imputation: {len(columns_with_correlation)}")
    if columns_with_correlation:
        log_message(f"Sample columns: {list(columns_with_correlation)[:5]}")
    
    # Save data after correlation-based imputation
    correlation_output_file = os.path.join(output_folder, "3_after_correlation.csv")
    log_message(f"Saving dataset after correlation-based imputation to {correlation_output_file}")
    df.to_csv(correlation_output_file, index=False)
    
    # Track progress after correlation imputation
    null_count_after_correlation = df[columns_to_interpolate].isna().sum().sum()
    filled_correlation = null_count_after_spatial - null_count_after_correlation
    log_message(f"Missing values filled by correlation-based imputation: {filled_correlation} ({filled_correlation/null_count_original*100:.2f}%)")
    
    # Step 4: Forward and backward fill for any remaining missing values
    log_message("\n" + "="*80)
    log_message("STEP 4: Performing forward/backward fill for remaining missing values...")
    time_start = time.time()
    
    # Process in chunks to reduce memory usage
    chunk_size = 10
    for i in range(0, len(columns_to_interpolate), chunk_size):
        chunk_cols = columns_to_interpolate[i:i+chunk_size]
        log_message(f"  Processing columns {i+1}-{min(i+chunk_size, len(columns_to_interpolate))} of {len(columns_to_interpolate)}")
        
        for col in chunk_cols:
            # Count missing values before
            missing_before = df[col].isna().sum()
            if missing_before == 0:
                log_message(f"    Skipping {col}: No missing values")
                continue
                
            log_message(f"    Filling {col} ({missing_before} missing values)")
            
            # Apply forward fill
            df[col] = df[col].fillna(method='ffill')
            
            # Count missing after forward fill
            missing_after_ffill = df[col].isna().sum()
            filled_ffill = missing_before - missing_after_ffill
            if filled_ffill > 0:
                log_message(f"      Forward fill: {filled_ffill} values filled")
            
            # Apply backward fill
            df[col] = df[col].fillna(method='bfill')
            
            # Count missing after backward fill
            missing_after_bfill = df[col].isna().sum()
            filled_bfill = missing_after_ffill - missing_after_bfill
            if filled_bfill > 0:
                log_message(f"      Backward fill: {filled_bfill} values filled")
            
            # Total filled
            total_filled = filled_ffill + filled_bfill
            log_message(f"      Total filled: {total_filled} values, {missing_after_bfill} still missing")
        
        if (i + chunk_size) % 30 == 0 or i + chunk_size >= len(columns_to_interpolate):
            log_message(f"  Processed {min(i + chunk_size, len(columns_to_interpolate))}/{len(columns_to_interpolate)} columns...")
            # Update memory usage
            memory_info = psutil.Process().memory_info()
            log_message(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    time_end = time.time()
    log_message(f"Forward/backward fill completed in {time_end - time_start:.2f} seconds")
    
    # Save data after forward/backward fill
    fillna_output_file = os.path.join(output_folder, "4_after_fillna.csv")
    log_message(f"Saving dataset after forward/backward fill to {fillna_output_file}")
    df.to_csv(fillna_output_file, index=False)
    
    # Track progress after forward/backward fill
    null_count_after_fillna = df[columns_to_interpolate].isna().sum().sum()
    filled_fillna = null_count_after_correlation - null_count_after_fillna
    log_message(f"Missing values filled by forward/backward fill: {filled_fillna} ({filled_fillna/null_count_original*100:.2f}%)")
    
    # Step 5: Fill remaining missing values with column median
    log_message("\n" + "="*80)
    log_message("STEP 5: Filling remaining gaps with column medians...")
    time_start = time.time()
    
    # Calculate all medians at once
    log_message("Calculating column medians...")
    medians = df[columns_to_interpolate].median()
    
    # Process in chunks
    chunk_size = 10
    for i in range(0, len(columns_to_interpolate), chunk_size):
        chunk_cols = columns_to_interpolate[i:i+chunk_size]
        log_message(f"  Processing columns {i+1}-{min(i+chunk_size, len(columns_to_interpolate))} of {len(columns_to_interpolate)}")
        
        for col in chunk_cols:
            # Count missing values before
            missing_before = df[col].isna().sum()
            if missing_before == 0:
                log_message(f"    Skipping {col}: No missing values")
                continue
                
            if pd.isna(medians[col]):
                log_message(f"    Skipping {col}: No valid median available")
                continue
                
            log_message(f"    Filling {col} with median {medians[col]:.4f} ({missing_before} missing values)")
            df[col] = df[col].fillna(medians[col])
            
            # Count missing after median fill
            missing_after = df[col].isna().sum()
            filled = missing_before - missing_after
            log_message(f"      Filled {filled} values, {missing_after} still missing")
        
        if (i + chunk_size) % 30 == 0 or i + chunk_size >= len(columns_to_interpolate):
            log_message(f"  Processed {min(i + chunk_size, len(columns_to_interpolate))}/{len(columns_to_interpolate)} columns...")
            # Update memory usage
            memory_info = psutil.Process().memory_info()
            log_message(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    time_end = time.time()
    log_message(f"Median fill completed in {time_end - time_start:.2f} seconds")
    
    # Calculate interpolation effectiveness
    null_count_after = df[columns_to_interpolate].isna().sum().sum()
    
    # Save final data
    final_output_file = os.path.join(output_folder, "5_final_interpolated.csv")
    log_message(f"Saving final interpolated dataset to {final_output_file}")
    df.to_csv(final_output_file, index=False)
    
    # Generate summary of still-missing values
    if null_count_after > 0:
        log_message("Columns with remaining missing values:")
        cols_still_missing = [(col, df[col].isna().sum()) for col in columns_to_interpolate if df[col].isna().any()]
        cols_still_missing.sort(key=lambda x: x[1], reverse=True)
        for col, count in cols_still_missing[:20]:  # Show top 20
            log_message(f"  {col}: {count} missing values")
    
    # Calculate overall effectiveness
    if null_count_original > 0:
        temporal_effectiveness = (null_count_original - null_count_after_temporal) / null_count_original * 100
        spatial_effectiveness = (null_count_after_temporal - null_count_after_spatial) / null_count_original * 100
        correlation_effectiveness = (null_count_after_spatial - null_count_after_correlation) / null_count_original * 100
        fillna_effectiveness = (null_count_after_correlation - null_count_after_fillna) / null_count_original * 100
        median_effectiveness = (null_count_after_fillna - null_count_after) / null_count_original * 100
        overall_effectiveness = (null_count_original - null_count_after) / null_count_original * 100
    else:
        temporal_effectiveness = spatial_effectiveness = correlation_effectiveness = fillna_effectiveness = median_effectiveness = overall_effectiveness = 0
    
    log_message(f"\n" + "="*80)
    log_message(f"INTERPOLATION SUMMARY:")
    log_message(f"  Missing values before interpolation: {null_count_original}")
    log_message(f"  Values filled by temporal interpolation: {null_count_original - null_count_after_temporal} ({temporal_effectiveness:.2f}%)")
    log_message(f"  Values filled by spatial interpolation: {null_count_after_temporal - null_count_after_spatial} ({spatial_effectiveness:.2f}%)")
    log_message(f"  Values filled by correlation-based interpolation: {null_count_after_spatial - null_count_after_correlation} ({correlation_effectiveness:.2f}%)")
    log_message(f"  Values filled by forward/backward fill: {null_count_after_correlation - null_count_after_fillna} ({fillna_effectiveness:.2f}%)")
    log_message(f"  Values filled by median imputation: {null_count_after_fillna - null_count_after} ({median_effectiveness:.2f}%)")
    log_message(f"  Total missing values after all interpolation steps: {null_count_after} ({overall_effectiveness:.2f}% filled)")
    
    total_time = time.time() - start_time
    log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace these with your actual folder and output file paths
    input_file = "./data/weather/processing/1_rounded_2017_to_2023.csv"
    output_folder = "./data/weather/processing/interpolation_steps"
    
    interpolate_weather_data(input_file, output_folder)