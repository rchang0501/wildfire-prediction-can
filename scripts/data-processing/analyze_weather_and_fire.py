import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

def analyze_wildfire_weather_data(input_file, output_dir):
    """
    Performs comprehensive statistical analysis on merged wildfire-weather data,
    addressing missing values and preparing insights for ML tasks.
    
    Args:
        input_file (str): Path to the merged wildfire-weather CSV file
        output_dir (str): Directory to save analysis outputs and processed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("WILDFIRE-WEATHER DATA ANALYSIS")
    print("="*80)
    
    # Load the data
    print("\nLoading data from", input_file)
    df = pd.read_csv(input_file)
    
    # Convert date to datetime for time-based analysis
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Map months to seasons (fix for duplicate label error)
    season_map = {
        1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall',
        12: 'Winter'
    }
    df['Season'] = df['Month'].map(season_map)
    
    # Basic dataset information
    print("\n1. BASIC DATASET INFORMATION")
    print("-"*50)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Save the dataframe info to a text file
    with open(os.path.join(output_dir, 'data_info.txt'), 'w') as f:
        df.info(buf=f)
    print(f"Column information saved to {os.path.join(output_dir, 'data_info.txt')}")
    
    # 2. MISSING VALUE ANALYSIS
    print("\n2. MISSING VALUE ANALYSIS")
    print("-"*50)
    
    # Calculate missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_data = pd.concat([missing, missing_percent], axis=1, keys=['Total', 'Percent'])
    
    # Filter to only show columns with missing values
    missing_data = missing_data[missing_data['Total'] > 0]
    print(f"Found {len(missing_data)} columns with missing values")
    
    # Save missing data analysis
    missing_data.to_csv(os.path.join(output_dir, 'missing_values_analysis.csv'))
    print(f"Missing values analysis saved to {os.path.join(output_dir, 'missing_values_analysis.csv')}")
    
    # Visualize missing data for top 20 columns with missing values
    if len(missing_data) > 0:
        plt.figure(figsize=(10, 8))
        missing_data_top20 = missing_data.head(20)
        sns.barplot(x=missing_data_top20['Percent'], y=missing_data_top20.index)
        plt.title('Percentage of Missing Values by Column (Top 20)')
        plt.xlabel('Missing Value Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values_plot.png'))
        print(f"Missing values visualization saved to {os.path.join(output_dir, 'missing_values_plot.png')}")
    
    # 3. IDENTIFY KEY COLUMNS FOR ANALYSIS
    print("\n3. IDENTIFYING KEY COLUMNS FOR ANALYSIS")
    print("-"*50)
    
    # A. Identify columns with high coverage (low missing values)
    coverage_threshold = 70  # Only consider columns with at least 70% data
    high_coverage_cols = missing_percent[missing_percent < (100 - coverage_threshold)].index.tolist()
    print(f"Found {len(high_coverage_cols)} columns with at least {coverage_threshold}% data coverage")
    
    # B. Identify numeric columns (potential features for ML)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric columns (potential ML features)")
    
    # C. Key columns intersection (high coverage numeric columns)
    key_numeric_cols = list(set(high_coverage_cols).intersection(set(numeric_cols)))
    key_numeric_cols = [col for col in key_numeric_cols if col not in ['Longitude', 'Latitude', 'Year', 'Month']]
    print(f"Identified {len(key_numeric_cols)} key numeric columns with good coverage for ML tasks")
    
    # Save key columns list
    with open(os.path.join(output_dir, 'key_columns.txt'), 'w') as f:
        f.write("KEY NUMERIC COLUMNS FOR ML:\n")
        f.write('\n'.join(key_numeric_cols))
    print(f"Key columns list saved to {os.path.join(output_dir, 'key_columns.txt')}")
    
    # 4. STATISTICAL SUMMARY
    print("\n4. STATISTICAL SUMMARY")
    print("-"*50)
    
    # Generate comprehensive statistical summary for key numeric columns
    stats_summary = df[key_numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    
    # Add additional statistics
    stats_summary['skew'] = df[key_numeric_cols].skew()
    stats_summary['kurtosis'] = df[key_numeric_cols].kurtosis()
    stats_summary['missing_percent'] = df[key_numeric_cols].isnull().mean() * 100
    stats_summary['unique_values'] = df[key_numeric_cols].nunique()
    stats_summary['zeros_percent'] = (df[key_numeric_cols] == 0).mean() * 100
    
    # Save statistical summary
    stats_summary.to_csv(os.path.join(output_dir, 'statistical_summary.csv'))
    print(f"Statistical summary saved to {os.path.join(output_dir, 'statistical_summary.csv')}")
    
    # 5. TARGET VARIABLE ANALYSIS (FIRE_SIZE_HA)
    print("\n5. TARGET VARIABLE ANALYSIS (FIRE_SIZE_HA)")
    print("-"*50)
    
    if 'FIRE_SIZE_HA' in df.columns:
        # Basic statistics
        print("Fire size statistics:")
        print(df['FIRE_SIZE_HA'].describe())
        
        # Plot fire size distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df['FIRE_SIZE_HA'], bins=50, kde=True)
        plt.title('Fire Size Distribution')
        plt.xlabel('Fire Size (hectares)')
        
        # Log-transformed distribution for skewed data
        plt.subplot(1, 2, 2)
        sns.histplot(np.log1p(df['FIRE_SIZE_HA']), bins=50, kde=True)
        plt.title('Log-Transformed Fire Size Distribution')
        plt.xlabel('Log(Fire Size + 1)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fire_size_distribution.png'))
        print(f"Fire size distribution plots saved to {os.path.join(output_dir, 'fire_size_distribution.png')}")
        
        # Fire size by season
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Season', y='FIRE_SIZE_HA', data=df)
        plt.title('Fire Size by Season')
        plt.ylabel('Fire Size (hectares)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fire_size_by_season.png'))
        print(f"Fire size by season plot saved to {os.path.join(output_dir, 'fire_size_by_season.png')}")
        
        # Fire size by year
        plt.figure(figsize=(12, 6))
        yearly_fire_size = df.groupby('Year')['FIRE_SIZE_HA'].agg(['mean', 'median', 'sum', 'count'])
        yearly_fire_size.plot(kind='bar', y='sum', figsize=(12, 6))
        plt.title('Total Fire Size by Year')
        plt.ylabel('Total Fire Size (hectares)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fire_size_by_year.png'))
        print(f"Fire size by year plot saved to {os.path.join(output_dir, 'fire_size_by_year.png')}")
        
        # Save yearly fire statistics
        yearly_fire_size.to_csv(os.path.join(output_dir, 'yearly_fire_statistics.csv'))
        print(f"Yearly fire statistics saved to {os.path.join(output_dir, 'yearly_fire_statistics.csv')}")
    else:
        print("Warning: FIRE_SIZE_HA column not found in the dataset")
    
    # 6. CORRELATION ANALYSIS
    print("\n6. CORRELATION ANALYSIS")
    print("-"*50)
    
    # Check if we have enough key columns for correlation analysis
    if len(key_numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[key_numeric_cols].corr(method='spearman').round(2)
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        print(f"Correlation matrix saved to {os.path.join(output_dir, 'correlation_matrix.csv')}")
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        print(f"Correlation heatmap saved to {os.path.join(output_dir, 'correlation_heatmap.png')}")
        
        # If FIRE_SIZE_HA is in the dataset, show top correlated features
        if 'FIRE_SIZE_HA' in key_numeric_cols:
            fire_corr = corr_matrix['FIRE_SIZE_HA'].sort_values(ascending=False)
            fire_corr = fire_corr[fire_corr.index != 'FIRE_SIZE_HA']  # Remove self-correlation
            fire_corr_top = fire_corr.head(10)
            fire_corr_bottom = fire_corr.tail(10)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=fire_corr_top.values, y=fire_corr_top.index, palette='Reds_r')
            plt.title('Top 10 Features Positively Correlated with Fire Size')
            plt.xlabel('Correlation Coefficient (Spearman)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_positive_correlations.png'))
            print(f"Top positive correlations plot saved to {os.path.join(output_dir, 'top_positive_correlations.png')}")
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=fire_corr_bottom.values, y=fire_corr_bottom.index, palette='Blues_r')
            plt.title('Top 10 Features Negatively Correlated with Fire Size')
            plt.xlabel('Correlation Coefficient (Spearman)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_negative_correlations.png'))
            print(f"Top negative correlations plot saved to {os.path.join(output_dir, 'top_negative_correlations.png')}")
    else:
        print("Not enough key numeric columns for correlation analysis")
    
    # 7. HANDLING MISSING VALUES
    print("\n7. HANDLING MISSING VALUES")
    print("-"*50)
    
    # Create different versions of cleaned datasets
    
    # A. Dataset with only key columns
    df_key = df[['Date', 'Longitude', 'Latitude', 'Year', 'Month', 'Season'] + key_numeric_cols]
    df_key.to_csv(os.path.join(output_dir, 'wildfire_weather_key_columns.csv'), index=False)
    print(f"Dataset with only key columns saved to {os.path.join(output_dir, 'wildfire_weather_key_columns.csv')}")
    
    # B. Simple imputation - median for numeric columns
    df_median_imputed = df_key.copy()
    for col in key_numeric_cols:
        df_median_imputed[col] = df_median_imputed[col].fillna(df_median_imputed[col].median())
    
    df_median_imputed.to_csv(os.path.join(output_dir, 'wildfire_weather_median_imputed.csv'), index=False)
    print(f"Dataset with median imputation saved to {os.path.join(output_dir, 'wildfire_weather_median_imputed.csv')}")
    
    # C. Advanced imputation - KNN imputer for numeric columns
    if len(df_key) > 2:  # Only if we have enough samples
        try:
            # Prepare data for KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            # Only impute key numeric columns
            numeric_data = df_key[key_numeric_cols].copy()
            imputed_data = imputer.fit_transform(numeric_data)
            
            # Create new dataframe with imputed values
            df_knn_imputed = df_key.copy()
            df_knn_imputed[key_numeric_cols] = imputed_data
            
            df_knn_imputed.to_csv(os.path.join(output_dir, 'wildfire_weather_knn_imputed.csv'), index=False)
            print(f"Dataset with KNN imputation saved to {os.path.join(output_dir, 'wildfire_weather_knn_imputed.csv')}")
        except Exception as e:
            print(f"KNN imputation failed: {e}")
    else:
        print("Not enough samples for KNN imputation")
    
    # D. Advanced imputation - Iterative imputer (uses ML to predict missing values)
    if len(df_key) > 5:  # Only if we have enough samples
        try:
            # Prepare data for iterative imputation
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10),
                max_iter=10,
                random_state=42
            )
            # Only impute key numeric columns
            numeric_data = df_key[key_numeric_cols].copy()
            imputed_data = imputer.fit_transform(numeric_data)
            
            # Create new dataframe with imputed values
            df_iterative_imputed = df_key.copy()
            df_iterative_imputed[key_numeric_cols] = imputed_data
            
            df_iterative_imputed.to_csv(os.path.join(output_dir, 'wildfire_weather_iterative_imputed.csv'), index=False)
            print(f"Dataset with iterative imputation saved to {os.path.join(output_dir, 'wildfire_weather_iterative_imputed.csv')}")
        except Exception as e:
            print(f"Iterative imputation failed: {e}")
    else:
        print("Not enough samples for iterative imputation")
    
    # 8. GEOGRAPHICAL ANALYSIS
    print("\n8. GEOGRAPHICAL ANALYSIS")
    print("-"*50)
    
    # Analyze fire occurrences by geographical location
    location_counts = df.groupby(['Longitude', 'Latitude']).size().reset_index(name='count')
    location_counts = location_counts.sort_values('count', ascending=False)
    
    print("Top 5 locations with most fire occurrences:")
    print(location_counts.head(5))
    
    # Save location analysis
    location_counts.to_csv(os.path.join(output_dir, 'fire_locations_frequency.csv'), index=False)
    print(f"Location frequency analysis saved to {os.path.join(output_dir, 'fire_locations_frequency.csv')}")
    
    # 9. TEMPORAL ANALYSIS
    print("\n9. TEMPORAL ANALYSIS")
    print("-"*50)
    
    # Analyze fire occurrences by month
    monthly_counts = df.groupby('Month').size().reset_index(name='count')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='count', data=monthly_counts)
    plt.title('Fire Occurrences by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Fires')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fires_by_month.png'))
    print(f"Fires by month plot saved to {os.path.join(output_dir, 'fires_by_month.png')}")
    
    # Analyze fire occurrences by season
    seasonal_counts = df.groupby('Season').size().reset_index(name='count')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Season', y='count', data=seasonal_counts)
    plt.title('Fire Occurrences by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Fires')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fires_by_season.png'))
    print(f"Fires by season plot saved to {os.path.join(output_dir, 'fires_by_season.png')}")
    
    # 10. SUMMARY AND RECOMMENDATIONS
    print("\n10. SUMMARY AND RECOMMENDATIONS")
    print("-"*50)
    
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("WILDFIRE-WEATHER DATA ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. Dataset Overview:\n")
        f.write(f"   - Total records: {df.shape[0]}\n")
        f.write(f"   - Total features: {df.shape[1]}\n")
        f.write(f"   - Key numeric features: {len(key_numeric_cols)}\n\n")
        
        f.write("2. Missing Data Summary:\n")
        f.write(f"   - Columns with missing values: {len(missing_data)}\n")
        if len(missing_data) > 0:
            f.write(f"   - Top 5 columns with most missing values:\n")
            for idx, (col, row) in enumerate(missing_data.head(5).iterrows()):
                f.write(f"     {idx+1}. {col}: {row['Percent']:.2f}%\n")
        f.write("\n")
        
        f.write("3. Recommendations for Machine Learning:\n")
        f.write("   - Consider using the iterative imputation dataset for best results with missing values\n")
        f.write("   - Alternative approach: Use only columns with high data coverage\n")
        
        if 'FIRE_SIZE_HA' in df.columns:
            f.write("   - Target variable (FIRE_SIZE_HA) is highly skewed - consider log transformation\n")
            
            if len(key_numeric_cols) > 1 and 'FIRE_SIZE_HA' in key_numeric_cols:
                f.write("   - Key predictive features based on correlation:\n")
                for idx, (col, val) in enumerate(fire_corr_top.head(5).items()):
                    f.write(f"     {idx+1}. {col}: {val:.3f}\n")
        
        f.write("\n4. Feature Engineering Ideas:\n")
        f.write("   - Create temperature differential features (day/night, max/min)\n")
        f.write("   - Calculate drought indices from precipitation data\n")
        f.write("   - Combine wind speed and direction into vector components\n")
        f.write("   - Extract features like 'days since last precipitation'\n")
        f.write("   - Consider adding external data sources (vegetation indices, fuel moisture)\n")
    
    print(f"Analysis summary and recommendations saved to {os.path.join(output_dir, 'analysis_summary.txt')}")
    print("\nAnalysis complete! All outputs saved to", output_dir)
    
    return df_key, df_median_imputed

if __name__ == "__main__":
    input_file = "./data/wildfire_weather_interpolated_merged.csv"  # Path to your merged data file
    output_dir = "./data/analysis/"                    # Directory to save analysis outputs
    
    df_key, df_imputed = analyze_wildfire_weather_data(input_file, output_dir)
    
    print("\nAnalysis script execution completed successfully!")


"""
1. Basic Dataset Information

Provides shape, data types, and basic structure
Adds useful time features (Year, Month, Season)

2. Missing Value Analysis

Identifies all columns with missing values and their percentages
Creates visualizations of missing data patterns
Generates a detailed report of missing values

3. Key Column Identification

Identifies columns with good data coverage (low missing values)
Selects numeric columns suitable for machine learning
Creates a list of high-quality feature candidates

4. Statistical Summary

Generates comprehensive statistics for all key columns
Includes percentiles, skewness, kurtosis, and unique value counts
Identifies columns with high zero counts

5. Target Variable Analysis (FIRE_SIZE_HA)

Analyzes the distribution of fire sizes
Shows fire size variations by season and year
Creates visualizations for better understanding

6. Correlation Analysis

Calculates correlation matrix using Spearman method (robust to outliers)
Identifies features most correlated with fire size
Creates visualizations of the most important relationships

7. Missing Value Handling

Creates multiple datasets with different imputation strategies:

Dataset with only key columns
Simple median imputation
KNN-based imputation (using similar data points)
Advanced iterative imputation using RandomForest



8. Geographical Analysis

Identifies hotspots where fires occur most frequently
Creates a geographical frequency distribution

9. Temporal Analysis

Analyzes fire patterns by month and season
Creates visualizations of temporal trends

10. Summary and Recommendations

Provides a comprehensive summary of findings
Suggests feature engineering ideas
Recommends best approaches for machine learning
"""