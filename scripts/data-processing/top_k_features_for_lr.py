import pandas as pd
import os


def filter_and_save_data(input_path, output_path):
    # Features to keep based on the image
    selected_features = [
        "HUMIDITY",
        "Wetness",
        "Pressurembar",
        "cum_pcpn_amt",
        "Rainmm",
        "min_air_temp_snc_last_reset",
        "CURRENT_AIR_TEMPERATURE1",
        "Longitude",
        "SnowWaterEquivalent",
        "TempC",
        "relative_humidity",
        "air_temp",
        "avg_rel_hum_pst1hr",
        "air_temperature_yesterday_high",
        "snwfl_amt_pst1hr",
        "DewPtC",
        "snw_dpth",
        "TEMP_MEAN",
        "SolarRadiation",
        "Date",
        "FIRE_SIZE_HA"
    ]

    # Load the data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Keep only selected columns that exist in the dataset
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = set(selected_features) - set(available_features)

    if missing_features:
        print(
            f"Warning: The following selected features were not found in the dataset and will be skipped:"
        )
        for col in missing_features:
            print(f"  - {col}")

    df_filtered = df[available_features]

    # Save to output CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")


# Example usage
if __name__ == "__main__":
    input_file = "./data/wildfire_weather_interpolated_merged_cleaned.csv"
    output_file = "./data/lr_top_20_wildfire_weather_interpolated_merged_cleaned.csv"
    filter_and_save_data(input_file, output_file)
