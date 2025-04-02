import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_xgboost_model(input_file, output_dir, target_column='FIRE_SIZE_HA'):
    """
    Trains an XGBoost model on wildfire data, implements best practices for model training,
    evaluation, and interpretation.
    
    Args:
        input_file (str): Path to the processed wildfire-weather CSV file (ideally with imputed values)
        output_dir (str): Directory to save model, performance metrics, and visualizations
        target_column (str): Name of the target column to predict (default: 'FIRE_SIZE_HA')
    
    Returns:
        dict: Dictionary containing trained model and performance metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    print("="*80)
    print("WILDFIRE PREDICTION XGBOOST MODEL TRAINING")
    print("="*80)
    
    # 1. LOAD AND PREPARE DATA
    print("\n1. LOADING AND PREPARING DATA")
    print("-"*50)
    
    # Load the dataset
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Convert date to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # Extract additional temporal features if not already present
        if 'Year' not in df.columns:
            df['Year'] = df['Date'].dt.year
        if 'Month' not in df.columns:
            df['Month'] = df['Date'].dt.month
        if 'DayOfYear' not in df.columns:
            df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")
    
    # Log transform the target variable if it's skewed (common for fire size)
    print(f"Analyzing target variable: {target_column}")
    skewness = df[target_column].skew()
    print(f"Target skewness: {skewness:.2f}")
    
    # Apply log transformation if target is positively skewed
    if skewness > 1.0:
        print("Target is positively skewed. Applying log transformation.")
        # Add small constant to handle zeros: log(x + 1)
        df['target_original'] = df[target_column]
        df[target_column] = np.log1p(df[target_column])
        results['target_transformation'] = 'log1p'
    else:
        print("Target distribution is not highly skewed. Using original values.")
        df['target_original'] = df[target_column]
        results['target_transformation'] = 'none'
    
    # 2. FEATURE ENGINEERING AND SELECTION
    print("\n2. FEATURE ENGINEERING AND SELECTION")
    print("-"*50)
    
    # Drop non-feature columns
    exclude_cols = ['Date']
    
    # Identify numeric columns (potential features)
    feature_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in exclude_cols and 
                   col != target_column and col != 'target_original']
    
    print(f"Initial feature count: {len(feature_cols)}")
    
    # Handle missing values in features if any
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            print(f"Warning: Column {col} has {df[col].isnull().sum()} missing values.")
    
    # Check for constant or near-constant features
    near_constant_threshold = 0.99
    near_constant_features = []
    
    for col in feature_cols:
        # Calculate the percentage of the most common value
        most_common_pct = df[col].value_counts(normalize=True).max()
        if most_common_pct > near_constant_threshold:
            near_constant_features.append(col)
    
    if near_constant_features:
        print(f"Removing {len(near_constant_features)} near-constant features:")
        for col in near_constant_features:
            print(f"  - {col}: {df[col].value_counts(normalize=True).max():.2%} most common value")
        
        # Remove these features
        feature_cols = [col for col in feature_cols if col not in near_constant_features]
    
    print(f"Features after removing near-constants: {len(feature_cols)}")
    
    # 3. DATA SPLITTING
    print("\n3. SPLITTING DATA INTO TRAIN/VALIDATION/TEST SETS")
    print("-"*50)
    
    # Get features and target
    X = df[feature_cols]
    y = df[target_column]
    
    # Save feature list for later use
    with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # First split: 80% train+validation, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Second split: 75% train, 25% validation (resulting in 60%/20%/20% split overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_SEED
    )
    
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # 4. CREATE BASELINE MODEL
    print("\n4. TRAINING BASELINE XGBOOST MODEL")
    print("-"*50)
    
    # Basic XGBoost regressor with default hyperparameters
    baseline_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )
    
    # Train the baseline model
    baseline_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Get baseline predictions
    y_train_pred_baseline = baseline_model.predict(X_train)
    y_val_pred_baseline = baseline_model.predict(X_val)
    
    # Calculate baseline metrics
    baseline_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_baseline))
    baseline_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_baseline))
    baseline_train_r2 = r2_score(y_train, y_train_pred_baseline)
    baseline_val_r2 = r2_score(y_val, y_val_pred_baseline)
    
    print(f"Baseline model training RMSE: {baseline_train_rmse:.4f}")
    print(f"Baseline model validation RMSE: {baseline_val_rmse:.4f}")
    print(f"Baseline model training R²: {baseline_train_r2:.4f}")
    print(f"Baseline model validation R²: {baseline_val_r2:.4f}")
    
    # 5. FEATURE IMPORTANCE AND FEATURE SELECTION
    print("\n5. FEATURE IMPORTANCE ANALYSIS")
    print("-"*50)
    
    # Get feature importance from the baseline model
    feature_importance = baseline_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance to CSV
    feature_importance_df.to_csv(os.path.join(output_dir, 'baseline_feature_importance.csv'), index=False)
    print(f"Feature importance saved to {os.path.join(output_dir, 'baseline_feature_importance.csv')}")
    
    # Plot feature importance (top 20)
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Features by Importance (Baseline Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_feature_importance.png'))
    
    # Select the most important features (top 80% of cumulative importance)
    cumulative_importance = feature_importance_df['Importance'].cumsum() / feature_importance_df['Importance'].sum()
    importance_threshold = 0.8
    selected_features_count = (cumulative_importance > importance_threshold).idxmax() + 1
    
    # Ensure we have at least 5 features (to avoid overly simplistic model)
    selected_features_count = max(selected_features_count, 5)
    selected_features = feature_importance_df['Feature'].iloc[:selected_features_count].tolist()
    
    print(f"Selected top {selected_features_count} features (cumulative importance > {importance_threshold})")
    print("Top 5 features:")
    for i, feature in enumerate(selected_features[:5]):
        print(f"  {i+1}. {feature}: {feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0]:.4f}")
    
    # Create reduced feature datasets
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # 6. HYPERPARAMETER TUNING
    print("\n6. HYPERPARAMETER TUNING")
    print("-"*50)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    # Use reduced parameter grid if training dataset is small (to avoid overfitting)
    if X_train_selected.shape[0] < 500:
        print("Small training dataset detected. Using reduced parameter grid.")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    
    # Create base model for tuning
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_SEED
    )
    
    # Create cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Perform grid search with cross-validation
    print("Starting hyperparameter search with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_selected, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print("Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # 7. TRAIN FINAL MODEL
    print("\n7. TRAINING FINAL MODEL WITH TUNED HYPERPARAMETERS")
    print("-"*50)
    
    # Create final model with best parameters
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_SEED,
        **best_params
    )
    
    # Train final model on combined train+validation data
    X_train_full = pd.concat([X_train_selected, X_val_selected])
    y_train_full = pd.concat([y_train, y_val])
    
    print(f"Training final model on {len(X_train_full)} samples with {len(selected_features)} features")
    final_model.fit(X_train_full, y_train_full)
    
    # Save the model
    model_filename = os.path.join(model_dir, 'xgboost_wildfire_model.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to {model_filename}")
    
    # Save the model in XGBoost's native format
    model_xgb_filename = os.path.join(model_dir, 'xgboost_wildfire_model.json')
    final_model.save_model(model_xgb_filename)
    print(f"Model saved in XGBoost format to {model_xgb_filename}")
    
    # Save selected features and best parameters
    with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
        f.write("XGBOOST WILDFIRE PREDICTION MODEL\n")
        f.write(f"Trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BEST HYPERPARAMETERS:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nSELECTED FEATURES:\n")
        for feature in selected_features:
            f.write(f"{feature}\n")
        
        if results['target_transformation'] != 'none':
            f.write(f"\nTARGET TRANSFORMATION: {results['target_transformation']}\n")
            f.write("Remember to apply inverse transformation to predictions!\n")
    
    # 8. EVALUATE FINAL MODEL
    print("\n8. EVALUATING FINAL MODEL PERFORMANCE")
    print("-"*50)
    
    # Get predictions on test set
    y_test_pred = final_model.predict(X_test_selected)
    
    # Calculate metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_explained_var = explained_variance_score(y_test, y_test_pred)
    
    # If target was transformed, also evaluate on original scale
    if results['target_transformation'] == 'log1p':
        # Get original scale test target
        y_test_original = df.loc[y_test.index, 'target_original']
        
        # Convert predictions back to original scale
        y_test_pred_original = np.expm1(y_test_pred)
        
        # Calculate metrics on original scale
        test_mae_original = mean_absolute_error(y_test_original, y_test_pred_original)
        test_rmse_original = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
        test_r2_original = r2_score(y_test_original, y_test_pred_original)
        
        print("Performance metrics on transformed scale:")
    
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test Explained Variance: {test_explained_var:.4f}")
    
    if results['target_transformation'] == 'log1p':
        print("\nPerformance metrics on original scale:")
        print(f"Test MAE: {test_mae_original:.4f}")
        print(f"Test RMSE: {test_rmse_original:.4f}")
        print(f"Test R²: {test_r2_original:.4f}")
    
    # Save metrics to file
    metrics = {
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_explained_variance': test_explained_var
    }
    
    if results['target_transformation'] == 'log1p':
        metrics.update({
            'test_mae_original': test_mae_original,
            'test_rmse_original': test_rmse_original,
            'test_r2_original': test_r2_original
        })
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'model_performance_metrics.csv'), index=False)
    print(f"Performance metrics saved to {os.path.join(output_dir, 'model_performance_metrics.csv')}")
    
    # 9. VISUALIZE ACTUAL VS PREDICTED
    print("\n9. VISUALIZING PREDICTIONS")
    print("-"*50)
    
    # Choose the right y values based on transformation
    if results['target_transformation'] == 'log1p':
        plot_y_test = y_test_original
        plot_y_pred = y_test_pred_original
        plot_title = "Actual vs Predicted Fire Size (Original Scale)"
        plot_xlabel = "Actual Fire Size (hectares)"
        plot_ylabel = "Predicted Fire Size (hectares)"
    else:
        plot_y_test = y_test
        plot_y_pred = y_test_pred
        plot_title = "Actual vs Predicted Fire Size"
        plot_xlabel = "Actual Fire Size"
        plot_ylabel = "Predicted Fire Size"
    
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(plot_y_test, plot_y_pred, alpha=0.5)
    plt.plot([plot_y_test.min(), plot_y_test.max()], [plot_y_test.min(), plot_y_test.max()], 'r--')
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    
    # Residual plot
    plt.figure(figsize=(10, 8))
    residuals = plot_y_test - plot_y_pred
    plt.scatter(plot_y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
    
    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'))
    
    # 10. MODEL EXPLAINABILITY WITH SHAP
    print("\n10. MODEL EXPLAINABILITY WITH SHAP")
    print("-"*50)
    
    try:
        # Create explainer
        explainer = shap.Explainer(final_model)
        
        # Calculate SHAP values for test set (limit to 100 samples if dataset is large)
        max_samples = min(100, X_test_selected.shape[0])
        X_test_sample = X_test_selected.iloc[:max_samples]
        
        print(f"Calculating SHAP values for {max_samples} test samples...")
        shap_values = explainer(X_test_sample)
        
        # Plot summary
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        
        # Plot detailed SHAP values for top features
        top_features_to_plot = min(5, len(selected_features))
        for i in range(top_features_to_plot):
            feature = selected_features[i]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values.values, X_test_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}.png'))
        
        print(f"SHAP visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Warning: SHAP analysis failed with error: {str(e)}")
    
    # 11. CONCLUSION AND SUMMARY
    print("\n11. MODEL TRAINING SUMMARY")
    print("-"*50)
    
    # Store results
    results.update({
        'model': final_model,
        'best_params': best_params,
        'selected_features': selected_features,
        'metrics': metrics
    })
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_training_summary.txt'), 'w') as f:
        f.write("WILDFIRE PREDICTION MODEL TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. Dataset Information:\n")
        f.write(f"   - Total samples: {len(df)}\n")
        f.write(f"   - Initial features: {len(feature_cols)}\n")
        f.write(f"   - Selected features: {len(selected_features)}\n\n")
        
        f.write("2. Model Performance:\n")
        f.write(f"   - Test RMSE: {test_rmse:.4f}\n")
        f.write(f"   - Test R²: {test_r2:.4f}\n")
        if results['target_transformation'] == 'log1p':
            f.write(f"   - Test RMSE (original scale): {test_rmse_original:.4f}\n")
            f.write(f"   - Test R² (original scale): {test_r2_original:.4f}\n")
        f.write("\n")
        
        f.write("3. Top 5 Most Important Features:\n")
        for i, feature in enumerate(selected_features[:5]):
            importance_val = feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0]
            f.write(f"   {i+1}. {feature}: {importance_val:.4f}\n")
        f.write("\n")
        
        f.write("4. Model Configuration:\n")
        for param, value in best_params.items():
            f.write(f"   - {param}: {value}\n")
    
    print(f"Model training summary saved to {os.path.join(output_dir, 'model_training_summary.txt')}")
    print("\nModel training complete! All outputs saved to", output_dir)
    
    return results

if __name__ == "__main__":
    # Path to your processed and imputed wildfire-weather dataset
    input_file = "./data/analysis/2_interpolation/wildfire_weather_iterative_imputed.csv"
    
    # If the specific file doesn't exist, try to find an alternative in the analysis output
    if not os.path.exists(input_file):
        analysis_dir = "./data/analysis/"
        potential_files = [
            "./data/wildfire_weather_knn_imputed.csv",
            "./data/wildfire_weather_median_imputed.csv",
            "./data/wildfire_weather_key_columns.csv",
            "./data/wildfire_weather_merged.csv"
        ]
        
        for file in potential_files:
            if os.path.exists(file):
                input_file = file
                print(f"Using alternative input file: {input_file}")
                break
    
    # Output directory for model artifacts
    output_dir = "./models/xgboost/train_2"
    
    # Train the model
    results = train_xgboost_model(input_file, output_dir)
    
    print("\nScript execution completed successfully!")