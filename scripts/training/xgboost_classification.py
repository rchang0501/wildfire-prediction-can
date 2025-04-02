import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

import xgboost as xgb

# Try to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP package not found. Model explainability section will be skipped.")
    print("To enable SHAP analysis, install it with: pip install shap")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_xgboost_classifier(input_file, output_dir, target_column='FIRE_SIZE_HA', threshold=1.0):
    """
    Trains an XGBoost classifier on wildfire data to predict if fire size exceeds a threshold.
    
    Args:
        input_file (str): Path to the processed wildfire-weather CSV file
        output_dir (str): Directory to save model, performance metrics, and visualizations
        target_column (str): Name of the column used to create the binary target (default: 'FIRE_SIZE_HA')
        threshold (float): Threshold in hectares for classifying fires (default: 1.0)
                          Fires >= threshold will be class 1, < threshold will be class 0
    
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
    print(f"WILDFIRE CLASSIFICATION XGBOOST MODEL TRAINING (Threshold: {threshold} ha)")
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
    
    # Create binary classification target based on threshold
    binary_target = f"{target_column}_binary"
    df[binary_target] = (df[target_column] >= threshold).astype(int)
    
    # Calculate class distribution
    class_counts = df[binary_target].value_counts()
    class_percentages = class_counts / len(df) * 100
    
    print(f"Created binary target '{binary_target}' with threshold {threshold} hectares")
    print(f"Class distribution:")
    print(f"  Class 0 (< {threshold} ha): {class_counts[0]} samples ({class_percentages[0]:.2f}%)")
    print(f"  Class 1 (>= {threshold} ha): {class_counts[1]} samples ({class_percentages[1]:.2f}%)")
    
    # If classes are very imbalanced, log a warning
    if min(class_percentages) < 10:
        print(f"WARNING: Classes are imbalanced. Consider using class weights or resampling techniques.")
    
    # 2. FEATURE ENGINEERING AND SELECTION
    print("\n2. FEATURE ENGINEERING AND SELECTION")
    print("-"*50)
    
    # Drop non-feature columns
    exclude_cols = ['Date']
    
    # Identify numeric columns (potential features)
    feature_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in exclude_cols and 
                   col != target_column and col != binary_target]
    
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
    y = df[binary_target]
    
    # Save feature list for later use
    with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # First split: 80% train+validation, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: 75% train, 25% validation (resulting in 60%/20%/20% split overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_val
    )
    
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Verify class distribution in the splits
    print("Class distribution in splits:")
    print(f"  Training set: {y_train.value_counts(normalize=True) * 100}")
    print(f"  Validation set: {y_val.value_counts(normalize=True) * 100}")
    print(f"  Test set: {y_test.value_counts(normalize=True) * 100}")
    
    # 4. CREATE BASELINE MODEL
    print("\n4. TRAINING BASELINE XGBOOST MODEL")
    print("-"*50)
    
    # Calculate class weights to handle potential imbalance
    class_weights = None
    if min(class_percentages) < 20:
        # Calculate weight for class 1 based on inverse class frequencies
        weight_1 = class_counts[0] / class_counts[1]
        class_weights = {0: 1.0, 1: weight_1}
        print(f"Using class weights due to imbalance: {class_weights}")
    
    # Basic XGBoost classifier with default hyperparameters
    baseline_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        scale_pos_weight=weight_1 if class_weights else 1.0
    )
    
    # Train the baseline model
    baseline_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Get baseline predictions
    y_train_pred_proba = baseline_model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = baseline_model.predict_proba(X_val)[:, 1]
    
    # Convert to binary predictions
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    
    # Calculate baseline metrics
    baseline_train_accuracy = accuracy_score(y_train, y_train_pred)
    baseline_val_accuracy = accuracy_score(y_val, y_val_pred)
    baseline_train_f1 = f1_score(y_train, y_train_pred)
    baseline_val_f1 = f1_score(y_val, y_val_pred)
    baseline_train_auc = roc_auc_score(y_train, y_train_pred_proba)
    baseline_val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"Baseline model training accuracy: {baseline_train_accuracy:.4f}")
    print(f"Baseline model validation accuracy: {baseline_val_accuracy:.4f}")
    print(f"Baseline model training F1: {baseline_train_f1:.4f}")
    print(f"Baseline model validation F1: {baseline_val_f1:.4f}")
    print(f"Baseline model training AUC: {baseline_train_auc:.4f}")
    print(f"Baseline model validation AUC: {baseline_val_auc:.4f}")
    
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
    
    # Create base model for tuning (with appropriate class weight if needed)
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_SEED,
        scale_pos_weight=weight_1 if class_weights else 1.0
    )
    
    # Create cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Perform grid search with cross-validation
    print("Starting hyperparameter search with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  # Focus on F1 score for imbalanced classification
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
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_SEED,
        scale_pos_weight=weight_1 if class_weights else 1.0,
        **best_params
    )
    
    # Train final model on combined train+validation data
    X_train_full = pd.concat([X_train_selected, X_val_selected])
    y_train_full = pd.concat([y_train, y_val])
    
    print(f"Training final model on {len(X_train_full)} samples with {len(selected_features)} features")
    final_model.fit(X_train_full, y_train_full)
    
    # Save the model
    model_filename = os.path.join(model_dir, 'xgboost_wildfire_classifier.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to {model_filename}")
    
    # Save the model in XGBoost's native format
    model_xgb_filename = os.path.join(model_dir, 'xgboost_wildfire_classifier.json')
    final_model.save_model(model_xgb_filename)
    print(f"Model saved in XGBoost format to {model_xgb_filename}")
    
    # Save selected features and best parameters
    with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
        f.write("XGBOOST WILDFIRE CLASSIFICATION MODEL\n")
        f.write(f"Trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"CLASSIFICATION THRESHOLD: {threshold} hectares\n\n")
        f.write(f"Class 0: Fire size < {threshold} hectares\n")
        f.write(f"Class 1: Fire size >= {threshold} hectares\n\n")
        
        f.write("BEST HYPERPARAMETERS:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nSELECTED FEATURES:\n")
        for feature in selected_features:
            f.write(f"{feature}\n")
        
        if class_weights:
            f.write("\nCLASS WEIGHTS:\n")
            for cls, weight in class_weights.items():
                f.write(f"Class {cls}: {weight}\n")
    
    # 8. EVALUATE FINAL MODEL
    print("\n8. EVALUATING FINAL MODEL PERFORMANCE")
    print("-"*50)
    
    # Get predictions on test set
    y_test_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Generate and print confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.4f}")
    
    # Generate classification report
    class_report = classification_report(y_test, y_test_pred, target_names=[f'<{threshold}ha', f'>={threshold}ha'])
    print("\nClassification Report:")
    print(class_report)
    
    # Save metrics to file
    metrics = {
        'threshold': threshold,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_specificity': specificity,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'model_performance_metrics.csv'), index=False)
    print(f"Performance metrics saved to {os.path.join(output_dir, 'model_performance_metrics.csv')}")
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Classification Report (Threshold: {threshold} hectares)\n")
        f.write("="*50 + "\n\n")
        f.write(class_report)
    
    # 9. VISUALIZE MODEL PERFORMANCE
    print("\n9. VISUALIZING MODEL PERFORMANCE")
    print("-"*50)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'<{threshold}ha', f'>={threshold}ha'],
                yticklabels=[f'<{threshold}ha', f'>={threshold}ha'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # 10. MODEL EXPLAINABILITY 
    print("\n10. MODEL EXPLAINABILITY")
    print("-"*50)
    
    if SHAP_AVAILABLE:
        print("Performing SHAP analysis for model explainability...")
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
    else:
        print("SHAP package not available. Using feature importance as alternative:")
        
        # Create feature importance plot as alternative
        plt.figure(figsize=(12, 10))
        xgb.plot_importance(final_model, max_num_features=20, importance_type='gain', title='Feature Importance (Gain)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_gain.png'))
        
        plt.figure(figsize=(12, 10))
        xgb.plot_importance(final_model, max_num_features=20, importance_type='weight', title='Feature Importance (Weight)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_weight.png'))
        
        print(f"Feature importance visualizations saved to {output_dir}")
    
    # 11. CONCLUSION AND SUMMARY
    print("\n11. MODEL TRAINING SUMMARY")
    print("-"*50)
    
    # Store results
    results.update({
        'model': final_model,
        'best_params': best_params,
        'selected_features': selected_features,
        'metrics': metrics,
        'threshold': threshold
    })
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_training_summary.txt'), 'w') as f:
        f.write("WILDFIRE CLASSIFICATION MODEL TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Classification Task: Predict if fire size is >= {threshold} hectares\n\n")
        
        f.write("1. Dataset Information:\n")
        f.write(f"   - Total samples: {len(df)}\n")
        f.write(f"   - Class 0 (< {threshold} ha): {class_counts[0]} samples ({class_percentages[0]:.2f}%)\n")
        f.write(f"   - Class 1 (>= {threshold} ha): {class_counts[1]} samples ({class_percentages[1]:.2f}%)\n")
        f.write(f"   - Initial features: {len(feature_cols)}\n")
        f.write(f"   - Selected features: {len(selected_features)}\n\n")
        
        f.write("2. Model Performance:\n")
        f.write(f"   - Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"   - Test Precision: {test_precision:.4f}\n")
        f.write(f"   - Test Recall: {test_recall:.4f}\n")
        f.write(f"   - Test F1 Score: {test_f1:.4f}\n")
        f.write(f"   - Test AUC: {test_auc:.4f}\n\n")
        
        f.write("3. Top 5 Most Important Features:\n")
        for i, feature in enumerate(selected_features[:5]):
            importance_val = feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0]
            f.write(f"   {i+1}. {feature}: {importance_val:.4f}\n")
        f.write("\n")
        
        f.write("4. Model Configuration:\n")
        for param, value in best_params.items():
            f.write(f"   - {param}: {value}\n")
        
        if class_weights:
            f.write("\n5. Class Weights Used:\n")
            for cls, weight in class_weights.items():
                f.write(f"   - Class {cls}: {weight}\n")
    
    print(f"Model training summary saved to {os.path.join(output_dir, 'model_training_summary.txt')}")
    print("\nModel training complete! All outputs saved to", output_dir)
    
    return results

if __name__ == "__main__":
    # Path to your processed wildfire-weather dataset
    input_file = "./data/wildfire_weather_interpolated_merged.csv"
    
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
    output_dir = "./models/xgboost/classification_5"
    
    # Define classification threshold in hectares
    # 1.0 hectare = 2.47 acres, which is a common threshold for "significant" wildfires
    # threshold = 1.0  
    threshold = 0.01  
    
    # Train the classification model
    results = train_xgboost_classifier(input_file, output_dir, threshold=threshold)
    
    print("\nScript execution completed successfully!")