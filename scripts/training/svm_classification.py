import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # Add imputer for handling missing values

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_svm_classifier(input_file, output_dir, target_column='FIRE_SIZE_HA', threshold=0.01):
    """
    Trains an SVM classifier on wildfire data to predict if fire size exceeds a threshold.
    
    Args:
        input_file (str): Path to the processed wildfire-weather CSV file
        output_dir (str): Directory to save model, performance metrics, and visualizations
        target_column (str): Name of the column used to create the binary target (default: 'FIRE_SIZE_HA')
        threshold (float): Threshold in hectares for classifying fires (default: 0.01)
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
    print(f"WILDFIRE CLASSIFICATION SVM MODEL TRAINING (Threshold: {threshold} ha)")
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
        # Handle NaN values by using dropna
        most_common_pct = df[col].dropna().value_counts(normalize=True).max() if not df[col].empty else 0
        if most_common_pct > near_constant_threshold:
            near_constant_features.append(col)
    
    if near_constant_features:
        print(f"Removing {len(near_constant_features)} near-constant features:")
        for col in near_constant_features:
            print(f"  - {col}: {df[col].dropna().value_counts(normalize=True).max():.2%} most common value")
        
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
    
    # 4. FEATURE SELECTION AND IMPUTE MISSING VALUES
    print("\n4. FEATURE SELECTION AND IMPUTE MISSING VALUES")
    print("-"*50)
    
    # Create imputer to handle missing values
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    
    # Apply imputer to training data
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    # Feature selection using ANOVA F-value
    print("Performing univariate feature selection...")
    
    # Number of features to select initially
    num_features = min(40, len(feature_cols))
    
    # Initialize the feature selector
    selector = SelectKBest(f_classif, k=num_features)
    X_train_selected = selector.fit_transform(X_train_imputed, y_train)
    
    # Get scores and p-values
    f_scores = selector.scores_
    p_values = selector.pvalues_
    
    # Create DataFrame for feature importance
    min_length = min(len(feature_cols), len(f_scores), len(p_values))
    feature_scores = pd.DataFrame({
        'Feature': feature_cols[:min_length],
        'F_Score': f_scores[:min_length],
        'P_Value': p_values[:min_length]
    })
    
    # Sort by F-score (higher is better)
    feature_scores = feature_scores.sort_values('F_Score', ascending=False)
    
    # Get the selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    
    # Save feature importance to CSV
    feature_scores.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print(f"Feature importance saved to {os.path.join(output_dir, 'feature_importance.csv')}")
    
    # Plot feature importance (top 20)
    plt.figure(figsize=(12, 8))
    top_features = feature_scores.head(20)
    sns.barplot(x='F_Score', y='Feature', data=top_features)
    plt.title('Top 20 Features by F-Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    print(f"Selected top {len(selected_features)} features based on F-scores")
    print("Top 5 features:")
    for i, feature in enumerate(selected_features[:5]):
        score = feature_scores.loc[feature_scores['Feature'] == feature, 'F_Score'].values[0]
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Extract the selected features after imputation
    X_train_selected = X_train_imputed[:, selected_indices]
    X_val_selected = X_val_imputed[:, selected_indices]
    X_test_selected = X_test_imputed[:, selected_indices]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 5. TRAIN BASELINE SVM MODEL
    print("\n5. TRAINING BASELINE SVM MODEL")
    print("-"*50)
    
    # Calculate class weights to handle potential imbalance
    class_weights = None
    if min(class_percentages) < 20:
        # Calculate weight based on inverse class frequencies
        class_weights = {
            0: len(y_train) / (2 * (len(y_train) - sum(y_train))),
            1: len(y_train) / (2 * sum(y_train))
        }
        print(f"Using class weights due to imbalance: {class_weights}")
    
    # Basic SVM classifier with default hyperparameters
    baseline_model = SVC(
        # kernel='rbf',
        kernel='poly',
        C=1.0,
        gamma='scale',
        probability=True,
        class_weight=class_weights,
        random_state=RANDOM_SEED
    )
    
    # Train the baseline model
    print("Training baseline SVM model...")
    baseline_model.fit(X_train_scaled, y_train)
    
    # Get baseline predictions
    y_train_pred = baseline_model.predict(X_train_scaled)
    y_val_pred = baseline_model.predict(X_val_scaled)
    
    # Get probability predictions if available
    try:
        y_train_pred_proba = baseline_model.predict_proba(X_train_scaled)[:, 1]
        y_val_pred_proba = baseline_model.predict_proba(X_val_scaled)[:, 1]
        proba_available = True
    except:
        print("Warning: Probability predictions not available for baseline model.")
        proba_available = False
    
    # Calculate baseline metrics
    baseline_train_accuracy = accuracy_score(y_train, y_train_pred)
    baseline_val_accuracy = accuracy_score(y_val, y_val_pred)
    baseline_train_f1 = f1_score(y_train, y_train_pred)
    baseline_val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Baseline model training accuracy: {baseline_train_accuracy:.4f}")
    print(f"Baseline model validation accuracy: {baseline_val_accuracy:.4f}")
    print(f"Baseline model training F1: {baseline_train_f1:.4f}")
    print(f"Baseline model validation F1: {baseline_val_f1:.4f}")

    if proba_available:
        baseline_train_auc = roc_auc_score(y_train, y_train_pred_proba)
        baseline_val_auc = roc_auc_score(y_val, y_val_pred_proba)
        print(f"Baseline model training AUC: {baseline_train_auc:.4f}")
        print(f"Baseline model validation AUC: {baseline_val_auc:.4f}")

    # Include AUC metrics if probability is available
    baseline_metrics = {
        'baseline_train_accuracy': baseline_train_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'baseline_train_f1': baseline_train_f1,
        'baseline_val_f1': baseline_val_f1
    }

    if proba_available:
        baseline_metrics['baseline_train_auc'] = baseline_train_auc
        baseline_metrics['baseline_val_auc'] = baseline_val_auc

    # Create metrics dataframe and save to CSV
    baseline_metrics_df = pd.DataFrame([baseline_metrics])
    baseline_metrics_df.to_csv(os.path.join(output_dir, 'baseline_model_metrics.csv'), index=False)
    print(f"Baseline metrics saved to {os.path.join(output_dir, 'baseline_model_metrics.csv')}")

    # Also write a more readable text version
    with open(os.path.join(output_dir, 'baseline_model_metrics.txt'), 'w') as f:
        f.write("BASELINE SVM MODEL METRICS\n")
        f.write("=" * 30 + "\n\n")
        f.write("Default Parameters:\n")
        f.write(f"  kernel: rbf\n")
        f.write(f"  C: 1.0\n")
        f.write(f"  gamma: scale\n")
        f.write(f"  probability: True\n")
        f.write(f"  class_weight: {class_weights}\n\n")
        f.write(f"Training Accuracy: {baseline_train_accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {baseline_val_accuracy:.4f}\n\n")
        f.write(f"Training F1 Score: {baseline_train_f1:.4f}\n")
        f.write(f"Validation F1 Score: {baseline_val_f1:.4f}\n\n")
        if proba_available:
            f.write(f"Training AUC: {baseline_train_auc:.4f}\n")
            f.write(f"Validation AUC: {baseline_val_auc:.4f}\n")
    print(f"Baseline metrics text summary saved to {os.path.join(output_dir, 'baseline_model_metrics.txt')}")

    
    # 6. HYPERPARAMETER TUNING
    print("\n6. HYPERPARAMETER TUNING")
    print("-"*50)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        # 'kernel': ['rbf', 'poly', 'sigmoid']
        'kernel': ['poly']
    }
    
    # # Use reduced parameter grid if training dataset is small
    # if X_train_scaled.shape[0] < 500:
    #     print("Small training dataset detected. Using reduced parameter grid.")
    #     param_grid = {
    #         'C': [0.1, 1, 10],
    #         'gamma': ['scale', 0.1],
    #         'kernel': ['rbf', 'linear']
    #     }
    
    # Create base model for tuning (with appropriate class weight if needed)
    svm_model = SVC(
        probability=True,
        class_weight=class_weights,
        random_state=RANDOM_SEED
    )
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Perform grid search with cross-validation
    print("Starting hyperparameter search with 5-fold stratified cross-validation...")
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  # Keep focus on F1 score for imbalanced classification
        n_jobs=-1,
        verbose=1,
        return_train_score=True  # Add this to get training scores
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print("Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    best_model = grid_search.best_estimator_
    tuned_train_pred = best_model.predict(X_train_scaled)
    tuned_val_pred = best_model.predict(X_val_scaled)

    # Get probability predictions if available
    try:
        tuned_train_pred_proba = best_model.predict_proba(X_train_scaled)[:, 1]
        tuned_val_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
        tuned_proba_available = True
    except:
        print("Warning: Probability predictions not available for tuned model.")
        tuned_proba_available = False

    # Calculate metrics for tuned model
    tuned_train_accuracy = accuracy_score(y_train, tuned_train_pred)
    tuned_val_accuracy = accuracy_score(y_val, tuned_val_pred)
    tuned_train_f1 = f1_score(y_train, tuned_train_pred)
    tuned_val_f1 = f1_score(y_val, tuned_val_pred)
    tuned_val_precision = precision_score(y_val, tuned_val_pred)
    tuned_val_recall = recall_score(y_val, tuned_val_pred)

    # Print metrics
    print(f"\nTuned model metrics:")
    print(f"Training Accuracy: {tuned_train_accuracy:.4f}  |  Validation Accuracy: {tuned_val_accuracy:.4f}")
    print(f"Training F1 Score: {tuned_train_f1:.4f}  |  Validation F1 Score: {tuned_val_f1:.4f}")
    print(f"Validation Precision: {tuned_val_precision:.4f}")
    print(f"Validation Recall: {tuned_val_recall:.4f}")

    # Add AUC if probability predictions are available
    if tuned_proba_available:
        tuned_train_auc = roc_auc_score(y_train, tuned_train_pred_proba)
        tuned_val_auc = roc_auc_score(y_val, tuned_val_pred_proba)
        print(f"Training AUC: {tuned_train_auc:.4f}  |  Validation AUC: {tuned_val_auc:.4f}")

    # Write tuned model metrics to file
    tuned_metrics = {
        'tuned_train_accuracy': tuned_train_accuracy,
        'tuned_val_accuracy': tuned_val_accuracy,
        'tuned_train_f1': tuned_train_f1,
        'tuned_val_f1': tuned_val_f1,
        'tuned_val_precision': tuned_val_precision,
        'tuned_val_recall': tuned_val_recall
    }

    # Add AUC metrics if available
    if tuned_proba_available:
        tuned_metrics['tuned_train_auc'] = tuned_train_auc
        tuned_metrics['tuned_val_auc'] = tuned_val_auc

    # Create metrics dataframe and save to CSV
    tuned_metrics_df = pd.DataFrame([tuned_metrics])
    tuned_metrics_df.to_csv(os.path.join(output_dir, 'tuned_model_metrics.csv'), index=False)
    print(f"Tuned model metrics saved to {os.path.join(output_dir, 'tuned_model_metrics.csv')}")

    # Also write a more readable text version
    with open(os.path.join(output_dir, 'tuned_model_metrics.txt'), 'w') as f:
        f.write("TUNED SVM MODEL METRICS\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        f.write(f"Training Accuracy: {tuned_train_accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {tuned_val_accuracy:.4f}\n\n")
        f.write(f"Training F1 Score: {tuned_train_f1:.4f}\n")
        f.write(f"Validation F1 Score: {tuned_val_f1:.4f}\n\n")
        f.write(f"Validation Precision: {tuned_val_precision:.4f}\n")
        f.write(f"Validation Recall: {tuned_val_recall:.4f}\n")
        if tuned_proba_available:
            f.write(f"\nTraining AUC: {tuned_train_auc:.4f}\n")
            f.write(f"Validation AUC: {tuned_val_auc:.4f}\n")
    print(f"Tuned model metrics text summary saved to {os.path.join(output_dir, 'tuned_model_metrics.txt')}")
    
    # 7. TRAIN FINAL MODEL
    print("\n7. TRAINING FINAL MODEL WITH TUNED HYPERPARAMETERS")
    print("-"*50)
    
    # Create final model with best parameters
    final_model = SVC(
        probability=True,
        class_weight=class_weights,
        random_state=RANDOM_SEED,
        **best_params
    )
    
    # Train final model on combined train+validation data
    X_train_val_selected = np.vstack((X_train_scaled, X_val_scaled))
    y_train_val_combined = pd.concat([y_train, y_val])
    
    print(f"Training final model on {len(X_train_val_selected)} samples with {len(selected_features)} features")
    final_model.fit(X_train_val_selected, y_train_val_combined)
    
    # Create a complete pipeline that includes imputation, feature selection, scaling, and the model
    # This will allow for proper handling of new data with missing values
    pipeline = Pipeline([
        ('imputer', imputer),
        ('selector', selector),
        ('scaler', scaler),
        ('svm', final_model)
    ])
    
    # Save the pipeline (includes imputer, selector, scaler, and model)
    model_filename = os.path.join(model_dir, 'svm_wildfire_classifier.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model pipeline saved to {model_filename}")
    
    # Save selected features and best parameters
    with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
        f.write("SVM WILDFIRE CLASSIFICATION MODEL\n")
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
    y_test_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = final_model.predict(X_test_scaled)
    
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
    
    # 10. FEATURE IMPORTANCE ANALYSIS
    print("\n10. FEATURE IMPORTANCE ANALYSIS")
    print("-"*50)
    
    # SVM doesn't provide feature importance directly
    # Use permutation importance instead
    print("Calculating permutation feature importance...")
    
    perm_importance = permutation_importance(
        final_model, X_test_scaled, y_test,
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    # Create DataFrame for permutation importance
    perm_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Save permutation importance to CSV
    perm_importance_df.to_csv(os.path.join(output_dir, 'permutation_importance.csv'), index=False)
    print(f"Permutation importance saved to {os.path.join(output_dir, 'permutation_importance.csv')}")
    
    # Plot permutation importance (top 20)
    plt.figure(figsize=(12, 8))
    top_features = perm_importance_df.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features, xerr=top_features['Std'])
    plt.title('Permutation Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'permutation_importance.png'))
    
    print("Top 5 features by permutation importance:")
    for i, (_, row) in enumerate(perm_importance_df.head(5).iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")
    
    # 11. VISUALIZATION OF DECISION BOUNDARY (IF POSSIBLE)
    # print("\n11. DECISION BOUNDARY VISUALIZATION")
    # print("-"*50)
    
    # # Only attempt if dimensionality is not too high
    # if len(selected_features) > 2:
    #     # Use PCA to reduce dimensions for visualization
    #     print("Applying PCA for decision boundary visualization...")
        
    #     pca = PCA(n_components=2)
    #     X_test_pca = pca.fit_transform(X_test_scaled)
        
    #     # Calculate explained variance
    #     explained_var = pca.explained_variance_ratio_
    #     print(f"Explained variance ratio for 2 PCA components: {explained_var[0]:.2f}, {explained_var[1]:.2f}")
    #     print(f"Total explained variance: {sum(explained_var):.2f}")
        
    #     # Train a new SVM model on PCA components for visualization
    #     svm_pca = SVC(
    #         probability=True,
    #         class_weight=class_weights,
    #         random_state=RANDOM_SEED,
    #         **best_params
    #     )
    #     svm_pca.fit(pca.fit_transform(X_train_val_selected), y_train_val_combined)
        
    #     # Create meshgrid for decision boundary
    #     h = 0.02  # step size in the mesh
    #     x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    #     y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
    #     # Plot decision boundary
    #     plt.figure(figsize=(10, 8))
    #     Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        
    #     # Plot test samples
    #     scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
    #                         edgecolors='k', cmap=plt.cm.RdBu)
    #     plt.legend(*scatter.legend_elements(),
    #             title="Classes", loc="upper right")
    #     plt.xlabel(f"PCA Component 1 ({explained_var[0]:.2%} variance)")
    #     plt.ylabel(f"PCA Component 2 ({explained_var[1]:.2%} variance)")
    #     plt.title('Decision Boundary (PCA Projection)')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, 'decision_boundary.png'))
    #     print(f"Decision boundary visualization saved to {os.path.join(output_dir, 'decision_boundary.png')}")
    # else:
    #     print("Skipping decision boundary visualization (feature space already 2D or less)")
    
    # 12. CONCLUSION AND SUMMARY
    print("\n12. MODEL TRAINING SUMMARY")
    print("-"*50)
    
    # Store results
    results.update({
        'model': final_model,
        'imputer': imputer,
        'selector': selector,
        'scaler': scaler,
        'best_params': best_params,
        'selected_features': selected_features,
        'metrics': metrics,
        'threshold': threshold
    })
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_training_summary.txt'), 'w') as f:
        f.write("WILDFIRE CLASSIFICATION SVM MODEL TRAINING SUMMARY\n")
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
        
        f.write("3. Top 5 Most Important Features (by permutation importance):\n")
        for i, (_, row) in enumerate(perm_importance_df.head(5).iterrows()):
            f.write(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}\n")
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
    input_file = "data/wildfire_weather_interpolated_merged_cleaned.csv"
    
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
    output_dir = "./models/svm/classification_6"
    
    # Define classification threshold in hectares (same as XGBoost for comparison)
    threshold = 0.01
    
    # Train the SVM classification model
    results = train_svm_classifier(input_file, output_dir, threshold=threshold)
    
    print("\nScript execution completed successfully!")