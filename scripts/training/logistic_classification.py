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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_logistic_classifier(input_file, output_dir, target_column='FIRE_SIZE_HA', threshold=0.01):
    """
    Trains a Logistic Regression classifier on wildfire data to predict if fire size exceeds a threshold.
    
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
    print(f"WILDFIRE CLASSIFICATION LOGISTIC REGRESSION MODEL TRAINING (Threshold: {threshold} ha)")
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
    
    # 4. FEATURE SELECTION AND SCALING
    print("\n4. FEATURE SELECTION AND SCALING")
    print("-"*50)
    
    # Feature selection using ANOVA F-value
    print("Performing univariate feature selection...")
    
    # Calculate maximum number of features for logistic regression
    # For logistic regression, we should have fewer features than data points to avoid overfitting
    # Rule of thumb: min(n_samples / 10, n_features)
    max_features = min(X_train.shape[0] // 10, X_train.shape[1])
    max_features = max(max_features, 10)  # Ensure we have at least 10 features
    
    # Initialize the feature selector
    selector = SelectKBest(f_classif, k=max_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Get scores and p-values
    f_scores = selector.scores_
    p_values = selector.pvalues_
    
    # Create DataFrame for feature importance
    feature_scores = pd.DataFrame({
        'Feature': feature_cols,
        'F_Score': f_scores,
        'P_Value': p_values
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
    
    # Create feature subsets for modeling
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 5. TRAIN BASELINE LOGISTIC REGRESSION MODEL
    print("\n5. TRAINING BASELINE LOGISTIC REGRESSION MODEL")
    print("-"*50)
    
    # Calculate class weights to handle potential imbalance
    class_weights = None
    if min(class_percentages) < 20:
        # Use 'balanced' option which automatically adjusts weights inversely proportional to class frequencies
        class_weights = 'balanced'
        print(f"Using balanced class weights due to imbalance")
    
    # Basic Logistic Regression classifier with default hyperparameters
    baseline_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        class_weight=class_weights,
        random_state=RANDOM_SEED,
        max_iter=1000
    )
    
    # Train the baseline model
    print("Training baseline Logistic Regression model...")
    baseline_model.fit(X_train_scaled, y_train)
    
    # Get baseline predictions
    y_train_pred = baseline_model.predict(X_train_scaled)
    y_val_pred = baseline_model.predict(X_val_scaled)
    
    # Get probability predictions
    y_train_pred_proba = baseline_model.predict_proba(X_train_scaled)[:, 1]
    y_val_pred_proba = baseline_model.predict_proba(X_val_scaled)[:, 1]
    
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

    # Write baseline metrics to file
    baseline_metrics = {
        'baseline_train_accuracy': baseline_train_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'baseline_train_f1': baseline_train_f1,
        'baseline_val_f1': baseline_val_f1,
        'baseline_train_auc': baseline_train_auc,
        'baseline_val_auc': baseline_val_auc
    }

    # Create metrics dataframe and save to CSV
    baseline_metrics_df = pd.DataFrame([baseline_metrics])
    baseline_metrics_df.to_csv(os.path.join(output_dir, 'baseline_model_metrics.csv'), index=False)
    print(f"Baseline metrics saved to {os.path.join(output_dir, 'baseline_model_metrics.csv')}")

    # Also write a more readable text version
    with open(os.path.join(output_dir, 'baseline_model_metrics.txt'), 'w') as f:
        f.write("BASELINE LOGISTIC REGRESSION MODEL METRICS\n")
        f.write("="*45 + "\n\n")
        f.write("Default Parameters:\n")
        f.write(f"  penalty: l2\n")
        f.write(f"  C: 1.0\n")
        f.write(f"  solver: liblinear\n")
        f.write(f"  class_weight: {class_weights}\n")
        f.write(f"  max_iter: 1000\n\n")
        f.write(f"Training Accuracy: {baseline_train_accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {baseline_val_accuracy:.4f}\n\n")
        f.write(f"Training F1 Score: {baseline_train_f1:.4f}\n")
        f.write(f"Validation F1 Score: {baseline_val_f1:.4f}\n\n")
        f.write(f"Training AUC: {baseline_train_auc:.4f}\n")
        f.write(f"Validation AUC: {baseline_val_auc:.4f}\n")
    print(f"Baseline metrics text summary saved to {os.path.join(output_dir, 'baseline_model_metrics.txt')}")
    
    # Look at coefficient magnitude for feature importance
    coefficients = np.abs(baseline_model.coef_[0])
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefficients
    }).sort_values('Coefficient', ascending=False)
    
    # Plot top coefficients
    plt.figure(figsize=(12, 8))
    top_coeffs = coef_df.head(20)
    sns.barplot(x='Coefficient', y='Feature', data=top_coeffs)
    plt.title('Top 20 Features by Coefficient Magnitude (Baseline Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_coefficients.png'))

    
    # 6. HYPERPARAMETER TUNING
    print("\n6. HYPERPARAMETER TUNING")
    print("-"*50)
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],  # liblinear supports both L1 and L2 regularization
        'class_weight': [None, 'balanced'] if class_weights is None else ['balanced']
    }
    
    # Use reduced parameter grid if feature set is large
    if X_train_scaled.shape[1] > 50:
        print("Large feature set detected. Using simplified parameter grid.")
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'class_weight': [None, 'balanced'] if class_weights is None else ['balanced']
        }
    
    # Create cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Perform grid search with cross-validation
    print("Starting hyperparameter search with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  # Focus on F1 score for imbalanced classification
        n_jobs=-1,
        verbose=1
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
    tuned_train_pred_proba = best_model.predict_proba(X_train_scaled)[:, 1]
    tuned_val_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]

    # Calculate metrics for tuned model
    tuned_train_accuracy = accuracy_score(y_train, tuned_train_pred)
    tuned_val_accuracy = accuracy_score(y_val, tuned_val_pred)
    tuned_train_f1 = f1_score(y_train, tuned_train_pred)
    tuned_val_f1 = f1_score(y_val, tuned_val_pred)
    tuned_train_auc = roc_auc_score(y_train, tuned_train_pred_proba)
    tuned_val_auc = roc_auc_score(y_val, tuned_val_pred_proba)
    tuned_val_precision = precision_score(y_val, tuned_val_pred)
    tuned_val_recall = recall_score(y_val, tuned_val_pred)

    # Print metrics
    print(f"\nTuned model metrics:")
    print(f"Training Accuracy: {tuned_train_accuracy:.4f}  |  Validation Accuracy: {tuned_val_accuracy:.4f}")
    print(f"Training F1 Score: {tuned_train_f1:.4f}  |  Validation F1 Score: {tuned_val_f1:.4f}")
    print(f"Training AUC: {tuned_train_auc:.4f}  |  Validation AUC: {tuned_val_auc:.4f}")
    print(f"Validation Precision: {tuned_val_precision:.4f}")
    print(f"Validation Recall: {tuned_val_recall:.4f}")

    # Write tuned model metrics to file
    tuned_metrics = {
        'tuned_train_accuracy': tuned_train_accuracy,
        'tuned_val_accuracy': tuned_val_accuracy,
        'tuned_train_f1': tuned_train_f1,
        'tuned_val_f1': tuned_val_f1,
        'tuned_train_auc': tuned_train_auc,
        'tuned_val_auc': tuned_val_auc,
        'tuned_val_precision': tuned_val_precision,
        'tuned_val_recall': tuned_val_recall
    }

    # Create metrics dataframe and save to CSV
    tuned_metrics_df = pd.DataFrame([tuned_metrics])
    tuned_metrics_df.to_csv(os.path.join(output_dir, 'tuned_model_metrics.csv'), index=False)
    print(f"Tuned model metrics saved to {os.path.join(output_dir, 'tuned_model_metrics.csv')}")

    # Also write a more readable text version
    with open(os.path.join(output_dir, 'tuned_model_metrics.txt'), 'w') as f:
        f.write("TUNED LOGISTIC REGRESSION MODEL METRICS\n")
        f.write("="*45 + "\n\n")
        f.write(f"Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        f.write(f"Training Accuracy: {tuned_train_accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {tuned_val_accuracy:.4f}\n\n")
        f.write(f"Training F1 Score: {tuned_train_f1:.4f}\n")
        f.write(f"Validation F1 Score: {tuned_val_f1:.4f}\n\n")
        f.write(f"Training AUC: {tuned_train_auc:.4f}\n")
        f.write(f"Validation AUC: {tuned_val_auc:.4f}\n\n")
        f.write(f"Validation Precision: {tuned_val_precision:.4f}\n")
        f.write(f"Validation Recall: {tuned_val_recall:.4f}\n")
    print(f"Tuned model metrics text summary saved to {os.path.join(output_dir, 'tuned_model_metrics.txt')}")
    
    # 7. TRAIN FINAL MODEL
    print("\n7. TRAINING FINAL MODEL WITH TUNED HYPERPARAMETERS")
    print("-"*50)
    
    # Create final model with best parameters
    final_model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
        **best_params
    )
    
    # Train final model on combined train+validation data
    X_train_full = np.vstack((X_train_scaled, X_val_scaled))
    y_train_full = pd.concat([y_train, y_val])
    
    print(f"Training final model on {len(X_train_full)} samples with {len(selected_features)} features")
    final_model.fit(X_train_full, y_train_full)
    
    # Create a pipeline that includes scaling and the model
    pipeline = Pipeline([
        ('scaler', scaler),
        ('logistic', final_model)
    ])
    
    # Save the pipeline (includes scaler and model)
    model_filename = os.path.join(model_dir, 'logistic_wildfire_classifier.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model pipeline saved to {model_filename}")
    
    # Save selected features and best parameters
    with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
        f.write("LOGISTIC REGRESSION WILDFIRE CLASSIFICATION MODEL\n")
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
        
        # Get coefficients for interpretation
        coef_info = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': final_model.coef_[0],
            'Abs_Coefficient': np.abs(final_model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        f.write("\nTOP 10 COEFFICIENTS:\n")
        for i, (_, row) in enumerate(coef_info.head(10).iterrows()):
            f.write(f"{i+1}. {row['Feature']}: {row['Coefficient']:.4f}\n")
    
    # 8. EVALUATE FINAL MODEL
    print("\n8. EVALUATING FINAL MODEL PERFORMANCE")
    print("-"*50)
    
    # Get predictions on test set
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
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
    
    # Logistic Regression coefficients
    coefficients = final_model.coef_[0]
    abs_coefficients = np.abs(coefficients)
    
    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefficients,
        'Abs_Coefficient': abs_coefficients
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Save coefficients to CSV
    coef_df.to_csv(os.path.join(output_dir, 'model_coefficients.csv'), index=False)
    print(f"Model coefficients saved to {os.path.join(output_dir, 'model_coefficients.csv')}")
    
    # Plot coefficients (top 20 by absolute value)
    plt.figure(figsize=(12, 10))
    top_coeffs = coef_df.head(20)
    # Create a color map based on coefficient sign
    colors = ['red' if c < 0 else 'blue' for c in top_coeffs['Coefficient']]
    plt.barh(top_coeffs['Feature'], top_coeffs['Coefficient'], color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Top 20 Features by Coefficient Magnitude')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coefficient_importance.png'))
    
    # Create a second plot showing actual coefficient values (not absolute)
    plt.figure(figsize=(12, 10))
    top_by_value = coef_df.sort_values('Coefficient', ascending=False).head(20)
    colors = ['blue' for _ in range(len(top_by_value))]
    plt.barh(top_by_value['Feature'], top_by_value['Coefficient'], color=colors)
    plt.title('Top 20 Features with Positive Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positive_coefficients.png'))
    
    # Create a third plot showing the most negative coefficients
    plt.figure(figsize=(12, 10))
    bottom_by_value = coef_df.sort_values('Coefficient').head(20)
    colors = ['red' for _ in range(len(bottom_by_value))]
    plt.barh(bottom_by_value['Feature'], bottom_by_value['Coefficient'], color=colors)
    plt.title('Top 20 Features with Negative Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'negative_coefficients.png'))
    
    print("Top 5 positive coefficient features:")
    top_positive = coef_df.sort_values('Coefficient', ascending=False).head(5)
    for i, (_, row) in enumerate(top_positive.iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
    
    print("\nTop 5 negative coefficient features:")
    top_negative = coef_df.sort_values('Coefficient').head(5)
    for i, (_, row) in enumerate(top_negative.iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
    
    # 11. DECISION BOUNDARY VISUALIZATION (IF POSSIBLE)
    print("\n11. DECISION BOUNDARY VISUALIZATION")
    print("-"*50)
    
    # Only attempt if we have many features that need dimensional reduction
    if len(selected_features) > 2:
        # Use PCA to reduce dimensions for visualization
        print("Applying PCA for decision boundary visualization...")
        
        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(X_test_scaled)
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        print(f"Explained variance ratio for 2 PCA components: {explained_var[0]:.2f}, {explained_var[1]:.2f}")
        print(f"Total explained variance: {sum(explained_var):.2f}")
        
        # Train a new logistic model on PCA components for visualization
        logistic_pca = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            **best_params
        )
        logistic_pca.fit(pca.fit_transform(X_train_full), y_train_full)
        
        # Create meshgrid for decision boundary
        h = 0.02  # step size in the mesh
        x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
        y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Plot decision boundary
        plt.figure(figsize=(10, 8))
        Z = logistic_pca.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        
        # Plot test samples
        scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                            edgecolors='k', cmap=plt.cm.RdBu)
        plt.legend(*scatter.legend_elements(),
                title="Classes", loc="upper right")
        plt.xlabel(f"PCA Component 1 ({explained_var[0]:.2%} variance)")
        plt.ylabel(f"PCA Component 2 ({explained_var[1]:.2%} variance)")
        plt.title('Decision Boundary (PCA Projection)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'decision_boundary.png'))
        print(f"Decision boundary visualization saved to {os.path.join(output_dir, 'decision_boundary.png')}")
    else:
        print("Skipping decision boundary visualization (feature space already 2D or less)")
    
    # 12. CONCLUSION AND SUMMARY
    print("\n12. MODEL TRAINING SUMMARY")
    print("-"*50)
    
    # Store results
    results.update({
        'model': final_model,
        'scaler': scaler,
        'best_params': best_params,
        'selected_features': selected_features,
        'metrics': metrics,
        'threshold': threshold
    })
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_training_summary.txt'), 'w') as f:
        f.write("WILDFIRE CLASSIFICATION LOGISTIC REGRESSION MODEL TRAINING SUMMARY\n")
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
        
        f.write("3. Top 5 Most Influential Features (by coefficient magnitude):\n")
        for i, (_, row) in enumerate(coef_df.head(5).iterrows()):
            f.write(f"   {i+1}. {row['Feature']}: {row['Coefficient']:.4f}\n")
        f.write("\n")
        
        f.write("4. Model Configuration:\n")
        for param, value in best_params.items():
            f.write(f"   - {param}: {value}\n")
    
    print(f"Model training summary saved to {os.path.join(output_dir, 'model_training_summary.txt')}")
    print("\nModel training complete! All outputs saved to", output_dir)
    
    return results

if __name__ == "__main__":
    # Path to your processed wildfire-weather dataset
    input_file = "./data/lr_top_20_wildfire_weather_interpolated_merged_cleaned.csv"
    
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
    output_dir = "./models/logistic/classification_3"
    
    # Define classification threshold in hectares (same as other models for comparison)
    threshold = 0.01
    
    # Train the Logistic Regression classification model
    results = train_logistic_classifier(input_file, output_dir, threshold=threshold)
    
    print("\nScript execution completed successfully!")