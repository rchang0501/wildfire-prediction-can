import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.inspection import permutation_importance

# Import PyTorch TabNet implementation
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("Warning: pytorch_tabnet is not installed. Please install it with:")
    print("pip install pytorch-tabnet")
    print("Exiting script.")
    import sys
    sys.exit(1)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def train_tabnet_classifier(input_file, output_dir, target_column='FIRE_SIZE_HA', threshold=0.01):
    """
    Trains a TabNet classifier on wildfire data to predict if fire size exceeds a threshold.
    
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
    print(f"WILDFIRE CLASSIFICATION TABNET MODEL TRAINING (Threshold: {threshold} ha)")
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
    
    # Save feature list for later use
    with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # 3. DATA SPLITTING
    print("\n3. SPLITTING DATA INTO TRAIN/VALIDATION/TEST SETS")
    print("-"*50)
    
    # Get features and target
    X = df[feature_cols].values
    y = df[binary_target].values
    
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
    print(f"  Training set: {np.bincount(y_train) / len(y_train) * 100}")
    print(f"  Validation set: {np.bincount(y_val) / len(y_val) * 100}")
    print(f"  Test set: {np.bincount(y_test) / len(y_test) * 100}")
    
    # 4. FEATURE SCALING
    print("\n4. FEATURE SCALING")
    print("-"*50)
    
    # TabNet generally performs better with normalized features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. DEFINE TABNET BASELINE MODEL
    print("\n5. DEFINING BASELINE TABNET MODEL")
    print("-"*50)
    
    # Calculate class weight if needed
    if min(class_percentages) < 20:
        # Compute weights based on class frequencies
        class_weights = len(y_train) / (2 * np.bincount(y_train))
        print(f"Using class weights due to imbalance: {class_weights}")
    else:
        class_weights = None
    
    # Initialize TabNet with baseline parameters
    baseline_model = TabNetClassifier(
        n_d=16,  # Width of the decision prediction layer
        n_a=16,  # Width of the attention embedding for each step
        n_steps=3,  # Number of steps in the architecture
        gamma=1.5,  # Scaling factor for attention
        lambda_sparse=1e-3,  # Strength of the sparsity regularization
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(
            mode="min", patience=5, min_lr=1e-5, factor=0.5
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax',  # "entmax" or "sparsemax"
        verbose=0,
        seed=RANDOM_SEED
    )
    
    # Train the baseline model
    print("Training baseline TabNet model...")
    
    # Set class weights if needed
    baseline_model.class_weight = class_weights
    
    # Fit the model - TabNet's fit method works similar to sklearn
    baseline_model.fit(
        X_train=X_train_scaled, y_train=y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric=['accuracy', 'logloss'],
        max_epochs=50,
        patience=15,
        batch_size=256
    )
    
    # Get baseline predictions
    y_train_pred = baseline_model.predict(X_train_scaled)
    y_val_pred = baseline_model.predict(X_val_scaled)
    
    # Get probability predictions
    y_train_pred_proba = baseline_model.predict_proba(X_train_scaled)[:,1]
    y_val_pred_proba = baseline_model.predict_proba(X_val_scaled)[:,1]
    
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
        f.write("BASELINE TABNET MODEL METRICS\n")
        f.write("=" * 30 + "\n\n")
        f.write("Default Parameters:\n")
        f.write(f"  n_d: 16\n")
        f.write(f"  n_a: 16\n")
        f.write(f"  n_steps: 3\n")
        f.write(f"  gamma: 1.5\n")
        f.write(f"  lambda_sparse: 1e-3\n")
        f.write(f"  batch_size: 256\n")
        f.write(f"  max_epochs: 50\n")
        f.write(f"  patience: 15\n")
        f.write(f"  class_weight: {class_weights}\n\n")
        f.write(f"Training Accuracy: {baseline_train_accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {baseline_val_accuracy:.4f}\n\n")
        f.write(f"Training F1 Score: {baseline_train_f1:.4f}\n")
        f.write(f"Validation F1 Score: {baseline_val_f1:.4f}\n\n")
        f.write(f"Training AUC: {baseline_train_auc:.4f}\n")
        f.write(f"Validation AUC: {baseline_val_auc:.4f}\n")
    print(f"Baseline metrics text summary saved to {os.path.join(output_dir, 'baseline_model_metrics.txt')}")
    
    # Get the feature importance scores
    feature_importance = baseline_model.feature_importances_
    
    # Create DataFrame for feature importance
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
    plt.title('Top 20 Features by Importance (Baseline TabNet)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_feature_importance.png'))
    
    # 6. HYPERPARAMETER TUNING
    print("\n6. HYPERPARAMETER TUNING")
    print("-"*50)
    
    # Define parameter grid (smaller than usual due to TabNet's training time)
    param_grid = [
        {
            'n_d': [8, 16, 32],
            'n_a': [8, 16, 32],
            'n_steps': [3, 5],
            'gamma': [1.0, 1.5],
            'lambda_sparse': [1e-4, 1e-3, 1e-2],
            'batch_size': [256, 512]
        }
    ]
    
    best_val_f1 = 0
    best_params = None
    best_model = None
    
    print("Starting hyperparameter search...")
    
    # Create a simpler grid for smaller datasets
    if X_train_scaled.shape[0] < 500:
        print("Small dataset detected. Using reduced parameter grid.")
        param_grid = [
            {
                'n_d': [8, 16],
                'n_a': [8, 16],
                'n_steps': [3],
                'gamma': [1.5],
                'lambda_sparse': [1e-3, 1e-2],
                'batch_size': [256]
            }
        ]
    
    # Create fold indices for cross-validation
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(cv.split(X_train_scaled))
    
    # Manual grid search with cross-validation
    for param_set in param_grid:
        # Create all parameter combinations
        param_keys = list(param_set.keys())
        param_values = list(param_set.values())
        
        # Get all parameter combinations
        import itertools
        param_combinations = list(itertools.product(*param_values))
        
        for params in param_combinations:
            param_dict = {param_keys[i]: params[i] for i in range(len(param_keys))}
            
            # Display current parameter set
            print(f"\nTrying parameters: {param_dict}")
            
            # Extract batch size for training
            batch_size = param_dict.pop('batch_size')
            
            # Cross-validation loop
            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
                # Prepare fold data
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Initialize TabNet with current parameters
                tabnet_fold = TabNetClassifier(
                    **param_dict,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    scheduler_params=dict(
                        mode="min", patience=5, min_lr=1e-5, factor=0.5
                    ),
                    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    mask_type='entmax',
                    verbose=0,
                    seed=RANDOM_SEED
                )
                
                # Set class weights if needed
                tabnet_fold.class_weight = class_weights
                
                # Fit on this fold
                tabnet_fold.fit(
                    X_train=X_fold_train, y_train=y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    eval_metric=['logloss'],
                    max_epochs=30,  # Reduced for grid search
                    patience=10,
                    batch_size=batch_size
                )
                
                # Evaluate on validation fold
                y_fold_val_pred = tabnet_fold.predict(X_fold_val)
                fold_f1 = f1_score(y_fold_val, y_fold_val_pred)
                cv_scores.append(fold_f1)
                
                print(f"  Fold {fold_idx+1}/3: F1 = {fold_f1:.4f}")
            
            # Calculate average F1 across folds
            avg_f1 = np.mean(cv_scores)
            print(f"  Average CV F1: {avg_f1:.4f}")
            
            # Check if this is the best model so far
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                param_dict['batch_size'] = batch_size  # Add batch_size back for final training
                best_params = param_dict
    
    # After grid search in section 6, add this code to evaluate the tuned model
    # Make a copy of best_params including batch_size for display
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    display_params = best_params.copy()
    display_params['batch_size'] = batch_size

    batch_size = best_params.pop('batch_size') if 'batch_size' in best_params else 256

    # Initialize a model with the best parameters to evaluate on the validation set
    tuned_model = TabNetClassifier(
        **best_params,  # This should not include batch_size
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(
            mode="min", patience=5, min_lr=1e-5, factor=0.5
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax',
        verbose=0,
        seed=RANDOM_SEED
    )

    # Set class weights if needed
    tuned_model.class_weight = class_weights

    # Fit the model on the training data
    tuned_model.fit(
        X_train=X_train_scaled, y_train=y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric=['accuracy', 'logloss'],
        max_epochs=50,
        patience=15,
        batch_size=batch_size  # Pass batch_size here
    )

    # Get predictions
    tuned_train_pred = tuned_model.predict(X_train_scaled)
    tuned_val_pred = tuned_model.predict(X_val_scaled)
    tuned_train_pred_proba = tuned_model.predict_proba(X_train_scaled)[:, 1]
    tuned_val_pred_proba = tuned_model.predict_proba(X_val_scaled)[:, 1]

    # Calculate metrics
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
        f.write("TUNED TABNET MODEL METRICS\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Best Parameters:\n")
        for param, value in display_params.items():
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
    final_model = TabNetClassifier(
        **best_params,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(
            mode="min", patience=10, min_lr=1e-5, factor=0.5
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax',
        verbose=1,
        seed=RANDOM_SEED
    )
    
    # Set class weights if needed
    final_model.class_weight = class_weights
    
    # Train final model on combined train+validation data
    X_train_full = np.vstack((X_train_scaled, X_val_scaled))
    y_train_full = np.concatenate((y_train, y_val))
    
    print(f"Training final model on {len(X_train_full)} samples with {X_train_full.shape[1]} features")
    final_model.fit(
        X_train=X_train_full, y_train=y_train_full,
        eval_set=[(X_test_scaled, y_test)],  # Use test set as eval set for final model
        eval_metric=['accuracy', 'logloss'],
        max_epochs=100,
        patience=20,
        batch_size=batch_size
    )
    
    # Save the model
    model_filename = os.path.join(model_dir, 'tabnet_wildfire_classifier.zip')
    final_model.save_model(model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save the scaler
    scaler_filename = os.path.join(model_dir, 'tabnet_scaler.pkl')
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_filename}")
    
    # Save selected features and best parameters
    with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
        f.write("TABNET WILDFIRE CLASSIFICATION MODEL\n")
        f.write(f"Trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"CLASSIFICATION THRESHOLD: {threshold} hectares\n\n")
        f.write(f"Class 0: Fire size < {threshold} hectares\n")
        f.write(f"Class 1: Fire size >= {threshold} hectares\n\n")
        
        f.write("BEST HYPERPARAMETERS:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"batch_size: {batch_size}\n")
        
        if class_weights is not None:
            f.write("\nCLASS WEIGHTS USED:\n")
            for i, weight in enumerate(class_weights):
                f.write(f"  Class {i}: {weight:.4f}\n")
    
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
    
    # TabNet provides feature importance via a mask that quantifies feature usage
    final_feature_importance = final_model.feature_importances_
    
    # Create DataFrame for feature importance
    final_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance to CSV
    final_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print(f"Feature importance saved to {os.path.join(output_dir, 'feature_importance.csv')}")
    
    # Plot feature importance (top 20)
    plt.figure(figsize=(12, 8))
    top_features = final_importance_df.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Features by Importance (Final TabNet Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    print("Top 5 features by TabNet importance:")
    for i, (_, row) in enumerate(final_importance_df.head(5).iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # 11. CONCLUSION AND SUMMARY
    print("\n11. MODEL TRAINING SUMMARY")
    print("-"*50)
    
    # Store results
    results.update({
        'model': final_model,
        'scaler': scaler,
        'best_params': best_params,
        'batch_size': batch_size,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'threshold': threshold
    })
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_training_summary.txt'), 'w') as f:
        f.write("WILDFIRE CLASSIFICATION TABNET MODEL TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Classification Task: Predict if fire size is >= {threshold} hectares\n\n")
        
        f.write("1. Dataset Information:\n")
        f.write(f"   - Total samples: {len(df)}\n")
        f.write(f"   - Class 0 (< {threshold} ha): {class_counts[0]} samples ({class_percentages[0]:.2f}%)\n")
        f.write(f"   - Class 1 (>= {threshold} ha): {class_counts[1]} samples ({class_percentages[1]:.2f}%)\n")
        f.write(f"   - Initial features: {len(feature_cols)}\n\n")
        
        f.write("2. Model Performance:\n")
        f.write(f"   - Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"   - Test Precision: {test_precision:.4f}\n")
        f.write(f"   - Test Recall: {test_recall:.4f}\n")
        f.write(f"   - Test F1 Score: {test_f1:.4f}\n")
        f.write(f"   - Test AUC: {test_auc:.4f}\n\n")
        
        f.write("3. Top 5 Most Important Features:\n")
        for i, (_, row) in enumerate(final_importance_df.head(5).iterrows()):
            f.write(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("4. Model Configuration:\n")
        for param, value in best_params.items():
            f.write(f"   - {param}: {value}\n")
        f.write(f"   - batch_size: {batch_size}\n")
        
        if class_weights is not None:
            f.write("\n5. Class Weights Used:\n")
            for i, weight in enumerate(class_weights):
                f.write(f"   - Class {i}: {weight:.4f}\n")
    
    print(f"Model training summary saved to {os.path.join(output_dir, 'model_training_summary.txt')}")
    print("\nModel training complete! All outputs saved to", output_dir)
    
    return results

if __name__ == "__main__":
    # Path to your processed wildfire-weather dataset
    input_file = "./data/wildfire_weather_interpolated_merged_cleaned.csv"
    
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
    output_dir = "./models/tabnet/classification_2"
    
    # Define classification threshold in hectares (same as XGBoost and SVM for comparison)
    threshold = 0.01
    
    # Train the TabNet classification model
    results = train_tabnet_classifier(input_file, output_dir, threshold=threshold)
    
    print("\nScript execution completed successfully!")