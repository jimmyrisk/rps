"""
Shared validation utilities for consistent model evaluation across all trainers.
Ensures comparable cross-validation results and preprocessing.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Optional, Dict, Any
import mlflow


def time_series_split(X: pd.DataFrame, y: pd.Series, 
                     test_size: int = 1000, 
                     val_size: int = 800) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                     pd.Series, pd.Series, pd.Series]:
    """
    Consistent time-aware data splitting with FIXED sizes (not fractions).
    
    Args:
        X: Features dataframe (already sorted chronologically)
        y: Target series
        test_size: Number of most recent events for testing (default: 1000)
        val_size: Number of events before test set for validation (default: 800)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        
    Example with 3000 events:
        - Training: events[0:1800]   (first 60%)
        - Validation: events[1800:2400]  (next 20%)
        - Test: events[2400:3000]   (last 20%)
    """
    n = len(X)
    
    # Calculate split points from the end
    test_start = max(0, n - test_size)
    val_start = max(0, test_start - val_size)
    
    # Split data
    X_train = X.iloc[:val_start].reset_index(drop=True)
    X_val = X.iloc[val_start:test_start].reset_index(drop=True)
    X_test = X.iloc[test_start:].reset_index(drop=True)
    
    y_train = y.iloc[:val_start].reset_index(drop=True)
    y_val = y.iloc[val_start:test_start].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)
    
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def game_stratified_cv_split(X: pd.DataFrame, y: pd.Series, events_df: pd.DataFrame, 
                              n_splits: int = 5, random_state: int = 42) -> list:
    """
    Create CV folds stratified by game_id (not by individual events).
    
    This ensures:
    - All events from the same game stay in the same fold
    - Folds are randomly partitioned (NOT chronological)
    - Fold sizes are approximately equal in number of events
    
    Args:
        X: Features DataFrame for training+validation combined
        y: Target Series for training+validation combined
        events_df: Original events DataFrame with game_id column (aligned with X indices)
        n_splits: Number of CV folds (default: 5)
        random_state: Random seed for reproducibility
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    from collections import defaultdict
    
    # Map each event index to its game_id
    game_ids = events_df['game_id'].values[:len(X)]
    
    # Group event indices by game_id
    game_to_indices = defaultdict(list)
    for idx, game_id in enumerate(game_ids):
        game_to_indices[game_id].append(idx)
    
    # Get unique games and shuffle them
    unique_games = list(game_to_indices.keys())
    np.random.seed(random_state)
    np.random.shuffle(unique_games)
    
    # Partition games into n_splits groups using greedy bin-packing
    # Goal: Balance fold sizes by number of events (not just number of games)
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    
    for game_id in unique_games:
        # Add game to the fold with fewest events so far
        min_fold_idx = np.argmin(fold_sizes)
        folds[min_fold_idx].append(game_id)
        fold_sizes[min_fold_idx] += len(game_to_indices[game_id])
    
    # Convert game-based folds to index-based folds
    cv_splits = []
    for val_fold_idx in range(n_splits):
        val_games = folds[val_fold_idx]
        train_games = [g for i, fold in enumerate(folds) if i != val_fold_idx for g in fold]
        
        # Get event indices for train and val
        train_indices = [idx for game in train_games for idx in game_to_indices[game]]
        val_indices = [idx for game in val_games for idx in game_to_indices[game]]
        
        cv_splits.append((train_indices, val_indices))
        print(f"  Fold {val_fold_idx+1}: Train={len(train_indices)} events ({len(train_games)} games), "
              f"Val={len(val_indices)} events ({len(val_games)} games)")
    
    return cv_splits


def get_time_series_cv(n_splits: int = 3) -> TimeSeriesSplit:
    """Get TimeSeriesSplit cross-validator for consistent CV across models."""
    return TimeSeriesSplit(n_splits=n_splits)


def prepare_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Consistent feature preprocessing with optional scaling.
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        scale_features: Whether to apply StandardScaler
        
    Returns:
        X_train_processed, X_val_processed, X_test_processed, scaler
    """
    if scale_features:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_val_processed = scaler.transform(X_val)
        X_test_processed = scaler.transform(X_test)
        print("✅ Applied StandardScaler to features")
    else:
        X_train_processed = X_train.values
        X_val_processed = X_val.values  
        X_test_processed = X_test.values
        scaler = None
        print("✅ Using raw features (no scaling)")
        
    return X_train_processed, X_val_processed, X_test_processed, scaler


def encode_labels(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, 
                 class_names: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Consistent label encoding across models.
    
    Returns:
        y_train_numeric, y_val_numeric, y_test_numeric, label_map
    """
    label_map = {move: idx for idx, move in enumerate(class_names)}
    
    y_train_numeric = y_train.map(label_map).values
    y_val_numeric = y_val.map(label_map).values
    y_test_numeric = y_test.map(label_map).values
    
    return y_train_numeric, y_val_numeric, y_test_numeric, label_map


def run_cross_validation(model_class, model_params: dict, X_train: np.ndarray, y_train: np.ndarray,
                        cv_splits: int = 3, class_names: list = None) -> Dict[str, float]:
    """
    Run consistent time-series cross-validation.
    
    Args:
        model_class: Model class or function that returns fitted model
        model_params: Parameters for model initialization
        X_train, y_train: Training data
        cv_splits: Number of CV splits
        class_names: Class names for reporting
        
    Returns:
        Dictionary with CV metrics
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    cv_scores = []
    
    print(f"Running {cv_splits}-fold time-series cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"  Fold {fold + 1}/{cv_splits}", end=" ")
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Create and train model (implementation depends on model type)
        if hasattr(model_class, 'fit'):
            # Sklearn-style model
            model = model_class(**model_params)
            model.fit(X_fold_train, y_fold_train)
            val_pred = model.predict(X_fold_val)
        else:
            # Custom training function
            val_pred = model_class(X_fold_train, y_fold_train, X_fold_val, **model_params)
        
        val_acc = accuracy_score(y_fold_val, val_pred)
        cv_scores.append(val_acc)
        
        print(f"Acc: {val_acc:.4f}")
        
        # Log individual fold metrics
        mlflow.log_metric(f"cv_fold_{fold + 1}_acc", val_acc)
    
    # Calculate summary statistics
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    cv_results = {
        'cv_mean_acc': mean_cv_score,
        'cv_std_acc': std_cv_score,
        'cv_scores': cv_scores
    }
    
    # Log summary metrics
    mlflow.log_metric("cv_mean_acc", mean_cv_score)
    mlflow.log_metric("cv_std_acc", std_cv_score)
    
    print(f"📊 CV Results: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    
    return cv_results


def evaluate_final_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                        class_names: list, model_name: str = "model") -> Dict[str, Any]:
    """
    Consistent final model evaluation on test set.
    
    Returns:
        Dictionary with test metrics and classification report
    """
    print(f"\n🎯 Final evaluation of {model_name}...")
    
    # Get predictions
    if hasattr(model, 'predict'):
        test_pred = model.predict(X_test)
    else:
        # Handle custom prediction logic
        test_pred = model(X_test)
    
    # Calculate metrics
    test_acc = accuracy_score(y_test, test_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, test_pred, target_names=class_names)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", test_acc)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(class_report)
    
    # Per-class accuracy for detailed analysis
    conf_matrix = confusion_matrix(y_test, test_pred)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    for i, class_name in enumerate(class_names):
        class_acc = per_class_acc[i]
        mlflow.log_metric(f"test_acc_{class_name.lower()}", class_acc)
        print(f"  {class_name} accuracy: {class_acc:.4f}")
    
    return {
        'test_accuracy': test_acc,
        'classification_report': class_report,
        'per_class_accuracy': dict(zip(class_names, per_class_acc)),
        'confusion_matrix': conf_matrix
    }


def log_common_params(algorithm: str, n_features: int, n_samples: int, 
                     lookback: int, **additional_params):
    """Log common parameters across all models for consistency."""
    common_params = {
        "algorithm": algorithm,
        "n_features": n_features, 
        "n_samples": n_samples,
        "lookback": lookback,
        "data_split": "time_series",
        "preprocessing": "scaled" if additional_params.get('scaled') else "raw"
    }
    common_params.update(additional_params)
    
    mlflow.log_params(common_params)


def run_time_series_cv(X_train: np.ndarray, y_train: np.ndarray, 
                       model_factory, model_params: Dict[str, Any], 
                       n_splits: int = 3) -> Tuple[list, float, float]:
    """
    Generic time-series cross validation for any model type.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        model_factory: Function that creates and returns a trained model
                      Signature: model_factory(X_fold_train, y_fold_train, X_fold_val, y_fold_val, **model_params)
                      Should return (model, val_predictions)
        model_params: Parameters to pass to model_factory
        n_splits: Number of CV folds
        
    Returns:
        cv_scores, mean_cv_score, std_cv_score
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    print(f"🔄 Running {n_splits}-fold time-series cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"  Fold {fold + 1}/{n_splits}", end=" ")
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Train model using the factory function
        model, val_pred = model_factory(X_fold_train, y_fold_train, X_fold_val, y_fold_val, **model_params)
        
        # Calculate validation accuracy
        val_acc = accuracy_score(y_fold_val, val_pred)
        cv_scores.append(val_acc)
        
        print(f"Acc: {val_acc:.4f}")
        mlflow.log_metric(f"cv_fold_{fold + 1}_acc", val_acc)
    
    # Log summary statistics
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    mlflow.log_metric("cv_mean_acc", mean_cv_score)
    mlflow.log_metric("cv_std_acc", std_cv_score)
    
    print(f"📊 CV Results: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    
    # Create CV scores visualization
    create_cv_plot(cv_scores, mean_cv_score, std_cv_score)
    
    return cv_scores, mean_cv_score, std_cv_score


def create_cv_plot(cv_scores: list, mean_score: float, std_score: float):
    """Create and log cross-validation scores visualization"""
    import tempfile
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    
    # Plot individual fold scores
    folds = range(1, len(cv_scores) + 1)
    plt.bar(folds, cv_scores, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Add mean line
    plt.axhline(y=mean_score, color='red', linestyle='--', 
                label=f'Mean: {mean_score:.4f} ± {std_score:.4f}')
    
    # Add error band
    plt.fill_between([0.5, len(cv_scores) + 0.5], 
                     mean_score - std_score, mean_score + std_score, 
                     alpha=0.2, color='red')
    
    plt.xlabel('CV Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Scores')
    plt.legend()
    plt.xticks(folds)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(f.name, "validation")
        plt.close()


def compare_models_summary():
    """Print a summary of model comparison capabilities"""
    print("📊 All models now use standardized validation:")
    print("  - Same chronological split (60/20/20) for train/val/test")
    print("  - Consistent cross-validation approach")
    print("  - Comparable metrics: cv_mean_acc ± cv_std_acc, hold-out validation & test accuracy")
    print("  - Per-class accuracies for Rock/Paper/Scissors")
    print("  - Check MLflow UI for direct comparison")

def print_validation_summary(model_name, cv_mean, cv_std, test_acc):
    """Print standardized validation summary for each model"""
    print(f"\n🎯 {model_name} Results Summary:")
    print(f"  Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Test accuracy:    {test_acc:.4f}")
    print(f"  Ready for MLflow comparison ✅")