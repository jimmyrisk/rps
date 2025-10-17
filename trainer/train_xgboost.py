import os
import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score

from trainer.base_model import BaseRPSModel
from trainer.model_defs.pyfunc_wrap import RpsUserNextMoveModel
from trainer.validation_utils import run_time_series_cv
# Import unified features module
from app.features import build_training_dataset


class XGBoostRPSModel(BaseRPSModel):
    """XGBoost model with consistent training interface"""
    
    def __init__(self):
        model_name = os.getenv("MLFLOW_MODEL_NAME", "rps_bot_xgboost")
        super().__init__(model_name, "xgboost")
        
    def get_hyperparameters(self):
        return {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": int(os.getenv("N_ESTIMATORS", "300")),
            "learning_rate": float(os.getenv("LEARNING_RATE", "0.05")),
            "max_depth": int(os.getenv("MAX_DEPTH", "3")),
            "min_child_weight": int(os.getenv("MIN_CHILD_WEIGHT", "2")),
            "subsample": float(os.getenv("SUBSAMPLE", "0.8")),
            "colsample_bytree": float(os.getenv("COLSAMPLE_BYTREE", "0.7")),
            "reg_lambda": float(os.getenv("REG_LAMBDA", "2.0")),  # L2 regularization
            "reg_alpha": float(os.getenv("REG_ALPHA", "0.1")),   # L1 regularization
            "gamma": float(os.getenv("GAMMA", "0.0")),
            "tree_method": "exact",  # Better for small datasets
            "early_stopping_rounds": 20,
        }
    
    def needs_feature_scaling(self):
        return True  # Tree models benefit from scaling for consistency
    
    def _xgb_cv_factory(self, X_fold_train, y_fold_train, X_fold_val, y_fold_val, **params):
        """Factory function for cross-validation"""
        # Remove early_stopping_rounds from model params
        model_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
        early_stopping = params.get('early_stopping_rounds', 20)
        
        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        val_pred = model.predict(X_fold_val)
        return model, val_pred
    
    def run_cross_validation(self, X_train, y_train, hyperparams, n_splits=3):
        """Run game-stratified cross-validation"""
        from trainer.validation_utils import game_stratified_cv_split
        import pandas as pd
        
        # Get game-stratified CV splits
        cv_splits = game_stratified_cv_split(
            pd.DataFrame(X_train),  # Convert numpy to DataFrame for consistency
            pd.Series(y_train),
            getattr(self, "training_events", self.train_val_events),  # Aligned events DataFrame
            n_splits=n_splits
        )
        
        cv_scores = []
        fold_weights = []  # Track number of events per fold for weighting
        fold_details = []
        
        print(f"🔄 Running {n_splits}-fold game-stratified cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"  Fold {fold + 1}/{n_splits}", end=" ")
            
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Train model using factory function
            model, val_pred = self._xgb_cv_factory(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val, **hyperparams
            )
            
            # Calculate validation accuracy
            val_acc = accuracy_score(y_fold_val, val_pred)
            cv_scores.append(val_acc)
            fold_weights.append(len(val_idx))
            fold_details.append({
                "fold": fold + 1,
                "accuracy": val_acc,
                "events": len(val_idx)
            })
            
            print(f"Acc: {val_acc:.4f}")
            mlflow.log_metric(f"cv_fold_{fold + 1}_acc", val_acc)
        
        # Calculate weighted mean and std dev
        weights = np.array(fold_weights) / sum(fold_weights)
        for detail, weight in zip(fold_details, weights):
            detail["weight"] = float(weight)
        self.cv_fold_details = fold_details
        mean_cv = np.average(cv_scores, weights=weights)
        
        # Weighted variance
        variance = np.average((np.array(cv_scores) - mean_cv)**2, weights=weights)
        std_cv = np.sqrt(variance)
        
        print(f"📊 Weighted CV Results: {mean_cv:.4f} ± {std_cv:.4f}")
        print(f"   Fold scores: {cv_scores}")
        print(f"   Fold weights: {weights.tolist()}")
        
        return cv_scores, mean_cv, std_cv
    
    def train_final_model(self, X_train, y_train, X_val, y_val, hyperparams):
        """Train XGBoost model with optional validation for early stopping"""
        # Remove early_stopping_rounds from model params
        model_params = {k: v for k, v in hyperparams.items() if k != 'early_stopping_rounds'}
        early_stopping = hyperparams.get('early_stopping_rounds', 20)
        
        # Check if validation data provided
        use_validation = X_val is not None and y_val is not None
        
        self.trained_model = xgb.XGBClassifier(**model_params)
        
        if use_validation:
            # Train with early stopping on validation set
            self.trained_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Log training and validation metrics
            train_pred = self.trained_model.predict(X_train)
            val_pred = self.trained_model.predict(X_val)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("val_acc", val_acc)
            
            print(f"  Training accuracy: {train_acc:.4f}")
            print(f"  Validation accuracy: {val_acc:.4f}")
        else:
            # No validation - train to max n_estimators without early stopping
            print("  No validation set - training to max n_estimators without early stopping")
            self.trained_model.fit(X_train, y_train, verbose=False)
            
            # Log training accuracy only
            train_pred = self.trained_model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            mlflow.log_metric("train_acc", train_acc)
            print(f"  Training accuracy: {train_acc:.4f}")
        
        # Create XGBoost-specific artifacts (only if we have feature importances)
        if use_validation:
            self.create_feature_importance_plot()
    
    def create_feature_importance_plot(self):
        """Create and log feature importance plot for XGBoost"""
        import tempfile
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if hasattr(self.trained_model, 'feature_importances_'):
            # Get real feature names from columns.txt
            feature_names = self.load_feature_names()
            
            # Ensure we have the right number of feature names
            if len(feature_names) != len(self.trained_model.feature_importances_):
                print(f"Warning: Feature names count ({len(feature_names)}) doesn't match model features ({len(self.trained_model.feature_importances_)})")
                feature_names = [f'feature_{i}' for i in range(len(self.trained_model.feature_importances_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.trained_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Plot top 20 features
            top_features = importance_df.tail(20)
            
            plt.figure(figsize=(12, 10))  # Make it slightly larger for longer feature names
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('XGBoost - Top 20 Feature Importances')
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, "plots")
                plt.close()
            
            # Save importance table
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                importance_df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, "feature_analysis")
    
    def predict(self, X):
        return self.trained_model.predict(X)
    
    def create_pyfunc_model(self, **kwargs):
        scaler = kwargs.get('scaler')
        scaler_params = None
        if scaler is not None:
            scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
            }

        return RpsUserNextMoveModel(
            sk_model=self.trained_model,  # Fix parameter name
            class_names=kwargs.get('class_names', ['rock', 'paper', 'scissors']),
            scaler_params=scaler_params,
        )


def main():
    model = XGBoostRPSModel()
    model.train()


if __name__ == "__main__":
    main()