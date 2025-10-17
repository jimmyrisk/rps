"""
Simplified Multinomial Logistic Regression trainer using BaseRPSModel to reduce code duplication.
"""
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow

from trainer.base_model import BaseRPSModel
from trainer.model_defs.pyfunc_wrap import RpsUserNextMoveModel, SoftmaxReg
from trainer.validation_utils import run_time_series_cv


class MNLogitRPSModel(BaseRPSModel):
    """Multinomial Logistic Regression model with consistent training interface"""
    
    def __init__(self):
        model_name = os.getenv("MLFLOW_MODEL_NAME", "rps_bot_mnlogit")
        super().__init__(model_name, "multinomial_logistic")
        
        # Device
        self.device = torch.device("cpu")
        
    def get_hyperparameters(self):
        return {
            "epochs": int(os.getenv("EPOCHS", "100")),  # Increased for proper convergence
            "batch_size": int(os.getenv("BATCH_SIZE", "64")),
            "lr": float(os.getenv("LR", "5e-3")),  # Larger learning rate is fine with L2 regularization
            "l2_lambda": float(os.getenv("L2_LAMBDA", "1.0")),  # Manual L2 regularization
            "use_class_weights": bool(os.getenv("USE_CLASS_WEIGHTS", "True")),  # Handle imbalanced classes
            "lambda_danger": float(os.getenv("LAMBDA_DANGER", "0.0")),  # Danger penalty (default: no penalty)
            "patience": int(os.getenv("PATIENCE", "25")),  # Early stopping patience on validation
            "reduce_lr_patience": int(os.getenv("REDUCE_LR_PATIENCE", "7")),  # Reduce-on-plateau patience
        }
    
    def needs_feature_scaling(self):
        return True  # Logistic regression SHOULD use feature scaling for better convergence
    
    def _torch_cv_factory(self, X_fold_train, y_fold_train, X_fold_val, y_fold_val, **params):
        """Factory function for cross-validation"""
        d_in, d_out = X_fold_train.shape[1], 3  # 3 moves: rock, paper, scissors
        model = SoftmaxReg(d_in, d_out).to(self.device)
        
        # Use Adam optimizer without weight_decay (we'll add manual L2 regularization)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        
        # Calculate class weights for imbalanced classes if enabled
        criterion = self._get_loss_function(y_fold_train, params)
        
        # Convert to tensors
        Xtr = torch.tensor(X_fold_train, dtype=torch.float32)
        ytr = torch.tensor(y_fold_train, dtype=torch.long)
        Xva = torch.tensor(X_fold_val, dtype=torch.float32)
        yva = torch.tensor(y_fold_val, dtype=torch.long)
        
        # Quick training (reduced epochs for CV)
        cv_epochs = min(20, params['epochs'] // 2)
        batch_size = params['batch_size']
        
        model.train()
        for epoch in range(cv_epochs):
            idx = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                sel = idx[i:i+batch_size]
                xb, yb = Xtr[sel], ytr[sel]
                
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                
                # Add manual L2 regularization
                l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += params['l2_lambda'] * l2_loss
                
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(Xva), dim=1).cpu().numpy()
        
        return model, pred
    
    def _get_loss_function(self, y_train, params):
        """Get loss function with optional class weighting and danger penalty"""
        import torch.nn.functional as F
        
        # Optional class weighting (unchanged behavior)
        weight = None
        if params.get('use_class_weights', True):
            # Calculate class weights for imbalanced classes
            from sklearn.utils.class_weight import compute_class_weight
            import numpy as np
            
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        
        ce = nn.CrossEntropyLoss(weight=weight)
        lam = float(params.get('lambda_danger', 0.0))
        
        # 0=rock, 1=paper, 2=scissors
        # Danger(y) = class that BEATS y (predicting it makes your counter-move lose):
        # Danger(R)=P, Danger(P)=S, Danger(S)=R
        danger = torch.tensor([1, 2, 0], dtype=torch.long, device=self.device)
        
        def loss_fn(logits, y):
            # Standard negative log-likelihood
            loss = ce(logits, y)
            
            # Penalty on probability mass assigned to the losing direction
            if lam > 0:
                q = F.softmax(logits, dim=-1)
                idx = torch.arange(y.size(0), device=logits.device)
                pen = q[idx, danger[y]].mean()
                return loss + lam * pen
            else:
                return loss
        
        return loss_fn
    
    def run_cross_validation(self, X_train, y_train, hyperparams, n_splits=3):
        """Run game-stratified cross-validation"""
        from sklearn.metrics import accuracy_score
        from trainer.validation_utils import game_stratified_cv_split

        cv_splits = game_stratified_cv_split(
            pd.DataFrame(X_train),
            pd.Series(y_train),
            getattr(self, "training_events", self.train_val_events),
            n_splits=n_splits
        )

        fold_accs = []
        fold_weights = []
        fold_details = []

        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits, start=1):
            print(f"  Fold {fold_idx}/{n_splits}: {len(train_indices)} train, {len(val_indices)} val events")

            X_fold_train = X_train[train_indices]
            y_fold_train = y_train[train_indices]
            X_fold_val = X_train[val_indices]
            y_fold_val = y_train[val_indices]

            model, preds = self._torch_cv_factory(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val, **hyperparams
            )

            fold_acc = accuracy_score(y_fold_val, preds)
            fold_accs.append(fold_acc)
            fold_weights.append(len(val_indices))
            fold_details.append({
                "fold": fold_idx,
                "accuracy": fold_acc,
                "events": len(val_indices)
            })

            print(f"    Fold {fold_idx} accuracy: {fold_acc:.4f}")
            mlflow.log_metric(f"cv_fold_{fold_idx}_acc", fold_acc)

        weights_arr = np.array(fold_weights, dtype=float)
        weights_arr /= weights_arr.sum()
        for detail, weight in zip(fold_details, weights_arr):
            detail["weight"] = float(weight)
        self.cv_fold_details = fold_details

        cv_mean = np.average(fold_accs, weights=weights_arr)
        cv_std = np.sqrt(np.average((np.array(fold_accs) - cv_mean)**2, weights=weights_arr))

        print(f"\nWeighted CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Fold weights: {weights_arr.tolist()}")

        return fold_accs, cv_mean, cv_std
    
    def train_final_model(self, X_train, y_train, X_val, y_val, hyperparams):
        """Train multinomial logistic model with optional validation"""
        d_in, d_out = X_train.shape[1], 3
        self.trained_model = SoftmaxReg(d_in, d_out).to(self.device)
        
        # Use Adam optimizer without weight_decay (manual L2 regularization)
        optimizer = torch.optim.Adam(self.trained_model.parameters(), lr=hyperparams['lr'])
        
        # Get loss function with optional class weighting
        criterion = self._get_loss_function(y_train, hyperparams)
        
        # Check if validation data provided
        use_validation = X_val is not None and y_val is not None
        
        # Convert training data to tensors
        Xtr = torch.tensor(X_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.long)
        
        # Validation data (if provided)
        if use_validation:
            Xva = torch.tensor(X_val, dtype=torch.float32)
            yva = torch.tensor(y_val, dtype=torch.long)
            patience = hyperparams.get('patience', 25)
            lr_patience = hyperparams.get('reduce_lr_patience', 7)
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_state = copy.deepcopy(self.trained_model.state_dict())
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=lr_patience,
                min_lr=1e-6
            )

            def acc(x, y):
                with torch.no_grad():
                    pred = torch.argmax(self.trained_model(x), dim=1)
                    return float((pred == y).float().mean().item())
        else:
            print("  No validation set - training without validation monitoring")
        
        # Training loop
        self.trained_model.train()
        for ep in range(1, hyperparams['epochs'] + 1):
            idx = torch.randperm(len(Xtr))
            epoch_loss = 0.0
            batch_count = 0
            for i in range(0, len(Xtr), hyperparams['batch_size']):
                sel = idx[i:i+hyperparams['batch_size']]
                xb, yb = Xtr[sel], ytr[sel]
                optimizer.zero_grad()
                logits = self.trained_model(xb)
                loss = criterion(logits, yb)
                
                # Add manual L2 regularization
                l2_loss = sum(p.pow(2.0).sum() for p in self.trained_model.parameters())
                loss += hyperparams['l2_lambda'] * l2_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
            
            if use_validation:
                self.trained_model.eval()
                with torch.no_grad():
                    train_logits = self.trained_model(Xtr)
                    train_pred = torch.argmax(train_logits, dim=1)
                    train_acc = float((train_pred == ytr).float().mean().item())
                    train_loss_avg = epoch_loss / max(1, batch_count)

                    val_logits = self.trained_model(Xva)
                    val_loss = criterion(val_logits, yva)
                    l2_val = sum(p.pow(2.0).sum() for p in self.trained_model.parameters())
                    val_loss = float(val_loss + hyperparams['l2_lambda'] * l2_val)
                    val_pred = torch.argmax(val_logits, dim=1)
                    val_acc = float((val_pred == yva).float().mean().item())

                if ep % 10 == 0:
                    mlflow.log_metrics({
                        "train_loss": train_loss_avg,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    }, step=ep)
                    print(f"  Epoch {ep}/{hyperparams['epochs']}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = copy.deepcopy(self.trained_model.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"  Early stopping at epoch {ep}")
                        break

                scheduler.step(val_loss)
                self.trained_model.train()
        
        # Final evaluation
        self.trained_model.eval()
        
        if use_validation:
            if 'best_state' in locals() and best_state is not None:
                self.trained_model.load_state_dict(best_state)
            train_acc = acc(Xtr, ytr)
            val_acc = acc(Xva, yva)
            mlflow.log_metrics({
                "train_acc_final": train_acc,
                "val_acc_final": val_acc
            })
            print(f"  Final training accuracy: {train_acc:.4f}")
            print(f"  Final validation accuracy: {val_acc:.4f}")
        else:
            # Just compute training accuracy
            with torch.no_grad():
                pred = torch.argmax(self.trained_model(Xtr), dim=1)
                train_acc = float((pred == ytr).float().mean().item())
            mlflow.log_metric("train_acc", train_acc)
            print(f"  Final training accuracy: {train_acc:.4f}")
    
    def create_coefficients_plot(self):
        """Create visualization of learned coefficients with proper feature names"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        # Get model weights (first layer is the only layer for logistic regression)
        weights = self.trained_model.linear.weight.detach().cpu().numpy()  # Shape: (3, n_features)
        
        # Load feature names
        feature_names = self.load_feature_names()
        
        # Ensure we have the right number of feature names
        if len(feature_names) != weights.shape[1]:
            print(f"Warning: Feature names count ({len(feature_names)}) doesn't match model features ({weights.shape[1]})")
            feature_names = [f"feature_{i}" for i in range(weights.shape[1])]
        
        # Create coefficient plot for each class
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        class_names = ['rock', 'paper', 'scissors']
        
        for i, (ax, class_name) in enumerate(zip(axes, class_names)):
            coeff_df = {
                'Feature': feature_names,
                'Coefficient': weights[i, :]
            }
            coeff_df = pd.DataFrame(coeff_df).sort_values('Coefficient', key=abs, ascending=False)
            
            # Plot top 15 most important coefficients
            top_15 = coeff_df.head(15)
            colors = ['red' if x < 0 else 'blue' for x in top_15['Coefficient']]
            
            ax.barh(range(len(top_15)), top_15['Coefficient'], color=colors)
            ax.set_yticks(range(len(top_15)))
            ax.set_yticklabels(top_15['Feature'])
            ax.set_xlabel('Coefficient Value')
            ax.set_title(f'{class_name.title()} - Top 15 Features')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multinomial_coefficients.png', dpi=150, bbox_inches='tight')
        
        # Log to MLflow
        mlflow.log_artifact('multinomial_coefficients.png')
        plt.close()
        
        # Also create a summary table of all coefficients
        coeff_summary = pd.DataFrame({
            'Feature': feature_names,
            'Rock_Coeff': weights[0, :],
            'Paper_Coeff': weights[1, :], 
            'Scissors_Coeff': weights[2, :]
        })
        coeff_summary['Max_Abs_Coeff'] = coeff_summary[['Rock_Coeff', 'Paper_Coeff', 'Scissors_Coeff']].abs().max(axis=1)
        coeff_summary = coeff_summary.sort_values('Max_Abs_Coeff', ascending=False)
        
        # Save coefficient table
        coeff_summary.to_csv('multinomial_coefficients_table.csv', index=False)
        mlflow.log_artifact('multinomial_coefficients_table.csv')
        
        print("📊 Created coefficient visualizations and saved to MLflow")
    
    def create_artifacts(self, y_test: np.ndarray, y_pred: np.ndarray, conf_matrix: np.ndarray):
        """Override base create_artifacts to add coefficient visualization"""
        # Call parent method for standard artifacts
        super().create_artifacts(y_test, y_pred, conf_matrix)
        
        # Add our custom coefficient plot
        self.create_coefficients_plot()
    
    def predict(self, X):
        Xte = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.trained_model(Xte), dim=1).numpy()
    
    def create_pyfunc_model(self, **kwargs):
        # Prepare model state for serving
        state_dict = {k: v.cpu() for k, v in self.trained_model.state_dict().items()}
        d_in = list(self.trained_model.parameters())[0].shape[1]
        
        scaler = kwargs.get('scaler')
        scaler_params = None
        if scaler is not None:
            scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
            }

        return RpsUserNextMoveModel(
            torch_state=state_dict,
            d_in=d_in,
            class_names=kwargs.get('class_names', ['rock', 'paper', 'scissors']),
            scaler_params=scaler_params,
        )


def run_l2_regularization_sweep():
    """Run multinomial logistic regression with different L2 regularization values"""
    l2_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]  # Extended range including higher regularization
    results = []
    
    print("🔍 Running L2 regularization sweep...")
    print(f"Testing L2 values: {l2_values}")
    
    # Ensure no active MLflow run
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    for l2_val in l2_values:
        print(f"\n--- Training with L2 = {l2_val} ---")
        
        # Set environment variable for this run
        os.environ["L2_LAMBDA"] = str(l2_val)
        
        # Create model instance
        model = MNLogitRPSModel()
        
        # Create experiment name with L2 value
        original_experiment = model.experiment
        model.experiment = f"{original_experiment}_l2_sweep"
        
        # Train the model (it will handle its own MLflow run)
        metrics = model.train()
        if metrics:
            results.append({
                'l2_lambda': l2_val,
                'cv_mean_acc': metrics.get('cv_mean_acc', 0),
                'cv_std_acc': metrics.get('cv_std_acc', 0),
                'test_accuracy': metrics.get('test_accuracy', 0)
            })
            
            print(f"✅ L2={l2_val}: CV={metrics.get('cv_mean_acc', 0):.4f}±{metrics.get('cv_std_acc', 0):.4f}, Test={metrics.get('test_accuracy', 0):.4f}")
    
    # Summary of results
    print(f"\n📊 L2 Regularization Sweep Summary:")
    print("-" * 60)
    for result in results:
        print(f"L2={result['l2_lambda']:6.3f}: CV={result['cv_mean_acc']:.4f}±{result['cv_std_acc']:.4f}, Test={result['test_accuracy']:.4f}")
    
    # Find best performing L2 value
    if results:
        best_result = max(results, key=lambda x: x['cv_mean_acc'])
        print(f"\n🏆 Best L2 regularization: {best_result['l2_lambda']} (CV: {best_result['cv_mean_acc']:.4f})")
    
    return results


def main():
    import sys
    
    # Check if user wants to run L2 regularization sweep
    if len(sys.argv) > 1 and sys.argv[1] == "--l2-sweep":
        run_l2_regularization_sweep()
    else:
        model = MNLogitRPSModel()
        model.train()


if __name__ == "__main__":
    main()