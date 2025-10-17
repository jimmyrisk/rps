"""
Simplified Feedforward NN trainer using BaseRPSModel to reduce code duplication.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow

from trainer.base_model import BaseRPSModel
from trainer.validation_utils import run_time_series_cv


class FeedforwardNN(nn.Module):
    """Simple feedforward neural network"""
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_size = input_size
        
        # Hidden layers with BatchNorm and Dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


class FeedforwardPyFuncModel(mlflow.pyfunc.PythonModel):
    """PyFunc wrapper for feedforward neural network"""
    
    def __init__(self, state_dict, input_size, hidden_sizes, num_classes, 
                 scaler_params=None, class_names=("rock", "paper", "scissors"), lambda_danger=0.0):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.scaler_params = scaler_params
        self.lambda_danger = lambda_danger  # Danger penalty coefficient
        
        # Reconstruct model
        self.model = FeedforwardNN(input_size, hidden_sizes, num_classes)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Reconstruct scaler if provided
        if scaler_params:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_params['mean'])
            self.scaler.scale_ = np.array(scaler_params['scale'])
        else:
            self.scaler = None
    
    def apply_danger_penalty(self, probs):
        """
        Apply danger penalty to predictions.
        
        For each predicted label:
        - Danger(rock=0) = paper=1 (predicting paper when user plays rock makes bot lose)
        - Danger(paper=1) = scissors=2
        - Danger(scissors=2) = rock=0
        
        Reduces probability mass on danger class and renormalizes.
        """
        if self.lambda_danger <= 0:
            return probs
        
        # Danger mapping
        danger_map = np.array([1, 2, 0])
        
        # Get most likely prediction for each sample
        pred_labels = np.argmax(probs, axis=1)
        
        # Apply penalty
        modified_probs = probs.copy()
        for i in range(len(probs)):
            danger_class = danger_map[pred_labels[i]]
            modified_probs[i, danger_class] *= np.exp(-self.lambda_danger)
        
        # Renormalize
        modified_probs /= modified_probs.sum(axis=1, keepdims=True)
        
        return modified_probs
    
    def predict(self, context=None, model_input=None):
        # Handle both signatures: predict(context, model_input) and predict(model_input)
        if model_input is None and context is not None:
            model_input = context
            
        if model_input is None:
            raise ValueError("model_input is required")
            
        # Ensure input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
            
        # Prepare input
        X = model_input.values.astype(np.float32)
        
        # Apply scaling if scaler exists
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Convert to tensor and predict
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()  # Shape: (n_samples, n_classes)
        
        # Apply danger penalty if enabled
        if self.lambda_danger > 0:
            probs_batch = self.apply_danger_penalty(probs_batch)
        
        # Get predicted indices after danger penalty applied
        pred_idx_batch = np.argmax(probs_batch, axis=1)  # Shape: (n_samples,)
        
        # For batch predictions, return list of results
        if len(model_input) > 1:
            results = []
            for i in range(len(model_input)):
                probs = probs_batch[i]
                pred_idx = pred_idx_batch[i]
                pick = self.class_names[pred_idx]
                results.append({
                    "classes": self.class_names,
                    "probs": probs.tolist(),
                    "pick": pick
                })
            return results
        else:
            # Single prediction - return dict (backward compatibility)
            probs = probs_batch[0]
            pred_idx = pred_idx_batch[0]
            pick = self.class_names[pred_idx]
            return {
                "classes": self.class_names,
                "probs": probs.tolist(),
                "pick": pick
            }


class FeedforwardRPSModel(BaseRPSModel):
    """Feedforward NN model with consistent training interface"""
    
    def __init__(self):
        model_name = os.getenv("MLFLOW_MODEL_NAME", "rps_bot_feedforward")
        super().__init__(model_name, "feedforward")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def get_hyperparameters(self):
        hidden_sizes = os.getenv("HIDDEN_SIZES", "64,32").split(",")  # Input(50) → 64 → 32 → 3
        return {
            "hidden_sizes": [int(h) for h in hidden_sizes],
            "dropout_rate": float(os.getenv("DROPOUT_RATE", "0.3")),  # Config 8 optimal dropout
            "epochs": int(os.getenv("EPOCHS", "200")),  # Up to 200 epochs
            "batch_size": int(os.getenv("BATCH_SIZE", "64")),  # Config 8 optimal batch size
            "lr": float(os.getenv("LR", "3e-3")),  # Config 8 optimal learning rate
            "weight_decay": float(os.getenv("WEIGHT_DECAY", "1e-4")),  # Config 8 optimal L2
            "patience": int(os.getenv("PATIENCE", "15")),  # Early stopping patience 10-20
            "reduce_lr_patience": int(os.getenv("REDUCE_LR_PATIENCE", "7")),  # Reduce-on-plateau
            "lambda_danger": float(os.getenv("LAMBDA_DANGER", "0.0")),  # Danger penalty (default: no penalty)
        }
    
    def needs_feature_scaling(self):
        return True  # Neural networks need feature scaling
    
    def _torch_cv_factory(self, X_fold_train, y_fold_train, X_fold_val, y_fold_val, **params):
        """Factory function for cross-validation"""
        input_size = X_fold_train.shape[1]
        model = FeedforwardNN(input_size, params['hidden_sizes'], 3, params['dropout_rate']).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=params['lr'], 
                                   weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        Xtr = torch.tensor(X_fold_train, dtype=torch.float32).to(self.device)
        ytr = torch.tensor(y_fold_train, dtype=torch.long).to(self.device)
        Xva = torch.tensor(X_fold_val, dtype=torch.float32).to(self.device)
        yva = torch.tensor(y_fold_val, dtype=torch.long).to(self.device)
        
        # Quick training (reduced epochs for CV)
        cv_epochs = min(30, params['epochs'] // 3)
        batch_size = params['batch_size']
        
        model.train()
        for epoch in range(cv_epochs):
            # Shuffle training data
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                batch_idx = perm[i:i+batch_size]
                xb, yb = Xtr[batch_idx], ytr[batch_idx]
                
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(Xva), dim=1).cpu().numpy()
        
        return model, pred
    
    def run_cross_validation(self, X_train, y_train, hyperparams, n_splits=3):
        """Run game-stratified cross-validation"""
        from trainer.validation_utils import game_stratified_cv_split
        
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
            model, val_pred = self._torch_cv_factory(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val, **hyperparams
            )
            
            # Calculate validation accuracy
            from sklearn.metrics import accuracy_score
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
        """Train model with optional validation for early stopping"""
        input_size = X_train.shape[1]
        self.trained_model = FeedforwardNN(
            input_size, 
            hyperparams['hidden_sizes'], 
            3, 
            hyperparams['dropout_rate']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.trained_model.parameters(),
                                   lr=hyperparams['lr'],
                                   weight_decay=hyperparams['weight_decay'])
        
        criterion = nn.CrossEntropyLoss()
        
        # Danger penalty configuration
        lambda_danger = hyperparams.get('lambda_danger', 0.0)
        # Danger mapping: [1, 2, 0] means danger(rock=0)=paper=1, danger(paper=1)=scissors=2, danger(scissors=2)=rock=0
        danger_map = torch.tensor([1, 2, 0], dtype=torch.long, device=self.device)
        
        def compute_loss_with_danger_penalty(logits, targets):
            """Compute cross-entropy loss with optional danger penalty"""
            ce_loss = criterion(logits, targets)
            
            if lambda_danger > 0:
                # Softmax probabilities
                probs = torch.softmax(logits, dim=1)
                
                # Get danger classes for each target
                danger_classes = danger_map[targets]
                
                # Penalty: mean probability assigned to danger class
                batch_indices = torch.arange(targets.size(0), device=self.device)
                danger_probs = probs[batch_indices, danger_classes]
                penalty = danger_probs.mean()
                
                return ce_loss + lambda_danger * penalty
            else:
                return ce_loss
        
        # Convert training data to tensors
        Xtr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        ytr = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        # Check if validation data provided (for early stopping)
        use_validation = X_val is not None and y_val is not None
        
        if use_validation:
            # Learning rate scheduler: reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=hyperparams['reduce_lr_patience'], 
                min_lr=1e-6
            )
            
            Xva = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            yva = torch.tensor(y_val, dtype=torch.long).to(self.device)
            
            # Early stopping and training history
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        else:
            # No validation - train to max epochs
            print("  No validation set - training to max epochs without early stopping")
            training_history = {'epoch': [], 'train_loss': [], 'train_acc': []}
        
        # Training loop
        self.trained_model.train()
        for epoch in range(1, hyperparams['epochs'] + 1):
            # Training
            perm = torch.randperm(len(Xtr))
            epoch_loss = 0
            for i in range(0, len(Xtr), hyperparams['batch_size']):
                batch_idx = perm[i:i+hyperparams['batch_size']]
                xb, yb = Xtr[batch_idx], ytr[batch_idx]
                
                optimizer.zero_grad()
                logits = self.trained_model(xb)
                loss = compute_loss_with_danger_penalty(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation every 10 epochs (or just logging if no validation)
            if epoch % 10 == 0:
                self.trained_model.eval()
                with torch.no_grad():
                    train_pred = torch.argmax(self.trained_model(Xtr), dim=1)
                    train_acc = (train_pred == ytr).float().mean().item()
                    
                    if use_validation:
                        val_logits = self.trained_model(Xva)
                        val_loss = compute_loss_with_danger_penalty(val_logits, yva).item()
                        val_pred = torch.argmax(val_logits, dim=1)
                        val_acc = (val_pred == yva).float().mean().item()
                        
                        mlflow.log_metrics({
                            "train_loss": epoch_loss / (len(Xtr) // hyperparams['batch_size']),
                            "val_loss": val_loss,
                            "train_acc": train_acc,
                            "val_acc": val_acc
                        }, step=epoch)
                        
                        # Store history for plotting
                        training_history['epoch'].append(epoch)
                        training_history['train_loss'].append(epoch_loss / (len(Xtr) // hyperparams['batch_size']))
                        training_history['val_loss'].append(val_loss)
                        training_history['train_acc'].append(train_acc)
                        training_history['val_acc'].append(val_acc)
                        
                        # Learning rate scheduler step
                        scheduler.step(val_loss)
                        
                        # Early stopping
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= hyperparams['patience']:
                                print(f"  Early stopping at epoch {epoch}")
                                break
                    else:
                        # No validation - just log training metrics
                        mlflow.log_metrics({
                            "train_loss": epoch_loss / (len(Xtr) // hyperparams['batch_size']),
                            "train_acc": train_acc
                        }, step=epoch)
                        
                        training_history['epoch'].append(epoch)
                        training_history['train_loss'].append(epoch_loss / (len(Xtr) // hyperparams['batch_size']))
                        training_history['train_acc'].append(train_acc)
                
                self.trained_model.train()
        
        self.trained_model.eval()
        
        # Create training progress plot (if we have history)
        if len(training_history['epoch']) > 0:
            self.create_training_plot(training_history, use_validation)

        # For production training (no validation set), log feature sensitivity artifact
        if not use_validation:
            self.create_feature_sensitivity_plot()
    
    def create_training_plot(self, history, use_validation=True):
        """Create and log training progress visualization"""
        import tempfile
        import matplotlib.pyplot as plt
        
        if use_validation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Loss plot
            ax1.plot(history['epoch'], history['train_loss'], label='Training Loss', color='blue')
            ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2.plot(history['epoch'], history['train_acc'], label='Training Accuracy', color='blue')
            ax2.plot(history['epoch'], history['val_acc'], label='Validation Accuracy', color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # No validation - single plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Loss plot
            ax1.plot(history['epoch'], history['train_loss'], label='Training Loss', color='blue')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss (No Validation)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2.plot(history['epoch'], history['train_acc'], label='Training Accuracy', color='blue')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Accuracy (No Validation)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "training")
            plt.close()
    
    def create_feature_sensitivity_plot(self):
        """Approximate feature importance using first layer weight magnitudes"""
        import tempfile
        import matplotlib.pyplot as plt

        if not hasattr(self.trained_model, 'network'):
            return

        first_layer = self.trained_model.network[0]
        if not isinstance(first_layer, nn.Linear):
            return

        weights = first_layer.weight.detach().cpu().numpy()
        importance = np.mean(np.abs(weights), axis=0)

        feature_names = getattr(self, 'feature_names', None)
        if not feature_names or len(feature_names) != len(importance):
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        top_n = df.head(20)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_n)), top_n['importance'][::-1])
        plt.yticks(range(len(top_n)), top_n['feature'][::-1])
        plt.xlabel('Mean |weight| across hidden units')
        plt.title('Feedforward NN - Top 20 Feature Sensitivities')
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "feature_analysis")
            plt.close()

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "feature_analysis")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = torch.argmax(self.trained_model(X_tensor), dim=1)
        return pred.cpu().numpy()
    
    def create_pyfunc_model(self, **kwargs):
        # Prepare model state for serving
        state_dict = {k: v.cpu() for k, v in self.trained_model.state_dict().items()}
        
        # Get scaler parameters if provided
        scaler_params = None
        scaler = kwargs.get('scaler')
        if scaler:
            scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        
        # Get danger penalty from current hyperparameters
        lambda_danger = self._current_hyperparams.get('lambda_danger', 0.0)
        
        return FeedforwardPyFuncModel(
            state_dict=state_dict,
            input_size=self.trained_model.input_size,
            hidden_sizes=self.trained_model.hidden_sizes,
            num_classes=self.trained_model.num_classes,
            scaler_params=scaler_params,
            class_names=kwargs.get('class_names', ['rock', 'paper', 'scissors']),
            lambda_danger=lambda_danger
        )


def run_feedforward_hyperparameter_sweep():
    """Run feedforward neural network with different hyperparameter combinations"""
    import itertools
    
    # Define hyperparameter ranges
    hidden_configs = [
        [32],      # Single layer, small
        [64],      # Single layer, medium  
        [128],     # Single layer, large
        [64, 32],  # Two layers, current config
        [128, 64], # Two layers, larger
    ]
    
    dropout_rates = [0.2, 0.3, 0.5]
    weight_decays = [1e-5, 1e-4, 1e-3]
    learning_rates = [3e-4, 1e-3, 3e-3, 5e-3, 1e-2]  # Extended higher LR range
    batch_sizes = [16, 32, 64, 128]  # Extended higher batch size range
    
    # Extended configurations focusing on higher LR and batch sizes
    configs = [
        # Previous best configuration and extensions
        {"hidden_sizes": [64, 32], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 3e-3, "batch_size": 64},  # Previous best
        
        # Higher learning rates with best architecture
        {"hidden_sizes": [64, 32], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 5e-3, "batch_size": 64},
        {"hidden_sizes": [64, 32], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 1e-2, "batch_size": 64},
        
        # Higher batch sizes with best LR
        {"hidden_sizes": [64, 32], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 3e-3, "batch_size": 128},
        {"hidden_sizes": [64, 32], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 5e-3, "batch_size": 128},
        
        # Test with single layer architectures at higher LR/batch
        {"hidden_sizes": [64], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 5e-3, "batch_size": 64},
        {"hidden_sizes": [32], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 5e-3, "batch_size": 64},
        
        # Lower regularization with higher LR (might need less regularization with better optimization)
        {"hidden_sizes": [64, 32], "dropout_rate": 0.2, "weight_decay": 1e-5, "lr": 5e-3, "batch_size": 64},
        {"hidden_sizes": [64, 32], "dropout_rate": 0.2, "weight_decay": 1e-5, "lr": 3e-3, "batch_size": 128},
        
        # Test larger architecture with better optimization
        {"hidden_sizes": [128, 64], "dropout_rate": 0.3, "weight_decay": 1e-4, "lr": 5e-3, "batch_size": 64},
        {"hidden_sizes": [128, 64], "dropout_rate": 0.2, "weight_decay": 1e-5, "lr": 3e-3, "batch_size": 128},
    ]
    
    results = []
    
    print("🔍 Running Feedforward Neural Network hyperparameter sweep...")
    print(f"Testing {len(configs)} different configurations...")
    
    # Ensure no active MLflow run
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Configuration {i}/{len(configs)} ---")
        print(f"Hidden: {config['hidden_sizes']}, Dropout: {config['dropout_rate']}, "
              f"L2: {config['weight_decay']}, LR: {config['lr']}, Batch: {config['batch_size']}")
        
        # Set environment variables for this run
        os.environ["HIDDEN_SIZES"] = ",".join(map(str, config['hidden_sizes']))
        os.environ["DROPOUT_RATE"] = str(config['dropout_rate'])
        os.environ["WEIGHT_DECAY"] = str(config['weight_decay'])
        os.environ["LR"] = str(config['lr'])
        os.environ["BATCH_SIZE"] = str(config['batch_size'])
        
        # Create model instance
        model = FeedforwardRPSModel()
        
        # Create experiment name for sweep
        original_experiment = model.experiment
        model.experiment = f"{original_experiment}_ffnn_sweep"
        
        # Train the model (it will handle its own MLflow run)
        metrics = model.train()
        if metrics:
            result = {
                'config_id': i,
                'hidden_sizes': config['hidden_sizes'],
                'dropout_rate': config['dropout_rate'],
                'weight_decay': config['weight_decay'],
                'lr': config['lr'],
                'batch_size': config['batch_size'],
                'cv_mean_acc': metrics.get('cv_mean_acc', 0),
                'cv_std_acc': metrics.get('cv_std_acc', 0),
                'test_accuracy': metrics.get('test_accuracy', 0)
            }
            results.append(result)
            
            print(f"✅ Config {i}: CV={metrics.get('cv_mean_acc', 0):.4f}±{metrics.get('cv_std_acc', 0):.4f}, "
                  f"Test={metrics.get('test_accuracy', 0):.4f}")
    
    # Summary of results
    print(f"\n📊 Feedforward Neural Network Sweep Summary:")
    print("-" * 80)
    print(f"{'ID':<3} {'Hidden':<12} {'Drop':<5} {'L2':<6} {'LR':<6} {'Batch':<5} {'CV Acc':<12} {'Test Acc':<8}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
        hidden_str = "x".join(map(str, result['hidden_sizes']))
        print(f"{result['config_id']:<3} {hidden_str:<12} {result['dropout_rate']:<5} "
              f"{result['weight_decay']:<6} {result['lr']:<6} {result['batch_size']:<5} "
              f"{result['cv_mean_acc']:.3f}±{result['cv_std_acc']:.3f}  {result['test_accuracy']:.3f}")
    
    # Find best performing configuration
    if results:
        best_result = max(results, key=lambda x: x['test_accuracy'])
        print(f"\n🏆 Best Configuration (Test Accuracy):")
        print(f"   Hidden: {best_result['hidden_sizes']}")
        print(f"   Dropout: {best_result['dropout_rate']}")
        print(f"   L2 Weight Decay: {best_result['weight_decay']}")
        print(f"   Learning Rate: {best_result['lr']}")
        print(f"   Batch Size: {best_result['batch_size']}")
        print(f"   Performance: {best_result['cv_mean_acc']:.4f}±{best_result['cv_std_acc']:.4f} CV, "
              f"{best_result['test_accuracy']:.4f} Test")
    
    return results


def main():
    import sys
    
    # Check if user wants to run hyperparameter sweep
    if len(sys.argv) > 1 and sys.argv[1] == "--hyperparam-sweep":
        run_feedforward_hyperparameter_sweep()
    else:
        model = FeedforwardRPSModel()
        model.train()


if __name__ == "__main__":
    main()