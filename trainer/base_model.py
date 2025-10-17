"""
Base model class to reduce code duplication across all trainers.
Handles common patterns: data loading, MLflow setup, evaluation, model registration.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from app.config import get_mlflow_tracking_uri, get_mlflow_production_alias
from app.features import build_training_dataset, MOVES
from trainer.validation_utils import (
    time_series_split, prepare_features, encode_labels,
    log_common_params, compare_models_summary
)


class BaseRPSModel(ABC):
    """Base class for all RPS models with common training patterns"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        
        # Common environment variables
        self.lookback = int(os.getenv("LOOKBACK", "3"))
        self.min_sup_rows = int(os.getenv("MIN_SUP_ROWS", "200"))
        
        # Handle both absolute and relative data paths
        data_path_env = os.getenv("DATA_PATH")
        if data_path_env:
            self.data_db = Path(data_path_env) / "rps.db"
        else:
            self.data_db = Path("local/rps.db")  # Default relative path
            
        self.experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "rps-bot")
        self.promote_stage = os.getenv("PROMOTE_STAGE", "Production")
        self.promote_alias = os.getenv("PROMOTE_ALIAS", get_mlflow_production_alias())

        # Optional dataset filters
        player_like_env = os.getenv("TRAINING_PLAYER_NAME_LIKE", "")
        self.training_player_like = [
            pattern.strip()
            for pattern in player_like_env.split(",")
            if pattern.strip()
        ]

        bot_policy_env = os.getenv("TRAINING_BOT_POLICY_IN", "")
        self.training_bot_policy_in = [
            policy.strip()
            for policy in bot_policy_env.split(",")
            if policy.strip()
        ]

        easy_mode_env = os.getenv("TRAINING_EASY_MODE_ONLY")
        self.training_easy_mode_only = None
        if easy_mode_env is not None:
            self.training_easy_mode_only = easy_mode_env.strip().lower() in {"1", "true", "yes"}
        
        # Local model storage configuration
        # Use local/models for development, /data/models in K8s
        default_models_dir = "/data/models" if Path("/data").exists() else "local/models"
        self.local_models_dir = Path(os.getenv("LOCAL_MODELS_DIR", default_models_dir))
        self.enable_local_storage = os.getenv("ENABLE_LOCAL_MODEL_STORAGE", "true").lower() == "true"
        
        if self.enable_local_storage:
            try:
                self.local_models_dir.mkdir(parents=True, exist_ok=True)
                print(f"💾 Local model storage enabled: {self.local_models_dir}")
            except Exception as e:
                print(f"⚠️  Could not create local models directory: {e}")
                self.enable_local_storage = False
        
        # Will be set during training
        self.X = None
        self.y = None
        self.trained_model = None
        self.feature_names = []
        self.cv_fold_details = None
        self.training_events = None
        self._current_hyperparams = {}
        self._active_config_id = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load events and games data from SQLite database, excluding test games and gambit rounds.

        Filters:
        1. Exclude test games (is_test=0)
        2. Exclude gambit rounds (step_no > 3)
        3. Limit to last 3000 usable rows chronologically
        """
        from app.config import get_training_data_since_date
        
        training_since = get_training_data_since_date()
        con = sqlite3.connect(self.data_db)
        
        # Load events joined with games, filtering:
        # 1. Test games (is_test=0)
        # 2. Gambit rounds (step_no > 3) - ML predictions start at round 4
        # Order chronologically by game creation time, then by step within game
        event_conditions = [
            "g.is_test = 0",
            "(g.created_ts_utc IS NULL OR g.created_ts_utc >= ?)",
            "e.step_no > 3",
        ]
        event_params = [training_since]

        if self.training_player_like:
            like_clause = " OR ".join("g.player_name LIKE ?" for _ in self.training_player_like)
            event_conditions.append(f"({like_clause})")
            event_params.extend(self.training_player_like)

        if self.training_bot_policy_in:
            placeholders = ",".join("?" for _ in self.training_bot_policy_in)
            event_conditions.append(f"g.bot_policy IN ({placeholders})")
            event_params.extend(self.training_bot_policy_in)

        if self.training_easy_mode_only is True:
            event_conditions.append("g.easy_mode = 1")
        elif self.training_easy_mode_only is False:
            event_conditions.append("g.easy_mode = 0")

        event_where = " AND ".join(event_conditions)

        ev = pd.read_sql_query(
            f"""SELECT e.*, g.easy_mode, g.created_ts_utc
               FROM events e
               INNER JOIN games g ON e.game_id = g.id
               WHERE {event_where}
               ORDER BY g.created_ts_utc ASC, e.game_id ASC, e.step_no ASC""",
            con,
            params=tuple(event_params)
        )
        
        total_events = len(ev)
        
        # Limit to last 3000 usable rows (or all if less than 3000)
        max_usable_rows = int(os.getenv("MAX_USABLE_ROWS", "3000"))
        if len(ev) > max_usable_rows:
            print(f"📊 Limiting from {len(ev)} to last {max_usable_rows} usable events (chronologically sorted)")
            ev = ev.tail(max_usable_rows).reset_index(drop=True)
        else:
            print(f"📊 Using all {len(ev)} usable events (after filtering gambits & test games)")
        
        try:
            # Load games with test exclusion and time filtering
            game_conditions = [
                "is_test = 0",
                "(created_ts_utc IS NULL OR created_ts_utc >= ?)",
            ]
            game_params = [training_since]

            if self.training_player_like:
                like_clause = " OR ".join("player_name LIKE ?" for _ in self.training_player_like)
                game_conditions.append(f"({like_clause})")
                game_params.extend(self.training_player_like)

            if self.training_bot_policy_in:
                placeholders = ",".join("?" for _ in self.training_bot_policy_in)
                game_conditions.append(f"bot_policy IN ({placeholders})")
                game_params.extend(self.training_bot_policy_in)

            if self.training_easy_mode_only is True:
                game_conditions.append("easy_mode = 1")
            elif self.training_easy_mode_only is False:
                game_conditions.append("easy_mode = 0")

            game_where = " AND ".join(game_conditions)

            gm = pd.read_sql_query(
                f"""SELECT id, rock_pts, paper_pts, scissors_pts 
                   FROM games 
                   WHERE {game_where}""",
                con,
                params=tuple(game_params)
            )
        except Exception:
            gm = pd.DataFrame()
        
        con.close()
        
        # Log filtering results
        if len(ev) > 0:
            print(f"   Excluded gambit rounds (step_no <= 3)")
            print(f"   Excluded test games (is_test=1)")
            print(f"   Chronologically sorted by game creation time")
            if total_events > max_usable_rows:
                print(f"   Discarded {total_events - len(ev)} oldest events")
            if self.training_player_like:
                print(f"   Filtered player_name LIKE {self.training_player_like}")
            if self.training_bot_policy_in:
                print(f"   Filtered bot_policy IN {self.training_bot_policy_in}")
            if self.training_easy_mode_only is not None:
                mode_str = "easy" if self.training_easy_mode_only else "standard"
                print(f"   Filtered easy_mode = {mode_str}")
        else:
            print(f"⚠️  No training events found (filters may be too restrictive)")
        
        return ev, gm
    
    def prepare_dataset(self) -> bool:
        """Load and prepare the dataset. Returns True if sufficient data."""
        ev, gm = self.load_data()
        self.events_df = ev  # Store for game-stratified CV
        self.X, self.y = build_training_dataset(ev, gm, lookback=self.lookback)
        self.feature_names = list(self.X.columns)
        
        if len(self.X) < self.min_sup_rows:
            print(f"Not enough supervised rows: {len(self.X)} < {self.min_sup_rows}")
            return False
        
        print(f"Dataset loaded with {len(self.X)} samples, {self.X.shape[1]} features")
        return True
    
    def split_and_prepare_data(self, scale_features: bool = True):
        """Split data chronologically into 60/20/20 and prepare features consistently."""
        total_samples = len(self.X)
        test_size = max(1, int(np.floor(total_samples * 0.20)))
        val_size = max(1, int(np.floor(total_samples * 0.20)))
        train_size = total_samples - test_size - val_size

        if train_size <= 0:
            # Ensure at least one training example by shrinking validation first, then test
            val_size = max(1, min(val_size, total_samples - 2))
            test_size = max(1, min(test_size, total_samples - val_size - 1))
            train_size = total_samples - test_size - val_size
            if train_size <= 0:
                raise ValueError("Insufficient samples to create 60/20/20 split")

        X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
            self.X, self.y, test_size=test_size, val_size=val_size
        )

        # Track aligned events for downstream analysis (training portion only)
        self.training_events = self.events_df.iloc[:len(X_train)].reset_index(drop=True)
        self.train_val_events = self.training_events  # Backwards compatibility

        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Prepare features
        X_train_proc, X_val_proc, X_test_proc, scaler = prepare_features(
            X_train, X_val, X_test, scale_features=scale_features
        )
        
        # Encode labels
        y_train_num, y_val_num, y_test_num, label_map = encode_labels(
            y_train, y_val, y_test, MOVES
        )
        
        return (X_train_proc, X_val_proc, X_test_proc, 
                y_train_num, y_val_num, y_test_num, 
                scaler, label_map)
    
    def _get_production_train_val_split(self, X_all: np.ndarray, y_all: np.ndarray):
        """
        Split full dataset into 80% train / 20% validation using game stratification.
        Every 5th game is held out for validation (similar to 5-fold CV logic).
        
        Returns: X_train, y_train, X_val, y_val
        """
        from collections import defaultdict
        
        # Get game_ids for all events
        game_ids = self.events_df['game_id'].values[:len(X_all)]
        
        # Group event indices by game_id
        game_to_indices = defaultdict(list)
        for idx, game_id in enumerate(game_ids):
            game_to_indices[game_id].append(idx)
        
        # Get unique games in order
        unique_games = []
        seen = set()
        for game_id in game_ids:
            if game_id not in seen:
                unique_games.append(game_id)
                seen.add(game_id)
        
        # Hold out every 5th game for validation
        val_games = unique_games[4::5]  # Every 5th game (indices 4, 9, 14, ...)
        train_games = [g for g in unique_games if g not in val_games]
        
        # Get event indices
        train_indices = [idx for game in train_games for idx in game_to_indices[game]]
        val_indices = [idx for game in val_games for idx in game_to_indices[game]]
        
        print(f"  Split: {len(train_games)} training games, {len(val_games)} validation games")
        
        return X_all[train_indices], y_all[train_indices], X_all[val_indices], y_all[val_indices]
    
    @abstractmethod
    def needs_feature_scaling(self) -> bool:
        """Return whether this model needs feature scaling"""
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return model-specific hyperparameters from environment variables"""
        pass
    
    @abstractmethod
    def run_cross_validation(self, X_train: np.ndarray, y_train: np.ndarray, 
                           hyperparams: Dict[str, Any], n_splits: int = 3) -> Tuple[list, float, float]:
        """Run time-series cross validation. Returns cv_scores, mean, std"""
        pass
    
    @abstractmethod
    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                         hyperparams: Dict[str, Any]):
        """
        Train the model. X_val/y_val can be None when training production model.
        
        For models with early stopping (XGBoost, NN):
        - If X_val is provided: use it for early stopping
        - If X_val is None: train without early stopping (use max epochs/estimators)
        """
        pass
        """Train the final model on full training data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        pass
    
    @abstractmethod
    def create_pyfunc_model(self, **kwargs):
        """Create MLflow pyfunc wrapper for the trained model"""
        pass
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model and return metrics"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        test_pred = self.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"🎯 Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred, target_names=MOVES))
        
        # Per-class accuracy
        conf_matrix = confusion_matrix(y_test, test_pred)
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        metrics = {"test_accuracy": test_acc}
        for i, class_name in enumerate(MOVES):
            class_acc = per_class_acc[i]
            metrics[f"test_acc_{class_name.lower()}"] = class_acc
            print(f"  {class_name} accuracy: {class_acc:.4f}")
        
        # Generate and log artifacts
        self.create_artifacts(y_test, test_pred, conf_matrix)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        return metrics
    
    def load_feature_names(self):
        """Load feature names from exports/columns.txt"""
        try:
            if self.feature_names:
                return self.feature_names
            columns_path = Path("exports/columns.txt")
            if columns_path.exists():
                with open(columns_path, 'r') as f:
                    feature_names = [line.strip() for line in f if line.strip()]
                return feature_names
            else:
                # Fallback to generic names if file doesn't exist
                return [f"feature_{i}" for i in range(self.X.shape[1])]
        except Exception as e:
            print(f"Warning: Could not load feature names from exports/columns.txt: {e}")
            return [f"feature_{i}" for i in range(self.X.shape[1])]

    def create_artifacts(self, y_test: np.ndarray, y_pred: np.ndarray, conf_matrix: np.ndarray):
        """Create and log visual artifacts for model validation"""
        import tempfile
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=MOVES, yticklabels=MOVES)
        plt.title(f'{self.model_type.title()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "plots")
            plt.close()
        
        # 2. Performance Summary Table
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, target_names=MOVES, output_dict=True)
        
        # Create a summary DataFrame
        summary_data = []
        for move in MOVES:
            summary_data.append({
                'Move': move.title(),
                'Precision': f"{report[move]['precision']:.3f}",
                'Recall': f"{report[move]['recall']:.3f}",
                'F1-Score': f"{report[move]['f1-score']:.3f}",
                'Support': int(report[move]['support'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            summary_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "reports")
        
        # 3. Model Parameters Summary
        hyperparams = getattr(self, "_current_hyperparams", None) or self.get_hyperparameters()
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': str(v)} for k, v in hyperparams.items()
        ])
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            params_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "config")

    def _log_cv_artifacts(self, cv_scores, mean_cv, std_cv):
        if not cv_scores:
            return

        import tempfile

        if self.cv_fold_details:
            df = pd.DataFrame(self.cv_fold_details)
        else:
            weight = 1.0 / len(cv_scores)
            df = pd.DataFrame({
                "fold": np.arange(1, len(cv_scores) + 1),
                "accuracy": cv_scores,
                "weight": [weight] * len(cv_scores)
            })

        df["cv_mean"] = mean_cv
        df["cv_std"] = std_cv

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "validation")

    def _log_training_summary(self, cv_mean, cv_std, holdout_val_acc, holdout_test_acc, 
                              prod_train_acc, prod_val_acc):
        import tempfile

        summary = pd.DataFrame([
            {
                "model_type": self.model_type,
                "cv_mean_acc": cv_mean,
                "cv_std_acc": cv_std,
                "holdout_val_acc": holdout_val_acc,
                "holdout_test_acc": holdout_test_acc,
                "production_train_acc": prod_train_acc,
                "production_val_acc": prod_val_acc,
                "timestamp_utc": pd.Timestamp.utcnow().isoformat()
            }
        ])

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            summary.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "validation")
    
    def log_model(self, pyfunc_model):
        """Log model to MLflow AND save locally for fast serving"""
        # Create input example for all models
        input_example_df = pd.DataFrame([self.X.iloc[0].to_dict()])
        
        # 1. Save model locally FIRST (if enabled)
        local_path = None
        if self.enable_local_storage:
            try:
                local_path = self._save_model_locally(input_example_df)
                print(f"💾 Model saved to local storage: {local_path}")
            except Exception as e:
                print(f"⚠️  Local save failed (continuing with MLflow): {e}")
        
        # 2. Log to MLflow (registry + optional artifacts)
        try:
            # ALL models should use pyfunc wrapper for consistent output format
            print(f"📦 Logging {self.model_type} model using mlflow.pyfunc format with custom wrapper...")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_model,
                input_example=input_example_df.values.tolist(),  # pyfunc expects list format
                registered_model_name=self.model_name
            )
                
            print(f"✅ Model logged to MLflow successfully!")
            
            # Store local path reference in MLflow
            if local_path:
                mlflow.log_param("local_model_path", str(local_path))
            
        except Exception as e:
            print(f"❌ MLflow logging failed: {e}")
            if not local_path:
                raise  # Fail if both local and MLflow fail
        
        # Add model identification tags
        mlflow.set_tag("model_name", self.model_name)
        mlflow.set_tag("model_type", self.model_type)
    
    def _save_model_locally(self, input_example_df: pd.DataFrame) -> Path:
        """Save model artifacts to local filesystem"""
        import json
        import torch
        
        # Get current run ID
        run = mlflow.active_run()
        if not run:
            raise ValueError("No active MLflow run")
        run_id = run.info.run_id
        
        # Create directory: /data/models/{model_name}/{run_id}/
        model_dir = self.local_models_dir / self.model_name / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file based on type
        if self.model_type == "xgboost":
            model_file = model_dir / "model.xgb"
            self.trained_model.save_model(str(model_file))
            print(f"  ✓ Saved XGBoost model: {model_file}")
        else:
            # Save PyTorch state dict
            model_file = model_dir / "model.pt"
            torch.save({
                'state_dict': self.trained_model.state_dict(),
                'model_class': self.trained_model.__class__.__name__,
            }, model_file)
            print(f"  ✓ Saved PyTorch model: {model_file}")
        
        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "run_id": run_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "feature_count": len(self.X.columns),
            "feature_names": list(self.X.columns),
            "mlflow_experiment": self.experiment,
        }
        
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata: {metadata_file}")
        
        # Save input example
        input_example_file = model_dir / "input_example.json"
        input_example_df.to_json(input_example_file, orient="records")
        print(f"  ✓ Saved input example: {input_example_file}")
        
        return model_dir
    
    def promote_model(self, run_id: str):
        """Promote model to specified alias and sync to MinIO for fast serving"""
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        mv = next(
            v
            for v in client.search_model_versions(f"name='{self.model_name}'")
            if v.run_id == run_id
        )

        alias_promoted = False
        
        if self.promote_alias and self.promote_alias.lower() not in ("none", "disable"):
            try:
                client.set_registered_model_alias(
                    name=self.model_name,
                    alias=self.promote_alias,
                    version=mv.version,
                )
                print(f"✅ Model alias '{self.promote_alias}' now points to version {mv.version}")
                alias_promoted = True
            except Exception as alias_err:
                print(
                    f"⚠️ Unable to set alias '{self.promote_alias}' for {self.model_name}: {alias_err}."
                    " Continuing without alias promotion."
                )

        if self.promote_stage and self.promote_stage.lower() not in ("none", "disable"):
            try:
                client.transition_model_version_stage(
                    name=self.model_name,
                    version=mv.version,
                    stage=self.promote_stage,
                    archive_existing_versions=True,
                )
                print(f"✅ Model promoted to {self.promote_stage} stage")
            except Exception as stage_err:
                print(
                    f"⚠️ Unable to promote {self.model_name} v{mv.version} to {self.promote_stage}: {stage_err}."
                    " Leaving model in its current stage."
                )
        
        # Sync promoted model to MinIO for fast serving
        if alias_promoted:
            self._sync_to_minio(run_id, mv.version)
    
    def _sync_to_minio(self, run_id: str, version: str):
        """Sync this model to MinIO after promotion"""
        print(f"\n📦 Syncing model v{version} to MinIO...")
        
        try:
            # Import sync function from app
            import sys
            from pathlib import Path
            app_path = Path(__file__).parent.parent / "app"
            if str(app_path) not in sys.path:
                sys.path.insert(0, str(app_path))
            
            from app.minio_sync import sync_promoted_models_to_minio
            
            # Sync all promoted models (clean old ones)
            synced, skipped, errors = sync_promoted_models_to_minio(clean=True, force=False)
            
            if errors == 0:
                print(f"✅ MinIO sync completed - {synced} models synced, {skipped} already present")
                print("   Model ready for fast serving from MinIO")
            else:
                print(f"⚠️  MinIO sync had {errors} errors (non-critical)")
                print("   Model is still available via MLflow (slower loading)")
                
        except Exception as e:
            print(f"⚠️  MinIO sync error (non-critical): {e}")
            print("   Model is still available via MLflow (slower loading)")

    
    def train(
        self,
        hyperparam_overrides: Optional[Dict[str, Any]] = None,
        config_id: Optional[str] = None,
        extra_tags: Optional[Dict[str, Any]] = None,
    ):
        """Main training pipeline with hold-out evaluation and 3-fold CV.

        Args:
            hyperparam_overrides: Optional mapping of hyperparameter names to override
                the defaults returned by ``get_hyperparameters``.
            config_id: Identifier for the active hyperparameter configuration. Logged as
                both a tag and parameter in MLflow for downstream traceability.
            extra_tags: Additional MLflow tags to attach to the training run (for
                example, information about the intended alias or experiment name).
        """
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        mlflow.set_experiment(self.experiment)
        
        # Prepare data
        if not self.prepare_dataset():
            return False
        self.cv_fold_details = None
        
        # Get model-specific parameters
        base_hyperparams = self.get_hyperparameters()
        hyperparams = dict(base_hyperparams)
        if hyperparam_overrides:
            hyperparams.update(hyperparam_overrides)
        self._current_hyperparams = dict(hyperparams)
        self._active_config_id = config_id
        
        # Split and prepare data
        scale_features = self.needs_feature_scaling()
        (X_train_proc, X_val_proc, X_test_proc, 
         y_train_num, y_val_num, y_test_num, 
         scaler, label_map) = self.split_and_prepare_data(scale_features)
        
        # Track run_id for promotion
        current_run_id = None
        training_succeeded = False
        holdout_val_acc = None
        holdout_test_acc = None
        train_acc_prod = None
        val_acc_prod = None
        cv_scores = []
        mean_cv = None
        std_cv = None
        
        with mlflow.start_run(run_name=f"{self.model_type}_training") as run:
            current_run_id = run.info.run_id
            
            if config_id:
                mlflow.set_tag("hyperparameter_config_id", config_id)
                mlflow.log_param("hyperparameter_config_id", config_id)

            if hyperparam_overrides:
                mlflow.set_tag("hyperparameters_overridden", "true")
            else:
                mlflow.set_tag("hyperparameters_overridden", "false")

            if extra_tags:
                for tag_key, tag_value in extra_tags.items():
                    if tag_value is not None:
                        mlflow.set_tag(tag_key, tag_value)

            # Log common parameters
            log_common_params(
                algorithm=self.model_type,
                n_features=X_train_proc.shape[1],
                n_samples=len(self.X),
                lookback=self.lookback,
                scaled=scale_features,
                **hyperparams
            )
            
            # ====================================================================
            # STEP 1: Train on 60% and evaluate on hold-out 20% test set
            # ====================================================================
            print("\n" + "="*70)
            print("STEP 1: Hold-out evaluation (train on 60%, monitor on validation, test on 20%)")
            print("="*70)

            self.train_final_model(X_train_proc, y_train_num, X_val_proc, y_val_num, hyperparams)

            val_pred = self.predict(X_val_proc)
            holdout_val_acc = accuracy_score(y_val_num, val_pred)
            mlflow.log_metric("val_acc_holdout", holdout_val_acc)
            print(f"✅ Validation Accuracy (20% chronological slice): {holdout_val_acc:.4f}")

            holdout_metrics = self.evaluate_model(X_test_proc, y_test_num)
            holdout_test_acc = holdout_metrics.get("test_accuracy")
            if holdout_test_acc is not None:
                mlflow.log_metric("test_acc_holdout", holdout_test_acc)

            # ====================================================================
            # STEP 2: Train production model (80% train / 20% val, every 5th game held out)
            # ====================================================================
            print("\n" + "="*70)
            print("STEP 2: Production model (80% train / 20% val, game-stratified)")
            print("="*70)

            # Combine all data for game-stratified split
            X_all = np.vstack([X_train_proc, X_val_proc, X_test_proc])
            y_all = np.concatenate([y_train_num, y_val_num, y_test_num])
            
            # Get game-stratified train/val split (every 5th game held out)
            X_prod_train, y_prod_train, X_prod_val, y_prod_val = self._get_production_train_val_split(
                X_all, y_all
            )
            
            print(f"  Production train: {len(X_prod_train)} events")
            print(f"  Production validation: {len(X_prod_val)} events (every 5th game)")
            
            # Train with validation for early stopping
            self.train_final_model(X_prod_train, y_prod_train, X_prod_val, y_prod_val, hyperparams)

            # Evaluate on production split
            train_pred = self.predict(X_prod_train)
            train_acc_prod = accuracy_score(y_prod_train, train_pred)
            
            val_pred = self.predict(X_prod_val)
            val_acc_prod = accuracy_score(y_prod_val, val_pred)
            
            mlflow.log_metric("train_acc_production", train_acc_prod)
            mlflow.log_metric("val_acc_production", val_acc_prod)
            
            print(f"  Production train accuracy: {train_acc_prod:.4f}")
            print(f"  Production val accuracy: {val_acc_prod:.4f}")

            # ====================================================================
            # STEP 3: 3-Fold CV on training subset (60%)
            # ====================================================================
            print("\n" + "="*70)
            print("STEP 3: 3-fold game-stratified CV on training subset (60%)")
            print("="*70)

            cv_scores, mean_cv, std_cv = self.run_cross_validation(
                X_train_proc, y_train_num, hyperparams, n_splits=3
            )

            if mean_cv is not None:
                mlflow.log_metric("cv_mean_acc", mean_cv)
            if std_cv is not None:
                mlflow.log_metric("cv_std_acc", std_cv)
            self._log_cv_artifacts(cv_scores, mean_cv, std_cv)

            print(f"✅ CV Results: {mean_cv:.4f} ± {std_cv:.4f}")

            # Log consolidated training summary
            self._log_training_summary(mean_cv, std_cv, holdout_val_acc, holdout_test_acc, 
                                      train_acc_prod, val_acc_prod)

            # Register PRODUCTION model (trained on full dataset)
            print("\n" + "="*70)
            print("Registering production model")
            print("="*70)

            pyfunc_model = self.create_pyfunc_model(
                scaler=scaler if scale_features else None,
                class_names=MOVES
            )

            self.log_model(pyfunc_model)

            # Mark training as successful
            training_succeeded = True
            print(f"\n🎯 {self.model_type} training completed!")
            print(f"   CV (60% data): {mean_cv:.4f} ± {std_cv:.4f}")
            print(f"   Hold-out Val: {holdout_val_acc:.4f}")
            print(f"   Hold-out Test: {holdout_test_acc:.4f}")
            print(f"   Production Train: {train_acc_prod:.4f}")
            print(f"   Production Val: {val_acc_prod:.4f}")
        
        # CRITICAL: Promotion happens OUTSIDE the with block
        # This ensures the run is properly closed before we assign aliases
        if training_succeeded and current_run_id:
            if hasattr(self, 'promote_stage') and (self.promote_stage or self.promote_alias):
                print(f"🔄 Promoting model (alias={self.promote_alias}, stage={self.promote_stage})...")
                # NO try/except - let errors bubble up so we know what failed
                self.promote_model(current_run_id)
                print(f"✅ Model promotion completed")
        
        # Summary
        compare_models_summary()
        
        return training_succeeded