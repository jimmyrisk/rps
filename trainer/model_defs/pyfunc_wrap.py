# trainer/model_defs/pyfunc_wrap.py
import pandas as pd
import numpy as np
import mlflow.pyfunc
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class SoftmaxReg(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.linear(x)

class RpsUserNextMoveModel(mlflow.pyfunc.PythonModel):
    """
    Wrap either:
      - a scikit-learn estimator with predict_proba, OR
      - a torch state_dict + (d_in, class_names) for softmax regression
    """
    def __init__(
        self,
        sk_model=None,
        torch_state=None,
        d_in=None,
        class_names=("rock", "paper", "scissors"),
        scaler_params=None,
    ):
        self.sk = sk_model
        self.state = torch_state
        self.d_in = d_in
        self.class_names = list(class_names)
        self.scaler = None
        if scaler_params:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.asarray(scaler_params.get("mean", []), dtype=np.float32)
            self.scaler.scale_ = np.asarray(scaler_params.get("scale", []), dtype=np.float32)
            # scikit-learn expects var_ attribute for some operations
            self.scaler.var_ = self.scaler.scale_ ** 2
        if self.state is not None:
            self.model = SoftmaxReg(d_in, len(self.class_names))
            self.model.load_state_dict(self.state)
            self.model.eval()
        else:
            self.model = None

    def predict(self, context=None, model_input=None):
        # Handle both signatures: predict(context, model_input) and predict(model_input)
        if model_input is None and context is not None:
            model_input = context
            
        if model_input is None:
            raise ValueError("model_input is required")
            
        # Ensure input is a DataFrame for downstream compatibility
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Convert to numpy for optional scaling
        input_array = model_input.values.astype(np.float32)
        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            input_array = scaler.transform(input_array)

        # Get probabilities for ALL samples (batch prediction)
        if self.sk is not None:
            probs_batch = self.sk.predict_proba(input_array)  # Shape: (n_samples, n_classes)
        else:
            with torch.no_grad():
                x = torch.tensor(input_array, dtype=torch.float32)
                logits = self.model(x)
                probs_batch = torch.softmax(logits, dim=1).cpu().numpy()  # Shape: (n_samples, n_classes)
        
        # For batch predictions, return list of results
        if len(model_input) > 1:
            results = []
            for i in range(len(model_input)):
                probs = probs_batch[i]
                pick = self.class_names[int(np.argmax(probs))]
                results.append({
                    "classes": self.class_names,
                    "probs": probs.tolist(),
                    "pick": pick
                })
            return results
        else:
            # Single prediction - return dict (backward compatibility)
            probs = probs_batch[0]
            pick = self.class_names[int(np.argmax(probs))]
            return {"classes": self.class_names, "probs": probs.tolist(), "pick": pick}
