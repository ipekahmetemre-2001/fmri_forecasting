import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import Dataset, DataLoader


class FMRIWindowDataset(Dataset):
    """PyTorch dataset for fMRI windowed sequences."""

    def __init__(self, X, Y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = None
        if Y is not None:
            self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        return self.X[idx]


class AdvancedLSTM(nn.Module):
    """
    Multi-layer LSTM for multi-step fMRI BOLD forecasting.
    Takes a sequence of M past ROI vectors and predicts H future steps.
    """
    def __init__(self, input_size, hidden_size=512, num_layers=3,
                 output_horizon=5, dropout=0.5):
        super().__init__()
        self.output_horizon = output_horizon
        self.input_size     = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_horizon * input_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]
        out    = self.fc(last_h)
        return out.view(-1, self.output_horizon, self.input_size)


class FmriPredictorAPI(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn compatible API for the fMRI LSTM model.

    Methods:
        predict(X)            — point prediction: x̂_t = argmax p̂(x_t|past)
        calibrate(X_cal,Y_cal)— estimate per-ROI residual std for proba
        predict_proba(X)      — returns (mean, std) of p̂(x_t|past) ~ N(μ,σ²)
        log_prob(X, Y)        — log p̂(y_t|past) under Gaussian assumption
    """

    def __init__(self, model_obj=None, M=50, H=5, device='cpu',
                 roi_std=None):
        self.model_obj = model_obj
        self.M         = M
        self.H         = H
        self.device    = device
        self.roi_std   = roi_std

        if self.model_obj is not None:
            self.model_obj.to(self.device)
            self.model_obj.eval()

    def fit(self, X, y=None):
        """Exists for Scikit-Learn compatibility. No training performed here."""
        return self

    def _run_inference(self, X, batch_size=512):
        """Shared inference loop."""
        if self.model_obj is None:
            raise ValueError("Model object is not initialized.")

        self.model_obj.eval()

        # Fix RNN contiguous memory warning
        if hasattr(self.model_obj, 'lstm'):
            self.model_obj.lstm.flatten_parameters()

        loader = DataLoader(
            FMRIWindowDataset(X),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        preds = []
        with torch.no_grad():
            for x_batch in loader:
                x_batch = x_batch.to(self.device)
                preds.append(self.model_obj(x_batch).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def predict(self, X, batch_size=512):
        """
        Point prediction: x̂_t = argmax_x p̂(x_t | past)
        Under Gaussian assumption this equals the mean.

        Input  X : (N, M, ROI)
        Output Y : (N, H, ROI)
        """
        return self._run_inference(X, batch_size)

    def calibrate(self, X_cal, Y_cal, batch_size=512):
        """
        Estimate per-ROI residual std from a calibration set.
        Must be called once before predict_proba or log_prob.

        X_cal : (N, M, ROI) — calibration inputs
        Y_cal : (N, H, ROI) — corresponding ground truth
        """
        Y_pred     = self.predict(X_cal, batch_size)
        residuals  = Y_cal - Y_pred
        flat       = residuals.reshape(-1, residuals.shape[-1])
        self.roi_std = flat.std(axis=0) + 1e-8
        print(f"Calibration done. Per-ROI std shape : {self.roi_std.shape}")
        print(f"Mean std across ROIs                : {self.roi_std.mean():.4f}")
        return self

    def predict_proba(self, X, batch_size=512):
        """
        Probabilistic prediction: p̂(x_t | past) ~ N(mean, std²)

        mean = LSTM point prediction
        std  = per-ROI residual std from calibrate()

        Input  X    : (N, M, ROI)
        Output mean : (N, H, ROI)
               std  : (N, H, ROI)
        """
        if self.roi_std is None:
            raise ValueError(
                "roi_std not set. Run calibrate(X_cal, Y_cal) first."
            )
        mean = self.predict(X, batch_size)
        std  = np.broadcast_to(
            self.roi_std[np.newaxis, np.newaxis, :],
            mean.shape
        ).copy()
        return mean, std

    def log_prob(self, X, Y, batch_size=512):
        """
        Compute log p̂(y_t | past) under Gaussian assumption.

        log p̂(x_t|past) = -0.5*((x_t - μ_t)²/σ²) - log(σ√2π)

        Input  X     : (N, M, ROI)
               Y     : (N, H, ROI)
        Output log_p : (N, H, ROI)
        """
        mean, std = self.predict_proba(X, batch_size)
        log_p = (
            -0.5 * ((Y - mean) ** 2) / (std ** 2)
            - np.log(std * np.sqrt(2 * np.pi))
        )
        return log_p

    def get_params(self, deep=True):
        return dict(model_obj=self.model_obj, M=self.M, H=self.H,
                    device=self.device, roi_std=self.roi_std)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
