LSTM Forward Model & Mutual Information Analysis
Trained LSTM model for fMRI BOLD forecasting on the NSD dataset, including probabilistic predictions and model-based mutual information estimation between ROIs.
Files

best_fmri_sklearn_api.joblib — Trained model weights (sklearn-compatible API)
LSTM_model_library_Sklearn.py — API class definitions (required to load .joblib)
TS_MRI_train_test_Latest.ipynb — Full training pipeline notebook (latest version)
TS_MRI_train_test_fixed_clean.ipynb — Cleaned training notebook
TS_MRI_train_test_Latest_clean.html — HTML export of the notebook with all outputs

Requirements
pip install torch numpy scikit-learn joblib
Usage
import joblib
model = joblib.load("best_fmri_sklearn_api.joblib")
X = ...                          # shape: (n_samples, 50, 19)
Y_pred = model.predict(X)        # shape: (n_samples, 3, 19)
mean, std = model.predict_proba(X)
Input / Output Specification
Input X shape: (n_samples, 50, 19) — 50 past TRs, 19 ROIs (BN19 atlas order)
Output Y shape: (n_samples, 3, 19) — 3 future TR predictions per ROI
Data must be z-scored per run before being passed to the model:
ts_norm = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-8)
Model Architecture

3-layer LSTM, hidden size 512, dropout 0.5
Lookback M = 50 TRs
Horizon H = 3 TRs
Loss: HuberLoss + delta-aware penalty (DeltaAwareLoss)
Best fold: subjxpYwO4azeZ (LOSO-CV)

Results (LOSO-CV, 6 subjects)

Mean LSTM RMSE: 0.825
Mean Naive RMSE: 1.023
Mean eta: 0.111
Beat naive: 6 / 6 folds
1 sigma coverage: 71.5% (expected 68%)
2 sigma coverage: 94.9% (expected 95%)

Notebook Structure
The training notebook covers:

Data loading from NSD .npz files via utils.parse_data
Subject-level normalization (run-level z-score, leak-free)
Sliding window construction (M=50, H=3)
LOSO cross validation across all subjects
Best-eta model selection
Sklearn-compatible API export (predict, predict_proba, log_prob)
Probabilistic forecast visualization with confidence intervals
Calibration check (per-ROI 1 sigma / 2 sigma coverage)
Model-based mutual information matrix between ROIs
