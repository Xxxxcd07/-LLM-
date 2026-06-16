import numpy as np


class RidgeResidualCorrection:
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.coef_ = None
        self.feature_names_ = None

    def fit(self, features: np.ndarray, residual: np.ndarray, feature_names: list[str]):
        xtx = features.T @ features
        penalty = self.alpha * np.eye(xtx.shape[0])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(xtx + penalty, features.T @ residual)
        self.feature_names_ = feature_names
        return self

    def predict_residual(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("RidgeResidualCorrection must be fitted before prediction.")
        return features @ self.coef_


def stack_residual_training_rows(data: dict, train_curves: list[str], base_predictions: dict, feature_fn):
    feature_blocks = []
    residual_blocks = []
    feature_names = None
    for name in train_curves:
        features, names = feature_fn(data[name]["lrs"], data[name]["step"])
        if feature_names is None:
            feature_names = names
        residual = data[name]["loss"] - base_predictions[name]
        feature_blocks.append(features)
        residual_blocks.append(residual)
    return np.vstack(feature_blocks), np.concatenate(residual_blocks), feature_names
