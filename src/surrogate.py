import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import fsl_light_features


def _parse_model_size(model_size: str | int | float) -> float:
    text = str(model_size).strip().lower().replace("m", "")
    value = float(text)
    if value <= 0.0:
        raise ValueError("model_size must be positive.")
    return value


def ncpl_surrogate_features(
    model_size: str | int | float,
    lrs: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    steps = np.asarray(steps, dtype=int)
    lrs = np.asarray(lrs, dtype=float)
    fsl_features, fsl_names = fsl_light_features(lrs, steps)

    max_step = max(float(np.max(steps)) if len(steps) else 0.0, 1.0)
    step_norm = steps.astype(float) / max_step
    log_step = np.log1p(steps.astype(float))
    model_column = np.full(len(steps), np.log(_parse_model_size(model_size)), dtype=float)

    selected = ["log_tau", "eta_norm", "decay_conv", "is_decay"]
    fsl_columns = [fsl_features[:, fsl_names.index(name)] for name in selected]
    features = np.column_stack([model_column, step_norm, log_step, *fsl_columns])
    names = ["log_model_size", "step_norm", "log_step", *selected]
    return features, names


class NCPLStyleSurrogate:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        alpha: float = 1e-3,
        max_iter: int = 2000,
        random_state: int = 0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_names_: list[str] | None = None
        self.model_ = None

    def fit(
        self,
        features: np.ndarray,
        loss: np.ndarray,
        feature_names: list[str],
    ) -> "NCPLStyleSurrogate":
        loss = np.asarray(loss, dtype=float)
        if np.any(loss <= 0.0):
            raise ValueError("NCPLStyleSurrogate requires positive loss values.")

        self.feature_names_ = list(feature_names)
        self.model_ = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                learning_rate_init=1e-3,
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_.fit(features, np.log(loss))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("NCPLStyleSurrogate must be fitted before prediction.")
        return np.exp(self.model_.predict(features))


def stack_surrogate_training_rows(data_by_size: dict, train_curves: list[str]):
    feature_blocks = []
    loss_blocks = []
    feature_names = None
    for size, data in data_by_size.items():
        for curve in train_curves:
            features, names = ncpl_surrogate_features(
                size,
                data[curve]["lrs"],
                data[curve]["step"],
            )
            if feature_names is None:
                feature_names = names
            feature_blocks.append(features)
            loss_blocks.append(data[curve]["loss"])
    return np.vstack(feature_blocks), np.concatenate(loss_blocks), feature_names
