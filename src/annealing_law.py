import numpy as np
from scipy.optimize import minimize
from scipy.special import huber

from src.features import cumulative_lr, tissue_s2


def _huber_loss(residual: np.ndarray, delta: float = 0.001) -> np.ndarray:
    return huber(delta, residual)


class TissueAnnealingLaw:
    def __init__(self, lambda_decay: float = 0.995):
        self.lambda_decay = lambda_decay
        self.params = None

    def predict_curve(self, lrs: np.ndarray, steps: np.ndarray, params=None) -> np.ndarray:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("TissueAnnealingLaw must be fitted before prediction.")
        l0, a, alpha, c = params
        s1 = cumulative_lr(lrs)[steps]
        s2 = tissue_s2(lrs, self.lambda_decay)[steps]
        return l0 + a * np.maximum(s1, 1e-12) ** (-alpha) - c * s2

    def fit(self, data: dict, train_curves: list[str]):
        min_loss = min(float(data[name]["loss"].min()) for name in train_curves)
        init = np.array([min_loss - 0.05, 0.5, 0.5, 0.5], dtype=float)

        def objective(params):
            if np.any(params <= 0):
                return 1e9
            total = 0.0
            for name in train_curves:
                pred = self.predict_curve(data[name]["lrs"], data[name]["step"], params=params)
                loss = data[name]["loss"]
                pred = np.maximum(pred, 1e-10)
                residual = np.log(loss) - np.log(pred)
                total += float(_huber_loss(residual).sum())
            return total

        result = minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=[(1e-8, None), (1e-8, None), (1e-8, 5.0), (1e-8, None)],
            options={"maxiter": 10000, "ftol": 1e-12},
        )
        self.params = result.x
        return self.params, float(result.fun)


def fit_best_tissue(data: dict, train_curves: list[str], lambdas=(0.99, 0.995, 0.999)):
    best = None
    for lambda_decay in lambdas:
        model = TissueAnnealingLaw(lambda_decay=lambda_decay)
        params, loss = model.fit(data, train_curves)
        if best is None or loss < best["loss"]:
            best = {"model": model, "params": params, "loss": loss, "lambda_decay": lambda_decay}
    return best
