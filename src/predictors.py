import numpy as np


def predict_mpl_curve(lrs: np.ndarray, steps: np.ndarray, params: list[float]) -> np.ndarray:
    l0, a, alpha, b, c, beta, gamma = params
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=float)
    lr_gap[1:] = np.diff(lrs)
    s1 = lr_sum[steps]
    ld = np.zeros(len(steps), dtype=float)
    for i, step in enumerate(steps):
        if step > 0:
            scaled = lrs[1 : step + 1] ** (-gamma) * (lr_sum[step] - lr_sum[:step])
            ld[i] = np.sum(lr_gap[1 : step + 1] * (1 - (1 + c * scaled) ** (-beta)))
    return l0 + a * np.maximum(s1, 1e-12) ** (-alpha) + b * ld
