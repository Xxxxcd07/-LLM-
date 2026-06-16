import numpy as np


def cumulative_lr(lrs: np.ndarray) -> np.ndarray:
    return np.cumsum(lrs)


def positive_decay(lrs: np.ndarray) -> np.ndarray:
    decay = np.zeros_like(lrs, dtype=float)
    decay[1:] = np.maximum(lrs[:-1] - lrs[1:], 0.0)
    return decay


def tissue_s2(lrs: np.ndarray, lambda_decay: float) -> np.ndarray:
    decay = positive_decay(lrs)
    momentum = np.zeros_like(lrs, dtype=float)
    for i in range(1, len(lrs)):
        momentum[i] = lambda_decay * momentum[i - 1] + decay[i]
    return np.cumsum(momentum)


def correction_features(lrs: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    s1 = cumulative_lr(lrs)
    decay = positive_decay(lrs)
    decay_cum = np.cumsum(decay)
    eta_max = max(float(np.max(lrs)), 1e-12)
    eta_norm = lrs / eta_max
    s1_sq = np.cumsum(lrs ** 2)
    mem_099 = tissue_s2(lrs, 0.99)
    mem_0995 = tissue_s2(lrs, 0.995)
    mem_0999 = tissue_s2(lrs, 0.999)
    total = max(len(lrs) - 1, 1)
    t_norm = np.arange(len(lrs), dtype=float) / total
    is_decay = (decay_cum > 0).astype(float)

    full = np.column_stack([
        np.ones(len(lrs)),
        np.log(np.maximum(s1, 1e-12)),
        eta_norm,
        s1_sq,
        decay_cum,
        mem_099,
        mem_0995,
        mem_0999,
        t_norm,
        is_decay,
    ])
    names = [
        "bias",
        "log_s1",
        "eta_norm",
        "s1_sq",
        "decay_cum",
        "decay_mem_099",
        "decay_mem_0995",
        "decay_mem_0999",
        "t_norm",
        "is_decay",
    ]
    return full[steps], names


def fsl_light_features(
    lrs: np.ndarray,
    steps: np.ndarray,
    lambda_decay: float = 0.995,
) -> tuple[np.ndarray, list[str]]:
    s1 = cumulative_lr(lrs)
    total_lr = float(s1[-1])
    tau = s1 / total_lr if total_lr > 0.0 else np.zeros_like(s1, dtype=float)
    decay_cum = np.cumsum(positive_decay(lrs))
    eta_max = max(float(np.max(lrs)), 1e-12)

    full = np.column_stack([
        np.ones(len(lrs)),
        np.log(np.maximum(tau, 1e-12)),
        lrs / eta_max,
        tissue_s2(lrs, lambda_decay),
        (decay_cum > 0).astype(float),
    ])
    names = ["bias", "log_tau", "eta_norm", "decay_conv", "is_decay"]
    return full[steps], names
