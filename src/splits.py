import numpy as np


SIZES = ["25", "100", "400"]

REPRODUCTION_TRAIN_CURVES = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsdcon_9.csv",
]

STRICT_COSINE_TRAIN_CURVES = [
    "cosine_24000.csv",
]

MAIN_TEST_CURVES = [
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
]

AUX_TEST_CURVES = [
    "constant_72000.csv",
    "cosine_72000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]

ALL_EVAL_CURVES = MAIN_TEST_CURVES + AUX_TEST_CURVES

PHASES = {
    "warmup_or_start": (0, 2160),
    "post_warmup_stable": (2160, 20000),
    "decay": (20000, 24000),
    "long_horizon": (24000, None),
}


def phase_mask(steps: np.ndarray, phase_name: str) -> np.ndarray:
    start, end = PHASES[phase_name]
    if end is None:
        return steps >= start
    return (steps >= start) & (steps < end)
