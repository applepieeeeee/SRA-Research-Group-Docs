import numpy as np

# TODO: BETA START AND END NEED TO BE CALIBRATED
def linear_beta(steps: int, beta_start: float = 0.1, beta_end: float = 3.0) -> np.ndarray:
    # Beta varied lienarly from hot (beta_start) to cold (beta_end)
    return np.linspace(beta_start, beta_end, steps, dtype = np.float32)

# The two schedules below are not necessary, but we can experiment with them if we want
def geometric_beta(steps: int, beta_start: float = 0.1, beta_end: float = 3.0) -> np.ndarray:
    # Beta multipled by a constant factor each step
    return np.geomspace(beta_start, beta_end, steps, dtype = np.float32)

def constant_beta(steps: int, beta: float = 1.0) -> np.ndarray:
    # Fixed temperature
    return np.full(steps, beta, dtype=np.float32)

