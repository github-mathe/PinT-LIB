import numpy as np

def epsilon(n: int) -> float:
    return 0.01 * n

def lambda_n(n: int) -> complex:
    return 1j * (1.0 + epsilon(n))

def f_ODE(
    t: float,
    u: complex,
    alpha: float,
    lam: complex,
) -> float:
    """Right-hand side function for the ODE u’ = u."""
    du_dt = lam * u + np.sin(alpha * t)
    return float(np.imag(du_dt))

