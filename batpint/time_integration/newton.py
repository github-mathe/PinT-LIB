import numpy as np
from batpint.time_integration.base_method import AlgebraicSolver

class Newton(AlgebraicSolver):
    """
    Newton's method for solving R(x) = 0.
    """

    def __init__(self, tol=1e-10, max_iter=20):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, residual, x0, residual_jacobian=None):
        """
        Solve R(x) = 0 using Newton's method.

        Parameters
        ----------
        residual : callable
            Residual function R(x).
        x0 : scalar or array-like
            Initial guess.
        residual_jacobian : callable
            Jacobian of the residual.

        Returns
        -------
        scalar or array-like
            Approximate solution of R(x) = 0.
        """
        if residual_jacobian is None:
            raise ValueError("Newton requires a residual Jacobian function.")

        x = np.copy(x0)
        for _ in range(self.max_iter):
            r = residual(x)
            if np.linalg.norm(r) < self.tol:
                return x
            J = residual_jacobian(x)
            if np.ndim(J) == 0:
                delta_x = -r / J
            else:
                delta_x = np.linalg.solve(J, -r)
            x = x + delta_x
        raise RuntimeError("Newton method did not converge.")
        