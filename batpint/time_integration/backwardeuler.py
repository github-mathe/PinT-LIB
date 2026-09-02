import numpy as np
from batpint.time_integration.base_method import IntegrationMethod

class BackwardEuler(IntegrationMethod):
    """
    Backward Euler method for solving ODEs of the form dy/dt = rhs(t,y)
    """

    def __init__(self, algebraic_solver):
        self.algebraic_solver = algebraic_solver
        super().__init__()

    def step(self, t, u, h, rhs, jacobian=None):
        """
        Perform a single Backward Euler step.
        
        Parameters:
        rhs : callable
            The right-hand side function of the ODE.
        t : float
            Current time.
        u : np.ndarray
            Current state vector.
        h : float
            Time step size.
        jacobian : callable, optional
            Jacobian J_f(t, u, ...) of the right-hand side.

        Returns:
        -------
        scalar or array-like
            Approximation u_{n+1} at time t + h.
        """
        t_new = t + h

        # Define the residual function for the implicit equation
        def residual(u_new):
            return u_new - u - h * rhs(t_new, u_new)
        residual_jacobian = None
        if jacobian is not None:
            def residual_jacobian(u_new):
                Jf = jacobian(t_new, u_new)
                if np.ndim(Jf) == 0:
                    return 1.0 - h * Jf
                return np.eye(Jf.shape[0]) - h * Jf      

        # Use the algebraic solver to solve for u_new
        return self.algebraic_solver.solve(
            residual,
            x0=np.copy(u),
            residual_jacobian=residual_jacobian
            )