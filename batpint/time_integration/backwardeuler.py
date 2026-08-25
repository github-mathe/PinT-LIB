import numpy as np
from base_method import IntegrationMethod

class BackwardEuler(IntegrationMethod):
    """
    Backward Euler method for solving ODEs of the form dy/dt = rhs(t,y)
    """
    def __init__(self, algebraic_solver):
        self.algebraic_solver = algebraic_solver
    def step(self,rhs, t, u, h, jacobian=None, **kwargs):
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
        
        Returns:
        np.ndarray
            The state vector after one Backward Euler step.
        """
        t_new = t + h
        # Define the residual function for the implicit equation
        def residual(u_new):
            return u_new - u - h * rhs(t_new, u_new)
        if jacobian is not None:
            def residual_jacobian(u_new):
                if np.isscalar(u_new):
                    return 1 - h * jacobian(t_new, u_new)
                else:
                    return np.eye(len(u_new)) - h * jacobian(t_new, u_new)
            kwargs['residual_jacobian'] = residual_jacobian         
        x0 = u  # Initial guess for the new state
        # Use the algebraic solver to solve for u_new
        return self.algebraic_solver.solve(residual, x0, **kwargs)