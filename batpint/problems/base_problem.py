import inspect
import numpy as np

class Problem:
    """
    The problem (IVP) is defined by
        du/dt = f(t, u,...), u(t0) = u0.
    Optional functions may be supplied for the Jacobian of the
    right-hand side and for event detection.
    All problem functions must begin with the arguments
        (t, u,...)
    Additional keyword arguments passed at construction are stored
    as fixed problem parameters "params".
    
    Parameters
    ----------
    t_start : float
        Initial time.
    u_start : scalar or array-like
        Initial state.
    rhs : callable
        Right-hand side with convention f(t, u, **kwargs).
    jacobian : callable, optional
        Jacobian with convention J(t, u, **kwargs).
    event : callable, optional
        Event function with convention g(t, u, **kwargs).
    terminate_step : callable, optional
        Function to determine if the integration should terminate at a given step.
    **params
        Fixed parameters of the mathematical problem.
    """

    def __init__(self, t_start, u_start, rhs, jacobian=None, event=None, terminate_step=None, **params):
        # Original initial condition of the IVP
        self.t_start = t_start
        self.u_start = u_start

        self.rhs = rhs
        self.jacobian = jacobian
        self.event = event
        self.terminate_step = terminate_step
        self.params = params

        self.rhs_params = self._get_params(rhs)
        self.jacobian_params = self._get_params(jacobian)
        self.event_params = self._get_params(event)
        self.terminate_step_params = self._get_params(terminate_step)

    @staticmethod
    def _get_params(func):
        
        if func is None:
            return ()
        
        names = tuple(inspect.signature(func).parameters)
        
        if names[:2] != ("t", "u"):
            raise ValueError("Problem functions must start with arguments (t, u).")

        return names[2:]