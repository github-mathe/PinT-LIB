import inspect
import numpy as np

class Problem:
    """
    The problem (IVP) is defined by
        du/dt = f(t, u), u(t0) = u0.
    Optional functions may be supplied for the Jacobian of the
    right-hand side and for event detection.
    Additional keyword arguments passed at construction are stored
    as fixed problem parameters.
    
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
    **params
        Fixed parameters of the mathematical problem.
    """

    def __init__(self, t_start, u_start, rhs, jacobian=None, event=None, terminate=None, **params):
        # Original initial condition of the IVP
        self.t_start = t_start
        self.u_start = u_start

        self.rhs = rhs
        self.jacobian = jacobian
        self.event = event
        self.terminate = terminate
        self.params = params

        self.rhs_params = self._get_params(rhs)
        self.jacobian_params = self._get_params(jacobian)
        self.event_params = self._get_params(event)
        self.terminate_params = self._get_params(terminate)

    @staticmethod
    def _get_params(func):
        
        if func is None:
            return ()
        names = tuple(inspect.signature(func).parameters)
        
        if names[:2] != ("t", "u"):
            raise ValueError("Problem functions must start with arguments (t, u).")

        return names[2:]
    
    def __call__(self, t, u):
        """
        Evaluate the right-hand side f(t, u).
        """
        
        kwargs = {name: self.params[name] for name in self.rhs_params}
        return self.rhs(t, u, **kwargs)

    def jacobian_value(self, t, u):
        """
        Evaluate the Jacobian of the right-hand side.
        """
        
        if self.jacobian is None: 
            raise ValueError("No Jacobian function defined for this problem.")
        
        kwargs = {name: self.params[name] for name in self.jacobian_params}
        return self.jacobian(t, u, **kwargs)

    def event_value(self, t, u):
        """
        Evaluate the event function g(t, u).
        """
        
        if self.event is None: 
            raise ValueError("No event function defined for this problem.")
        
        kwargs = {name: self.params[name] for name in self.event_params}
        return self.event(t, u, **kwargs)

    def termination_value(self, t, u):
        """
        Evaluate the termination function.
        """
        
        if self.terminate is None: 
            return False  # No termination function defined
        
        kwargs = {name: self.params[name] for name in self.terminate_params}
        return self.terminate(t, u, **kwargs)