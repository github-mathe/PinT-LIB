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
    t0 : float
        Initial time.
    u0 : scalar or array-like
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

    def __init__(self, t0, u0, rhs, jacobian=None, event=None, terminate=None, **params):
        # Original initial condition of the IVP
        self.t_start = t0
        self.u_start = u0
        
        self.rhs = rhs
        self.jacobian = jacobian
        self.event = event
        self.terminate = terminate
        self.params = params

    def __call__(self, t, u, **kwargs):
        """Evaluate the right-hand side f(t, u)."""
        return self.rhs(t, u, **self.params, **kwargs)

    def jacobian_value(self, t, u, **kwargs):
        """Evaluate the Jacobian of the right-hand side."""
        if self.jacobian is None: 
            raise ValueError("No Jacobian function defined for this problem.")
        return self.jacobian(t, u, **self.params, **kwargs)

    def event_value(self, t, u, **kwargs):
        """Evaluate the event function g(t, u)."""
        if self.event is None: 
            raise ValueError("No event function defined for this problem.")
        return self.event(t, u, **self.params, **kwargs)

    def termination_value(self, t, u, **kwargs):
        """Evaluate the termination function."""
        if self.terminate is None: 
            return False  # No termination function defined
        return self.terminate(t, u, **self.params, **kwargs)