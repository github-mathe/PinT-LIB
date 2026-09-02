import copy

class PropagationState: 
    """
    State associated with one propagation.

    Parameters
    ----------
    t : float
        Initial time of the propagation.
    u : Any
        Initial state of the propagation.
    cycle : int
        Cycle / propagation index.
    **kwargs : 
        Additional dynamic attributes associated with this propagation.
    """
    def __init__(self, t, u, cycle, **kwargs):
        self.t = t
        self.u = copy.deepcopy(u)
        self.cycle = cycle

        self.terminated_step = False
        self.terminated_cycle = False
        for key, value in kwargs.items():
            setattr(self, key, value)