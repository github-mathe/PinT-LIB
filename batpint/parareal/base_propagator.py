class Propagator(object):
    """Base class for propagation between pseudoperiod points."""
    def propagate(self, t, u, **kwargs):
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )