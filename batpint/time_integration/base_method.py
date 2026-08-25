class IntegrationMethod(object):
    """Base class for integration methods."""
    def step(self, rhs, t, u, h, *args, **kwargs):
        """Perform a single integration step."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class AlgebraicSolver(object):
    """Base class for algebraic solvers."""
    def solve(self, residual, x0, *args, **kwargs):
        """Solve the algebraic equation."""
        raise NotImplementedError("This method should be implemented by subclasses.")