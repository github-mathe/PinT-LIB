import numpy as np
from batpint.problems.base_problem import Problem
from batpint.time_integration.base_method import IntegrationMethod


class Dahlquist(Problem):
    """
    Dahlquist problem u' = λu + sin(αt), u(0) = u0, t ∈ [0, T]
    """
    
    def __init__(self, t_start, u_start, lam, alpha, event=None, terminate_step=None):
                
        def rhs(t, u, cycle, lam, alpha):
            return lam(cycle)*u + np.sin(alpha * t)
        
        def jacobian(t, u, cycle, lam):
            return lam(cycle)

        super().__init__(
            t_start,
            u_start,
            rhs=rhs,
            jacobian=jacobian,
            event=event,
            terminate_step=terminate_step,
            lam=lam,
            alpha=alpha,
        )
      
class DahlquistBE(IntegrationMethod):
    """
    Backward Euler method specialized for the Dahlquist problem.
    """

    def __init__(self):
        super().__init__()

    def step(self, t, u, h, rhs, jacobian, cycle, lam, alpha):
        t_next = t + h
        u_next = (u + h * np.sin(alpha * t_next)) / (1 - h * lam(cycle))

        return u_next
    
class DahlquistExact(IntegrationMethod):
    """
    Exact one-step method for the Dahlquist problem.
    """

    def __init__(self):
        super().__init__()

    def step(self, t, u, h, rhs, jacobian, cycle, lam, alpha):
        lam_cycle = lam(cycle)
        t_next = t + h

        denominator = alpha**2 + lam_cycle**2

        C = np.exp(-lam_cycle * t) * (
            u
            + (
                alpha * np.cos(alpha * t)
                + lam_cycle * np.sin(alpha * t)
            ) / denominator
        )

        return (
            C * np.exp(lam_cycle * t_next)
            - (
                alpha * np.cos(alpha * t_next)
                + lam_cycle * np.sin(alpha * t_next)
            ) / denominator
        )