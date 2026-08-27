import numpy as np
from dataclasses import dataclass
from batpint.problems.base_problem import Problem
from batpint.time_integration.base_method import IntegrationMethod


class Dahlquist(Problem):
    """
    Dahlquist problem u' = λu + sin(αt), u(0) = u0, t ∈ [0, T]
    """
    
    def __init__(self, t_start, u_start, lam, alpha, cycle, u_event, event=None, terminate=None):
                
        def rhs(t, u, lam, alpha, cycle):
            return lam(cycle)*u + np.sin(alpha * t)
        
        def jacobian(t, u, lam, cycle):
            return lam(cycle)

        super().__init__(
            t_start,
            u_start,
            rhs=rhs,
            jacobian=jacobian,
            event=event,
            terminate=terminate,
            lam=lam,
            alpha=alpha,
            cycle=cycle,
            u_event=u_event
        )
      
class DahlquistBE(IntegrationMethod):
    """
    Backward Euler method specialized for the Dahlquist problem.
    """

    def __init__(self, params):
        self.params = params

    def step(self, rhs, t, u, h, jacobian=None):
        cycle = self.params["cycle"]
        lam = self.params["lam"](cycle)
        alpha = self.params["alpha"]

        t_next = t + h
        u_next = (u + h * np.sin(alpha * t_next)) / (1 - h * lam)

        return u_next
    
class DahlquistExact(IntegrationMethod):
    """
    Exact one-step method for the Dahlquist problem.
    """

    def __init__(self, params):
        self.params = params

    def step(self, rhs, t, u, h, jacobian=None):
        cycle = self.params["cycle"]
        lam = self.params["lam"](cycle)
        alpha = self.params["alpha"]

        t_next = t + h

        denominator = alpha**2 + lam**2

        C = np.exp(-lam * t) * (
            u
            + (
                alpha * np.cos(alpha * t)
                + lam * np.sin(alpha * t)
            ) / denominator
        )

        return (
            C * np.exp(lam * t_next)
            - (
                alpha * np.cos(alpha * t_next)
                + lam * np.sin(alpha * t_next)
            ) / denominator
        )