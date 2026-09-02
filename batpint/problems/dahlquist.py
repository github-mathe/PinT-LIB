import numpy as np
from batpint.problems.base_problem import Problem
from batpint.time_integration.base_method import IntegrationMethod
from batpint.timestepping.timestepper import TimeStepper
from batpint.parareal.timestepper_propagator import TimeStepperPropagator

class Dahlquist(Problem):
    """
    Dahlquist problem u' = λu + sin(αt), u(0) = u0, t ∈ [0, T]
    """
    
    def __init__(self, t_start, u_start, lam, alpha, event=None, terminate_step=None):
        """
        Parameters
        ----------
        t_start : float
            Initial time.
        u_start : float
            Initial value of the solution.
        lam : callable  
          Function that returns the value of λ for a given cycle.
        alpha : float
            The frequency of the forcing term.
        event : callable, optional
            Event function.
        terminate_step : callable, optional
            Function to determine if the integration should terminate at a given step.
        """

        def rhs(t, u, cycle, lam, alpha):
            """
            Right-hand side of the Dahlquist problem.
            """
            return lam(cycle)*u + np.sin(alpha * t)
        
        def jacobian(t, u, cycle, lam):
            """
            Jacobian of the Dahlquist problem.
            """
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
        C = np.exp(-lam_cycle * t) * (u + (alpha * np.cos(alpha * t) + lam_cycle * np.sin(alpha * t)) / denominator)

        return (C * np.exp(lam_cycle * t_next) - (alpha * np.cos(alpha * t_next) + lam_cycle * np.sin(alpha * t_next)) / denominator)

def solve_dahlquist_cycles(problem, method, dt, num_cycles, make_state, terminate_cycle = None, direction=1, save_history=True):
    """
    solve_dahlquist solves the Dahlquist problem using the specified time stepper method.
    Args:
        dt (float): time step size
        num_cycles (int): number of cycles to simulate
        method (TimeStepper): time stepping method to use
        save_history (bool, optional): whether to save the history of the time stepping. Defaults to True.

    Returns:
        tuple: (t_all, u_all, t_ev, u_ev) where t_all and u_all are the time and solution arrays for the entire simulation, 
        and t_ev and u_ev are the time and solution arrays at the event points.
    """
    
    t_ev = np.zeros(num_cycles+1)
    t_ev[0] = problem.t_start
    u_ev = np.zeros(num_cycles+1, dtype=complex)
    u_ev[0] = problem.u_start
    states = []
    t_all = [problem.t_start]
    u_all = [problem.u_start]

    timestepper = TimeStepper(problem=problem, method=method, dt=dt, save_history=save_history)
    propagator = TimeStepperPropagator(timestepper=timestepper, direction=1, terminate_cycle = terminate_cycle)
    
    for ev in range(num_cycles):
        
        current_state = make_state(t_ev[ev], u_ev[ev], ev)
        states.append(current_state)
        t_new, u_new = propagator.propagate(state=current_state)

        t_ev[ev+1] = t_new
        u_ev[ev+1] = u_new
        
        # extract solution for the current cycle
        t_local = np.asarray(propagator.history['t'])
        u_local = np.asarray(propagator.history['u'])
        
        # save the solution for the current cycle
        t_all.extend(t_local[1:])
        u_all.extend(u_local[1:])
        
    return np.asarray(t_all), np.asarray(u_all), np.asarray(t_ev), np.asarray(u_ev)
