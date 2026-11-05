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
        
    def solve_dahlquist_exact_local(self, t, cycle, t_start, u_start):
        lam_cycle = self.params["lam"](cycle)
        denominator = self.params["alpha"]**2 + lam_cycle**2
        C = np.exp(-lam_cycle * t_start) * \
            (u_start + (self.params["alpha"] * np.cos(self.params["alpha"] * t_start) \
            + lam_cycle * np.sin(self.params["alpha"] * t_start)) / denominator)
        U = (C * np.exp(lam_cycle * (t)) - (self.params["alpha"] * np.cos(self.params["alpha"] * t) +\
            lam_cycle * np.sin(self.params["alpha"] * t)) / denominator)
        return U
    
    def solve_dahlquist_exact(self, t):
        if t < self.t_start:
            raise ValueError("Target time must be > = initial time")
        T = self.t_start
        U = self.u_start
        cycle = 0
        
        while True:
            lam_cycle = self.params["lam"](cycle)
            P = 2*np.pi/np.abs(lam_cycle)
            T_next = T+P
            if t <= T_next: 
                return self.solve_dahlquist_exact_local(t, cycle, T, U)
            U = self.solve_dahlquist_exact_local(T_next, cycle, T, U) 
            T = T_next
            cycle += 1
            
    def solve_dahlquist_exact_cycles(self, num_cycles, dt=1e-3):
        """Compute the exact solution over multiple cycles."""
    
        t_ev = np.zeros(num_cycles + 1)
        u_ev = np.zeros(num_cycles + 1, dtype=complex)
    
        t_ev[0] = self.t_start
        u_ev[0] = self.u_start
    
        t_all = []
        u_all = []
    
        for cycle in range(num_cycles):
    
            lam = self.params["lam"](cycle)
    
            # Homogeneous period
            P = 2 * np.pi / np.abs(lam)
    
            # Next cycle endpoint
            t_ev[cycle + 1] = t_ev[cycle] + P
    
            # Exact solution at the endpoint
            u_ev[cycle + 1] = self.solve_dahlquist_exact_local(
                t_ev[cycle + 1],
                cycle,
                t_ev[cycle],
                u_ev[cycle]
            )
    
            # Time grid for the current cycle
            n_steps = max(1, int(np.ceil(P / dt)))
    
            t_cycle = np.linspace(
                t_ev[cycle],
                t_ev[cycle + 1],
                n_steps + 1
            )
    
            # Exact solution on the local time grid
            u_cycle = self.solve_dahlquist_exact_local(
                t_cycle,
                cycle,
                t_ev[cycle],
                u_ev[cycle]
            )
    
            t_all.append(t_cycle)
            u_all.append(u_cycle)
        t_all = np.concatenate([
                t if i == 0 else t[1:]
                for i, t in enumerate(t_all)
            ])
            
        u_all = np.concatenate([
            u if i == 0 else u[1:]
            for i, u in enumerate(u_all)
        ])
        
        return t_all, u_all, t_ev, u_ev
                    
    
        
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
    
        
        
def solve_dahlquist_cycles(problem, method, dt, num_cycles, terminate_cycle = None, direction=1, save_history=True):
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
        
        current_state = propagator.make_state(t_ev[ev], u_ev[ev], ev)
        states.append(current_state)
        t_new, u_new = propagator.propagate(t_ev[ev], u_ev[ev], ev)

        t_ev[ev+1] = t_new
        u_ev[ev+1] = u_new
        
        # extract solution for the current cycle
        t_local = np.asarray(propagator.history['t'])
        u_local = np.asarray(propagator.history['u'])
        
        # save the solution for the current cycle
        t_all.extend(t_local[1:])
        u_all.extend(u_local[1:])
        
    return np.asarray(t_all), np.asarray(u_all), np.asarray(t_ev), np.asarray(u_ev)

