import numpy as np
from batpint.parareal.base_propagator import Propagator

class TimeStepper(object):
    """
    Controls time integration and event detection for an IVP.

    The TimeStepper stores the current numerical state and advances it
    using a supplied integration method.

    Parameters
    ----------
    problem : Problem
        IVP.
    method : IntegrationMethod
        Numerical one-step time integration method.
    dt : float
        Default time-step size.
    event_tol : float, optional
        Tolerance used for event localization.
    """
    
    def __init__(self, problem, method, dt, event_tol=1e-6):
        self.problem = problem
        self.method = method

        self.dt = dt  # Default time step
        self.event_tol = event_tol  # Tolerance for event detection

        # the current numerical state
        self.t = problem.t_start
        self.u = np.copy(problem.u_start)

    def set_state(self, t, u):
        """
        Set the current numerical state.
        """
        self.t = t
        self.u = np.copy(u)

    def step(self, h=None,**kwargs):
            """
            Perform a single time step using the specified integration method.
            """
            if h is None: 
                h = self.dt
            jacobian = None
            if self.problem.jacobian is not None:
                jacobian = self.problem.jacobian_value
            self.u = self.method.step(
                                    rhs=self.problem,
                                    t=self.t,
                                    u=self.u,
                                    h=h,
                                    jacobian = jacobian,
                                    **kwargs
                                    )
            self.t += h
            return self.t, self.u
    
    def advance_to_time(self, t_end,**kwargs):
        """
        Advance the solution to a specified end time.
        """
        while self.t < t_end:
            h = min(self.dt, t_end - self.t)  # Adjust step size if close to t_end
            self.step(h=h,**kwargs)
        return self.t, self.u

    @staticmethod
    def _event_crossed(g_old, g_new, direction):
        """
        Check if the event function has crossed zero.
        """
        if direction == 0:
            return g_old * g_new < 0  # Any crossing
        elif direction > 0:
            return g_old < 0 and g_new >= 0  # Crossing from negative to positive
        else:
            return g_old > 0 and g_new <= 0  # Crossing from positive to negative   

    @staticmethod
    def _interpolate_event(t_old, u_old, t_new, u_new, g_old, g_new):
        theta = abs(g_old) / (abs(g_old) + abs(g_new))

        t_event = t_old + theta * (t_new - t_old)
        u_event = u_old + theta * (u_new - u_old)

        return t_event, u_event
            
    def advance_to_event(self, direction=0, **kwargs):
        """
        Advance the solution until the next admissible event crossing.

        Parameters
        ----------
        direction : int, optional
            Crossing direction:
            0  -> any crossing,
            1  -> negative to positive,
            -1 -> positive to negative.
        max_steps : int, optional
            Maximum number of time steps before aborting.
        **kwargs
            Dynamic problem data forwarded to the integration method
            and event function.
        Returns
        -------
        t_event, u_event
            Approximate time and state at the detected event.
        """

        g_old = self.problem.event_value(self.t, self.u, **kwargs)

        # If the current state is already on the event surface,
        # move away from it before searching for the next event.
        if abs(g_old) <= self.event_tol:
            self.step(**kwargs)
            g_old = self.problem.event_value(self.t, self.u, **kwargs)

        while True:

            t_old = self.t
            u_old = np.copy(self.u)

            self.step(**kwargs)

            g_new = self.problem.event_value(self.t, self.u, **kwargs)

            if self._event_crossed(g_old, g_new, direction):

                # Linear interpolation to approximate event point
                t_event, u_event = self._interpolate_event(t_old, u_old, self.t, self.u, g_old, g_new)
                g_new = self.problem.event_value(t_event, u_event, **kwargs)

                if abs(g_new) > self.event_tol:
                    raise RuntimeError("Event not localized within the specified tolerance.")

                self.t = t_event
                self.u = u_event
                return self.t, self.u

            g_old = g_new
     
class TimeStepperPropagator(Propagator):
    """
    Propagator based on an event-driven TimeStepper.
    """

    def __init__(self, timestepper, direction=0):
        self.timestepper = timestepper
        self.direction = direction

    def propagate(self, t, u,**kwargs):
        self.timestepper.set_state(t, u)
        return self.timestepper.advance_to_event(direction=self.direction, **kwargs)


if __name__ == "__main__":
    from batpint.time_integration.backwardeuler import BackwardEuler
    from batpint.time_integration.newton import Newton
    from batpint.problems.base_problem import Problem

    # Dahlquist-type pseudoperiodic problem
    def lam(n):
        return 1j * (1.0 + 0.01 * n)

    def rhs(t, u, alpha, lam, cycle, **kwargs):
        return lam(cycle) * u + np.sin(alpha * t)

    def jacobian(t, u, alpha, lam, cycle, **kwargs):
        return lam(cycle)

    def event(t, u, u_start, **kwargs):
        return np.imag(u) - np.imag(u_start)

    # ============================================================
    # Initial data
    # ============================================================

    t_start = 0.0
    u_start = 1.0 + 0.0j
    period_start = 1
    alpha = 6.001 #1.1 
    dt = 0.000001
    # ============================================================
    # Create problem
    # ============================================================
    problem = Problem(
        t0=t_start,
        u0=u_start,
        rhs=rhs,
        jacobian=jacobian,
        event=event,
        alpha=alpha,
        lam=lam,
    )
    # ============================================================
    # Create TimeStepper
    # ============================================================
    method = BackwardEuler(
        Newton()
    )
    timestepper = TimeStepper(
        problem=problem,
        method=method,
        dt=dt,
    )
    # ============================================================
    # Create propagator
    # ============================================================
    propagator = TimeStepperPropagator(
        timestepper=timestepper,
        direction=1,
    )
    # ============================================================
    # Compute pseudoperiod points
    # ============================================================
    N = 2
    TP = np.zeros(N + 1)
    UP = np.zeros(N + 1, dtype=complex)
    TP[0] = t_start
    UP[0] = u_start
    for j in range(N):
        cycle = period_start + j
        terminate = abs(lam(cycle))**2 - alpha**2 == 0
        if terminate:
            print(f"Terminating at cycle {cycle} due to singularity.")
            break
        TP[j + 1], UP[j + 1] = propagator.propagate(
            TP[j],
            UP[j],
            cycle=cycle,
            u_start=UP[j],
            t_start=TP[j])
        print(
            f"period = {cycle:2d}, "
            f"lambda = {lam(cycle)}, "
            f"t = {TP[j + 1]:.8f}, "
            f"u = {UP[j + 1]}"
        )