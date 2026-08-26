from dataclasses import dataclass
import numpy as np
from batpint.parareal.base_propagator import Propagator

class TimeStepper:
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
    current_params : Parameters, optional
        Current values of the problem parameters.
    """
    
    def __init__(self, problem, method, dt, event_tol=1e-6, current_params=None):
        self.problem = problem
        self.method = method

        self.dt = dt  # Default time step
        self.event_tol = event_tol  # Tolerance for event detection

        if current_params is None:
            self.current_params = problem.params
        else:
            self.current_params = current_params

        self.rhs = lambda t, u: self.problem(t, u, self.current_params)

        self.jacobian = None
        if self.problem.jacobian is not None:
            self.jacobian = lambda t, u: self.problem.jacobian_value(t, u, self.current_params)

        self.event = None
        if self.problem.event is not None:
            self.event = lambda t, u: self.problem.event_value(t, u, self.current_params)

        self.terminate = None
        if self.problem.terminate is not None:
            self.terminate = lambda t, u: self.problem.termination_value(t, u, self.current_params)

    def step(self, h=None):
        if h is None:
            h = self.dt
        self.u = self.method.step(rhs=self.rhs, t=self.t, u=self.u, h=h, jacobian=self.jacobian)
        self.t += h
        return self.t, self.u
    
    def advance_to_time(self, t_end):
        """
        Advance the solution to a specified end time.
        """
        while self.t < t_end:
            h = min(self.dt, t_end - self.t)  # Adjust step size if close to t_end
            self.step(h=h)
        return self.t, self.u

    @staticmethod
    def _event_crossed(g_old, g_new, direction):
        """
        Check if the event function has crossed zero.
        """
        if direction == 0:
            return g_old * g_new <= 0  # Any crossing
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
            
    def advance_to_event(self, direction=0):
        """
        Advance the solution until the next admissible event crossing.

        Parameters
        ----------
        direction : int, optional
            Crossing direction:
            0  -> any crossing,
            1  -> negative to positive,
            -1 -> positive to negative.
        Returns
        -------
        t_event, u_event
            Approximate time and state at the detected event.
        """

        g_old = self.event(self.t, self.u)

        # If the current state is already on the event surface,
        # move away from it before searching for the next event.
        if abs(g_old) <= self.event_tol:
            self.step()
            g_old = self.event(self.t, self.u)

        while True:

            t_old = self.t
            u_old = np.copy(self.u)
            self.step()
            g_new = self.event(self.t, self.u)
            if self._event_crossed(g_old, g_new, direction):

                # Linear interpolation to approximate event point
                t_event, u_event = self._interpolate_event(t_old, u_old, self.t, self.u, g_old, g_new)
                g_event = self.event(t_event, u_event)

                if abs(g_event) > self.event_tol:
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

    def propagate(self):
        self.timestepper.t = self.timestepper.current_params.get("t_start")
        self.timestepper.u = self.timestepper.current_params.get("u_start")
        return self.timestepper.advance_to_event(direction=self.direction)

if __name__ == "__main__":
    from batpint.time_integration.backwardeuler import BackwardEuler
    from batpint.time_integration.newton import Newton
    from batpint.problems.base_problem import Problem, Parameters, Parameter, CurrentParameters

    # ============================================================
    # Dahlquist-type pseudoperiodic problem
    # ============================================================

    def lam(t, u, cycle):
        return 1j * (1.0 + 0.01 * cycle)

    def rhs(t, u, alpha, lam):
        return lam * u + np.sin(alpha * t)

    def jacobian(t, u, lam):
        return lam

    def event(t, u, u_start):
        return np.imag(u) - np.imag(u_start)

    def terminate(t, u, alpha, lam):
        return abs(alpha**2 - abs(lam)**2) < 1e-10

    params = Parameters(
    Parameter("alpha", value=6.001),
    Parameter("cycle", value=1),
    Parameter("t_start", value=0.0),
    Parameter("u_start", value=1.0 + 0.0j),
    Parameter("lam", function=lam),
    Parameter("period_start", value=1)
    )
    dt = 0.1

    problem = Problem(
    t0=0.0,
    u0=1.0 + 0.0j,
    rhs=rhs,
    params=params,
    jacobian=jacobian,
    event=event,
    terminate=terminate)

    current_params = CurrentParameters(problem.params)

    method = BackwardEuler(
    Newton()
    )

    timestepper = TimeStepper(
        problem=problem,
        method=method,
        dt=dt,
        current_params=current_params,
    )
    propagator = TimeStepperPropagator(
        timestepper=timestepper,
        direction=1,
    )
    # ============================================================
    # Compute pseudoperiod points
    # ============================================================

    N = 20

    TP = np.zeros(N + 1)
    UP = np.zeros(N + 1, dtype=complex)

    TP[0] = Parameters.get("t_start").value
    UP[0] = Parameters.get("u_start").value

    for j in range(N):

        cycle = Parameters.get("period_start").value + j

        current_params.set("cycle", cycle)
        current_params.set("t_start", TP[j])
        current_params.set("u_start", UP[j])

        if timestepper.terminate(TP[j], UP[j]):
            print(
                f"Termination condition met at period "
                f"{cycle}. Stopping propagation."
            )
            break

        TP[j + 1], UP[j + 1] = propagator.propagate()

        print(
            f"period = {cycle:2d}, "
            f"lambda = {current_params.get('lam', TP[j], UP[j])}, "
            f"t = {TP[j + 1]:.8f}, "
            f"u = {UP[j + 1]}"
        )