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
    save_history : bool, optional
        If True, store the history of time and state.
    """
    
    def __init__(self, problem, method, dt, event_tol=1e-6, save_history=False):
        self.problem = problem
        self.method = method

        self.dt = dt  # Default time step
        self.event_tol = event_tol  # Tolerance for event detection

        self.history = {"t": [], "u": []} if save_history else None
        
    def reset_history(self):
        """
        Reset the history of time and state.
        """
        if self.history is None:
            return
        self.history["t"] = [self.t]
        self.history["u"] = [np.copy(self.u)]
        
    def step(self, h=None):
        
        if h is None:
            h = self.dt
        jacobian = self.problem.jacobian_value if self.problem.jacobian is not None else None
        self.u = self.method.step(rhs=self.problem, t=self.t, u=self.u, h=h, jacobian=jacobian)
        self.t += h
        
        if self.history is not None:
            self.history["t"].append(self.t)
            self.history["u"].append(np.copy(self.u))
        
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

        g_old = self.problem.event_value(self.t, self.u)

        # If the current state is already on the event surface,
        # move away from it before searching for the next event.
        if abs(g_old) <= self.event_tol:
            self.step()
            g_old = self.problem.event_value(self.t, self.u)

        while True:
            t_old = self.t
            u_old = np.copy(self.u)
            self.step()
            g_new = self.problem.event_value(self.t, self.u)
            if self._event_crossed(g_old, g_new, direction):

                # Linear interpolation to approximate event point
                t_event, u_event = self._interpolate_event(t_old, u_old, self.t, self.u, g_old, g_new)
                g_event = self.problem.event_value(t_event, u_event)

                if abs(g_event) > self.event_tol:
                    raise RuntimeError("Event not localized within the specified tolerance.")

                self.t = t_event
                self.u = u_event
                if self.history is not None:
                    self.history["t"][-1] = self.t
                    self.history["u"][-1] = np.copy(self.u)
                
                return self.t, self.u

            g_old = g_new
     
     
class TimeStepperPropagator(Propagator):
    """
    Propagator based on an event-driven TimeStepper.
    """

    def __init__(self, timestepper, direction=0):
        self.timestepper = timestepper
        self.direction = direction

    @property
    def history(self):
        return self.timestepper.history

    def propagate(self, t, u):
        self.timestepper.t = t
        self.timestepper.u = u
        if self.timestepper.history is not None:
            self.timestepper.reset_history()
        return self.timestepper.advance_to_event(direction=self.direction)