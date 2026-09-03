import numpy as np
import copy

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
        self.state = None
        self.save_history = save_history
        self.history = {"t": [], "u": []} if save_history else None

    def reset_history(self):
        if self.save_history:
            self.history["t"] = [self.state.t]
            self.history["u"] = [copy.deepcopy(self.state.u)]

    def set_state(self, state):
        self.state = copy.deepcopy(state)
        self.state.terminated_step = False
        self.reset_history()

    def _resolve_params(self, names):
        kwargs = {}
        for name in names:
            if hasattr(self.state, name):
                kwargs[name] = getattr(self.state, name)
            elif name in self.problem.params:
                kwargs[name] = self.problem.params[name]
            else:
                raise KeyError(f"Parameter '{name}' could not be resolved.")
        return kwargs

    # wrapper for problem functions to include cycle and other parameters
    def rhs(self, t, u):
        kwargs = self._resolve_params(self.problem.rhs_params)
        return self.problem.rhs(t, u, **kwargs)
    
    def jacobian(self, t, u):
        if self.problem.jacobian is None:
            return None
        kwargs = self._resolve_params(self.problem.jacobian_params)
        return self.problem.jacobian(t, u, **kwargs)

    def event(self, t, u):
        kwargs = self._resolve_params(self.problem.event_params)
        return self.problem.event(t, u, **kwargs)

    def terminate_step(self, t, u):
        if self.problem.terminate_step is None:
            return False
        kwargs = self._resolve_params(self.problem.terminate_step_params)
        return self.problem.terminate_step(t, u, **kwargs)
    
    def step(self, h=None):
        if h is None:
            h = self.dt
        jacobian = self.jacobian if self.problem.jacobian else None
        method_kwargs = self._resolve_params(self.method.step_params)
        self.state.u = self.method.step(rhs=self.rhs, t=self.state.t, u=self.state.u, h=h, jacobian=jacobian, **method_kwargs)
        self.state.t += h

        if self.save_history:
            self.history["t"].append(self.state.t)
            self.history["u"].append(copy.deepcopy(self.state.u))

        if self.terminate_step(self.state.t, self.state.u):
            raise RuntimeError(
                f"STEP_TERMINATION: cycle={self.state.cycle}, "
                f"t={self.state.t}, u={self.state.u}"
            )
        return self.state
    
    def advance_to_time(self, t_end):
        """
        Advance the solution to a specified end time.
        """
        while self.state.t < t_end:
            h = min(self.dt, t_end - self.state.t)  # Adjust step size if close to t_end
            self.step(h=h)
        return self.state

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
        if self.problem.event is None:
            raise ValueError("No event function defined for this problem.")
        g_old = self.event(self.state.t, self.state.u)

        # If the current state is already on the event surface,
        # move away from it before searching for the next event.
        if abs(g_old) <= self.event_tol:
            self.step()
            g_old = self.event(self.state.t, self.state.u)

        while True:
            t_old = self.state.t
            u_old = copy.deepcopy(self.state.u)
            self.step()
            g_new = self.event(self.state.t, self.state.u)
            if self._event_crossed(g_old, g_new, direction):
                # Linear interpolation to approximate event point
                t_event, u_event = self._interpolate_event(t_old, u_old, self.state.t, self.state.u, g_old, g_new)
                g_event = self.event(t_event, u_event)

                if abs(g_event) > self.event_tol:
                    raise RuntimeError("Event not localized within the specified tolerance.")

                self.state.t = t_event
                self.state.u = u_event

                if self.save_history:
                    self.history["t"][-1] = self.state.t
                    self.history["u"][-1] = copy.deepcopy(self.state.u)

                return self.state

            g_old = g_new