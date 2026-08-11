import numpy as np
from collections.abc import Callable

from scripts.notebooks.DahlquistProblem.ModifiedProblem.modified_dahlquist import Period_end
class Dahlquist():
    """Dahlquist problem u' = λu + sin(alpha t), u(0) = u0, t ∈ [0, T]"""
    def __init__(self, u_start, t_start, P_start, alpha, lam):
        self.u_start = u_start
        self.t_start = t_start
        self.alpha = alpha
        if callable(lam):
            self.lam = lam
        else:
            self.lam = lambda P: lam
        self.P_start = P_start

    def f(self, u, t, P):
        """Evaluate the right-hand side of the Dahlquist problem at time t. Returns the value of u'."""
        return self.lam(P) * u + np.sin(self.alpha * t)

    def __str__(self):
        return f"Dahlquist problem with u_start={self.u_start}, t_start={self.t_start}, alpha={self.alpha}, lam={self.lam}, P_start={self.P_start}"

    def u_exact_local(self, t, P):
        """Compute the exact solution of the Dahlquist problem at time t."""
        t_array = np.atleast_1d(t)
        denominator = self.alpha**2 + self.lam(P)**2
        scale = max(1.0, self.alpha**2, abs(self.lam(P)) ** 2)
        if abs(denominator) <= 1e-14 * scale:
            raise ValueError("Resonance detected: alpha**2 + lam**2 is approximately zero.")
        C = np.exp(-self.lam(P)*self.t_start)*(self.u_start + (self.alpha*np.cos(self.alpha*self.t_start) + self.lam(P)*np.sin(self.alpha*self.t_start))/denominator)
        u = C*np.exp(self.lam(P)*t) - (self.alpha*np.cos(self.alpha*t) + self.lam(P)*np.sin(self.alpha*t))/denominator
        return u

    def check_event(self, tt , uu):
        """Check if the imaginary part is equal to the imaginary value of the initial condition."""
        t = np.asarray(tt[tt > self.t_start])
        u = np.asarray(uu[tt > self.t_start])
        dudt = self.f(u, t, self.P_start)

        # detect events
        event_values = np.imag(u) - np.imag(self.u_start)
        event_slopes = np.imag(dudt)
        initial_slope = np.imag(self.f(self.u_start, self.t_start, self.P_start))

        if abs(initial_slope) <= 1e-14:
            raise ValueError("Initial slope is approximately zero; "
            "crossing direction is undefined.")

        if u.size == 0:
            return (True, t, u) if (np.abs(event_values) <= 1e-14) & (initial_slope * event_slopes > 0.0) else (False, None, None)

        # Find points already close to the target level
        point_time = None
        point_value = None
        point_idx = np.flatnonzero((np.abs(event_values) <= 1e-14) & (initial_slope * event_slopes > 0.0))
        if point_idx.size > 0:
            point_index = point_idx[0]
            point_time = float(t[point_index])
            point_value = np.real(u[point_index]) + 1j * np.imag(self.u_start) 

        # Find the first valid interpolated crossing
        crossing_time = None
        crossing_value = None
        crossing_id = np.flatnonzero(event_values[:-1] * event_values[1:] < 0.0)
        if crossing_id.size:
            matching_crossings = np.flatnonzero(initial_slope * event_slopes[crossing_id] > 0.0) 
            if matching_crossings.size:
                left = int(crossing_id[matching_crossings[0]])
                right = left + 1
                crossing_time = float(tt[left] + (tt[right] - tt[left]) * (
                                -event_values[left] / (event_values[right] - event_values[left])))
                crossing_value = complex(uu[left] + (uu[right] - uu[left]) * (
                                    (crossing_time - tt[left]) / (tt[right] - tt[left])))

        events = [e for e in \
                [(point_time, point_value), (crossing_time, crossing_value)]\
                if e[0] is not None]
        return (True, *min(events)) if events else (False, None, None)

    def resetDahlquist(self, t, u, P):
        """Reset the problem to a new initial condition."""
        self.t_start = t
        self.u_start = u
        self.P_start = P

    def u_exact_global(self, 
                        T: float | None ,
                        dt: float | None ,
                        num_points: int | None ,
                        num_events: float | None,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the exact solution of the Dahlquist problem at time t."""
        if (T is None) == (num_events is None):
            raise ValueError("Provide exactly one of T or num_events.")
        if T is not None and T <= self.t_start:
            raise ValueError("T must be greater than t_start.")
        if (dt is None) == (num_points is None):
            raise ValueError("Provide exactly one of dt or num_points.")
        if num_points is not None and num_points < 2:
                raise ValueError("num_points must be at least 2.")
        if dt is not None and dt <= 0.0:
            raise ValueError("dt must be positive.")

        event_times = [self.t_start]
        event_values = [self.u_start]
        lambdas = list[complex] = []

        while True:
            current_event = len(event_times) - 1
            t_start = event_times[current_event]
            P_start = self.P_start
            current_lambda = self.lam(P_start)
            if num_events is not None and current_event >= num_events:
                break
            frequency = min(np.abs(current_lambda), np.abs(self.alpha)) if self.alpha != 0.0 else np.abs(current_lambda) 
            if frequency == 0.0:
                raise ValueError(
                "Cannot determine length of one pseudo-period because lambda and alpha are zero.")
            
            L = float(np.ceil(2.0 * np.pi /frequency))
            t_end = t_start + L if T is None else min(t_start + L, T)
            current_tt  =   np.linspace(t_start, t_end, 1e5, dtype=float) 
            current_uu  =   self.u_exact_local(current_tt, P_start)
            event_found, event_time, _ = self.check_event(current_tt, current_uu)
            if event_found:
                event_times.append(event_time)
                event_values.append(self.u_exact_local(event_time, P_start))
                self.resetDahlquist(event_time, event_values[-1], P_start + 1)
            lambdas.append(current_lambda)
            lambdas.append(current_lambda)