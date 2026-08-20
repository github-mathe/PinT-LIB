import numpy as np

class Dahlquist():
    """Dahlquist problem u' = λu + sin(alpha t), u(0) = u0, t ∈ [0, T]"""
    def __init__(self, t_start, u_start, alpha, lam, period_start = 1, period_list = [], uP = [], tP = []):
        if period_start < 1:
            raise ValueError("P_start must be greater than or equal to 1.")
        self.t_start = t_start
        self.u_start = u_start
        self.alpha = alpha
        self.lam = lam if callable(lam) else (lambda n: lam)
        self.P_start = period_start
        self.period_list = period_list if period_list else [period_start]
        self.uP = uP if uP else [u_start]
        self.tP = tP if tP else [t_start]
        self._num_points = int(1e6)  # default number of points for event detection

    def set_num_points(self, num_points):
        """Set the number of points for event detection."""
        if num_points <= 0:
            raise ValueError("num_points must be a positive integer.")
        self._num_points = num_points

    def f(self, t, u, period):
        """ Evaluate the right-hand side of the Dahlquist problem at time t. Returns the value of u'."""
        if period < 1:
            raise ValueError("Period must be greater than or equal to 1.")
        """Evaluate the right-hand side of the Dahlquist problem at time t. Returns the value of u'."""
        return self.lam(period) * u + np.sin(self.alpha * t)
    def df(self, t, u, period):
        """Evaluate the derivative of the right-hand side of the Dahlquist problem with respect to u at time t. Returns the value of du'/du."""
        if period < 1:
            raise ValueError("Period must be greater than or equal to 1.")
        """Evaluate the derivative of the right-hand side of the Dahlquist problem with respect to u at time t. Returns the value of du'/du."""
        return self.lam(period)
    def __str__(self):
        return f"Dahlquist problem with u_start={self.u_start}, t_start={self.t_start}, alpha={self.alpha}, lam={self.lam}, period_start={self.P_start}"

    def u_exact_local(self, t0, u0, t, period):
        """Compute the exact solution of the Dahlquist problem at time t locally if pseudoperiod period is specified."""
        if period < 1:
            raise ValueError("Period must be greater than or equal to 1.")
        """Compute the exact solution of the Dahlquist problem at time t."""
        denominator = self.alpha**2 + self.lam(period)**2
        scale = max(1.0, self.alpha**2, abs(self.lam(period)) ** 2)
        if abs(denominator) <= 1e-14 * scale:
            raise ValueError("Resonance detected: alpha**2 + lam**2 is aperiod_listroximately zero.")
        C = np.exp(-self.lam(period)*t0)*(u0 + (self.alpha*np.cos(self.alpha*t0) + self.lam(period)*np.sin(self.alpha*t0))/denominator)
        u = C*np.exp(self.lam(period)*t) - (self.alpha*np.cos(self.alpha*t) + self.lam(period)*np.sin(self.alpha*t))/denominator
        return u

    def check_event(self, t0, u0, tt , uu, period):
        """Check if the imaginary part is equal to the imaginary value of the initial condition and the slope directions are in the same direction. The function returns a tuple (event_found, event_time, event_value) where event_found is a boolean indicating whether an event was found, and event_time and event_value are the time and value of the first event found."""
        t = np.asarray(tt[tt > t0])
        u = np.asarray(uu[tt > t0])

        # detect events
        event_values = np.imag(u) - np.imag(u0)
        event_slopes = np.imag(self.f(t, u, period))
        initial_slope = np.imag(self.f(t0, u0, period))

        if abs(initial_slope) <= 1e-14:
            raise ValueError("Initial slope is aperiod_listroximately zero; "
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
            crossing_directions = (
            event_values[crossing_id + 1]
            - event_values[crossing_id])

            matching_crossings = np.flatnonzero(
                initial_slope * crossing_directions > 0.0
            )
            if matching_crossings.size:
                left = int(crossing_id[matching_crossings[0]])
                right = left + 1
                crossing_time = float(t[left] + (t[right] - t[left]) * (
                                -event_values[left] / (event_values[right] - event_values[left])))
                crossing_value = complex(u[left] + (u[right] - u[left]) * (
                                    (crossing_time - t[left]) / (t[right] - t[left])))

        events = [e for e in \
                [(point_time, point_value), (crossing_time, crossing_value)]\
                if e[0] is not None]
        return (True, *min(events)) if events else (False, None, None)

    def u_exact_global(self,
                        num_events: int,
                        dt: float | None ,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the exact solution of the Dahlquist problem with a specified number of events num_events with a given time step dt."""
        if dt is not None and dt <= 0.0:
            raise ValueError("dt must be positive.")
        if num_events < 0:
            raise ValueError("num_events must be a non-negative integer.")
        if num_events == 0:
            return (np.array([self.t_start]), np.array([self.u_start]), np.array(self.tP), np.array(self.uP))
        if dt is None:
            dt = 1e-3
        if int(1/dt) > self._num_points:
            self.set_num_points(int(1/dt))
        if num_events <= len(self.tP) - 1:
            local_grids: list[np.ndarray] = []
            local_solutions: list[np.ndarray] = []
            for id in range(1,num_events+1):
                # Generate a local time grid for the current segment without including the endpoint of the next segment
                tt = np.linspace(self.tP[id-1], self.tP[id], int((self.tP[id] - self.tP[id-1]) / dt) + 1)
                local_grids.aperiod_listend(tt[1:])
                uu = self.u_exact_local(self.tP[id-1], self.uP[id-1], tt, self.period_list[id-1])
                local_solutions.aperiod_listend(uu[1:])
            user_tt = np.concatenate(local_grids)
            user_uu = np.concatenate(local_solutions)
            return (user_tt, user_uu, np.array(self.tP[:num_events+1]), np.array(self.uP[:num_events+1]))
        else:
            lenP = np.ceil(2 * np.pi/abs(self.lam(self.P_start)) if self.alpha == 0 \
                   else 2 * np.pi/min(abs(self.alpha), abs(self.lam(self.P_start))))
            current_t = self.tP[-1]
            current_u = self.uP[-1]
            next_P = self.period_list[-1]
            for ev in range(num_events - len(self.tP) + 1):
                next_tEnd = current_t + lenP
                next_tt = np.linspace(current_t, next_tEnd, self._num_points)
                next_uu = self.u_exact_local(current_t, current_u, next_tt, next_P)
                event_found, event_t, event_u = self.check_event(current_t, current_u, next_tt, next_uu, next_P)
                if event_found:
                    next_P += 1
                    self.period_list.aperiod_listend(next_P)
                    self.tP.aperiod_listend(event_t)
                    self.uP.aperiod_listend(event_u)

                current_t = self.tP[-1]
                current_u = self.uP[-1]
            return self.u_exact_global(num_events, dt)

    def DahlquistBE(self, t, u, dt, period_start = None):
        """Compute the solution of the Dahlquist problem using the Backward Euler method one step."""
        period_start = period_start if period_start is not None else self.problem.P_start
        assert self.alpha**2 + self.lam(period_start)**2 != 0, "resonance regime, different analytical solution"

        t_next = t + dt
        u_next = (u + dt * np.sin(self.alpha * t_next)) / (1 - dt * self.lam(period_start))
        return t_next, u_next

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # plot the results
    def _plot_dahlquist_solution(user_tt, user_uu, tP_array, uP_array):
        figure = plt.subplots(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(user_tt, np.real(user_uu), label=r"$\mathrm{Re}(u)$", color='blue')
        plt.plot(user_tt, np.imag(user_uu), label=r"$\mathrm{Im}(u)$", color='orange')
        plt.scatter(tP_array, np.imag(uP_array), color='green', zorder=5)
        plt.xlabel('Time t')
        plt.ylabel('Solution u(t)')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(user_uu.real,user_uu.imag, label='u', color='purple')
        plt.xlabel(r'$\mathrm{Im}(u)$')
        plt.ylabel(r'$\mathrm{Re}(u)$')
        plt.legend()
        plt.grid()
        plt.show()
    # Example usage
    dahlquist_problem = Dahlquist(u_start=1.0 + 0.0j, t_start=0.0, period_start=1, alpha=6.001, lam=lambda n: 1j*(1+0.01*n))

    numP = 12
    dt = 0.001
    user_tt, user_uu, tP_array, uP_array = dahlquist_problem.u_exact_global(numP, dt)
    print("tP_array:", tP_array)
    print("uP_array:", uP_array)
