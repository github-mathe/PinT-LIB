from dbm import error

import numpy as np
from batpint.problems.dahlquist import Dahlquist

class DahlquistBackwardEuler(Dahlquist):
    def __init__(self, problem, dt = 1e-3):
        self.problem = problem
        self.dt = dt  # Default time step for the Backward Euler method

    def set_initial_conditions(self, u_start, t_start, P_start, num_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the solution of the Dahlquist problem using the Backward Euler method for a specified number of events."""
        self.u_start = u_start
        self.t_start = t_start
        self.P_start = P_start
        self._num_events = num_events
        self.P = [self.P_start]
        self.tP = [self.t_start]
        self.uP = [self.u_start]
        return 

    def advance(self,t,u, P_start = None):
        """Compute the solution of the Dahlquist problem using the Backward Euler method one step."""
        P_start = P_start if P_start is not None else self.problem.P_start
        t_next = t + self.dt
        u_next = (u + self.dt * np.sin(self.problem.alpha * t_next)) / (1 - self.dt * self.problem.lam(P_start))
        return t_next, u_next
    
    def advance_event(self, t_start = None, u_start = None, P_start = None) -> np.ndarray:
        """Compute the solution of the Dahlquist problem using the Backward Euler method one pseudoperiod."""
        t_start = t_start if t_start is not None else self.problem.t_start
        u_start = u_start if u_start is not None else self.problem.u_start
        P_start = P_start if P_start is not None else self.problem.P_start
        assert self.problem.alpha**2 + self.problem.lam(P_start)**2 != 0, "resonance regime, different analytical solution"
        t = [t_start]
        u = [u_start]

        while True:
            tt, uu = self.advance(t[-1], u[-1], P_start)
            t.append(tt)
            u.append(uu)
            event_found, t_event, u_event = self.problem.check_event(t_start, u_start, np.array(t[-2:]), np.array(u[-2:]), P_start)
            if event_found:
                t[-1] = t_event
                u[-1] = u_event
                break  # Exit the loop if an event is found
        return np.array(t), np.array(u)

    def solve(self, num_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the solution of the Dahlquist problem using the Backward Euler method for a specified number of events."""
        if num_events < 0:
            raise ValueError("num_events must be a non-negative integer.")
        if num_events == 0:
            return (np.array([self.problem.t_start]), np.array([self.problem.u_start]), np.array(self.tP), np.array(self.uP))
        self.set_initial_conditions(self.problem.u_start, self.problem.t_start, self.problem.P_start, num_events)
        self.t = np.array([self.t_start])
        self.u = np.array([self.u_start])
        for _ in range(num_events):
            tt,uu = self.advance_event(self.t[-1], self.u[-1], self.P[-1])
            self.t = np.concatenate((self.t, tt[1:]))  # Exclude the first point to avoid duplication
            self.u = np.concatenate((self.u, uu[1:]))  # Exclude the first point to avoid duplication
            self.P.append(self.P[-1] + 1)
            self.tP.append(self.t[-1])
            self.uP.append(self.u[-1])
        return self.t, self.u, np.array(self.tP), np.array(self.uP)

    def compute_event_error(self, num_events: int, dt_exact: float) -> float:
        """Compute the error between the exact solution and the Backward Euler solution for a specified number of events."""
        _, _, tP_exact, uP_exact = self.problem.u_exact_global(num_events, dt_exact)
        self.dt = dt_exact
        _,_, tP_num, uP_num = self.solve(num_events)
        assert len(tP_exact) == len(tP_num), "Number of events in exact and numerical solutions do not match."
        min_len = min(len(tP_exact), len(tP_num))
        uP_exact = uP_exact[:min_len]
        uP_num = uP_num[:min_len]
        tP_exact = tP_exact[:min_len]
        tP_num = tP_num[:min_len]
        error_uP = np.linalg.norm(uP_exact - uP_num, ord=np.inf)
        error_tP = np.linalg.norm(tP_exact - tP_num, ord=np.inf)
        return error_tP, error_uP 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    dahlquist_problem = Dahlquist(
                        u_start=1.0 + 0.0j,
                        t_start=0.0,
                        P_start=1,
                        alpha=6.001,
                        lam=lambda n: 1j*(1+0.01*n))
    num_events = 2
    dt_exact = 0.001
    dt_num = dt_exact

    t_ex,u_ex, tp_ex, up_ex = dahlquist_problem.u_exact_global(num_events=num_events, dt=dt_exact)

    dahlquist_be = DahlquistBackwardEuler(dahlquist_problem, dt = dt_num)

    dtVals = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,1e-6]
    error_tP = np.zeros_like(dtVals)
    error_uP = np.zeros_like(dtVals)

    for i,dt in enumerate(dtVals):
        error_tP[i], error_uP[i] = dahlquist_be.compute_event_error(num_events, dt)
        print(f"dt = {dt}, error_tP = {error_tP[i]}, error_uP = {error_uP[i]}")
    plt.figure()
    plt.loglog(dtVals, error_tP, "--", c="gray", label=r"$\|tP_{ex} -tP_{num}\|_\infty$ error")
    plt.loglog(dtVals, error_uP, "-*", c="gray", label=r"$\|uP_{ex} -uP_{num}\|_\infty$ error")
    plt.xlabel("$dt$"), plt.ylabel("Error"), plt.grid();plt.legend()
    plt.draw()
    plt.show()