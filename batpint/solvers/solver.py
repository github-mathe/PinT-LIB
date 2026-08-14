import numpy as np

class Solver(object):
    def __init__(self, problem, method, dt = 1e-3):
        self.problem = problem
        self.method = method
        self.dt = dt  # Default time step

    def set_initial_conditions(self, u_start, t_start, P_start) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the solution of the Dahlquist problem using the Backward Euler method for a specified number of events."""
        self.u_start = u_start
        self.t_start = t_start
        self.P_start = P_start
        self.P = [self.P_start]
        self.tP = [self.t_start]
        self.uP = [self.u_start]

    
    def advance_event(self, t_start = None, u_start = None, P_start = None) -> np.ndarray:
        """Compute the solution of the Dahlquist problem using the Backward Euler method one pseudoperiod."""
        t_start = t_start if t_start is not None else self.problem.t_start
        u_start = u_start if u_start is not None else self.problem.u_start
        P_start = P_start if P_start is not None else self.problem.P_start
        t = [t_start]
        u = [u_start]

        while True:
            tt, uu = self.method(t[-1], u[-1], self.dt, P_start)
            t.append(tt)
            u.append(uu)
            event_found, t_event, u_event = self.problem.check_event(t_start, u_start, np.array(t[-2:]), np.array(u[-2:]), P_start)
            if event_found:
                t[-1] = t_event
                u[-1] = u_event
                break  # Exit the loop if an event is found
        return np.array(t), np.array(u)

    def solve(self, num_events: int):
        """Solve the problem for a given number of events."""
        if num_events < 0:
            raise ValueError("num_events must be a non-negative integer.")

        if num_events == 0:
            return (
                np.asarray([self.problem.t_start]),
                np.asarray([self.problem.u_start]),
                np.asarray(self.tP if hasattr(self, "tP") else [self.problem.t_start]),
                np.asarray(self.uP if hasattr(self, "uP") else [self.problem.u_start]),
            )

        self.set_initial_conditions(self.problem.u_start, self.problem.t_start, self.problem.P_start)

        t_list = [self.t_start]
        u_list = [self.u_start]
        P_list = [self.P_start]
        tP_list = [self.t_start]
        uP_list = [self.u_start]

        for _ in range(num_events):
            tt, uu = self.advance_event(t_list[-1], u_list[-1], P_list[-1])

            # Skip the first point to avoid duplicating the current state
            t_list.extend(tt[1:].tolist())
            u_list.extend(uu[1:].tolist())

            P_list.append(P_list[-1] + 1)
            tP_list.append(t_list[-1])
            uP_list.append(u_list[-1])

        return np.asarray(t_list), np.asarray(u_list), np.asarray(tP_list), np.asarray(uP_list)

    def compute_event_error(self, num_events: int, dt: float, exact_solver) -> float:
        """Compute the error between the exact solution and the Backward Euler solution for a specified number of events."""
        _, _, tP_exact, uP_exact = exact_solver(num_events, dt)
        self.dt = dt
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
    from batpint.problems.dahlquist import Dahlquist
    from batpint.solvers.backwardeuler import BackwardEuler
    # Example usage
    dahlquist = Dahlquist(
                        u_start=1.0 + 0.0j,
                        t_start=0.0,
                        P_start=1,
                        alpha=6.001,
                        lam=lambda n: 1j*(1+0.01*n))
    num_events = 5
    dt_exact = 0.001
    dt_num = dt_exact

    t_ex,u_ex, tp_ex, up_ex = dahlquist.u_exact_global(num_events=num_events, dt=dt_exact)

    BE_modified = lambda t, u, dt, P_start: BackwardEuler(f=lambda t, u: dahlquist.f(t, u, P_start),df = lambda t, u: dahlquist.df(t, u, P_start), t0 = t, y0=u, dt=dt, T = t + dt)[0:2]
    solverBE = Solver(dahlquist, dt = dt_num, method = BE_modified)
    _,_,tpBE,upBE = solverBE.solve(num_events)

    dtVals = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    error_tP = np.zeros_like(dtVals)
    error_uP = np.zeros_like(dtVals)

    for i,dt in enumerate(dtVals):
        error_tP[i], error_uP[i] = solverBE.compute_event_error(num_events, dt, dahlquist.u_exact_global)
        print(f"dt = {dt}, error_tP = {error_tP[i]}, error_uP = {error_uP[i]}")
    plt.figure()
    plt.loglog(dtVals, error_tP, "--", c="gray", label=r"$\|tP_{ex} -tP_{num}\|_\infty$ error")
    plt.loglog(dtVals, error_uP, "-*", c="gray", label=r"$\|uP_{ex} -uP_{num}\|_\infty$ error")
    plt.xlabel("$dt$"), plt.ylabel("Error"), plt.grid();plt.legend()
    plt.draw()
    plt.show()