import numpy as np
from batpint.problems.dahlquist import Dahlquist

class DahlquistBackwardEuler(Dahlquist):
    def __init__(self, problem, dt = 1e-3, PP_num = None, uP_num = None, tP_num = None):
        self.problem = problem
        self.dt = dt  # Default time step for the Backward Euler method
        self.PP_num = PP_num if PP_num else [problem.P_start]
        self.uP_num = uP_num if uP_num else [problem.u_start]
        self.tP_num = tP_num if tP_num else [problem.t_start]
        self._num_points = 1000  # Default number of points for local solutions
        self.t = []
        self.u = []

    def advance(self) -> np.ndarray:
        """Compute the solution of the Dahlquist problem using the Backward Euler method one pseudoperiod."""
        t_start = self.tP_num[-1]
        u_start = self.uP_num[-1]
        lam = self.problem.lam(self.PP_num[-1])
        assert self.problem.alpha**2 + lam**2 != 0, "resonance regime, different analytical solution"

        while True:
            tt = np.linspace(t_start, t_start+self._num_points*self.dt, self._num_points)
            uu = np.zeros_like(tt, dtype=complex)
            uu[0] = u_start
            for i in range(self._num_points-1):
                uu[i+1] = (uu[i] + self.dt*np.sin(self.problem.alpha*tt[i+1])) / (1 - self.dt*lam)
            event_found, t_event, u_event = self.problem.check_event(self.tP_num[-1], self.uP_num[-1], tt, uu)
            if event_found:
                self.tP_num.append(t_event)
                self.uP_num.append(u_event)
                user_t = tt[tt < t_event] # Exclude the first point to avoid duplication
                user_u = uu[tt < t_event]  # Exclude the first point to avoid duplication
                self.t.append(user_t)
                self.u.append(user_u)
                self.PP_num.append(self.PP_num[-1]+1)
                break  # Exit the loop if an event is found
            t_start += self._num_points * self.dt
            u_start = uu[-1]
            self.t.append(tt[:-1])
            self.u.append(uu[:-1])

    def solve_global(self, num_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the solution of the Dahlquist problem using the Backward Euler method for a specified number of events."""
        if num_events < 0:
            raise ValueError("num_events must be a non-negative integer.")
        if num_events == 0:
            return (np.array([self.problem.t_start]), np.array([self.problem.u_start]), np.array(self.tP_num), np.array(self.uP_num))
        while len(self.tP_num) - 1 < num_events:
            self.advance()
        t_values = np.append(np.concatenate(self.t), self.tP_num[num_events])
        u_values = np.append(np.concatenate(self.u), self.uP_num[num_events])
        return t_values, u_values, np.array(self.tP_num[:num_events+1]), np.array(self.uP_num[:num_events+1]) 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    dahlquist_problem = Dahlquist(
                        u_start=1.0 + 0.0j,
                        t_start=0.0,
                        P_start=1,
                        alpha=6.001,
                        lam=lambda n: 1j*(1+0.01*n))
    dahlquist_be_solver = DahlquistBackwardEuler(dahlquist_problem, dt= 0.01)
    num_events = 3
    user_tt, user_uu, tP_array, uP_array = dahlquist_be_solver.solve_global(num_events)

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