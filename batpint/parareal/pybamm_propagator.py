import numpy as np
import pybamm

class PybammPropagator:
    """
    Propagate a numerical PyBaMM state through one complete experiment.

        (t_n, u_n) -> (t_{n+1}, u_{n+1}).
    """

    def __init__(
        self,
        model,
        experiment,
        parameter_values,
        var_pts,
        solver=None,
        inputs=None,
        experiment_model_mode="legacy",
    ):
        self.inputs = {} if inputs is None else dict(inputs)

        self.solver = solver or pybamm.IDAKLUSolver()

        self.sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
            solver=self.solver,
            var_pts=var_pts,
            experiment_model_mode=experiment_model_mode,
        )

        # Parameterise, mesh and discretise the experiment models once.
        self.sim.build(inputs=self.inputs)

        # Dimension of the complete differential + algebraic state.
        self._state_size = self.sim.built_model.len_rhs_and_alg
        
        self._initial_solution = self._compute_one_step()
        self._t_start, self._u_start, self._up_start = self.extract_state(self._initial_solution)
        
    @property
    def t_start(self):
        return self._t_start
    
    @property
    def u_start(self):
        return self._u_start
    @property
    def up_start(self):
        return self._up_start
    @property
    def u_full(self):
        return np.vstack(self._u_start, self._u_start)
    
    def _compute_one_step(self):
        first_solution = self.sim.step(dt = 1e-2)
        return first_solution.first_state
    
    def extract_state(self, solution):
        """
        Extract the numerical PinT interface state from a PyBaMM Solution.

        Parameters
        ----------
        solution : pybamm.Solution
            Solution whose final state is the experiment interface.

        Returns
        -------
        t : float
            Interface time [s].
        u : numpy.ndarray
            Complete discretised DAE state.
        """
        state = solution.last_state

        t = state.t[-1].copy()
        u = state.y[:,-1].copy()
        up = state.yp[:,-1].copy()
        if u.size != self._state_size:
            raise ValueError(
                "Returned interface-state size mismatch: "
                f"received {u.size}, expected {self.state_size}."
            )
        
        return t, u, up
    
    def make_starting_solution(self, u):
        dummy_solution = self._initial_solution.copy()
        
        # change states
        dummy_solution.y[:] = u[0].reshape(dummy_solution.y.shape)
        #dummy_solution.all_ys = [dummy_solution.y]
        dummy_solution.yp[:] = u[1].reshape(dummy_solution.yp.shape)
        #dummy_solution.all_yps = [dummy_solution.yp]
        
        # change times
        dummy_solution.t[-1] = 0
        dummy_solution.t_eval[-1] = 0
        dummy_solution.all_ts[-1][-1] = 0
        dummy_solution.all_t_evals[-1][-1] = 0
        return dummy_solution
        
        
    def propagate(self, t, u, cycle):
        """
        Propagate one complete experiment from (t, u).

        Parameters
        ----------
        t : float
            Starting numerical time [s].
        u : array-like
            Starting complete DAE state.
        cycle: int
            cycle number

        Returns
        -------
        t_next : float
            Dynamically determined end time of the experiment [s].
        u_next : numpy.ndarray
            Complete DAE state at the end of the cycle.
        up_next : numpy.ndarray
            derivatives of u.
        cycle_solution : pybamm.Solution
            Full solution of the newly propagated experiment.
        """
        
        init_sol = self.make_starting_solution(u)
        
        solution = self.sim.solve(
            starting_solution=init_sol,
            inputs=self.inputs,
            calc_esoh=False,
        )

        # starting_solution is represented by PyBaMM as an earlier
        # synthetic cycle. The final cycle is the newly computed one.
        cycle_solution = solution.cycles[-1]

        t_next_local, u_next, up_next = self.extract_state(cycle_solution)
        t_next = t + t_next_local
        
        return t_next, np.vstack((u_next, up_next))
    
