import numpy as np
import pybamm
import copy

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
        self.sim.build_for_experiment(inputs=self.inputs)
        self._last_step = self.sim.experiment.steps[-1].basic_repr()
        self._last_step_model = self.sim.steps_to_built_models[self._last_step]
        # Dimension of the complete differential + algebraic state.
        self._state_size = self._last_step_model.len_rhs_and_alg
        
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
        return np.vstack((self._u_start, self._up_start))
    
    def _compute_one_step(self):
        # compute the strating solution
        first_step_key = self.sim.experiment.steps[0].basic_repr()
        built_model = self.sim.steps_to_built_models[first_step_key]
        step_solver = self.sim.steps_to_built_solvers[first_step_key]
        
        first_solution = step_solver.step(
            old_solution = None,
            model=built_model,
            dt=1e-2,
            t_eval=np.array([0.0, 1e-2]),
            save=False,
            inputs=self.inputs,
            )
        return copy.deepcopy(first_solution.first_state)
    
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
        up: numpy.ndarray
            derivatives of u
        """
        state = solution.last_state

        t = state.t[-1].copy()
        u = state.y[:,-1].copy()
        up = state.yp[:,-1].copy()
        if u.size != self._state_size:
            raise ValueError(
                "Returned interface-state size mismatch: "
                f"received {u.size}, expected {self._state_size}."
            )
        
        return t, u, up
    
    def make_starting_solution(self, u):
        dummy_solution = pybamm.Solution(
            all_ts=np.array([0.0]), 
            all_ys=np.array(u[0].reshape(self._initial_solution.y.shape),copy=True),
            all_models = self._last_step_model,
            all_inputs = self.inputs,
            all_yps=np.array(u[1].reshape(self._initial_solution.yp.shape),copy=True),
            t_event=None,
            y_event=None,
            options={'compile': False},
            all_t_evals=None)
     
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
        Stacked:
            u_next : numpy.ndarray
                Complete DAE state at the end of the cycle.
            up_next : numpy.ndarray
                derivatives of u.
        """
        if cycle == 0:
            starting_solution = None
        else:
            starting_solution = self.make_starting_solution(u)
        
        solution = self.sim.solve(
            starting_solution=starting_solution,
        )
        cycle_solution = solution.cycles[-1]
        t_next_local, u_next, up_next = self.extract_state(cycle_solution)
        t_next = t + t_next_local
        
        return t_next, np.vstack((u_next, up_next))
    
    def solve(self, t, u, cycle):
        
        if cycle == 0:
            starting_solution = None
        else:
            starting_solution = self.make_starting_solution(u)
        
        solution = self.sim.solve(
            starting_solution=starting_solution,
            inputs=self.inputs
        )
        cycle_solution = solution.cycles[-1]
        t_next_local, u_next, up_next = self.extract_state(cycle_solution)
        t_next = t + t_next_local
        
        return t_next, np.vstack((u_next, up_next)), solution
