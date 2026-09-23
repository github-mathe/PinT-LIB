import pybamm
import numpy as np

class PybammPropagator:
    """Propagate a PyBaMM state through one complete experiment."""

    def __init__(
        self,
        model,
        experiment,
        parameter_values,
        var_pts,
        solver=None,
    ):
        self.solver = solver or pybamm.IDAKLUSolver()

        self.sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
            solver=self.solver,
            var_pts=var_pts,
        )

        # Build and discretise once
        self.sim.build_for_experiment()

    def propagate(self, starting_state=None):
        """
        Propagate one complete experiment.

        Parameters
        ----------
        starting_state : pybamm.Solution or None
            One-state Solution representing Y_n. If None, use the
            model's original initial conditions.

        Returns
        -------
        cycle_solution : pybamm.Solution
            Solution of the newly propagated experiment.
        final_state : pybamm.Solution
            One-state Solution containing Y_{n+1}.
        """
        solution = self.sim.solve(starting_solution=starting_state)

        # If starting_state was supplied, solution.cycles may also contain
        # the artificial cycle representing that starting state.
        cycle_solution = solution.cycles[-1]

        final_state = cycle_solution.last_state

        return cycle_solution, final_state
    
    @staticmethod
    def state_vector(state):
        """Return the flattened numerical solver state."""
        return np.asarray(state.y).reshape(-1)

    @staticmethod
    def state_time(state):
        """Return the absolute time associated with a one-state Solution."""
        return float(state.t[-1])
    

def make_state(reference_state, y, t=None):
    """Construct a one-state PyBaMM Solution from a numerical state vector."""
    reference_state = reference_state.last_state

    y = np.asarray(y).reshape(-1)

    y_reference = np.asarray(
        reference_state.y
    ).reshape(-1)

    if y.shape != y_reference.shape:
        raise ValueError(
            "State-vector size mismatch: "
            f"{y.shape} != {y_reference.shape}"
        )

    if t is None:
        t = float(reference_state.t[-1])

    state = pybamm.Solution(
        all_ts=np.array([t]),
        all_ys=y[:, np.newaxis],
        all_models=reference_state.all_models[-1],
        all_inputs=reference_state.all_inputs[-1],
        termination="final time",
        options=reference_state.user_options,
    )

    state.initial_start_time = (
        reference_state.initial_start_time
    )

    return state