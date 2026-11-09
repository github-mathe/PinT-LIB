import pybamm

import batpint.parameter.javid as bpar
from batpint.parareal.pybamm_propagator import PybammPropagator

DEFAULT_VAR_PTS = {
    "x_n": 1,
    "x_s": 10,
    "x_p": 10,
    "r_p": 30,
}


def make_dfm_model(half_cell=True):
    """Create a DFN model for half-cell or full-cell configuration."""
    args = ({"working electrode": "positive"},) if half_cell else ()
    return pybamm.lithium_ion.DFN(*args)


def make_parameter_values(half_cell=True):
    """Return ParameterValues for the half-cell or full-cell Javid parameter set."""
    return pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)


def make_propagator(half_cell=True, exp="CCCV", nCycles=1, var_pts=None, rtol=1e-6, eps_time=None, eps_state=None):
    """Create a PybammPropagator with the standard battery setup.

    Parameters
    ----------
    half_cell : bool
        Use half-cell (True) or full-cell (False) parameters.
    exp : str
        Experiment type: "CCCV" or "GITT".
    nCycles : int
        Number of cycles per experiment step.
    var_pts : dict, optional
        Spatial discretization points. Defaults to DEFAULT_VAR_PTS.
    rtol : float
        Relative tolerance for the IDAKLU solver.
    eps_time, eps_state : float, optional
        Perturbations applied to the propagated time/state (for testing).
    """
    if var_pts is None:
        var_pts = DEFAULT_VAR_PTS
    return PybammPropagator(
        model=make_dfm_model(half_cell),
        experiment=batteryExperiment(nCycles, expType=exp),
        parameter_values=make_parameter_values(half_cell),
        var_pts=var_pts,
        solver=pybamm.IDAKLUSolver(rtol=rtol),
        eps_time=eps_time,
        eps_state=eps_state,
    )


def batteryExperiment(nCycles=10, expType="GITT"):
    if expType == "GITT":
        # define GITT experiment: short pulse followed by long rest period
        pulse_duration = "10 minutes"  # duration of current pulse
        rest_duration = "2 hours"  # relaxation time
        current_rate = "0.05C"  # current during pulse
        voltage_min = 2.5  # V - discharge cutoff

        # Create GITT discharge experiment
        # Tuple groups discharge+rest as ONE cycle
        experiment = pybamm.Experiment(
            [(f"Discharge at {current_rate} for {pulse_duration}", f"Rest for {rest_duration}")] * nCycles,
            termination=f"{voltage_min} V",
        )
    elif expType == "CCCV":
        # define CCVV experiment: many cycles of discharge / charge
        experiment = pybamm.Experiment(
            [
                (
                    # "Rest for 5 minutes",
                    "Discharge at 2C until 3.5 V",
                    "Rest for 5 minutes",
                    "Charge at 0.1C until 4.2 V",
                    "Rest for 10 minutes",
                )
            ]
            * nCycles
        )
    else:
        raise NotImplementedError(f"{expType}")

    return experiment
