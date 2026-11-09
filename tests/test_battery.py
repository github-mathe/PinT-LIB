"""Tests for battery experiment creation, factory functions, and parameter structure."""

import pybamm
import pytest

from batpint.parameter.javid import Li_full, Li_half
from batpint.parameter.javid.common import BASE_PARAMS
from batpint.problems.battery import (
    DEFAULT_VAR_PTS,
    batteryExperiment,
    make_dfm_model,
    make_parameter_values,
)

# --- Parameter structure ---


def test_base_params_shared():
    """Both half and full cell PARAMS should contain all BASE_PARAMS keys."""
    for key in BASE_PARAMS:
        assert key in Li_half.PARAMS, f"Missing in Li_half: {key}"
        assert key in Li_full.PARAMS, f"Missing in Li_full: {key}"


def test_half_cell_negative_electrode_zero_thickness():
    assert Li_half.PARAMS["Negative electrode thickness [m]"] == 0


def test_full_cell_has_particle_params():
    assert Li_full.PARAMS["Negative electrode thickness [m]"] > 0
    assert "Negative particle radius [m]" in Li_full.PARAMS


def test_parameter_values_constructible():
    pv_half = pybamm.ParameterValues(Li_half.PARAMS)
    pv_full = pybamm.ParameterValues(Li_full.PARAMS)
    assert pv_half is not None
    assert pv_full is not None


# --- Experiment creation ---


def test_battery_experiment_gitt():
    exp = batteryExperiment(nCycles=5, expType="GITT")
    assert isinstance(exp, pybamm.Experiment)
    # GITT: each cycle has 2 steps (discharge + rest)
    assert len(exp.steps) == 2 * 5


def test_battery_experiment_cccv():
    exp = batteryExperiment(nCycles=3, expType="CCCV")
    assert isinstance(exp, pybamm.Experiment)
    # CCCV: each cycle has 4 steps (discharge, rest, charge, rest)
    assert len(exp.steps) == 4 * 3


def test_battery_experiment_invalid_type():
    with pytest.raises(NotImplementedError):
        batteryExperiment(nCycles=1, expType="INVALID")


# --- Factory functions ---


def test_make_dfm_model_half_cell():
    model = make_dfm_model(half_cell=True)
    assert isinstance(model, pybamm.lithium_ion.DFN)


def test_make_dfm_model_full_cell():
    model = make_dfm_model(half_cell=False)
    assert isinstance(model, pybamm.lithium_ion.DFN)


def test_make_parameter_values():
    pv_half = make_parameter_values(half_cell=True)
    pv_full = make_parameter_values(half_cell=False)
    assert isinstance(pv_half, pybamm.ParameterValues)
    assert isinstance(pv_full, pybamm.ParameterValues)


def test_default_var_pts():
    assert "x_n" in DEFAULT_VAR_PTS
    assert "x_s" in DEFAULT_VAR_PTS
    assert "x_p" in DEFAULT_VAR_PTS
    assert "r_p" in DEFAULT_VAR_PTS
    assert all(isinstance(v, int) and v > 0 for v in DEFAULT_VAR_PTS.values())


def test_make_propagator_constructs():
    """make_propagator should return a PybammPropagator with valid initial state."""
    from batpint.problems.battery import make_propagator

    prop = make_propagator(half_cell=True, exp="CCCV", nCycles=1, rtol=1e-4)
    assert prop.u_start is not None
    assert prop.u_full.shape[0] == 2  # stacked u and up
    assert prop.t_start == 0.0
