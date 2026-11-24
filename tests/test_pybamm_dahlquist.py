"""Numerical checks for scripts/runs/06_pybammDahlquist.py.

Run with ``python -m pytest tests/test_pybamm_dahlquist.py``.
The PyBaMM state is ordered as [imaginary part, real part].
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import pytest

from batpint.parareal.errors import compute_parareal_errors
from batpint.parareal.parareal import PararealModified
from batpint.problems.dahlquist import Dahlquist

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runs" / "06_pybammDahlquist.py"


def _load_script():
    """Import the numbered script using its path."""
    spec = importlib.util.spec_from_file_location("pybamm_dahlquist_run", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dahlquist_script():
    return _load_script()


@pytest.fixture
def make_propagator(dahlquist_script):
    def make(lam=lambda cycle: 1 + 0.01 * cycle, alpha=6.001, rtol=1e-9, atol=1e-11, u_start=1 + 0j, event_imag=None):
        model = dahlquist_script.pybammDahlquist(u_start, lam_f=lam, alpha_v=alpha)
        solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol, options={"max_order_bdf": 5})
        return dahlquist_script.MyPropagator(model=model, solver=solver, event_imag=event_imag)

    return make


def test_import_does_not_run_simulations_or_plot(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Importing the script must not run the demonstration")

    monkeypatch.setattr(pybamm, "IDAKLUSolver", unexpected_call)
    monkeypatch.setattr(Dahlquist, "solve_dahlquist_exact_cycles", unexpected_call)
    monkeypatch.setattr(plt, "subplots", unexpected_call)
    monkeypatch.setattr(plt, "show", unexpected_call)

    module = _load_script()

    assert callable(module.pybammDahlquist)
    assert callable(module.MyPropagator)
    assert callable(module.main)


@pytest.mark.parametrize("u_start", [1 + 0j, 2.25 - 0.75j])
def test_model_preserves_complex_initial_state(dahlquist_script, u_start):
    model = dahlquist_script.pybammDahlquist(u_start, lam_f=lambda cycle: 1, alpha_v=0)

    assert model.is_discretised
    assert model.len_rhs == 2
    assert model.len_alg == 0
    np.testing.assert_allclose(
        model.concatenated_initial_conditions.evaluate().reshape(-1),
        [u_start.imag, u_start.real],
    )


@pytest.mark.parametrize("cycle, alpha", [(0, 0.0), (4, 2.3)])
def test_model_uses_supplied_frequency_and_forcing(dahlquist_script, cycle, alpha):
    lam = lambda cycle: 1.5 + 0.25 * cycle
    model = dahlquist_script.pybammDahlquist(1 + 0j, lam_f=lam, alpha_v=alpha)
    t = 0.7
    u_im, u_re = -0.75, 2.25

    rhs = model.concatenated_rhs.evaluate(
        t=t,
        y=np.array([[u_im], [u_re]]),
        inputs={"cycle": cycle, "event_imag": 0, "event_direction": -1, "event_margin": 0},
    )

    np.testing.assert_allclose(
        rhs.reshape(-1),
        [lam(cycle) * u_re, -lam(cycle) * u_im + np.sin(alpha * t)],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "direction, margin, offset, expected",
    [(1, 1e-7, 0.0, 1e-7), (1, 1e-7, -1e-7, 0.0), (-1, 0, -0.4, 0.4), (-1, 0, 0.0, 0.0), (-1, 0, 0.4, -0.4)],
)
@pytest.mark.parametrize("event_imag", [-0.6, 0.0, 0.75])
def test_event_arms_below_target_then_detects_upward_crossing(
    dahlquist_script, direction, margin, offset, expected, event_imag
):
    model = dahlquist_script.pybammDahlquist(1 + 0j, lam_f=lambda cycle: 1, alpha_v=0)
    (event,) = model.events

    assert event.event_type == pybamm.EventType.TERMINATION
    value = event.expression.evaluate(
        t=0,
        y=np.array([[event_imag + offset], [1.0]]),
        inputs={"cycle": 0, "event_imag": event_imag, "event_direction": direction, "event_margin": margin},
    )
    assert float(np.asarray(value).item()) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("frequency", [0.1, 0.5, 1.0, 1.2, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("initial_imag", [-0.5, 0.0, 0.75])
def test_unforced_cycle_returns_to_starting_imaginary_level(make_propagator, frequency, initial_imag):
    propagator = make_propagator(lam=lambda cycle: frequency, alpha=0, u_start=1 + 1j * initial_imag)
    t_start = 2.0
    u_start = np.array([initial_imag, 1.0])

    t_event, u_event, solution = propagator.solve(t_start, u_start, cycle=3)

    assert solution.termination == "event: Upward imaginary-level crossing"
    # A fixed phase error corresponds to a larger time error at low frequency.
    assert t_event == pytest.approx(t_start + 2 * np.pi / frequency, abs=1e-7 / frequency)
    assert u_event.shape == (2,)
    np.testing.assert_allclose(u_event, u_start, rtol=0, atol=1e-7)
    assert t_event == solution.t_event[-1]
    np.testing.assert_array_equal(u_event, solution.y_event.reshape(-1))


@pytest.mark.parametrize("base_frequency", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("initial_imag", [-0.4, 0.0, 0.6])
def test_forced_consecutive_cycles_match_local_exact_solution(make_propagator, base_frequency, initial_imag):
    lam = lambda cycle: base_frequency + 0.01 * cycle
    propagator = make_propagator(lam=lam, u_start=1 + 1j * initial_imag)
    problem = Dahlquist(t_start=0, u_start=1 + 1j * initial_imag, lam=lambda cycle: 1j * lam(cycle), alpha=6.001)
    t_start = 0.0
    u_start = np.array([initial_imag, 1.0])

    for cycle in range(3):
        initial_state = u_start.copy()
        t_event, u_event, solution = propagator.solve(t_start, u_start, cycle)

        assert t_event > t_start
        assert solution.t[0] == t_start
        assert solution.termination == "event: Upward imaginary-level crossing"
        np.testing.assert_allclose(solution.y[:, 0], initial_state, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(u_start, initial_state)
        assert u_event[0] == pytest.approx(initial_state[0], abs=1e-8)
        assert lam(cycle) * u_event[1] > 0

        # The forced zero crossings differ from the homogeneous periods used by
        # solve_dahlquist_exact_cycles, so use the actual cycle start here.
        exact = problem.solve_dahlquist_exact_local(
            solution.t, cycle, t_start, initial_state[1] + 1j * initial_state[0]
        )
        numerical = solution.y[1, :] + 1j * solution.y[0, :]
        np.testing.assert_allclose(numerical, exact, rtol=0, atol=1e-6)

        t_start, u_start = t_event, u_event


def test_fixed_zero_target_from_already_negative_state(make_propagator):
    propagator = make_propagator(lam=lambda cycle: 2, alpha=0, event_imag=0)
    t_start = 1.0

    t_event, u_event, solution = propagator.solve(t_start, np.array([-1.0, 0.0]), cycle=0)

    assert solution.t[0] == t_start
    assert t_event == pytest.approx(t_start + np.pi / 4, abs=1e-7)
    np.testing.assert_allclose(u_event, [0.0, 1.0], rtol=0, atol=1e-7)


@pytest.mark.parametrize("initial_imag, phase", [(1.0, 3 * np.pi / 2), (-1.0, np.pi / 2)])
def test_downward_departure_returns_upward_to_starting_level(make_propagator, initial_imag, phase):
    propagator = make_propagator(lam=lambda cycle: 2, alpha=0)

    t_event, u_event, _ = propagator.solve(1.25, np.array([initial_imag, -1.0]), cycle=0)

    assert t_event == pytest.approx(1.25 + phase / 2, abs=1e-7)
    np.testing.assert_allclose(u_event, [initial_imag, 1.0], rtol=0, atol=1e-7)


@pytest.mark.parametrize("target", [-0.5, 0.5])
def test_fixed_imaginary_target_can_differ_from_initial_state(make_propagator, target):
    frequency = 2.0
    propagator = make_propagator(lam=lambda cycle: frequency, alpha=0, event_imag=target)
    for u_start in [np.array([0.0, 1.0]), np.array([0.2, 0.8])]:
        radius = np.linalg.norm(u_start)
        phase = (np.arcsin(target / radius) - np.arctan2(u_start[0], u_start[1])) % (2 * np.pi)

        t_event, u_event, solution = propagator.solve(1.25, u_start, cycle=0)

        assert t_event == pytest.approx(1.25 + phase / frequency, abs=1e-7)
        np.testing.assert_allclose(u_event, [target, np.sqrt(radius**2 - target**2)], rtol=0, atol=1e-7)
        np.testing.assert_allclose(solution.y[:, 0], u_start, rtol=0, atol=1e-12)
        assert all(inputs["event_imag"] == target for inputs in solution.all_inputs)


def test_unreachable_imaginary_target_raises_clear_error(make_propagator):
    propagator = make_propagator(lam=lambda cycle: 2, alpha=0, event_imag=1.5)

    with pytest.raises(RuntimeError, match="imaginary level 1.5"):
        propagator.solve(0.0, np.array([0.0, 1.0]), cycle=0)


@pytest.mark.parametrize("u_start", [[0.0, 1.0], [-0.1, 1.0]])
def test_missing_crossing_raises_clear_error(make_propagator, u_start):
    propagator = make_propagator(lam=lambda cycle: 0, alpha=0)

    with pytest.raises(RuntimeError, match="did not reach the requested event"):
        propagator.solve(0.0, np.array(u_start), cycle=0)


def test_reused_propagator_restarts_from_supplied_time_and_state(make_propagator):
    propagator = make_propagator()
    problem = Dahlquist(t_start=0, u_start=1 + 0j, lam=lambda cycle: 1j * (1 + 0.01 * cycle), alpha=6.001)

    # The second solve moves backwards in time, as can happen in Parareal.
    for t_start, initial_state, cycle in [(1.25, [0.3, 0.8], 2), (0.0, [-0.4, 1.2], 0), (0.0, [0.0, 1.0], 0)]:
        u_start = np.array(initial_state)
        _, u_event, solution = propagator.solve(t_start, u_start, cycle)

        assert solution.t[0] == t_start
        np.testing.assert_allclose(solution.y[:, 0], initial_state, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(u_start, initial_state)
        assert u_event[0] == pytest.approx(initial_state[0], abs=1e-8)
        assert all(inputs["event_imag"] == initial_state[0] for inputs in solution.all_inputs)
        exact = problem.solve_dahlquist_exact_local(
            solution.t, cycle, t_start, initial_state[1] + 1j * initial_state[0]
        )
        np.testing.assert_allclose(solution.y[1, :] + 1j * solution.y[0, :], exact, rtol=0, atol=1e-6)


def test_propagate_matches_solve_with_flat_state(make_propagator):
    propagator = make_propagator()
    u_start = np.array([0.0, 1.0])
    expected_t, expected_u, _ = propagator.solve(0.0, u_start, cycle=0)

    t_event, u_event = propagator.propagate(0.0, u_start, cycle=0)

    assert np.ndim(t_event) == 0
    assert u_event.shape == u_start.shape
    assert t_event == pytest.approx(expected_t, abs=1e-10)
    np.testing.assert_allclose(u_event, expected_u, rtol=0, atol=1e-10)


@pytest.mark.parametrize("initial_imag", [-0.4, 0.0, 0.6])
def test_parareal_recovers_sequential_fine_event_times_and_states(make_propagator, initial_imag):
    num_cycles = num_iterations = 3
    u_start = np.array([initial_imag, 1.0])
    reference = make_propagator(rtol=1e-8, atol=1e-10, u_start=1 + 1j * initial_imag)
    t_fine = np.zeros(num_cycles + 1)
    u_fine = np.empty((num_cycles + 1, 2))
    u_fine[0] = u_start
    for cycle in range(num_cycles):
        t_fine[cycle + 1], u_fine[cycle + 1] = reference.propagate(t_fine[cycle], u_fine[cycle], cycle)

    fine = make_propagator(rtol=1e-8, atol=1e-10, u_start=1 + 1j * initial_imag)
    coarse = make_propagator(rtol=1e-4, atol=1e-6, u_start=1 + 1j * initial_imag)
    parareal = PararealModified(fine=fine, coarse=coarse, coarse_s=coarse)
    times, states = parareal.solve(t0=0.0, u0=u_start, K=num_iterations, N=num_cycles)

    assert times.shape == (num_iterations + 1, num_cycles + 1)
    assert states.shape == (num_iterations + 1, num_cycles + 1, 2)
    np.testing.assert_array_equal(times[:, 0], np.zeros(num_iterations + 1))
    np.testing.assert_array_equal(states[:, 0], np.tile(u_start, (num_iterations + 1, 1)))

    for iteration in range(1, num_iterations + 1):
        np.testing.assert_allclose(times[iteration, : iteration + 1], t_fine[: iteration + 1], rtol=0, atol=1e-8)
        np.testing.assert_allclose(states[iteration, : iteration + 1], u_fine[: iteration + 1], rtol=0, atol=1e-8)

    complex_states = states[:, :, 1] + 1j * states[:, :, 0]
    complex_fine = u_fine[:, 1] + 1j * u_fine[:, 0]
    time_errors = compute_parareal_errors(times, t_fine)
    state_errors = compute_parareal_errors(complex_states, complex_fine)

    assert time_errors.shape == state_errors.shape == times.shape
    assert np.max(time_errors[-1]) < np.max(time_errors[0])
    assert np.max(state_errors[-1]) < np.max(state_errors[0])
    np.testing.assert_allclose(time_errors[-1], 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(state_errors[-1], 0, rtol=0, atol=1e-8)
