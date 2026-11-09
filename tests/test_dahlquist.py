"""Tests for the Dahlquist problem: exact solution, numerical methods, and Parareal."""

import numpy as np
import pytest

from batpint.parareal.parareal import PararealModified
from batpint.parareal.timestepper_propagator import TimeStepperPropagator
from batpint.problems.dahlquist import (
    Dahlquist,
    DahlquistBE,
    DahlquistExact,
    solve_dahlquist_cycles,
)
from batpint.timestepping.timestepper import TimeStepper


def _make_problem():
    lam = lambda cycle: 1j * (1 + 0.01 * cycle)
    alpha = 6.001

    def event(t, u, cycle, u_event):
        return np.imag(u) - np.imag(u_event)

    return Dahlquist(t_start=0, u_start=1 + 0j, lam=lam, alpha=alpha, event=event)


def _terminate(state):
    lam_val = 1j * (1 + 0.01 * state.cycle)
    return abs(6.001**2 + lam_val**2) < 1e-10


# --- Exact solution tests ---


def test_dahlquist_exact_matches_analytical():
    """The exact cycle solver should match the per-point exact solver."""
    problem = _make_problem()
    _, _, t_ev, u_ev = problem.solve_dahlquist_exact_cycles(num_cycles=3, dt=1e-3)
    assert np.all(np.diff(t_ev) > 0), "Event times must be monotonically increasing"
    for t, u in zip(t_ev, u_ev):
        u_check = problem.solve_dahlquist_exact(t)
        assert abs(u - u_check) < 1e-10


def test_dahlquist_exact_rejects_past_time():
    """solve_dahlquist_exact should raise for t < t_start."""
    problem = _make_problem()
    with pytest.raises(ValueError, match="Target time"):
        problem.solve_dahlquist_exact(-1.0)


def test_dahlquist_exact_step_is_exact():
    """DahlquistExact.step should reproduce the analytical solution at t+h."""
    problem = _make_problem()
    method = DahlquistExact()
    lam = problem.params["lam"]
    alpha = problem.params["alpha"]

    # Start at t=0, u=1+0j; step to t=0.5
    t, u, h, cycle = 0.0, 1.0 + 0j, 0.5, 0
    u_next = method.step(t=t, u=u, h=h, rhs=None, jacobian=None, cycle=cycle, lam=lam, alpha=alpha)
    u_exact = problem.solve_dahlquist_exact(t + h)
    assert abs(u_next - u_exact) < 1e-12, f"Exact step mismatch: {u_next} vs {u_exact}"

    # Step from a mid-cycle point
    t2, u2 = 1.0, problem.solve_dahlquist_exact(1.0)
    u_next2 = method.step(t=t2, u=u2, h=0.3, rhs=None, jacobian=None, cycle=0, lam=lam, alpha=alpha)
    u_exact2 = problem.solve_dahlquist_exact(t2 + 0.3)
    assert abs(u_next2 - u_exact2) < 1e-12


# --- Backward Euler convergence ---


def test_backward_euler_converges_to_exact():
    """Backward Euler with decreasing dt should converge to the exact solution."""
    problem = _make_problem()
    num_cycles = 3
    dt_vals = [1e-1, 1e-2, 1e-3]
    errors = []

    _, _, _, ue_exact = solve_dahlquist_cycles(
        problem=problem,
        method=DahlquistExact(),
        dt=1e-4,
        num_cycles=num_cycles,
        terminate_cycle=_terminate,
        save_history=False,
    )

    for dt in dt_vals:
        _, _, _, ue_num = solve_dahlquist_cycles(
            problem=problem,
            method=DahlquistBE(),
            dt=dt,
            num_cycles=num_cycles,
            terminate_cycle=_terminate,
            save_history=False,
        )
        errors.append(np.max(np.abs(ue_num - ue_exact)))

    assert errors[-1] < errors[0], f"BE should converge as dt decreases: {errors}"


def test_dahlquist_be_step_formula():
    """DahlquistBE step should match the analytical one-step formula."""
    method = DahlquistBE()
    lam = lambda cycle: 1j
    alpha = 6.0
    t, u, h, cycle = 0.0, 1.0 + 0j, 0.01, 0
    u_next = method.step(t=t, u=u, h=h, rhs=None, jacobian=None, cycle=cycle, lam=lam, alpha=alpha)
    expected = (u + h * np.sin(alpha * (t + h))) / (1 - h * lam(cycle))
    assert abs(u_next - expected) < 1e-15


# --- Parareal on Dahlquist ---


def test_parareal_converges_on_dahlquist():
    """Parareal should converge: error at last iteration < error at first iteration."""
    problem = _make_problem()
    N, K = 3, 3
    dtF, dtG = 1e-3, 1e-2

    tsF = TimeStepper(problem=problem, method=DahlquistBE(), dt=dtF, save_history=False)
    tsG = TimeStepper(problem=problem, method=DahlquistBE(), dt=dtG, save_history=False)
    pF = TimeStepperPropagator(timestepper=tsF, direction=1, terminate_cycle=_terminate)
    pG = TimeStepperPropagator(timestepper=tsG, direction=1, terminate_cycle=_terminate)

    # Fine reference
    _, _, teFine, _ = solve_dahlquist_cycles(
        problem=problem,
        method=DahlquistBE(),
        dt=dtF,
        num_cycles=N,
        terminate_cycle=_terminate,
        save_history=False,
    )

    parareal = PararealModified(fine=pF, coarse=pG, coarse_s=pG)
    TT, U = parareal.solve(t0=0, u0=1 + 0j, K=K, N=N)

    assert TT.shape == (K + 1, N + 1)
    assert U.shape == (K + 1, N + 1)

    # Error in event times should decrease across iterations
    err_per_iter = [np.max(np.abs(TT[k, 1:] - teFine[1:])) for k in range(K + 1)]
    assert err_per_iter[-1] <= err_per_iter[0] + 1e-12, f"Parareal should not get worse: {err_per_iter}"


# --- Problem validation ---


def test_problem_param_validation():
    """Problem functions must start with (t, u) arguments."""
    from batpint.problems.base_problem import Problem

    with pytest.raises(ValueError, match="must start with arguments"):
        Problem(t_start=0, u_start=1, rhs=lambda x, y: x + y)


def test_problem_stores_params():
    """Extra kwargs should be stored in params dict."""
    from batpint.problems.base_problem import Problem

    p = Problem(t_start=0, u_start=1, rhs=lambda t, u, alpha: alpha * u, alpha=2.0)
    assert p.params["alpha"] == 2.0
    assert p.rhs_params == ("alpha",)
