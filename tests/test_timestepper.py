"""Tests for the TimeStepper: stepping, event detection, and advance methods."""

import pytest

from batpint.problems.base_problem import Problem
from batpint.time_integration.base_method import IntegrationMethod
from batpint.timestepping.timestepper import TimeStepper


class ForwardEuler(IntegrationMethod):
    """Simple explicit method for testing: u_next = u + h * rhs(t, u)."""

    def __init__(self):
        super().__init__()

    def step(self, t, u, h, rhs, jacobian=None):
        return u + h * rhs(t, u)


def _make_exp_decay_problem():
    """u' = -u, u(0) = 1, with event at u = 0.5."""

    def rhs(t, u):
        return -u

    def event(t, u):
        return u - 0.5

    return Problem(t_start=0.0, u_start=1.0, rhs=rhs, event=event)


def test_timestepper_step_advances_time():
    problem = _make_exp_decay_problem()
    ts = TimeStepper(problem=problem, method=ForwardEuler(), dt=0.1)
    from batpint.parareal.propagation_state import PropagationState

    ts.set_state(PropagationState(t=0.0, u=1.0, cycle=0))
    ts.step()
    assert ts.state.t == pytest.approx(0.1)
    assert ts.state.u == pytest.approx(0.9)


def test_timestepper_advance_to_time():
    problem = _make_exp_decay_problem()
    ts = TimeStepper(problem=problem, method=ForwardEuler(), dt=0.1)
    from batpint.parareal.propagation_state import PropagationState

    ts.set_state(PropagationState(t=0.0, u=1.0, cycle=0))
    ts.advance_to_time(0.35)
    assert ts.state.t == pytest.approx(0.35)
    # After 3 full steps + 1 partial step of 0.05
    u_expected = 1.0 * 0.9**3 * 0.95
    assert ts.state.u == pytest.approx(u_expected, rel=1e-10)


def test_timestepper_advance_to_event():
    """advance_to_event should stop near u = 0.5 for u' = -u."""
    problem = _make_exp_decay_problem()
    ts = TimeStepper(problem=problem, method=ForwardEuler(), dt=0.1)
    from batpint.parareal.propagation_state import PropagationState

    ts.set_state(PropagationState(t=0.0, u=1.0, cycle=0))
    ts.advance_to_event(direction=-1)  # u crosses 0.5 from above
    assert ts.state.u == pytest.approx(0.5, abs=0.05)
    # Exact event time: t = -ln(0.5) ≈ 0.693
    assert ts.state.t == pytest.approx(0.693, abs=0.1)


def test_timestepper_no_event_raises():
    def rhs(t, u):
        return -u

    problem = Problem(t_start=0.0, u_start=1.0, rhs=rhs)  # no event
    ts = TimeStepper(problem=problem, method=ForwardEuler(), dt=0.1)
    from batpint.parareal.propagation_state import PropagationState

    ts.set_state(PropagationState(t=0.0, u=1.0, cycle=0))
    with pytest.raises(ValueError, match="No event function"):
        ts.advance_to_event()


def test_timestepper_save_history():
    problem = _make_exp_decay_problem()
    ts = TimeStepper(problem=problem, method=ForwardEuler(), dt=0.1, save_history=True)
    from batpint.parareal.propagation_state import PropagationState

    ts.set_state(PropagationState(t=0.0, u=1.0, cycle=0))
    ts.step()
    ts.step()
    assert len(ts.history["t"]) == 3  # initial + 2 steps
    assert ts.history["t"][0] == 0.0
    assert ts.history["t"][-1] == pytest.approx(0.2)


def test_event_crossed_directions():
    """Test _event_crossed static method for all directions."""
    # direction=0: any crossing
    assert TimeStepper._event_crossed(-1, 1, 0)
    assert TimeStepper._event_crossed(1, -1, 0)
    # direction=1: negative to positive
    assert TimeStepper._event_crossed(-1, 1, 1)
    assert not TimeStepper._event_crossed(1, -1, 1)
    # direction=-1: positive to negative
    assert TimeStepper._event_crossed(1, -1, -1)
    assert not TimeStepper._event_crossed(-1, 1, -1)


def test_interpolate_event():
    t_old, u_old = 0.0, 1.0
    t_new, u_new = 1.0, 0.0
    g_old, g_new = -1.0, 1.0
    t_ev, u_ev = TimeStepper._interpolate_event(t_old, u_old, t_new, u_new, g_old, g_new)
    assert t_ev == pytest.approx(0.5)
    assert u_ev == pytest.approx(0.5)
