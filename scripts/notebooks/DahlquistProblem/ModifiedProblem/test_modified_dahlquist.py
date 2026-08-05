import numpy as np
import pytest
from modified_dahlquist import (f_ODE,
                                check_event,
                                exact_local_solution,
                                exact_global_solution,)

u0 = 1.0 + 0.0j
t0 = 0.0
alpha = 6.001
lam = lambda n: 1j*(1+0.01*n)  
Period_start = 1
Period_end = 7
T = 15
dt = 1e-2
num_points=100

def test_initial_condition_local():
    lam_start = lam(1)
    local_u     =   exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t0,
                    alpha=alpha,
                    lam=lam_start,
                    )

    assert np.isclose(local_u, u0)

def test_initial_condition_global():
    _,global_u,_,_     =   exact_global_solution(
                    u_start=u0,
                    t_start=t0,
                    T = 1,
                    alpha=alpha,
                    lam=lam,
                    dt=dt,
                    )

    assert np.isclose(global_u[0], u0) 

def test_local_solution_slope():
    lam_start = lam(1)
    t = 0.5
    h = 1e-6
    u_left     =   exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t - h,
                    alpha=alpha,
                    lam=lam_start,
                    )
    u = exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t,
                    alpha=alpha,
                    lam=lam_start,
                    )
    u_right     =   exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t + h,
                    alpha=alpha,
                    lam=lam_start,
                    )
    slope = (u_right - u_left)/(2*h)
    exact_slope = f_ODE(t, u, alpha, lam_start)
    assert np.isclose(slope, exact_slope)

def test_continuity_local_solution():
    lam_start = lam(1)
    t1 = 1
    t2 = 2
    u1     =   exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t1,
                    alpha=alpha,
                    lam=lam_start,
                    )
    u2     =   exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t2,
                    alpha=alpha,
                    lam=lam_start,
                    )
    u2_from_u1     =   exact_local_solution(
                    u_start=u1,
                    t_start=t1,
                    t=t2,
                    alpha=alpha,
                    lam=lam_start,
                    )
    assert np.isclose(u2, u2_from_u1)
    assert np.isclose(u1, u2, atol=1e-6) == False

def test_local_solution_array():
    lam_start = lam(1)
    t = np.linspace(0, 1, num_points)
    u     =   exact_local_solution(
                    u_start=u0,
                    t_start=t0,
                    t=t,
                    alpha=alpha,
                    lam=lam_start,
                    )
    assert u.shape == t.shape

def test_local_solution_rejects_resonance():
    alpha = 2.0
    lam = 2.0j

    with pytest.raises(ValueError):
        exact_local_solution(
            u_start=1.0 + 0.0j,
            t_start=0.0,
            t=1.0,
            alpha=alpha,
            lam=lam,
        )
def test_global_solution_requires_output_resolution():
    with pytest.raises(ValueError):
        exact_global_solution(
            u_start=1.0 + 0.0j,
            t_start=0.0,
            alpha=2.0,
            lam=lam,
            T=2.0,
        )

def test_global_solution_rejects_two_output_resolutions():
    with pytest.raises(ValueError):
        exact_global_solution(
            u_start=1.0 + 0.0j,
            t_start=0.0,
            alpha=2.0,
            lam=lam,
            T=2.0,
            dt=1e-3,
            num_points=100,
        )
def test_global_solution_requires_stopping_condition():
    with pytest.raises(ValueError):
        exact_global_solution(
            u_start=1.0 + 0.0j,
            t_start=0.0,
            alpha=2.0,
            lam=lam,
            dt=1e-3,
        )
def test_global_solution_rejects_two_stopping_conditions():
    with pytest.raises(ValueError):
        exact_global_solution(
            u_start=1.0 + 0.0j,
            t_start=0.0,
            alpha=2.0,
            lam=lam,
            T=2.0,
            Period_end=3,
            dt=1e-3,
        )

def test_global_solution_output_shapes():
    t, u, event_times, event_values = exact_global_solution(
        u_start=1.0 + 0.0j,
        t_start=0.0,
        alpha=6.001,
        lam=lam,
        T=T,
        num_points=100,
    )

    assert isinstance(t, np.ndarray)
    assert isinstance(u, np.ndarray)
    assert isinstance(event_times, np.ndarray)
    assert isinstance(event_values, np.ndarray)

    assert np.all(np.diff(t) > 0.0)
    assert np.all(np.diff(event_times) > 0.0)
    assert t.ndim == 1
    assert u.ndim == 1
    assert event_times.ndim == 1
    assert event_values.ndim == 1

    assert t.size == u.size
    assert event_times.size == event_values.size
    assert np.iscomplexobj(u)
    assert np.iscomplexobj(event_values)
    for event_time, event_value in zip(event_times, event_values):
        assert np.any(np.isclose(t, event_time))
        assert np.any(np.isclose(u, event_value))

def test_check_event_in_event_interval():
    lam_start = lam(1)
    t=np.linspace(0, 7, num_points)
    local_u     =   exact_local_solution(
                        u_start=u0,
                        t_start=t0,
                        t=t,
                        alpha=alpha,
                        lam=lam_start,
                        )
    found, t_event, u_event = check_event(
                        t_start     =   t0,
                        u_start     =   u0,
                        t           =   t,
                        u           =   local_u,
                        alpha       =   alpha,
                        lam         =   lam(1),
                    )
    assert found

def test_check_event_in_event_interval():
    lam_start = lam(1)
    t=np.linspace(0, 1, num_points)
    local_u     =   exact_local_solution(
                        u_start=u0,
                        t_start=t0,
                        t=t,
                        alpha=alpha,
                        lam=lam_start,
                        )
    found, t_event, u_event = check_event(
                        t_start     =   t0,
                        u_start     =   u0,
                        t           =   t,
                        u           =   local_u,
                        alpha       =   alpha,
                        lam         =   lam(1),
                    )
    assert not found


    