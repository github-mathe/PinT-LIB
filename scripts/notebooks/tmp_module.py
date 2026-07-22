#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:40:40 2026

@author: yessima
"""
import numpy as np
from scipy.optimize import brentq
import warnings
from DahlquistParameters import f_ODE, lambda_n, epsilon

# Analytical solution for one step
def exact_local_solution(
        u0: complex,
        t0: float,
        t: float | np.ndarray,
        alpha: float,
        lam: complex,
        resonance_tolerance: float = 1e-12,
        ) -> complex | np.ndarray:
    """
    Exact propagation of

        u'(t) = lam*u(t) + sin(alpha*t),
        u(t0) = u0,

    assuming lam is constant on [t0, t1].
    """
    t_array = np.asarray(t, dtype=float)

    denominator = alpha**2 + lam**2
    scale = max(1.0, alpha**2, abs(lam) ** 2)

    if abs(denominator) <= resonance_tolerance * scale:
        raise ValueError(
            "Resonance detected: alpha**2 + lam**2 is approximately zero."
            )
    
    C = np.exp(-lam*t0)*(u0 + (alpha*np.cos(alpha*t0) + lam*np.sin(alpha*t0))/denominator)
    
    u = C*np.exp(lam*t_array) - (alpha*np.cos(alpha*t_array) + lam*np.sin(alpha*t_array))/denominator
    
    if u.ndim == 0: return complex(u.item())
    
    return np.asarray(u, dtype=complex)

def check_event(
    t_start: float,
    u_start: complex,
    t: np.ndarray,
    u: np.ndarray,
    alpha: float,
    lam: complex,
    level_tolerance: float = 1e-12,
    slope_tolerance: float = 1e-12,
    ) -> tuple[bool, float | None, complex | None]:
    """
    Find the first t_event > t_start such that

        Im(u(t_event)) ~= Im(u_start)

    and

        Im(u'(t_event))'Im(u'(t_start))>0

    Exact sampled matches and interpolated crossings are both considered.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=complex)
    
    sample_t = t[1:]
    sample_u = u[1:]
    
    # Determine the target slope sign and event values
    event_values = np.imag(sample_u) - np.imag(u_start)
    event_slopes = np.imag(f_ODE(sample_t, sample_u, alpha, lam))
    initial_slope = np.imag(f_ODE(t_start, u_start, alpha, lam))
    
    if abs(initial_slope) <= slope_tolerance:
        warnings.warn(
            "Initial slope is approximately zero; "
            "crossing direction is undefined.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False, None, None
    
    # Find points already close to the target level
    point_time = None
    point_value = None
    point_idx = np.flatnonzero((np.abs(event_values) <= level_tolerance) & (initial_slope * event_slopes > 0.0))
    if point_idx.size > 0:
        point_index = point_idx[0]
        point_time = float(sample_t[point_index])
        point_value = complex(sample_u[point_index])
    
    # Find the first valid interpolated crossing
    crossing_time = None
    crossing_value = None
    crossing_id = np.flatnonzero(event_values[:-1] * event_values[1:] < 0.0)
    if crossing_id.size:
        crossing_directions = (
            event_values[crossing_id + 1]
            - event_values[crossing_id]
        )

        matching_crossings = np.flatnonzero(
            initial_slope * crossing_directions > 0.0
        )

        if matching_crossings.size:
            left = int(
                crossing_id[matching_crossings[0]]
            )
            right = left + 1
            crossing_time = float(sample_t[left] + (sample_t[right] - sample_t[left]) * (
                            -event_values[left] / (event_values[right] - event_values[left])))
            crossing_value = complex(sample_u[left] + (sample_u[right] - sample_u[left]) * (
                            (crossing_time - sample_t[left]) / (sample_t[right] - sample_t[left])))

    if point_time is None and crossing_time is None:
        return False, None, None
    if point_time is not None and crossing_time is not None:
        t_event, u_event = (point_time, point_value) if point_time < crossing_time else (crossing_time, crossing_value)
    else:
        t_event, u_event = (point_time, point_value) if point_time is not None else (crossing_time, crossing_value)
    return True, t_event, u_event


def exact_solution_global(
        u0: complex,
        t0: float,
        alpha: float,
        lam: complex,
        T: float,
        dt: float = 1e-5,
        Period_start: int = 1,
        Period_end: int = 1,
        num_points: int = 100_000,
        resonance_tolerance: float = 1e-12,
        level_tolerance: float = 1e-12,
        slope_tolerance: float = 1e-12,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact propagation of

        u'(t) = lam*u(t) + sin(alpha*t),
        u(t0) = u0,
    for t in [t0, t0 + T] with a time step dt.
    """
    frequency = max(abs(lam(Period_start)), abs(alpha))
    L = int(np.ceil(2 * np.pi / frequency))
    if Period_end < Period_start:
        raise ValueError("Period_end must be greater than or equal to Period_start.")
    final_time = (Period_end - Period_start + 1) * L if Period_end is not None else T
    
    
    current_t0 = t0
    current_u0 = u0
    current_lam = lam(Period_start)
    current_t = np.linspace(current_t0, final_time, num_points, dtype=float)    
    event_times = [t0]
    event_values = [u0]
    
    while current_t0 < final_time:
        current_u = exact_local_solution(u0 = current_u0, t0 = current_t0, t=current_t, alpha=alpha, current_lam=current_lam, resonance_tolerance=resonance_tolerance)
        if current_u.shape != current_t.shape:
            raise ValueError(
                "local_solver must return one value for each time."
            )
    
        event_found, event_time, event_value = check_event(
            t0=current_t0,
            u0=current_u0,
            t=current_t,
            u=current_u,
            alpha=alpha,
            lam=current_lam,
            level_tolerance=level_tolerance,
            slope_tolerance=slope_tolerance,)
        if event_found:
                event_times.append(float(event_time))
                event_values.append(complex(event_value))
                current_t0 = float(event_time)
                current_u0 = complex(event_value)
                current_lam = lam(Period_start+1)
                current_t = np.linspace(current_t0, final_time, num_points, dtype=float)
        else:
                event_times.append(current_t[-1])0
                event_values.append(complex(current_u[-1]))
