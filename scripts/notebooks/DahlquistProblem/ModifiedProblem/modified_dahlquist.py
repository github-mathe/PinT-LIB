#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:40:40 2026

@author: yessima
"""
from collections.abc import Callable
import numpy as np
import warnings

_EVENT_SEARCH_POINTS = 1_000_000
# ODE right-hand side function
def f_ODE(
            t: float| np.ndarray,
            u: complex| np.ndarray,
            alpha: float,
            lam: complex,
            ) -> complex| np.ndarray:
    """Right-hand side function for the ODE 
    u’ = lam * u + sin(alpha * t).
    """
    return lam * u + np.sin(alpha * t)

# Exact local solution
def exact_local_solution(
    u_start: complex,
    t_start: float,
    t: float | np.ndarray,
    alpha: float,
    lam: complex,
    resonance_tolerance: float = 1e-14,
    ) -> complex | np.ndarray:
    """Exact local solution of
        u'(t) = lam*u(t) + sin(alpha*t),
        u(t_start) = u_start,
    assuming lam is constant on [t_start, t].
    """
    t_array = np.asarray(t, dtype=float)
    denominator = alpha**2 + lam**2
    scale = max(1.0, alpha**2, abs(lam) ** 2)
    if abs(denominator) <= resonance_tolerance * scale:
        raise ValueError("Resonance detected: alpha**2 + lam**2 is approximately zero.")
    C = np.exp(-lam*t_start)*(u_start + (alpha*np.cos(alpha*t_start) + lam*np.sin(alpha*t_start))/denominator)
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
    level_tolerance: float = 1e-14,
    slope_tolerance: float = 1e-14,
    ) -> tuple[bool, float | None, complex | None]:
    """ Find the first t_event > t_start such that
        Im(u(t_event)) ~= Im(u_start) and Im(u'(t_event))'Im(u'(t_start))>0
        Exact sampled matches and interpolated crossings are both considered.
    """
    
    denominator = alpha**2 + lam**2
    scale = max(1.0, alpha**2, abs(lam) ** 2)
    if abs(denominator) <= 1e-14 * scale:
        raise ValueError("Resonance detected: alpha**2 + lam**2 is approximately zero.")
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=complex)
    
    sample_t = t[1:]
    sample_u = u[1:] 
    
    # Determine the target slope sign and event values
    event_values = np.imag(sample_u) - np.imag(u_start)
    event_slopes = np.imag(f_ODE(sample_t, sample_u, alpha, lam))
    initial_slope = np.imag(f_ODE(t_start, u_start, alpha, lam))
    
    # Change that to an error --> not implemented yet
    if abs(initial_slope) <= slope_tolerance:
        raise ValueError(
            "Initial slope is approximately zero; "
            "crossing direction is undefined."
        )
            
    # Find points already close to the target level
    point_time = None
    point_value = None
    point_idx = np.flatnonzero((np.abs(event_values) <= level_tolerance) & (initial_slope * event_slopes > 0.0))
    if point_idx.size > 0:
        #print(f"The interval [{t_start}, {sample_t[-1]}] has {point_idx.size} points that are close to the pseudo-period.")
        point_index = point_idx[0]
        point_time = float(sample_t[point_index])
        point_value = np.real(sample_u[point_index]) + 1j * np.imag(u_start)
    
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
            #print(f"Detected interval [{t_start}, {sample_t[-1]}] with the pseudo-period.")
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

# Exact global solution
def exact_global_solution(
    u_start: complex,
    t_start: float,
    alpha: float,
    lam: Callable[[int], complex],
    *,
    Period_start: int = 1,
    Period_end: int | None = None,
    T: float | None = None,
    dt: float | None = None,
    num_points: int | None = None,
    resonance_tolerance: float = 1e-14,
    level_tolerance: float = 1e-14,
    slope_tolerance: float = 1e-14,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Exact global solution of
    u'(t) = lam*u(t) + sin(alpha*t),
    u(t0) = u0, for
    - t in [t0, t0 + T] 
    - t in [t0, t0 + T*], where T* is the end point for pseudo-period Period_end,
    with a time step dt or num_points per pseudo-period.
    """
    if (T is None) == (Period_end is None):
        raise ValueError("Provide exactly one of T or Period_end.")

    if Period_end is not None and Period_end < Period_start:
        raise ValueError("Period_end must be greater than or equal to Period_start.")

    if T is not None and T <= t_start:
        raise ValueError("T must be greater than t_start.")

    if (dt is None) == (num_points is None):
        raise ValueError("Provide exactly one of dt or num_points.")

    if dt is not None and dt <= 0.0:
        raise ValueError("dt must be positive.")

    if num_points is not None and num_points < 2:
        raise ValueError("num_points must be at least 2.")

    # Number of complete pseudo-periods requested.
    requested_windows = (
        Period_end - Period_start + 1
        if Period_end is not None
        else None
    )
    current_t0 = t_start
    current_u0 = u_start
    current_period = Period_start
      
    event_times = [t_start]
    event_values = [u_start]

    # Lambda used on each interval [event_times[i], event_times[i + 1]].
    interval_lambdas: list[complex] = []

    # ==========================================================
    # Phase 1: discover pseudo-period boundaries
    # ==========================================================
    while True:
        windows_found = len(interval_lambdas)
        current_lam = complex(lam(current_period))
        if (
            requested_windows is not None
            and windows_found >= requested_windows
        ):
            break

        if T is not None and current_t0 >= T:
            break

        frequency = min(abs(current_lam),abs(alpha),)

        if frequency == 0.0:
            raise ValueError(
                "Cannot determine L because lambda and alpha are zero."
            )

        L = float(np.ceil(2.0 * np.pi /frequency))


        window_end  =   current_t0 + L if T is None else min(current_t0 + L, T)

        current_tt  =   np.linspace(current_t0, window_end, _EVENT_SEARCH_POINTS, dtype=float)    
        current_uu  =   exact_local_solution(
                                            u_start = current_u0,
                                            t_start = current_t0,
                                            t       = current_tt,
                                            alpha   = alpha,
                                            lam     = current_lam,
                                            resonance_tolerance = resonance_tolerance)

        if current_uu.shape != current_tt.shape:
            raise ValueError(
                "local_solver must return one value for each time."
            )
    
        event_found, event_time,_ = check_event(
                                                t_start =   current_t0,
                                                u_start =   current_u0,
                                                t       =   current_tt,
                                                u       =   current_uu,
                                                alpha   =   alpha,
                                                lam     =   current_lam,
                                                level_tolerance = level_tolerance,
                                                slope_tolerance = slope_tolerance,)
        if event_found:
            event_time  =   float(event_time)
            if event_time <= current_t0 + level_tolerance:
                    raise RuntimeError("Event detection did not advance in time.")
            event_value     =   exact_local_solution(
                                                    u_start = current_u0,
                                                    t_start = current_t0,
                                                    t       = event_time,
                                                    alpha   = alpha,
                                                    lam     = current_lam,
                                                    resonance_tolerance = resonance_tolerance)
            interval_lambdas.append(current_lam)
            event_times.append(event_time)
            event_values.append(event_value)

            current_t0 = event_time
            current_u0 = event_value
            current_period += 1
            continue
        if T is not None and window_end >= T:
            final_value = complex(current_uu[-1])
            interval_lambdas.append(current_lam)
            event_times.append(float(T))
            event_values.append(final_value)
            break

        # In period-count mode, failure to find an expected event is an error.
        raise RuntimeError("No pseudo-period event was found in the search window "+f"[{current_t0}, {window_end}].")

    event_times = np.asarray(event_times, dtype=float)
    event_values = np.asarray(event_values, dtype=complex)
    interval_lambdas = np.asarray(interval_lambdas,dtype=complex,)
    num_windows = event_times.size - 1

    if interval_lambdas.size != num_windows:
        raise RuntimeError("The number of lambda values does not match the number of pseudo-windows.")
    
    # ==========================================================
    # Phase 2: construct the user grid on each pseudo-window
    # ==========================================================
    local_grids: list[np.ndarray] = []
    for i in range(num_windows):
        left = event_times[i]
        right = event_times[i + 1]

        # Use enough subdivisions that the actual spacing is <= dt.
        n_steps = max(1,int(np.ceil((right - left) / dt)),) if dt is not None else num_points-1 
        local_t = np.linspace(left,right,n_steps + 1,endpoint=True,dtype=float,)
        # Avoid duplicating the shared event boundary.
        local_t = local_t[1:] if i > 0 else local_t
        local_grids.append(local_t)

    user_t = np.concatenate(local_grids)
    user_u = np.empty_like(user_t, dtype=complex)

    # Locate each event boundary in the global grid.
    window_indices = np.searchsorted(user_t,event_times,)

    # Set event values exactly.
    user_u[window_indices] = event_values
    
    
    # ==========================================================
    # Phase 3: evaluate the interior of every pseudo-window
    # ==========================================================
    for i in range(num_windows):
        window_start = window_indices[i] + 1
        window_end = window_indices[i + 1]

        if window_start >= window_end:
            continue

        user_u[window_start:window_end] = exact_local_solution(
                                                                u_start=event_values[i],
                                                                t_start=event_times[i],
                                                                t=user_t[window_start:window_end],
                                                                alpha=alpha,
                                                                lam=interval_lambdas[i],
                                                                resonance_tolerance=resonance_tolerance,
                                                            )
    return (user_t,user_u,event_times,event_values,)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    u0 = 1.0 + 0.0j
    t0 = 0.0
    alpha = 6.001
    lam = lambda n: 1j*(1+0.01*n)  # Example lambda function
    Period_start = 1
    Period_end = 20
    num_points = 1000
    print("================================")
    print("Computing exact global solution for the ODE with parameters:")
    print(f"u0 = {u0},\nt0 = {t0},\nalpha = {alpha},\nlam(n) = 1j*(1+0.01*n),\nPeriod_start = {Period_start},\nPeriod_end = {Period_end},\n{num_points} points in each pseudo-window" )
    t_global, u_global, t_event, u_event = exact_global_solution(
        u_start = u0,
        t_start = t0,
        alpha = alpha,
        lam = lam,
        Period_start = Period_start,
        Period_end = Period_end,
        num_points = num_points
    )
