#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:40:40 2026

@author: yessima
"""
import numpy as np
from scipy.optimize import brentq

def f(
    t: float,
    u: complex,
    alpha: float,
    lam: complex,
) -> float:
    """Right-hand side function for the ODE u’ = u."""
    du_dt = lam * u + np.sin(alpha * t)
    return float(np.imag(du_dt))


# Analytical solution for one step
def exact_one_step(
        u0: complex,
        t0: float,
        t1: float | np.ndarray,
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
    t1_array = np.asarray(t1, dtype=float)

    denominator = alpha**2 + lam**2
    scale = max(1.0, alpha**2, abs(lam) ** 2)

    if abs(denominator) <= resonance_tolerance * scale:
        raise ValueError(
            "Resonance detected: alpha**2 + lam**2 is approximately zero."
            )
    
    C = np.exp(-lam*t0)*(u0 + (alpha*np.cos(alpha*t0) + lam*np.sin(alpha*t0))/denominator)
    
    u = C*np.exp(lam*t1_array) - (alpha*np.cos(alpha*t1_array) + lam*np.sin(alpha*t1_array))/denominator
    
    if u.ndim == 0: return complex(u.item())
    
    return u.astype(complex)

def epsilon(n: int) -> float:
    return 0.01 * n

def lambda_n(n: int) -> complex:
    return 1j * (1.0 + epsilon(n))

def check_event(
    t_left: float,
    t_right: float,
    t_start: float,
    u_start: complex,
    alpha: float,
    lam: complex,
    root_tolerance: float = 1e-11,
    level_tolerance: float = 1e-12,
    slope_tolerance: float = 1e-12,
) -> tuple(bool, float | None, complex | None):
    """
    Check whether a complete pseudo-period event occurs inside
    [t_left, t_right].

    A valid event requires:

        Im(u(t_event)) = Im(u_start)

    and

        Im(u'(t_event)) * Im(u'(t_start)) > 0.
    """
    reference_imaginary_part = float(np.imag(u_start))

    def event_value(t: float) -> float:
        u = exact_one_step(
            u0=u_start,
            t0=t_start,
            t1=t,
            alpha=alpha,
            lam=lam,
        )
        return float(np.imag(u) - reference_imaginary_part)

    g_left = event_value(t_left)
    g_right = event_value(t_right)

    # Check whether the event surface is crossed.
    if abs(g_left) <= level_tolerance:
        t_event = t_left
    elif abs(g_right) <= level_tolerance:
        t_event = t_right
    elif g_left * g_right < 0.0:
        t_event = brentq(
            event_value,
            t_left,
            t_right,
            xtol=root_tolerance,
            rtol=4.0 * np.finfo(float).eps,
        )
    else:
        return False, None, None

    # Exclude the trivial root at the beginning of the window.
    if t_event <= t_start + root_tolerance:
        return False, None, None

    u_event = complex(
        exact_one_step(
            u0=u_start,
            t0=t_start,
            t1=t_event,
            alpha=alpha,
            lam=lam,
        ))

    start_slope = float(
        np.imag(
            f(
                t_start,
                u_start,
                alpha,
                lam,
            )
        )
    )

    event_slope = float(
        np.imag(
            f(
                t_event,
                u_event,
                alpha,
                lam,
            )
        )
    )

    if abs(start_slope) <= slope_tolerance:
        raise ValueError(
            "The initial crossing direction is undefined because "
            "the initial event slope is approximately zero."
        )
        
    if abs(event_slope) <= slope_tolerance:
        return False, t_event, u_event
    
    occured = start_slope * event_slope > 0.0
    
    return occured, t_event, u_event



