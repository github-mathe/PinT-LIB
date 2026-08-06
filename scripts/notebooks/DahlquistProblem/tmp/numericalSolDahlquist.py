#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:03:28 2026

@author: yessima
"""

import numpy as np

def timeOneStep(u0,alpha, t1, dt, lam):
    u = (u0 + dt*np.sin(alpha*t1))/(1-dt*lam)
    return u

def timeStepperPeriod(u0,alpha,lam_f, t0,nP,dt):
    u = [u0]  
    tt = [t0]
    lam_nP = lam_f(nP)
    assert lam_nP**2 + alpha**2 != 0, "resonance regime, different analytical solution"
    #UP = np.zeros(1, dtype=complex)  # to store the period point for the current period
    #TP = np.zeros(1, dtype=float)  # to store the period point for the current period
    occ = 0
    steps = 0
    while True:
        t_next = tt[-1] + dt
        u_next = timeOneStep(u[-1],alpha, t_next, dt, lam_nP)
        
        u.append(u_next)
        tt.append(t_next)
        
        occ += u[-1].imag*u[-2].imag < 0 # check for sign change in the imaginary part
        steps += 1
        if occ == 2: # we completed a full period
            tP = np.interp(0, [u[-2].imag, u[-1].imag], [tt[-2], tt[-1]])
            uP_real = np.interp(tP, [tt[-2], tt[-1]], [u[-2].real, u[-1].real])
            UP = uP_real + 0j
            TP = tP
            steps += 1
            u[-1]= uP_real + 0j
            tt[-1]= tP  # insert tP in the correct position to maintain sorted order
            break
    return tt, u, TP, UP, steps

def timeStepperAll(u0, alpha,lam_f, t0, dt, pStart, pEnd):
    nP = pStart
    tt = [t0]
    u = [u0]
    steps = 0
    TP = np.zeros(pEnd - pStart + 1, dtype=float)
    UP = np.zeros_like(TP, dtype=complex)
    while nP < pEnd + 1:
        t1, u1, TP1, UP1, steps = timeStepperPeriod(u[-1], alpha, lam_f, tt[-1], nP, dt)
        
        tt.extend(t1[1:])
        u.extend(u1[1:])
        TP[nP - pStart] = TP1
        UP[nP - pStart] = UP1
             
        nP += 1
        steps += steps

    return (np.asarray(tt, dtype=float), np.asarray(u, dtype=complex), TP, UP, steps)

def DahlquistBE(
    *args,
    t0,
    y0,
    dt,
    T,
    **kwargs
):
    alpha = kwargs.get("alpha")
    lam = kwargs.get("lam")
    assert alpha is not None, "alpha must be provided in kwargs"
    assert lam is not None, "lam must be provided in kwargs"
    assert alpha**2 + lam**2 != 0, "resonance regime, different analytical solution"
    y0 = np.atleast_1d(y0)
    if T<t0: raise ValueError("T must be bigger than t0")
    if dt <= 0:
        raise ValueError("dt must be positive")
    
    num_steps = int(round((T - t0) / dt))
    
    if num_steps <= 0:
        raise ValueError("(T - t0) must be an integer multiple of dt")
    t   =    np.linspace(t0, T, num_steps+1, endpoint=True)
    y   =    np.zeros((num_steps + 1, len(y0)), dtype = y0.dtype)
    y[0,:]  =    y0
    for i in range(num_steps):
        y[i+1] = (y[i] + dt*np.sin(alpha*t[i+1])) / (1 - dt*lam)
    y = np.squeeze(y)
    return t, y

def DahlquistBEEvent( 
    *args,
    t0,
    y0,
    dt,
    event_func,
    num_steps=1_000,
    **kwargs
):
    alpha = kwargs.get("alpha")
    lam = kwargs.get("lam")
    assert alpha is not None, "alpha must be provided in kwargs"
    assert lam is not None, "lam must be provided in kwargs"
    assert alpha**2 + lam**2 != 0, "resonance regime, different analytical solution"
    
    t_values = []
    y_values = []

    t = t0
    y = y0.copy() if isinstance(y0, np.ndarray) else np.array([y0])

    while True:
        # Perform a single Backward Euler step
        T = t + num_steps * dt
        t_next,y_next = DahlquistBE(*args, t0 = t, y0=y, dt=dt, T=T, **kwargs)  
        is_event, t_event, y_event = event_func(t_next, y_next)

        # Check for event detection
        if is_event:
            y_values.append(y_next[t_next<t_event])  # Store the last value before the event
            t_values.append(t_next[t_next<t_event])  # Store the
            break

        # Store the new values
        t_values.append(t_next[:-1])
        y_values.append(y_next[:-1])

        # Update time and solution
        y = y_next[-1]
        t = t_next[-1]   
    t_values = np.concatenate(t_values, casting = "no")
    y_values = np.concatenate(y_values, casting = "no")
    return t_values, y_values, t_event, y_event   