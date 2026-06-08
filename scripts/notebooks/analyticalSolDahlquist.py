#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:36:24 2026

@author: yessima
"""
import numpy as np

# Analytical solution for one step
def analytical_one_step(u0, alpha, t0, t1, lam):
    C = np.exp(-lam*t0)*(u0 + (alpha*np.cos(alpha*t0) + lam*np.sin(alpha*t0))/(alpha**2 + lam**2))
    u = C*np.exp(lam*t1) - (alpha*np.cos(alpha*t1) + lam*np.sin(alpha*t1))/(alpha**2 + lam**2)
    return u

# Analytical solution until period N
def analytical_all(u0, alpha,lam_f, t0, dt, N, plot_sol = False):
    u = [u0]
    nP = 1
    occ = 0
    t = [t0]
    TP  =   np.zeros(N, dtype=float)
    UP  =   np.zeros_like(TP, dtype=complex)
    while nP < N+1: # we want to capture at least two periods to be sure about the period points
        lam_nP = lam_f(nP)
        assert lam_nP**2 + alpha**2 != 0, "resonance regime, different analytical solution"
        t_next = t[-1] + dt
        t.append(t_next)
        u_next = analytical_one_step(u[-1], alpha, t[-2], t[-1], lam_nP)
        u.append(u_next)
        occ += u[-1].imag*u[-2].imag < 0 # check for sign change in the imaginary part
        if occ == 2: # we completed a full period
            tP = np.interp(0, [u[-2].imag, u[-1].imag], [t[-2], t[-1]])
            uP_real = np.interp(tP, [t[-2], t[-1]], [u[-2], u[-1]])
            TP[nP-1] = tP  # insert tP in the correct position to maintain sorted order
            UP[nP-1] = uP_real + 0j
            t.insert(-1, tP)  
            u.insert(-1, uP_real + 0j)
            nP += 1
            occ = 0
    return np.squeeze(t), np.squeeze(u), TP, UP
