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
            u.insert(-1, uP_real + 0j)
            tt.insert(-1, tP)  # insert tP in the correct position to maintain sorted order
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