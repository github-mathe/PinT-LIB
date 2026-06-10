#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:30:43 2026

@author: yessima
"""
import numpy as np
import matplotlib.pyplot as plt
from scripts.notebooks.analyticalSolDahlquist import analytical_all, analytical_one_step
from scripts.notebooks.numericalSolDahlquist import *
from scripts.notebooks.lamDahlquist import lam
from scripts.notebooks.error_ExactNum import compute_Linf
from scripts.notebooks.pararealSolDahlquist import pararealADahlquist


# Dahlquist test problem parameters
alpha = 6.001   # to avoid resonance regime
u0 = 1+0j
t0 = 0
pStart = 1


# analytical solution
dt = 0.001
N = 3
t, uTh, tP, uP = analytical_all(u0, alpha,lam, t0, dt, N)

# plot Re vs Im parts of the analytical solution
plt.figure()
plt.plot(uTh.real, uTh.imag, label="Analytical")
plt.plot(uP.real, uP.imag, "o", label="Period point")
plt.legend(), plt.xlabel(r"$\Re(u)$"), plt.ylabel(r"$\Im(u)$"), plt.grid();

# plot Re/Im-solution parts vs time
plt.figure()
plt.plot(t, uTh.real, label=r"$\Re(u)$")
plt.plot(t, uTh.imag, label=r"$\Im(u)$")
plt.plot(tP, np.array(uP).imag, 'o', label="Period points")
plt.legend(loc="upper left"), plt.xlabel("time"), plt.ylabel("solution"), plt.grid();


# numerical solution
dt = 0.001
T, uNum, TP, UP, steps = timeStepperAll(u0, alpha, lam, t0, dt, pStart, N)

# plot Re vs Im parts of the 
# analytical and numerical solutions 
plt.figure()
plt.plot(uNum.real, uNum.imag, label="Numerical")
plt.plot(uTh.real, uTh.imag, "--", label="Analytical")
plt.plot(UP.real, UP.imag, 'o', label="Points")
plt.legend(loc="lower left"), plt.xlabel(r"$\Re(u)$"), plt.ylabel(r"$\Im(u)$"), plt.grid();


# compute L_inf error between solutions
dtVals  =  1./(10**np.arange(7))
exact   = lambda dt: analytical_all(u0, alpha,lam, t0, dt, N)
num     = lambda dt: timeStepperAll(u0,alpha,lam, t0, dt, pStart, N)
error_sol, error_solP, error_timeP = compute_Linf(dtVals, exact,num)

plt.figure()
plt.loglog(dtVals, error_sol, label=r"$\|U_{ex} - U_{num}\|$")
plt.loglog(dtVals, error_timeP, "--", c="gray", label=r"$\|tP_{ex} -tP_{num}\|$ error")
plt.loglog(dtVals, error_solP, "-*", c="gray", label=r"$\|uP_{ex} -tP_{num}\|$ error")
plt.xlabel("$dt$"), plt.ylabel("Error"), plt.grid();
plt.legend();


# parareal setup
N       =   5
K       =   N
dtF     =   1/1000
dtG     =   1
dtInit  =   0.01

# Parareal implementation 
F = lambda u0, t0, dtF, nP: timeStepperAll(u0, alpha, lam, t0, dtF, nP, nP)[2:] # fine solver
G = lambda u0, t0, dtG, nP: timeStepperAll(u0, alpha, lam, t0, dtG, nP, nP)[2:] # coarse solver

TP_para, UP_para, T_coarse_fine, U_coarse_fine,steps_para = pararealADahlquist(F, G, u0, t0, N, K, dtInit, dtF,dtG)
TP_para     =   TP_para[:,1:]
UP_para      =   UP_para[:,1:]
_, _, TP_ex, UP_ex, steps_ex = timeStepperAll(u0, alpha, lam, t0, dtF, pStart, N)



# compute L_inf error in 
error_TP_para = np.zeros(K+1)                    # time between Parareal and exact
error_UP_para = np.zeros(K+1)                    # solution between Parareal and exact
error_TP_coarse_fine = np.zeros((3,K+1))    # time between fine, coarse and sequential coarse solvers
error_UP_coarse_fine = np.zeros((3,K+1))    # solution between fine, coarse and sequential coarse solvers


for k in range(K+1):
    error_TP_para[k] = np.linalg.norm(TP_para[k,:]-TP_ex,ord=np.inf)
    error_UP_para[k] = np.linalg.norm(UP_para[k,:]-UP_ex,ord=np.inf)

    error_TP_coarse_fine[:,k]   = np.linalg.norm(T_coarse_fine[:,k,1:]-TP_ex,ord=np.inf,axis=1)
    error_UP_coarse_fine[:,k]   = np.linalg.norm(U_coarse_fine[:,k,1:]-UP_ex,ord=np.inf,axis=1)

iterK = np.repeat(np.arange(K+1).reshape(1,6), repeats=4, axis=0)
errors_time = np.concatenate((error_TP_para.reshape(1,6), error_TP_coarse_fine), axis =0)
errors_sol = np.concatenate((error_UP_para.reshape(1,6), error_UP_coarse_fine), axis =0)

fig = plt.figure(figsize=(15, 8))

plt.subplot(1,2, 1)
plt.semilogy(iterK.T,errors_time.T)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend(["Error in Parareal grid","Error in fine grid","Error in coarse grid", "Error in coarse sequential grid"])

plt.subplot(1,2, 2)
plt.semilogy(iterK.T,errors_sol.T)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend(["Error in Parareal solution","Error in fine solution","Error in coarse solution", "Error in coarse sequential solution"], loc='best')

