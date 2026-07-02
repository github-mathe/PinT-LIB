#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:30:43 2026

@author: yessima
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scripts.notebooks.analyticalSolDahlquist import analytical_all, analytical_one_step
from scripts.notebooks.numericalSolDahlquist import *
from scripts.notebooks.lamDahlquist import lam
from scripts.notebooks.error_ExactNum import compute_Linf
from scripts.notebooks.pararealSolDahlquist import pararealADahlquist

# %%
# Dahlquist test problem parameters
alpha = 6.001   # to avoid resonance regime
u0 = 1+0j
t0 = 0
pStart = 1


# analytical solution
dt = 0.001
N = 5
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

#%%

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


# %%
# compute L_inf error between solutions
dtVals  =  1./(10**np.arange(6))
exact   = lambda dt: analytical_all(u0, alpha,lam, t0, dt, N)
num     = lambda dt: timeStepperAll(u0,alpha,lam, t0, dt, pStart, N)
#error_sol, error_solP, error_timeP = compute_Linf(dtVals, exact,timeStepperAll)

error_timeP = np.zeros_like(dtVals)
error_solP = np.zeros_like(dtVals)
TP_num = np.zeros((len(dtVals),N+1), dtype=float)
UP_num = np.zeros((len(dtVals),N+1), dtype = complex)
TP_num[:,0] = t0
UP_num[:,0] = u0
TP_th = np.zeros((len(dtVals),N+1), dtype=float)
UP_th = np.zeros((len(dtVals),N+1), dtype = complex)
TP_th[:,0] = t0
UP_th[:,0] = u0

for i, dt in enumerate(dtVals):
    u0 = 1+0j
    t0 = 0
    tTh, uTh,tpTh,upTh = analytical_all(u0, alpha,lam, t0, dt, N)
    TP_th[i,1:] = tpTh
    UP_th[i,1:] = upTh
    for n in range(N):
        _, _, tp_ex, up_ex, steps_ex = timeStepperAll(u0, alpha, lam, t0, dt, n+1, n+1)
        t0 = tp_ex[0]
        u0 = up_ex[0]
        TP_num[i,n+1] = t0
        UP_num[i,n+1] = u0
    #min_len = min(len(uTh), len(uNum))
    #uNum = uNum[:min_len]
    #uTh = uTh[:min_len]
    error_solP[i] = np.linalg.norm(UP_th[i,:]-UP_num[i,:], ord=np.inf)
    error_timeP[i] = np.linalg.norm(TP_num[i,:]-TP_th[i,:], ord=np.inf)  


plt.figure()
#plt.loglog(dtVals, error_sol, label=r"$\|U_{ex} - U_{num}\|$")
plt.loglog(dtVals, error_timeP, "--", c="gray", label=r"$\|tP_{ex} -tP_{num}\|_\infty$ error")
plt.loglog(dtVals, error_solP, "-*", c="gray", label=r"$\|uP_{ex} -uP_{num}\|_\infty$ error")
plt.xlabel("$dt$"), plt.ylabel("Error"), plt.grid();
plt.legend();


# %%
# parareal setup
#N       =   7

alpha = 6.001   # to avoid resonance regime
u0 = 1+0j
t0 = 0
pStart = 1

K       =   10
dtF     =   1/10000
dtG     =   1/100
#dtInit  =   1

# Parareal implementation 
F = lambda u0, t0, nP: timeStepperAll(u0, alpha, lam, t0, dtF, nP, nP)#[2:] # fine solver
G = lambda u0, t0, nP: timeStepperAll(u0, alpha, lam, t0, dtG, nP, nP)#[2:] # coarse solver

TP_para, UP_para,steps_para = pararealADahlquist(F, G, u0, t0, N, K)
#TP_para     =   TP_para[:,1:]
#UP_para      =   UP_para[:,1:]


# compute L_inf error in 
dtF_index = np.where(dtVals == dtF)[0][0]
#thres_U = error_sol[dtF_index]
thres_TP = error_timeP[dtF_index]
thres_UP = error_solP[dtF_index]


error_TP_para = np.zeros(K+1)                    # time between Parareal and exact
error_UP_para = np.zeros(K+1)                    # solution between Parareal and exact
#error_TP_coarse_fine = np.zeros((3,K+1))    # time between fine, coarse and sequential coarse solvers
#error_UP_coarse_fine = np.zeros((3,K+1))    # solution between fine, coarse and sequential coarse solvers


for k in range(K+1):
    error_TP_para[k] = np.linalg.norm(TP_para[k,:]-TP_num[dtF_index,:],ord=np.inf)
    error_UP_para[k] = np.linalg.norm(UP_para[k,:]-UP_num[dtF_index,:],ord=np.inf)

   # error_TP_coarse_fine[:,k]   = np.linalg.norm(T_coarse_fine[:,k,1:]-TP_ex,ord=np.inf,axis=1)
   # error_UP_coarse_fine[:,k]   = np.linalg.norm(U_coarse_fine[:,k,1:]-UP_ex,ord=np.inf,axis=1)

iterK = np.arange(K+1)
#errors_time = np.concatenate((error_TP_para.reshape(1,K+1), error_TP_coarse_fine), axis =0)
#errors_sol = np.concatenate((error_UP_para.reshape(1,K+1), error_UP_coarse_fine), axis =0)

labelsT = [r'$\|T_{Parareal} - T_{numEx}\|_\infty$']
labelsSol = [r'$\|Sol_{Parareal} - Sol_{numEx}\|_\infty$']


# %%
fig = plt.figure(figsize=(15, 8))
plt.subplot(1,2, 1)
plt.semilogy(iterK, error_TP_para, label=labelsT)
plt.axhline(y=thres_TP, color='red', linestyle='--', label="Fine solver error")
plt.xticks(iterK)
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend()

plt.subplot(1,2, 2)
plt.semilogy(iterK, error_UP_para, label=labelsSol)
plt.xticks(iterK)
plt.yscale('symlog', linthresh=1e-14)
plt.axhline(y=thres_UP, color='red', linestyle='--', label="Fine solver error")
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend()
plt.show()

# %% pointwise error 
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
for k in range(K+1):
    err_T = np.abs(TP_para[k,:]-TP_num[dtF_index,:])
    plt.semilogy(np.arange(N+1), err_T, label=f"k={k}")
plt.xticks(np.arange(1,N+1))
plt.yscale('symlog', linthresh=1e-14)

plt.xlabel("$n_{th}$ time period solution $T_n$"), plt.ylabel(r"$\|T_{n, Parareal} - T_{n,numEx}\|$")
plt.grid(True)
plt.legend()


plt.subplot(1,2,2)
for k in range(K+1):
    err_U = np.abs(UP_para[k,:]-UP_num[dtF_index,:])
    plt.semilogy(np.arange(N+1), err_U, label=f"k={k}")
plt.xticks(np.arange(1,N+1))
plt.yscale('symlog', linthresh=1e-14)

plt.xlabel("$n_{th}$ periodic solution $U_n$"), plt.ylabel(r'$\|Sol_{n,Parareal} - Sol_{n,numEx}\|$')
plt.grid(True)
plt.legend()
plt.show()

# %%








