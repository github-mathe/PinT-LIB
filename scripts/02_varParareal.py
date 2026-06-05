#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script of Parareal with variable time-windows
"""
import numpy as np
import matplotlib.pyplot as plt

# Dahlquist test problem parameters
eps = lambda n: 0.01*n
lam = lambda n: 1j*(1 + eps(n))
Period =lambda n: 2*np.pi/abs((lam(n)))
t0 = 0
tEnd = 12
alpha = 6.001   # to avoid resonance regime
u0 = 1+0j

# linear interpolation function
def linInterp(x1,x2,y1,y2,x):
    m = (y2-y1)/(x2-x1)
    y=y1+m*(x-x1)
    return y

def timeOneStep(u0, t1, dt, lam):
    u = (u0 + dt*np.sin(alpha*t1))/(1-dt*lam)
    return u

def timeStepperPeriod(u0,t0,nP,dt):
    u = [u0]
    tt = [t0]
    lam_nP = lam(nP)

    #UP = np.zeros(1, dtype=complex)  # to store the period point for the current period
    #TP = np.zeros(1, dtype=float)  # to store the period point for the current period
    occ = 0
    steps = 0
    while True:
        t_next = tt[-1] + dt
        u_next = timeOneStep(u[-1], t_next, dt, lam_nP)

        u.append(u_next)
        tt.append(t_next)

        occ += u[-1].imag*u[-2].imag < 0 # check for sign change in the imaginary part
        steps += 1
        if occ == 2: # we completed a full period
            #nP += 1
            #lam_nP = lam(nP)

            tP = linInterp(u[-2].imag, u[-1].imag, tt[-2], tt[-1], 0)
            uP_real = linInterp(tt[-2], tt[-1], u[-2].real, u[-1].real, tP)
            UP = uP_real + 0j
            TP = tP
            steps += 1
            u.insert(-1, uP_real + 0j)
            tt.insert(-1, tP)  # insert tP in the correct position to maintain sorted order
            break
    return tt, u, TP, UP, steps

def timeStepperAll(u0, t0, dt, pStart, pEnd):
    nP = pStart
    tt = [t0]
    u = [u0]
    steps = 0
    TP = np.zeros(pEnd - pStart + 1, dtype=float)
    UP = np.zeros_like(TP, dtype=complex)
    while nP < pEnd + 1:
        t1, u1, TP1, UP1, steps = timeStepperPeriod(u[-1], tt[-1], nP, dt)
        tt.extend(t1[1:])
        u.extend(u1[1:])
        TP[nP - pStart] = TP1
        UP[nP - pStart] = UP1
        nP += 1
        steps += steps
    return (np.asarray(tt, dtype=float),
            np.asarray(u, dtype=complex),
            TP,
            UP,
            steps)


N = 10 # time windows - coarse time grid
K = N # Parareal iterations
dtF = 1/1000 # Fine solver's time steps
dtG = 1/10 # Coarse solver's time steps
#tEnd = 12.39

pStart = 1
T = 0
TT =  np.zeros((K+1, N+1), dtype=float)
TT_p = np.zeros((K+1, N), dtype=float)
U_p = np.zeros((K+1, N), dtype=complex)


# Parareal implementation
F = lambda u0, t0, nP: timeStepperAll(u0, t0, dtF, nP, nP) # fine solver
G = lambda u0, t0, nP: timeStepperAll(u0, t0, dtG, nP, nP) # Coarse solver

U = np.zeros((K+1, N+1), dtype=complex)
U[:, 0] = u0
steps_para = np.zeros((K+1, N), dtype=int)
# Initial guess
p = pStart

for n in range(N):
    TG, UG, TPG, UPG, steps = G(U[0, n], TT[0,n], n+1)
    TT[0,n+1] = TPG[0]#TG[-1]
    U[0, n+1] = UPG[0]#UG[-1]
    TT_p[0,n] = TPG[0]
    U_p[0,n] = UPG[0]
    steps_para[0, n] = steps


for k in range(K):  # Parareal iterations
    for n in range(N):
        TF, UF, TPF, UPF, steps = F(U[k, n], TT[k,n],n+1)
        TG, UG, TPG, UPG, _ = G(U[k, n], TT[k,n],n+1)
        TG_seq, UG_seq, TPG_seq, UPG_seq,_ = G(U[k+1, n], TT[k+1,n],n+1)
        TT[k+1, n+1] = TPF[0] + TPG[0] - TPG_seq[0]
        U[k+1, n+1] = UPF[0] + UPG[0] - UPG_seq[0]
        TT_p[k+1, n] = TPF[0]
        U_p[k+1, n] = UPF[0]
        steps_para[k+1, n] = steps


# exact solution
t_ex, U_exact, tP_ex, uP_ex, steps_num = timeStepperAll(u0, t0, dtF/N,pStart, N)
steps_num, sum(steps_para[-1,:])


errors_para_TT = []
errors_para_U = []
for k in range(K+1):
    #errors_para_tP.append(np.linalg.norm(TT_p[k]-np.array(tP_ex),ord=np.inf))
    #errors_para_uP.append(np.linalg.norm(U_p[k]-np.array(uP_ex),ord=np.inf))
    errors_para_U.append(np.linalg.norm(U[k,1:]-uP_ex,ord=np.inf))
    errors_para_TT.append(np.linalg.norm(TT[k,1:]-tP_ex,ord=np.inf))
plt.semilogy(range(K+1), errors_para_TT)
plt.semilogy(range(K+1), errors_para_U)
plt.xticks(range(K+1))
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend(["Error in coarse grid", "Error in coarse solution"])
# tTh, uTh,tpTh,upTh = analytical_all(u0, t0, dtF/N, N)
# min_len = min(len(uTh), len(U_exact))
# U_exact = U_exact[:min_len]
# uTh = uTh[:min_len]
# error_seq = np.linalg.norm(uTh-U_exact, ord=np.inf)
# error_seq_tP = np.linalg.norm(tpTh-tP_ex, ord=np.inf)

print(f"Error in coarse time: {errors_para_TT[K]:.6e}")
print(f"Error in coarse solution: {errors_para_U[K]:.6e}")
