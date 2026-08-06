#%%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
root = Path.cwd().resolve()
while root != root.parent and not (root / "scripts" / "notebooks").exists():
    root = root.parent
root = root / "scripts" / "notebooks"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))    
from DahlquistProblem.tmp.numericalSolDahlquist import DahlquistBE, DahlquistBEEvent
from Solver.AdaptiveParareal import PararealModified
from ModifiedProblem.modified_dahlquist import exact_local_solution, exact_global_solution, f_ODE, check_event
import time 

# Dahlquist test problem parameters
alpha = 6.001   # to avoid resonance regime
u0 = 1+0j
t0 = 0
pStart = 1
lam = lambda n: 1j*(1+0.01*n)
PPeriod = [1,5,10,15]

print(f"alpha = {alpha}, u0 = {u0}, t0 = {t0}, pStart = {pStart}, lam(n) = 1j*(1+0.01*n)")

#%%
# error between theoretical and numerical solution
dtVals  =  1./(10**np.arange(1,6))
errorT_th_num = np.zeros((len(dtVals), len(PPeriod)))
errorU_th_num = np.zeros((len(dtVals), len(PPeriod)), dtype = complex)

plt.figure(figsize=(12, 5))
for N in PPeriod:
    TP_num = np.zeros((len(dtVals),N + 1), dtype=float)
    UP_num = np.zeros((len(dtVals),N + 1), dtype = complex)
    TP_num[:,0] = t0
    UP_num[:,0] = u0.copy() if isinstance(u0, np.ndarray) else u0
        
    TP_th = np.zeros((len(dtVals),N + 1), dtype=float)
    UP_th = np.zeros((len(dtVals),N + 1), dtype = complex)
    for i, dt in enumerate(dtVals):
        # exact solution
        tpTh,upTh = exact_global_solution(
                u_start = u0,
                t_start = t0,
                alpha = alpha,
                lam = lam,
                Period_start = pStart,
                Period_end = pStart + N - 1,
                dt = dt
                )[2:]
        TP_th[i,:] = tpTh
        UP_th[i,:] = upTh
        
        # numerical solution
        for numevent in range(N):
            tpnum, upnum = DahlquistBEEvent(y0=UP_num[i,numevent], t0=TP_num[i,numevent], dt=dt, alpha=alpha, lam=lam(numevent+1),event_func=lambda t,u:check_event(t0, u0, t, u, alpha, lam(numevent+1)))[2:]
            TP_num[i,numevent+1] = tpnum
            UP_num[i,numevent+1] = upnum.copy() if isinstance(upnum, np.ndarray) else upnum
    
    errorT_th_num[:, PPeriod.index(N)] = np.linalg.norm(TP_th - TP_num, ord=np.inf, axis=1)
    errorU_th_num[:, PPeriod.index(N)] = np.linalg.norm(UP_th - UP_num, ord=np.inf, axis=1)
    ax = plt.subplot(2,2, PPeriod.index(N)+1)
    ax.loglog(dtVals, errorT_th_num[:, PPeriod.index(N)], "--", c="gray", label=r"$\|TP_{th} -TP_{num}\|_\infty$ error")
    ax.loglog(dtVals, errorU_th_num[:, PPeriod.index(N)], "-*", c="black", label=r"$\|UP_{th} -UP_{num}\|_\infty$ error")
    ax.set_xlabel('dt')
    ax.set_ylabel('Error')
    ax.legend()
    ax.set_title(f'N={N}')

# %%
#Parareal setup
dtF = 1e-3
dtG = 1e-1
dtF_index = np.where(dtVals == dtF)[0][0]
K = max(PPeriod) + 1
# Parareal implementation 
check_pseudo_period = lambda u0,t0,nP:lambda t, u: check_event(t0, u0, t, u, alpha, lam(nP))
F2 = lambda u0, t0, nP: DahlquistBEEvent(y0=u0, t0=t0, dt=dtF, alpha=alpha, lam=lam(nP),event_func=check_pseudo_period(u0, t0, nP))[2:] # fine solver
G2 = lambda u0, t0, nP: DahlquistBEEvent(y0=u0, t0=t0, dt=dtG, alpha=alpha, lam=lam(nP),event_func=check_pseudo_period(u0, t0, nP))[2:] # coarse solver
errT_Para_ex = np.zeros((K+1, len(PPeriod)), dtype=float)
errU_Para_ex = np.zeros((K+1, len(PPeriod)), dtype=float)
for N in PPeriod:
    
    thres_TP = errorT_th_num[dtF_index, PPeriod.index(N)]                   # threshold for time error between analytical and fine numerical
    thres_UP = errorU_th_num[dtF_index, PPeriod.index(N)]                   # threshold for solution error between analytical and fine numerical

    print(f"Running Parareal for nP = {N}")
    time_start = time.time()
    TP_Para, UP_Para = PararealModified(F2, G2, u0, t0, N, K)
    time_end = time.time()
    print(f"Parareal completed for nP = {N} in {time_end - time_start:.4f} seconds")
    
    # fine exact solution
    TP_num = np.zeros(N + 1, dtype=float)
    UP_num = np.zeros(N + 1, dtype = complex)
    TP_num[0] = t0
    UP_num[0] = u0.copy() if isinstance(u0, np.ndarray) else u0
    for numevent in range(N):
        tpnum, upnum = DahlquistBEEvent(y0=UP_num[numevent], t0=TP_num[numevent], dt=dtF, alpha=alpha, lam=lam(numevent+1),event_func=lambda t,u:check_event(t0, u0, t, u, alpha, lam(numevent+1)))[2:]
        TP_num[numevent+1] = tpnum
        UP_num[numevent+1] = upnum.copy() if isinstance(upnum, np.ndarray) else upnum
    
    
    for k in range(K+1):
        errT_Para_ex[k, PPeriod.index(N)] = np.linalg.norm(TP_Para[k,:]-TP_num[:],ord=np.inf)
        errU_Para_ex[k, PPeriod.index(N)] = np.linalg.norm(UP_Para[k,:]-UP_num[:],ord=np.inf)

    # plot
    iterK = np.arange(K+1)
    labelsT = [r'$\|T_{Parareal} - T_{ex}\|_\infty$']
    labelsSol = [r'$\|U_{Parareal} - U_{ex}\|_\infty$']

    fig = plt.figure(figsize=(15, 8))
    plt.subplot(1,2, 1)
    plt.semilogy(iterK, errT_Para_ex[:, PPeriod.index(N)], label=labelsT)
    plt.axhline(y=thres_TP, color='red', linestyle='--', label="Fine solver error")
    plt.xticks(iterK)
    plt.yscale('symlog', linthresh=1e-14)
    plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
    plt.legend()

    plt.subplot(1,2, 2)
    plt.semilogy(iterK, errU_Para_ex[:, PPeriod.index(N)], label=labelsSol)
    plt.axhline(y=thres_UP, color='red', linestyle='--', label="Fine solver error")
    plt.xticks(iterK)
    plt.yscale('symlog', linthresh=1e-14)
    plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
    plt.legend()
    plt.title(f'N={N}')
    plt.show()
# %%
