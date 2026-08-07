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
from dahlquist_adaptive_parareal import *

#%%
alpha, lam, u0, t0, Period_start, _ = init_parametersDahlquist()
print(f"alpha = {alpha}, lam = {lam}, u0 = {u0}, t0 = {t0}, Period_start = {Period_start}")

dtVals = np.array([1e-1, 1e-2, 1e-3, 1e-4])
dtF     =   1/10000
dtG     =   1/100
F = lambda u0, t0, P: DahlquistBEEvent(y0=u0, t0=t0, dt=dtF, alpha=alpha, lam=lam(P),event_func=local_event_func(t0, u0, alpha, lam(P)))[2:] # fine solver
G = lambda u0, t0, P: DahlquistBEEvent(y0=u0, t0=t0, dt=dtG, alpha=alpha, lam=lam(P),event_func=local_event_func(t0, u0, alpha, lam(P)))[2:] # coarse solver
dtF_index = np.where(dtVals == dtF)[0][0]
Periods = [2,10]
lenPeriods = len(Periods)

error_TP_Periods = np.zeros((len(dtVals), lenPeriods))
error_UP_Periods = np.zeros((len(dtVals), lenPeriods), dtype = complex)

for idxN, N in enumerate(Periods):
    exact = lambda dt: solve_theoretical_solution(
            u_start = u0,
            t_start = t0,
            alpha = alpha,
            lam = lam,
            Period_start = Period_start,
            Period_end = N+Period_start-1,
            dt = dt
        )[2:]
    num = lambda dt: solve_numerical_solution(
            u_start = u0,
            t_start = t0,
            alpha = alpha,
            lam = lam,
            Period_start = Period_start,
            Period_end = N+Period_start-1,
            dt = dt,
            num_steps = 1000
        )[2:]
        
    error_TP, error_UP, TP_num, UP_num, _, _ = compute_errorDahlquist(
            dtVals = dtVals,
            exact = exact,
            num = num, 
            plot = False,
            Period_start = Period_start,
            Period_end = N+Period_start-1
        )
    thres_TP = error_TP[dtF_index]                   # threshold for time error between analytical and fine numerical
    thres_UP = error_UP[dtF_index] 
    error_TP_Periods[:, Periods.index(N)] = error_TP
    error_UP_Periods[:, Periods.index(N)] = error_UP
    TP_Para, UP_Para = PararealModified(F, G, u0, t0, N, N+2)
    error_TP_Para, error_UP_Para, _, _ = compute_errorPararealDahlquist(TP_Para, UP_Para, TP_num[dtF_index,:], UP_num[dtF_index,:], thres_TP, thres_UP, plot=True)    

#%%
fig,ax = plt.subplots(lenPeriods, 1, figsize=(12, 5))
for idxN, N in enumerate(Periods):
    ax[idxN].loglog(dtVals, error_TP_Periods[:, Periods.index(N)], "--", c="gray", label=r"$\|TP_{th} -TP_{num}\|_\infty$ error")
    ax[idxN].loglog(dtVals, error_UP_Periods[:, Periods.index(N)], "-*", c="black", label=r"$\|UP_{th} -UP_{num}\|_\infty$ error")
    ax[idxN].set_xlabel('dt')
    ax[idxN].set_ylabel('Error')
    ax[idxN].legend()
    ax[idxN].set_title(f'N={N}')
    plt.draw()
plt.show()