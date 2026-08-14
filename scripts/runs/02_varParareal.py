import numpy as np
import matplotlib.pyplot as plt
from batpint.problems.dahlquist import Dahlquist
from batpint.solvers.solver import Solver
from batpint.solvers.parareal import PararealModified

# Dahlquist test problem parameters
eps = lambda n: 0.01*n
lam = lambda n: 1j*(1 + eps(n))
t0 = 0
alpha = 6.001   # to avoid resonance regime
u0 = 1+0j
pStart = 1
numP = 12

# Time discretization parameters 
dtEx = 0.001
dtNum = 0.01

dahlquist = Dahlquist(u_start=u0, t_start=t0, P_start=pStart, alpha=alpha, lam=lam)
tTh, uTh, tpTh, upTh = dahlquist.u_exact_global(numP, dtEx)
solver = Solver(problem = dahlquist, method = dahlquist.DahlquistBE , dt = dtNum)
tNum, uNum, tpNum, upNum = solver.solve(num_events=numP)
# Plotting the results
plt.plot(uNum.real, uNum.imag, label="Numerical")
plt.plot(uTh.real, uTh.imag, "--", label="Analytical")
plt.plot(upNum.real, upNum.imag, 'o', label="upNum")
plt.legend(loc="lower left"), plt.xlabel(r"$\Re(u)$"), plt.ylabel(r"$\Im(u)$"), plt.grid();

# error analysis
dtVals =  1./(10**np.arange(5))
error_tP = np.zeros_like(dtVals)
error_uP = np.zeros_like(dtVals)

for i,dt in enumerate(dtVals):
    error_tP[i], error_uP[i] = solver.compute_event_error(num_events=numP, dt=dt, exact_solver=dahlquist.u_exact_global)
    print(f"dt = {dt}, error_tP = {error_tP[i]}, error_uP = {error_uP[i]}")
# Plotting the error
plt.figure()
plt.loglog(dtVals, error_tP, "--", c="gray", label=r"$\|tp_{ex} -tp_{num}\|_\infty$ error")
plt.loglog(dtVals, error_uP, "-*", c="gray", label=r"$\|up_{ex} -up_{num}\|_\infty$ error")
plt.xlabel("$dt$"), plt.ylabel("Error"), plt.grid();plt.legend()
plt.draw()
plt.show()

# Adaptive Parareal

N = numP # time windows - coarse time grid
K = N+1 # Parareal iterations
dtF = 1/1000 # Fine solver's time steps
dtG = 1/100  # Coarse solver's time steps
dahlquistBE_F = Solver(problem = dahlquist, method = dahlquist.DahlquistBE , dt = dtF)
dahlquistBE_G = Solver(problem = dahlquist, method = dahlquist.DahlquistBE , dt = dtG)
# Parareal implementation 
F = lambda t, u, P: [res[-1] for res in dahlquistBE_F.advance_event(t,u,P)] # fine solver
G = lambda t, u, P: [res[-1] for res in dahlquistBE_G.advance_event(t,u,P)] # Coarse solver
tpPara, upPara = PararealModified(F, G,dahlquist.t_start, dahlquist.u_start, N, K)

# compare with fine exact solution
tFine,uFine,tpFine, upFine = dahlquistBE_F.solve(num_events=N)
dtF_index = np.where(dtVals == dtF)[0][0]
thres_tp = error_tP[dtF_index]                   # threshold for time error between analytical and fine numerical
thres_up = error_uP[dtF_index]  
error_tp_per_window = np.zeros((K, N+1), dtype=float)
error_up_per_window = np.zeros((K, N+1), dtype=float)
for k in range(K):
    error_tp_per_window[k,:] = np.abs(tpPara[k,:]-tpFine)
    error_up_per_window[k,:] = np.abs(upPara[k,:]-upFine)
error_tp_per_iteration = np.max(error_tp_per_window, axis=1)
error_up_per_iteration = np.max(error_up_per_window, axis=1)

#plotting the error per iteration
fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Adaptive Parareal Error for N={N+1}")
plt.subplot(1,2, 1)
plt.semilogy(range(K), error_tp_per_iteration, label=r'$\|tp_{Parareal} - tp_{FineNum}\|_\infty$')
plt.axhline(y=thres_tp, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(K))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend()

plt.subplot(1,2, 2)
plt.semilogy(range(K), error_up_per_iteration, label=r'$\|up_{Parareal} - up_{FineNum}\|_\infty$')
plt.axhline(y=thres_up, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(K))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend()  
plt.tight_layout()  

# plotting the error vs N points
fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Adaptive Parareal Error for N={N+1}")
plt.subplot(1,2, 1)
for k in range(K):
    plt.semilogy(range(N+1), error_tp_per_window[k,:], label=f"k={k}")
plt.axhline(y=thres_tp, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(N+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Time window n"), plt.ylabel("Error"), plt.grid();


plt.subplot(1,2, 2)
for k in range(K):
    plt.semilogy(range(N+1), error_up_per_window[k,:], label=f"k={k}")
plt.axhline(y=thres_up, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(N+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Time window n"), plt.ylabel("Error"), plt.grid();
plt.tight_layout()
    