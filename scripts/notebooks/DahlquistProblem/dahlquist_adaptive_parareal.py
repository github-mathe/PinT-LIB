import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable

root = Path.cwd().resolve()
while root != root.parent and not (root / "scripts" / "notebooks").exists():
    root = root.parent
root = root / "scripts" / "notebooks"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
    
from tmp.numericalSolDahlquist import DahlquistBEEvent
from Solver.AdaptiveParareal import PararealModified
from ModifiedProblem.modified_dahlquist import exact_global_solution, check_event

def init_parametersDahlquist(
    alpha=6.001,
    lam = lambda n: 1j*(1+0.01*n),
    u0 = 1+0j,
    t0 = 0.0,
    Period_start = 1,
    Period_end = 3
    ):
    """_summary_

    Args:
        alpha (float, optional): _description_. Defaults to 6.001.
        lam (_type_, optional): _description_. Defaults to lambdan:1j*(1+0.01*n).
        u0 (_type_, optional): _description_. Defaults to 1+0j.
        t0 (float, optional): _description_. Defaults to 0.0.
        Period_start (int, optional): _description_. Defaults to 1.
        Period_end (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    alpha = alpha
    lam = lam
    u0 = u0
    t0 = t0
    Period_start = Period_start
    Period_end = Period_end
    return alpha, lam, u0, t0, Period_start, Period_end
def solve_theoretical_solution(alpha, lam, u_start, t_start, Period_start, Period_end, dt=1e-3, user_action: Callable = None):
    tEx, uEx, tpEx, upEx = exact_global_solution(
        u_start = u_start,
        t_start = t_start,
        alpha = alpha,
        lam = lam,
        Period_start = Period_start,
        Period_end = Period_end,
        dt = dt
    )
    if user_action is not None:
        user_action(tEx, uEx, tpEx, upEx)
    return tEx, uEx, tpEx, upEx
def local_event_func(t_start, u_start, alpha, lam_P):
    return lambda t, u: check_event(
        t_start=t_start,
        u_start=u_start,
        t = t,
        u = u,
        alpha=alpha,
        lam=lam_P,
    )
def solve_numerical_solution(alpha, lam, u_start, t_start, Period_start, Period_end, dt=1e-3,num_steps: int = 1000, user_action: Callable = None):
    alpha = alpha
    lam = lam
    u_start = u_start.copy() if isinstance(u_start, np.ndarray) else u_start
    t_start = t_start
    Period_start = Period_start
    Period_end = Period_end
    dt = dt
    num_steps = num_steps
    window_grids: list[np.ndarray] = []
    window_solutions: list[np.ndarray] = []
    tpNum = np.zeros(Period_end - Period_start + 2, dtype=float)
    tpNum[0] = t_start
    upNum = np.empty_like(tpNum, dtype=complex)
    upNum[0] = u_start
    for P in range(Period_start, Period_end+1):
        tt, uu, tp, up = DahlquistBEEvent(
            y0 = upNum[P-Period_start],
            t0 = tpNum[P-Period_start],
            alpha = alpha,
            lam = lam(P),
            dt = dt,
            event_func = local_event_func(
                                t_start=tpNum[P-Period_start],
                                u_start=upNum[P-Period_start],
                                alpha=alpha,
                                lam_P=lam(P)
                                ),
            num_steps = num_steps
        )
        window_grids.append(tt)
        window_solutions.append(uu)
        tpNum[P - Period_start+1] = float(tp)
        upNum[P - Period_start+1] = up
    tNum = np.concatenate(window_grids)
    uNum = np.concatenate(window_solutions)
    if user_action is not None:
        user_action(tNum, uNum, tpNum, upNum)
    return tNum, uNum, tpNum, upNum 
def plot_Dahlquist(t,u, tp, up, **kwargs):
    plt.figure(figsize=(15, 6))
    # plot Re vs Im parts 
    plt.subplot(1, 2, 1)
    plt.plot(u.real, u.imag, label=kwargs.get("label", "solution"))
    plt.plot(up.real, up.imag, "o", label="PseudoPeriod points")
    plt.legend(), plt.xlabel(r"$\Re(u)$"), plt.ylabel(r"$\Im(u)$"), plt.grid();

    # plot Re/Im-solution parts vs time
    plt.subplot(1, 2, 2)
    plt.plot(t, u.real, label=r"$\Re(u)$")
    plt.plot(t, u.imag, label=r"$\Im(u)$")
    plt.plot(tp, np.array(up).imag, 'o', label="PseudoPeriod points")
    plt.legend(loc="upper left"), plt.xlabel("time"), plt.ylabel("solution"), plt.grid();
    plt.draw()
    plt.pause(0.001)
def compute_errorDahlquist(dtVals, exact, num, plot=False, **kwargs):
    Period_start = kwargs.get("Period_start", 1)
    Period_end = kwargs.get("Period_end", 3)
    N = Period_end - Period_start + 1
    error_TP = np.zeros_like(dtVals)
    error_UP = np.zeros_like(dtVals)

    TP_num = np.zeros((len(dtVals),N + 1), dtype=float)
    UP_num = np.zeros((len(dtVals),N + 1), dtype = complex)

    TP_th = np.zeros((len(dtVals),N + 1), dtype=float)
    UP_th = np.zeros((len(dtVals),N + 1), dtype = complex)
    for i, dt in enumerate(dtVals):
        TP_th[i,:], UP_th[i,:] = exact(dt = dt)
        TP_num[i,:], UP_num[i,:] = num(dt = dt)
        error_UP[i] = np.linalg.norm(UP_num[i,:]-UP_th[i,:], ord=np.inf)
        error_TP[i] = np.linalg.norm(TP_num[i,:]-TP_th[i,:], ord=np.inf)  
    if plot:
        plt.figure()
        plt.loglog(dtVals, error_TP, "--", c="gray", label=r"$\|tP_{ex} -tP_{num}\|_\infty$ error")
        plt.loglog(dtVals, error_UP, "-*", c="gray", label=r"$\|uP_{ex} -uP_{num}\|_\infty$ error")
        plt.xlabel("$dt$"), plt.ylabel("Error"), plt.grid();
        plt.legend();
        plt.draw()
        plt.pause(0.001)
    return error_TP, error_UP, TP_num, UP_num, TP_th, UP_th
def compute_errorPararealDahlquist(TP_Para, UP_Para, TP_ex, UP_ex, thresT, thresU, plot = False):
    K = TP_Para.shape[0]
    N = TP_Para.shape[1]
    error_TP = [np.linalg.norm(TP_Para[k,:]-TP_ex[:], ord=np.inf) for k in range(K)]
    error_UP = [np.linalg.norm(UP_Para[k,:]-UP_ex[:], ord=np.inf) for k in range(K)]
    error_TP_per_window = np.zeros((K, N), dtype=float)
    error_UP_per_window = np.zeros((K, N), dtype=float)
    for k in range(K):
        error_TP_per_window[k,:] = np.abs(TP_Para[k,:]-TP_ex[:])
        error_UP_per_window[k,:] = np.abs(UP_Para[k,:]-UP_ex[:])
    if plot:
        iterK = np.arange(K)
        labelsT = [r'$\|T_{Parareal} - T_{FineNum}\|_\infty$']
        labelsSol = [r'$\|U_{Parareal} - U_{FineNum}\|_\infty$']

        plt.figure(figsize=(15, 8))
        plt.title(f"Adaptive Parareal Error for N={N-1}")

        plt.subplot(2,2, 1)
        plt.semilogy(iterK, error_TP, label=labelsT)
        plt.axhline(y=thresT, color='red', linestyle='--', label="Fine solver error")
        plt.xticks(iterK)
        plt.yscale('symlog', linthresh=1e-14)
        plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
        plt.legend()

        plt.subplot(2,2, 2)
        plt.semilogy(iterK, error_UP, label=labelsSol)
        plt.xticks(iterK)
        plt.yscale('symlog', linthresh=1e-14)
        plt.axhline(y=thresU, color='red', linestyle='--', label="Fine solver error")
        plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
        plt.legend()
        
        plt.subplot(2,2, 3)
        for k in range(K):
            plt.semilogy(np.arange(N), error_TP_per_window[k,:], label=f"k={k}")
        plt.xticks(np.arange(N))
        plt.axhline(y=thresT, color='red', linestyle='--', label="Fine solver error")
        plt.yscale('symlog', linthresh=1e-12)

        plt.xlabel("$n_{th}$ time period solution $T_n$"), plt.ylabel(r"$\|T_{n, Parareal} - T_{n,FineNum}\|$")
        plt.grid(True)
        plt.legend(loc="upper left")

        plt.subplot(2,2, 4)
        for k in range(K):
            plt.semilogy(np.arange(N), error_UP_per_window[k,:], label=f"k={k}")
            plt.draw()
        plt.xticks(np.arange(N))
        plt.axhline(y=thresU, color='red', linestyle='--', label="Fine solver error")
        plt.yscale('symlog', linthresh=1e-12)
        plt.xlabel("$n_{th}$ time period solution $T_n$"), plt.ylabel(r"$\|T_{n, Parareal} - T_{n,FineNum}\|$")
        plt.grid(True)
        plt.legend(loc="upper left")
        
        plt.draw()
        plt.pause(0.01)
    return error_TP, error_UP, error_TP_per_window, error_UP_per_window

#%%
if __name__ == "__main__":
    
    alpha, lam, u0, t0, Period_start, Period_end = init_parametersDahlquist(Period_end=10)
    print("Parameters initialized:\n alpha = {}\n lam = {}\n u0 = {}\n t0 = {}\n Period_start = {}\n Period_end = {}".format(alpha, lam, u0, t0, Period_start, Period_end))
    # Solve exact solution 
    dtTh = 1e-3
    tEx, uEx, tpEx, upEx = solve_theoretical_solution(
    u_start = u0,
    t_start = t0,
    alpha = alpha,
    lam = lam,
    Period_start = Period_start,
    Period_end = Period_end,
    dt = dtTh,
    user_action = lambda t, u, tp, up: plot_Dahlquist(t, u, tp, up, label="theoretical")
    )

    #solve numerical solution with direct Backward Euler method
    tNum, uNum, tpNum, upNum = solve_numerical_solution(
    u_start = u0,
    t_start = t0,
    alpha = alpha,
    lam = lam,
    Period_start = Period_start,
    Period_end = Period_end,
    dt = dtTh,
    num_steps = 1000,
    user_action = lambda t, u, tp, up: plot_Dahlquist(t, u, tp, up, label = "numerical")
    )
    
    plt.figure(figsize=(12,5))
    plt.plot(uNum.real, uNum.imag, '--',label="$numerical$")
    plt.plot(uEx.real, uEx.imag, label=r"$analytical$")
    plt.legend(loc="lower left"), plt.xlabel(r"$\Re(u)$"), plt.ylabel(r"$\Im(u)$"), plt.grid();    plt.show()
    
    # Compute error between exact and numerical solution for different dt values
    exact = lambda dt: solve_theoretical_solution(
        u_start = u0,
        t_start = t0,
        alpha = alpha,
        lam = lam,
        Period_start = Period_start,
        Period_end = Period_end,
        dt = dt
    )[2:]
    num = lambda dt: solve_numerical_solution(
        u_start = u0,
        t_start = t0,
        alpha = alpha,
        lam = lam,
        Period_start = Period_start,
        Period_end = Period_end,
        dt = dt,
        num_steps = 1000
    )[2:]
    
    dtVals = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    
    error_TP, error_UP, TP_num, UP_num, TP_th, UP_th = compute_errorDahlquist(
        dtVals = dtVals,
        exact = exact,
        num = num, 
        plot = True,
        Period_start = Period_start,
        Period_end = Period_end
    )
    
    # Parareal method with adaptive time-stepping
    N = Period_end - Period_start + 1
    K = N+2
    dtF     =   1/1000
    dtG     =   1/100

    F = lambda u0, t0, P: DahlquistBEEvent(y0=u0, t0=t0, dt=dtF, alpha=alpha, lam=lam(P),event_func=local_event_func(t0, u0, alpha, lam(P)))[2:] # fine solver
    G = lambda u0, t0, P: DahlquistBEEvent(y0=u0, t0=t0, dt=dtG, alpha=alpha, lam=lam(P),event_func=local_event_func(t0, u0, alpha, lam(P)))[2:] # coarse solver
    TP_Para, UP_Para = PararealModified(F, G, u0, t0, N, K)
    
    dtF_index = np.where(dtVals == dtF)[0][0]
    thres_TP = error_TP[dtF_index]                   # threshold for time error between analytical and fine numerical
    thres_UP = error_UP[dtF_index]                   # threshold for solution error between analytical and fine numerical
    error_TP_Para, error_UP_Para, error_TP_per_window, error_UP_per_window = compute_errorPararealDahlquist(TP_Para, UP_Para, TP_num[dtF_index,:], UP_num[dtF_index,:], thres_TP, thres_UP, plot=True)