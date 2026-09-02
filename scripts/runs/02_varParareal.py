# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from batpint.timestepping import timestepper
from batpint.timestepping.timestepper import TimeStepper, TimeStepperPropagator, PropagationState
from batpint.problems.dahlquist import Dahlquist, DahlquistBE, DahlquistExact
from batpint.parareal.parareal import PararealModified

# Dahlquist test problem parameters
t_start = 0
u_start = 1+0j
ev_Start = 1
dtNum = 0.001
dtEx = 0.0001

# define lam and alpha
def lam(cycle):
    return 1j*(1 + 0.01*(cycle+ev_Start))
alpha = 6.001   # to avoid resonance regime

# define event and termination functions
def event(t, u, cycle, u_event):
    return np.imag(u) - np.imag(u_event)

def terminate(state):
    return abs(alpha**2 + lam(state.cycle)**2) < 1e-10

# Create Dahlquist problem and Backward Euler method
problem = Dahlquist(t_start=t_start, u_start=u_start, lam=lam, alpha=alpha, event=event)

methodNum = DahlquistBE()
methodEx = DahlquistExact()

make_state = lambda t, u, cycle: PropagationState(t=t, u=u, cycle=cycle, u_event=u)

# solve the problem
def solve_dahlquist(dt, num_events, method, save_history=True):
    """
    solve_dahlquist solves the Dahlquist problem using the specified time stepper method.
    Args:
        dt (float): time step size
        num_events (int): number of events to simulate
        method (TimeStepper): time stepping method to use
        save_history (bool, optional): whether to save the history of the time stepping. Defaults to True.

    Returns:
        tuple: (t_all, u_all, t_ev, u_ev) where t_all and u_all are the time and solution arrays for the entire simulation, 
        and t_ev and u_ev are the time and solution arrays at the event points.
    """
    
    t_ev = np.zeros(num_events+1)
    t_ev[0] = problem.t_start
    u_ev = np.zeros(num_events+1, dtype=complex)
    u_ev[0] = problem.u_start
    states = []
    t_all = [problem.t_start]
    u_all = [problem.u_start]

    timestepper = TimeStepper(problem=problem, method=method, dt=dt, save_history=save_history)
    propagator = TimeStepperPropagator(timestepper=timestepper, direction=1, terminate_cycle = terminate)
    
    for ev in range(num_events):
        
        current_state = make_state(t_ev[ev], u_ev[ev], ev)
        states.append(current_state)
        t_new, u_new = propagator.propagate(state=current_state)

        t_ev[ev+1] = t_new
        u_ev[ev+1] = u_new
        
        # extract solution for the current cycle
        t_local = np.asarray(propagator.history['t'])
        u_local = np.asarray(propagator.history['u'])
        
        # save the solution for the current cycle
        t_all.extend(t_local[1:])
        u_all.extend(u_local[1:])
        
    return np.asarray(t_all), np.asarray(u_all), np.asarray(t_ev), np.asarray(u_ev)

# %%
num_events = 5 # number of cycles
try:    
    tNum, uNum, t_evNum, u_evNum = solve_dahlquist(dt=dtNum, num_events=num_events, method=methodNum, save_history=True)
    tEx, uEx, t_evEx, u_evEx = solve_dahlquist(dt=dtEx, num_events=num_events, method=methodEx, save_history=True)
    # Plotting the results
    plt.plot(uNum.real, uNum.imag, label="Numerical")
    plt.plot(uEx.real, uEx.imag, "--", label="Analytical")
    plt.plot(u_evNum.real, u_evNum.imag, 'o', label=r"$u_{\mathrm{ev}}$ (Numerical)")
    plt.legend(loc="lower left"), plt.xlabel(r"$\Re(u)$"), plt.ylabel(r"$\Im(u)$"), plt.grid();
except Exception as e:
    print(f"An error occurred during the simulation: {e}")

# %%

# error analysis
dtVals =  1./(10**np.arange(1,6))
error_t_ev = np.zeros_like(dtVals)
error_u_ev = np.zeros_like(dtVals)

for i,dt in enumerate(dtVals):
    try:
        tNum, uNum, t_evNum, u_evNum = solve_dahlquist(dt=dt, num_events=num_events, method=methodNum, save_history=True)
        tEx, uEx, t_evEx, u_evEx = solve_dahlquist(dt=dt, num_events=num_events, method=methodEx, save_history=True)
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        continue
    error_t_ev[i] = np.max(np.abs(t_evNum - t_evEx))
    error_u_ev[i] = np.max(np.abs(u_evNum - u_evEx))
    print(f"dt = {dt}, error_t_ev = {error_t_ev[i]}, error_u_ev = {error_u_ev[i]}")

# %%
# Plotting the error
plt.figure()

plt.loglog(
    dtVals,
    error_t_ev,
    "--o",
    label=r"$\|t_{\mathrm{ev}}^{\mathrm{ex}}"
          r"-t_{\mathrm{ev}}^{\mathrm{num}}\|_\infty$",
)

plt.loglog(
    dtVals,
    error_u_ev,
    "-*",
    label=r"$\|u_{\mathrm{ev}}^{\mathrm{ex}}"
          r"-u_{\mathrm{ev}}^{\mathrm{num}}\|_\infty$",
)

plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$L_\infty$ error")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%

# Adaptive Parareal

N = num_events # time windows - coarse time grid
K = N+1 # Parareal iterations
dtF = 1/10000 # Fine solver's time steps
dtG = 1/100  # Coarse solver's time steps

timestepperF = TimeStepper(problem=problem, method=methodNum, dt=dtF, save_history=False)
timestepperG = TimeStepper(problem=problem, method=methodNum, dt=dtG, save_history=False)
propagatorF = TimeStepperPropagator(timestepper=timestepperF, direction=1, terminate_cycle = terminate)
propagatorG = TimeStepperPropagator(timestepper=timestepperG, direction=1, terminate_cycle = terminate)


parareal = PararealModified(fine=propagatorF, coarse=propagatorG, make_state=make_state)
time_start = time.time()
try:
    TT, U = parareal.solve(t0=t_start, u0=u_start, K=K, N=N)
except Exception as e:
    print(f"Parareal execution failed: {e}")
time_end = time.time()
total_time_parareal = time_end - time_start
print(f"Parareal execution ended in {total_time_parareal:.2f} seconds")

# %%
# solve with fine solver for comparison
time_start = time.time()
try:
    tFine, uFine, t_evFine, u_evFine = solve_dahlquist(dt=dtF, num_events=num_events, method=methodNum, save_history=True)
except Exception as e:
    print(f"An error occurred during the fine solver simulation: {e}")
time_end = time.time()
total_time_fine = time_end - time_start
print(f"Fine solver computation completed in {total_time_fine:.2f} seconds")

# %%
# error analysis for Parareal
dtF_index = np.where(dtVals == dtF)[0][0]

# error at the event points for the fine solver
err_t_ev_fine = error_t_ev[dtF_index]
err_u_ev_fine = error_u_ev[dtF_index]

# error at the event points for the Parareal solver
err_t_ev_per_cycle = np.zeros((K+1, N+1), dtype=float)
err_u_ev_per_cycle = np.zeros((K+1, N+1), dtype=complex)
for k in range(K+1):
    err_t_ev_per_cycle[k, :] = np.abs(TT[k, :] - t_evFine)
    err_u_ev_per_cycle[k, :] = np.abs(np.asarray(list(U[k, :]),dtype=complex) - u_evFine)
err_t_ev_parareal = np.max(err_t_ev_per_cycle, axis=1)
err_u_ev_parareal = np.max(err_u_ev_per_cycle, axis=1)

# %%
fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Adaptive Parareal Error for N={N+1}")
plt.subplot(1,2, 1)
plt.semilogy(range(K+1), err_t_ev_parareal, label=r'$\|tp_{Parareal} - tp_{FineNum}\|_\infty$')
plt.axhline(y=err_t_ev_fine, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(K+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend()

plt.subplot(1,2, 2)
plt.semilogy(range(K+1), err_u_ev_parareal, label=r'$\|up_{Parareal} - up_{FineNum}\|_\infty$')
plt.axhline(y=err_u_ev_fine, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(K+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Parareal iteration k"), plt.ylabel("Error"), plt.grid();
plt.legend()  
plt.tight_layout()  

# plotting the error vs N points
fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Adaptive Parareal Error for N={N+1}")
plt.subplot(1,2, 1)
for k in range(K+1):
    plt.semilogy(range(N+1), err_t_ev_per_cycle[k,:], label=f"k={k}")
plt.axhline(y=err_t_ev_fine, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(N+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Time window n"), plt.ylabel("Error"), plt.grid();
plt.legend()

plt.subplot(1,2, 2)
for k in range(K+1):
    plt.semilogy(range(N+1), err_u_ev_per_cycle[k,:], label=f"k={k}")
plt.axhline(y=err_u_ev_fine, color='red', linestyle='--', label="Fine solver error")
plt.xticks(range(N+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Time window n"), plt.ylabel("Error"), plt.grid();
plt.tight_layout()