import pybamm
import numpy as np
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
import matplotlib.pyplot as plt


# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 10

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()

# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

# parameters
parameter_values=pybamm.ParameterValues(
    bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)
# solver
solver = pybamm.IDAKLUSolver()

# models
model_step = pybamm.lithium_ion.DFN(*args)
model_all = pybamm.lithium_ion.DFN(*args)

# experiments
experiment_step = batteryExperiment(nCycles, expType=exp)
experiment_all = batteryExperiment(num_cycles,expType=exp)

# simulations
sim_all = pybamm.Simulation(model_all,
                            experiment=experiment_all,
                            parameter_values=parameter_values,
                            var_pts=var_pts,
                            solver=solver)
sim_step = pybamm.Simulation(model_step,
                            experiment=experiment_step,
                            parameter_values=parameter_values,
                            var_pts=var_pts,
                            solver=pybamm.IDAKLUSolver())

#solve for all cycles
solution_all = sim_all.solve()

# solve stepwise
# no pybamm time correction
init_sol = None
solution_step = []
t_step = []
t_delta = 0.0
for cycle in range(num_cycles):
    
    sol = sim_step.solve(starting_solution=init_sol)
    
    cycle_sol = sol.cycles[-1]
    t_global = cycle_sol.t + t_delta
    t_step.append(t_global)
    t_delta = t_global[-1]
    solution_step.append(cycle_sol)
    
    init_sol = cycle_sol.last_state.copy()
    init_sol.t[-1] = 0
    init_sol.t_eval[-1] = 0
    init_sol.all_ts[-1][-1] = 0
    init_sol.all_t_evals[-1][-1] = 0

    
print(f"Solved {num_cycles} cycles.")
#%%
# check the size of the states
# processed models
built_model_all = solution_all.cycles[0].first_state.all_models[0]
built_model_step = solution_step[0].first_state.all_models[0]
len_t_all = np.array([sol.t.shape[0] for sol in solution_all.cycles])
len_t_step = np.array([t.shape[0] for t in t_step])
check_time_lengths = np.where(len_t_all != len_t_step)[0]
print("Same state vector size: ", built_model_all.len_rhs_and_alg== built_model_step.len_rhs_and_alg)
print("Same differential state vector size: ", built_model_all.len_rhs== built_model_step.len_rhs)
print("Same algebraic state vector size: ", built_model_all.len_alg== built_model_step.len_alg)
print("Differ in time lengths: ", check_time_lengths.size != 0, check_time_lengths)
# data at the time interface = time at the switch between cycles

#%%
# interface times
t_start_all = [sol.t[0] for sol in solution_all.cycles]
t_last_all = [sol.t[-1] for sol in solution_all.cycles]
t_switch_all = [np.abs(t1-t2) for t1,t2 in zip(t_start_all[1:],t_last_all[:-1])]

t_last_step = [t[-1] for t in t_step]
t_start_step = [t[0] for t in t_step]
t_switch_step = [t1-t2 for t1,t2 in zip(t_start_step[1:], t_last_step[:-1])]

# difference in time at the time interface
err_t_last = [np.abs(t1-t2) for t1,t2 in zip(t_last_all,t_last_step)]
err_t_start = [np.abs(t1-t2) for t1,t2 in zip(t_start_all,t_start_step)]

#interface states

y_all = [sol.y for sol in solution_all.cycles]
y_last_all = [sol.y[:,-1] for sol in solution_all.cycles]
y_start_all = [sol.y[:,0] for sol in solution_all.cycles]
y_switch_all = [y1-y2 for y1,y2 in zip(y_last_all[:-1], y_start_all[1:])]
err_y_switch_all = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_switch_all]

y_diff_last_all = [sol.y[:built_model_all.len_rhs,-1] for sol in solution_all.cycles]
y_diff_start_all = [sol.y[:built_model_all.len_rhs,0] for sol in solution_all.cycles]
y_diff_switch_all = [y1-y2 for y1,y2 in zip(y_diff_last_all[:-1],y_diff_start_all[1:])]
err_y_diff_switch_all = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_diff_switch_all]

y_alg_last_all = [sol.y[built_model_all.len_rhs:,-1] for sol in solution_all.cycles]
y_alg_start_all = [sol.y[built_model_all.len_rhs:,0] for sol in solution_all.cycles]
y_alg_switch_all = [y1-y2 for y1,y2 in zip(y_alg_last_all[:-1],y_alg_start_all[1:])]
err_y_alg_switch_all = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_alg_switch_all]


y_step = [sol.y for sol in solution_step]
y_last_step = [sol.y[:,-1] for sol in solution_step]
y_start_step = [sol.y[:,0] for sol in solution_step]
y_switch_step = [y1-y2 for y1,y2 in zip(y_last_step[:-1],y_start_step[1:])]
err_y_switch_step = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_switch_step]


y_diff_last_step = [sol.y[:built_model_step.len_rhs,-1] for sol in solution_step]
y_diff_start_step = [sol.y[:built_model_step.len_rhs,0] for sol in solution_step]
y_diff_switch_step = [y1-y2 for y1,y2 in zip(y_diff_last_step[:-1],y_diff_start_step[1:])]
err_y_diff_switch_step = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_diff_switch_step]


y_alg_last_step =  [sol.y[built_model_step.len_rhs:,-1] for sol in solution_step]
y_alg_start_step = [sol.y[built_model_step.len_rhs:,0] for sol in solution_step]
y_alg_switch_step = [y1-y2 for y1,y2 in zip(y_alg_last_step[:-1],y_alg_start_step[1:])]
err_y_alg_switch_step = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_alg_switch_step]

#%%
PRINT = False
if PRINT:
    print("\n" + "=" * 50)
    print("CYCLE COMPARISON between sequential and step solutions")
    print("=" * 50)
    
    
    for cycle in range(num_cycles):
        print("\n" + "=" * 50)
        print(f"Cycle {cycle}")
        print("starting time difference:", err_t_start[cycle])
        print("ending time difference:", err_t_last[cycle])
        
        # print("starting state difference:", err_y[cycle][0])
        # print("starting differential state difference:", err_y_diff[cycle][0])
        # print("starting algebraic state difference:", err_y_alg[cycle][0])
        
        # print("ending state difference:", err_y[cycle][-1])
        # print("ending differential state difference:", err_y_diff[cycle][-1])
        # print("ending algebraic state difference:", err_y_alg[cycle][-1])
        
        #start_error = np.linalg.norm(y_start_step[cycle] - y_start_all[cycle], ord=np.inf)
        #end_error = np.linalg.norm(y_last_step[cycle] - y_last_all[cycle], ord=np.inf)
        #print(start_error == err_y[cycle][0], end_error == err_y[cycle][-1])
        print("=" * 50)  
    
#%%    
# Interface time difference for solver 'all'"since it has time correction on the interface
# step solvers time interfaces are same
plt.figure()
plt.semilogy(t_switch_all, "-o", color="red")
plt.semilogy(t_switch_step, "-o")
plt.xlabel(r"$t_{\Gamma}$")
plt.ylabel(r"$t_+ - t_-$")
plt.title("Interface time difference for solver 'all'")

# difference of the differential and algebraic states at the time inteface
fig, (ax1, ax2) = plt.subplots(1,2, layout="constrained")
fig.subplots_adjust(hspace=1)
ax1.semilogy(err_y_diff_switch_all, "-o", color = "red", label=r"$\|y_{+,d} - y_{-,d}\|_{\infty,all}$")
ax1.semilogy(err_y_diff_switch_step, "-o", label=r"$\|y_{+,d} - y_{-,d}\|_{\infty,step}$")
ax1.set_xlabel(r"$t_{\Gamma}$ [s]")
ax1.set_ylabel(r"$\|y_{+,d} - y_{-,d}\|_{\infty}$")
ax1.set_title("Differential state difference at the time interface")

ax2.plot(err_y_alg_switch_all, "-o", color="red", label = r"all")
ax2.plot(err_y_alg_switch_step, "-o",label = r"step")
ax2.set_xlabel(r"$t_{\Gamma}$ [s]")
ax2.set_ylabel(r"$\|y_{+,a} - y_{-,a}\|_{\infty}$")
ax2.legend()
ax2.set_title("Algebraic state difference at the time interface")
fig.tight_layout()
plt.show()
#%%
# difference in states between solvers at the time interface
# difference between interface times for solvers all and step
plt.figure()
plt.semilogy(err_t_last, "o", label = r"$t_{-, all} - t_{-, step}$")
plt.semilogy(err_t_start, "x", label = r"$t_{+, all} - t_{+, step}$")
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel(r"$t_{\Gamma}$")
plt.ylabel(r"$\Delta t_{\Gamma}$")
plt.title("Interface time difference per cycle")
plt.xticks(range(0,num_cycles,round(num_cycles/5)))
plt.legend()


#%%
if check_time_lengths.size == 0:
    # L_Inf error of the state vectors at all times 
    err_y = [np.linalg.norm(y1-y2, ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
    err_y_diff = [np.linalg.norm(y1[:built_model_step.len_rhs,:]-y2[:built_model_step.len_rhs,:], ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
    err_y_alg = [np.linalg.norm(y1[built_model_step.len_rhs:,:]-y2[built_model_step.len_rhs:,:], ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
    
    fig, (ax3, ax4) = plt.subplots(1,2, layout="constrained")
    fig.subplots_adjust(hspace=1)
    ax3.semilogy([np.max(err_per_cycle) for err_per_cycle in err_y_diff], "--",  label=r"$\|y_{+,d, all} - y_{-,d, step}\|_{\infty}$")
    ax3.set_xlabel(r"$t_{\Gamma}$ [s]")
    ax3.set_ylabel(r"$\|y_{+,d} - y_{-,d}\|_{\infty}$")
    ax3.set_title("Differential state")
    ax3.set_xticks(range(0,num_cycles,round(num_cycles/5)))
    
    
    ax4.plot([np.max(err_per_cycle) for err_per_cycle in err_y_alg], "-", label = r"$\|y_{+,a,all} - y_{-,a,all}\|_{\infty}$")
    ax4.set_xlabel(r"$t_{\Gamma}$ [s]")
    ax4.set_ylabel(r"$\|y_{+,a} - y_{-,a}\|_{\infty}$")
    ax4.legend()
    ax4.set_title("Algebraic state")
    ax4.set_xticks(range(0,num_cycles,round(num_cycles/5)))
    
    fig.tight_layout()
    fig.suptitle("Error between states per cycle")
    plt.show()

#%%
# ------------------------------------------------------------
# Mesh points
# ------------------------------------------------------------
# Spatial coordinates
mesh = sim_step.mesh

r_p = mesh["positive particle"].nodes
x_p = mesh["positive electrode"].nodes
x_s = mesh["separator"].nodes
x = mesh["separator", "positive electrode"].nodes

results = {"all": solution_all.cycles[:], "step": solution_step[:]}

state_variable = lambda cycle, sol_type, state: results[sol_type][cycle][state]
c_s_p = lambda cycle, sol_type: state_variable(cycle,sol_type, "Positive particle concentration [mol.m-3]")
eps_c_e = lambda cycle, sol_type: state_variable(cycle,sol_type, "Porosity times concentration [mol.m-3]")
ce = lambda cycle, sol_type: state_variable(cycle,sol_type, "Electrolyte concentration [mol.m-3]")

phi_s_p = lambda cycle, sol_type: state_variable(cycle,sol_type,"Positive electrode potential [V]")
phi_e = lambda cycle, sol_type: state_variable(cycle,sol_type,"Electrolyte potential [V]")
Q_abs = lambda cycle, sol_type: state_variable(cycle,sol_type,"Throughput capacity [A.h]")
Q_discharge = lambda cycle, sol_type: state_variable(cycle,sol_type,"Discharge capacity [A.h]")
current_profile = lambda cycle, sol_type: state_variable(cycle,sol_type,"Current [A]")
# interface states
# solid 

def plot_cs(ax, cycle, sol_type, t, x=None, label=None):
    x0=x_p[-1] if x is None else x
    c_s = c_s_p(cycle, sol_type)(t=t, x=x0, r=r_p)
    ax.plot(r_p*1e6, c_s, label=label)
    ax.set_xlabel(r"$\mu m$")
    ax.set_ylabel("c_s")
    return ax

def plot_ce(ax, cycle, sol_type, t, label=None):
    c_e = ce(cycle, sol_type)(t=t)
    ax.plot(x*1e6, c_e, label=label)
    ax.set_xlabel(r"$\mu m$")
    ax.set_ylabel("c_e")
    return ax
def plot_eps_ce(ax, cycle, sol_type, t, label=None):
    c_e = eps_c_e(cycle, sol_type)(t=t)
    ax.plot(x*1e6, c_e, label=label)
    ax.set_xlabel(r"$\mu m$")
    ax.set_ylabel("c_e")
    return ax

def plot_phi_s(ax,cycle,sol_type,t,label=None):
    phi_s=phi_s_p(cycle,sol_type)(t=t)
    ax.plot(x_p*1e6,phi_s,label=label)
    ax.set_xlabel(r"$x_p$ [$\mu$m]")
    ax.set_ylabel(r"$\phi_{s,p}$ [V]")
    return ax

def plot_phi_e(ax,cycle,sol_type,t,label=None):
    phi=phi_e(cycle,sol_type)(t=t)
    ax.plot(x*1e6,phi,label=label)
    ax.set_xlabel(r"$x$ [$\mu$m]")
    ax.set_ylabel(r"$\phi_e$ [V]")
    return ax

def plot_Q_discharge(ax,cycle,sol_type,t,label=None):
    Q=Q_discharge(cycle,sol_type)(t=t)
    ax.plot(Q,"o",label=label)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$Q_{\mathrm{discharge}}$ [A.h]")
    return ax

def plot_Q_abs(ax,cycle,sol_type,t,label=None):
    Q=Q_abs(cycle,sol_type)(t=t)
    ax.plot(Q,"o",label=label)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$Q_{\mathrm{throughput}}$ [A.h]")
    return ax
def plot_current_profile(ax,cycle,sol_type,t,label=None):
    current = current_profile(cycle,sol_type)(t=t)
    ax.plot(current,"o",label=label)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(f"Applied current for cycle {cycle} [A]")
    return ax
#%%
x0 = x_p[0]
cycle = num_cycles-2
t_minus = t_last_all[cycle]
t_plus = t_start_all[cycle+1]

t_step_local = lambda cycle, idx: results["step"][cycle].t[idx]

fig, ax = plt.subplots()
plot_cs(ax, cycle, "all", t = t_minus, label=r"all: $t_\Gamma^-$", x=x0)
plot_cs(ax, cycle+1, "all", t = t_plus, label=r"all: $t_\Gamma^+$",x=x0)
plot_cs(ax, cycle, "step", t =  t_step_local(cycle, -1), label=r"step: $t_\Gamma^-$",x=x0)
plot_cs(ax, cycle, "step", t =  t_step_local(cycle+1, 0), label=r"step: $t_\Gamma^+$",x=x0)
ax.set_title(f"Cycle {cycle}: positive particle concentrtaion at x = {x0*1e6}")
ax.legend()
plt.show()

fig,ax=plt.subplots()
plot_phi_s(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_phi_s(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_phi_s(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_phi_s(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}: positive electrode potential")
ax.legend()
plt.show()

fig,ax=plt.subplots()
plot_phi_e(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_phi_e(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_phi_e(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_phi_e(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}: electrolyte potential")
ax.legend()
plt.show()

fig,ax=plt.subplots()
plot_eps_ce(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_eps_ce(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_eps_ce(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_eps_ce(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}:porosity times electrolyte concentration")
ax.legend()
plt.show()

fig,ax=plt.subplots()
plot_ce(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_ce(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_ce(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_ce(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}: electrolyte concentration")
ax.legend()
plt.show()

fig,ax=plt.subplots()
plot_Q_discharge(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_Q_discharge(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_Q_discharge(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_Q_discharge(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}: discharge capacity")
ax.set_xticks(range(2))

ax.legend()
plt.show()

fig,ax=plt.subplots()
plot_Q_abs(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_Q_abs(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_Q_abs(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_Q_abs(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}: throughput capacity")
ax.set_xticks(range(2))
ax.legend()

fig,ax=plt.subplots()
plot_current_profile(ax,cycle,"all",t_minus,label=r"all: $t_\Gamma^-$")
plot_current_profile(ax,cycle+1,"all",t_plus,label=r"all: $t_\Gamma^+$")
plot_current_profile(ax,cycle,"step",t_step_local(cycle,-1),label=r"step: $t_\Gamma^-$")
plot_current_profile(ax,cycle+1,"step",t_step_local(cycle+1,0),label=r"step: $t_\Gamma^+$")
ax.set_title(f"Cycle {cycle}: applied current")
ax.set_xticks(range(2))
ax.legend()
plt.show()