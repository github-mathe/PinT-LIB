#%%
# plot steps
experiment_steps = fine_propagator.sim.experiment.cycles[-1]

state_slices = fine_propagator._last_step_model.y_slices

cycle = N-1
fig, ax = plt.subplots(1,len(experiment_steps))
for i, (step_solF, step_solC) in enumerate(zip(solF[cycle].steps, solC[cycle].steps)):
    ax[i].set_title(f"{experiment_steps[i]}")
    ax[i].plot(step_solF.t+tF[cycle], step_solF["Voltage [V]"].entries,"o-", label=f"Fine solution")
    ax[i].plot(step_solC.t+tF[cycle], step_solC["Voltage [V]"].entries, "s-", label=f"Coarse solution")
    ax[i].set_xlabel("t [s]")
    ax[i].set_ylabel("Voltage [V]")
    ax[i].legend(loc="best")
    ax[i].grid(True)

    ax[i].grid(True)
fig.suptitle(f"Cycle {cycle}")


#%%
# plot all cycles
fig, ax = plt.subplots(1,N)

for i, (cycleF, cycleC) in enumerate(zip(solF, solC)):
    print(f"Cycle {i}\n")
    times_F = cycleF.t+tF[i]
    times_C = cycleC.t+tC[i]
    
    ax[i].plot(cycleF['Average positive particle stoichiometry'].entries, cycleF["Voltage [V]"].entries,"o", label=f"Fine solution at cycle {i}")
    ax[i].plot(cycleC['Average positive particle stoichiometry'].entries, cycleC["Voltage [V]"].entries,"s", label=f"Coarse solution at cycle {i}", color="red")
    print("Start voltage of fine: ", cycleF["Voltage [V]"].entries[0])
    print("Start time:", times_F[0])
    
    print("End voltage of fine: ", cycleF["Voltage [V]"].entries[-1])
    print("End time: ", times_F[-1])
    print("===="*3)

fig.suptitle(f"Voltage vs. time")
plt.show()



fig, ax = plt.subplots(2,N, sharey=True)
fig.subplots_adjust(wspace=0)

start_old = solF[0].yp[:,0]
end_old = solF[0].y[2:902,-1]
for i, (cycleF, cycleC) in enumerate(zip(solF, solC)):
    ax[0,i].plot(cycleF.t+tF[i], cycleF["Positive particle concentration [mol.m-3]"].entries[-1,-1,:], label=f"Fine solution at cycle {i}")
    ax[1,i].plot(cycleC.t+tF[i], cycleC["Positive particle concentration [mol.m-3]"].entries[-1,-1,:], label=f"Coarse solution at cycle {i}", color="red")
for ax in fig.get_axes():
    ax.label_outer()
fig.suptitle(f"Concentration vs. time")
plt.show()