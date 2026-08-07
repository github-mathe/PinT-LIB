import pybamm


def batteryProblem(nCycles=10, expType="GITT"):
    if expType == "GITT":
        # define GITT experiment: short pulse followed by long rest period
        pulse_duration = "10 minutes"  # duration of current pulse
        rest_duration = "2 hours"      # relaxation time
        current_rate = "0.05C"         # current during pulse
        voltage_min = 2.5              # V - discharge cutoff

        # WARNING : this parameter is not used !
        voltage_max = 4.2              # V - charge cutoff

        # Create GITT discharge experiment
        # Tuple groups discharge+rest as ONE cycle
        experiment = pybamm.Experiment(
            [
                (
                    f"Discharge at {current_rate} for {pulse_duration}",
                    f"Rest for {rest_duration}"
                )
            ] * nCycles,
            termination=f"{voltage_min} V"
        )
    elif expType == "CCCV":
        # define CCVV experiment: many cycles of discharge / charge
        experiment = pybamm.Experiment(
            [
                (
                #"Rest for 5 minutes",
                "Discharge at 2C until 3.5 V",
                "Rest for 5 minutes",
                "Charge at 0.1C until 4.2 V",
                "Rest for 10 minutes"
                )
            ]
            * nCycles
            )
    else:
        raise NotImplementedError(f"{expType}")

    return experiment