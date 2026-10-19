#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:25:18 2026

@author: yessima
"""

# -*- coding: utf-8 -*-
import pybamm
import numpy as np
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator
from batpint.parareal.parareal import PararealModified

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy


# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 1
# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

rtolCoarse = 1e-5
rtolFine = 1e-6

fine_propagator = PybammPropagator(
    model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
    experiment = batteryExperiment(nCycles, expType=exp),
    parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
    var_pts = var_pts,
    solver=pybamm.IDAKLUSolver(rtol=rtolFine))

tStart = 0
uStart = copy.deepcopy(fine_propagator.u_full)

len_rhs = fine_propagator._last_step_model.len_rhs

coarse_propagator = PybammPropagator(
        model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
        experiment = batteryExperiment(nCycles, expType=exp),
        parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
        var_pts = var_pts,
        solver=pybamm.IDAKLUSolver(rtol=rtolCoarse)
        )



uF = np.empty(num_cycles+1, dtype=object)
uC = np.empty(num_cycles+1,dtype=object)

uF[0] = uStart
uC[0] = uStart


solF, solC = [], []
tF = np.empty(num_cycles+1, dtype=object)
tC = np.empty(num_cycles+1,dtype=object)

tF[0] = tStart
tC[0] = tStart

for cycle in range(num_cycles):
    tFine, uFine, solFine = fine_propagator.solve(t=tF[cycle], u=uF[cycle], cycle=cycle)
    tCoarse, uCoarse, solCoarse = coarse_propagator.solve(t=tC[cycle], u=uC[cycle], cycle=cycle)
    
    tF[cycle+1] = tFine
    tC[cycle+1] = tCoarse
    
    uF[cycle+1] = uFine
    uC[cycle+1] = uCoarse
    
    solF.append(solFine)
    solC.append(solCoarse)
#%%
