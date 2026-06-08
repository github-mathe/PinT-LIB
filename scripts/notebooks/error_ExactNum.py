#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:36:17 2026

@author: yessima
"""
import numpy as np

def compute_Linf(dtVals, Exact, Num):
    error_sol = np.zeros_like(dtVals)
    error_timeP = np.zeros_like(dtVals)
    error_solP = np.zeros_like(dtVals)

    for i, dt in enumerate(dtVals):
        tTh, uTh,tpTh,upTh = Exact(dt)
        tNum, uNum,tpNum,upNum, steps = Num(dt)
        min_len = min(len(uTh), len(uNum))
        uNum = uNum[:min_len]
        uTh = uTh[:min_len]
        error_sol[i] = np.linalg.norm(uNum-uTh, ord=np.inf)
        error_solP[i] = np.linalg.norm(upTh-upNum, ord=np.inf)
        error_timeP[i] = np.linalg.norm(tpNum-tpTh, ord=np.inf)  
    return error_sol, error_solP, error_timeP