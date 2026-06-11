# -*- coding: utf-8 -*-
import numpy as np


def pararealADahlquist(F, G, u0, t0, N, K, dtStart, dtF, dtG):
    
    TT          =   np.zeros((K+1, N+1), dtype = float)
    TT[:,0]     =   t0
    
    T_coarse_fine       =   np.zeros((3,K+1,N+1), dtype = float)
    U_coarse_fine       =   np.zeros_like(T_coarse_fine, dtype=complex)
    
    U           =   np.zeros((K+1, N+1), dtype = complex)  
    U[:, 0]     =   u0
    
    steps_para  = np.zeros((K+1, N), dtype=int) 
    
    for n in range(N):
        TPG, UPG, steps     =   G(U[0, n], TT[0,n], dtStart, n+1)
        TT[0,n+1]           =   TPG[0]
        U[0, n+1]           =   UPG[0]
        T_coarse_fine[:,0,n+1]  =    TPG[0]

        steps_para[0, n]    =   steps 

    for k in range(K):  # Parareal iterations
        for n in range(N):
            TPF, UPF, steps         =    F(U[k, n], TT[k,n], dtF, n+1) 
            TPG, UPG, _             =    G(U[k, n], TT[k,n], dtG, n+1)
            TPG_s, UPG_s,_          =    G(U[k+1, n], TT[k+1,n], dtG, n+1)
            TT[k+1, n+1]            =    TPF[0] - TPG[0] + TPG_s[0] 
            U[k+1, n+1]             =    UPF[0] - UPG[0] + UPG_s[0]
            T_coarse_fine[:,k+1,n+1]  =    TPF[0], TPG[0], TPG_s[0]
            U_coarse_fine[:,k+1,n+1]  =    UPF[0], UPG[0], UPG_s[0]
            steps_para[k+1, n]      =    steps
    return TT, U,T_coarse_fine,U_coarse_fine, steps_para