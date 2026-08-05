# -*- coding: utf-8 -*-
import numpy as np
def pararealADahlquist(F, G, u0, t0, N, K):
    
    TT          =   np.zeros((K+1, N+1), dtype = float)
    TT[:,0]     =   t0
    
    U           =   np.zeros((K+1, N+1), dtype = complex)  
    U[:, 0]     =   u0
    
    for n in range(N):
        _,_,TPG, UPG,_    =   G(U[0, n], TT[0,n], n+1)
        TT[0,n+1]           =   TPG
        U[0, n+1]           =   UPG

    for k in range(K):  # Parareal iterations
        for n in range(N):
            _,_,TPF, UPF,_    =    F(U[k, n], TT[k,n], n+1) 
            _,_,TPG, UPG,_    =    G(U[k, n], TT[k,n], n+1)
            _,_,TPG_s, UPG_s,_ =    G(U[k+1, n], TT[k+1,n], n+1)
            TT[k+1, n+1]            =    TPF - TPG + TPG_s 
            U[k+1, n+1]             =    UPF - UPG + UPG_s   
    return TT, U