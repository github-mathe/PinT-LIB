import numpy as np

def PararealModified(F, G, t0, u0, N, K):
    
    TT          =   np.zeros((K+1, N+1), dtype = float)
    TT[:,0]     =   t0
    
    U           =   np.zeros((K+1, N+1), dtype = complex)  
    U[:, 0]     =   u0
    
    for n in range(N):
        TPG, UPG    =   G(TT[0, n], U[0,n], n+1)
        TT[0,n+1]           =   TPG
        U[0, n+1]           =   UPG

    for k in range(K):  # Parareal iterations
        for n in range(N):
            TPF, UPF     =    F(TT[k, n], U[k, n], n+1) 
            TPG, UPG     =    G(TT[k, n], U[k, n], n+1)
            TPG_s, UPG_s =    G(TT[k+1, n], U[k+1, n], n+1)
            TT[k+1, n+1] =    TPF - TPG + TPG_s 
            U[k+1, n+1]  =    UPF - UPG + UPG_s   
    return TT, U