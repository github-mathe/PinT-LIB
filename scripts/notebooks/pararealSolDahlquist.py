# -*- coding: utf-8 -*-
import numpy as np
#from scripts.notebooks.numericalSolDahlquist import *
#from scripts.notebooks.lamDahlquist import lam



def pararealADahlquist(F, G, u0, t0, N, K):
    
    TT          =   np.zeros((K+1, N+1), dtype = float)
    TT[:,0]     =   t0
    
    #T_coarse_fine       =   np.zeros((3,K+1,N+1), dtype = float)
    #U_coarse_fine       =   np.zeros_like(T_coarse_fine, dtype=complex)
    
    U           =   np.zeros((K+1, N+1), dtype = complex)  
    U[:, 0]     =   u0
    
    steps_para  = np.zeros((K+1, N), dtype=int) 
    
    for n in range(N):
        ttG,uuG,TPG, UPG, steps     =   G(U[0, n], TT[0,n], n+1)
        TT[0,n+1]           =   TPG[0]
        U[0, n+1]           =   UPG[0]
 #       T_coarse_fine[:,0,n+1]  =    TPG[0]

        steps_para[0, n]    =   steps 

    for k in range(K):  # Parareal iterations
  #      u0 = U[k,0]
   #     t0 = TT[k,0]
        for n in range(N):
            ttF,uuF,TPF, UPF, steps         =    F(U[k, n], TT[k,n], n+1) 
            ttG,uuG,TPG, UPG, _             =    G(U[k, n], TT[k,n], n+1)
            ttGs,uuGs,TPG_s, UPG_s,_          =    G(U[k+1, n], TT[k+1,n], n+1)
            
            TT[k+1, n+1]            =    TPF[0] - TPG[0] + TPG_s[0] 
            U[k+1, n+1]             =    UPF[0] - UPG[0] + UPG_s[0]
            
    #        T_coarse_fine[:,k+1,n+1]  =    TPF[0], TPG[0], TPG_s[0]
     #       U_coarse_fine[:,k+1,n+1]  =    UPF[0], UPG[0], UPG_s[0]
            steps_para[k+1, n]      =    steps
      #      T, uNum, TP, UP, steps=timeStepperAll(u0, 6.001, lam, t0, 1/1000, n+1, n+1)
       #     u0=uNum[-1]
        #    t0=T[-1]
            
    return TT, U, steps_para #T_coarse_fine,U_coarse_fine, 