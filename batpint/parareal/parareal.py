import numpy as np
import copy

class PararealModified:
    """
    Generic Parareal solver operating on Propagator objects.

    Parameters
    ----------
    fine : Propagator
        Fine propagator F.
    coarse : Propagator
        Coarse propagator G.
    make_state : callable
        Function
            make_state(t, u, n)
        returning the propagation state for cycle n.
    """

    def __init__(self, fine, coarse, make_state):
        self.fine = fine
        self.coarse = coarse
        self.make_state = make_state

    def solve(self, t0, u0, K, N):
        """
        Perform Parareal iterations.

        Parameters
        ----------
        K : int
            Number of Parareal iterations.
        N : int
            Number of time windows (coarse time grid).
        t0 : float
            Initial time.
        u0 : object
            Initial condition.
        Returns
        -------
        TT : np.ndarray
            Array of shape (K+1, N+1) containing the time values for each iteration and window.
        U : np.empty
            Array of shape (K+1, N+1) containing the solution values for each iteration and window.
        """
        print("Initializing Parareal solver...")
        TT          =   np.zeros((K+1, N+1), dtype = float)
        TT[:,0]     =   t0
        
        U           =   np.empty((K+1, N+1), dtype = object)  

        def parareal_state(k, n):
            return self.make_state(TT[k, n], U[k, n], n)

        for k in range(K+1):
            U[k, 0]     =   copy.deepcopy(u0)

        print("Initialization completed.")
        
        # First coarse propagation
        print("Starting first coarse propagation...")
        for n in range(N):
            TPG, UPG    =   self.coarse.propagate(parareal_state(0, n))
            TT[0,n+1]           =   TPG
            U[0, n+1]           =   copy.deepcopy(UPG)

        print("First coarse propagation completed.")
        print("Starting Parareal iterations...")
        # Parareal iterations
        for k in range(K):  
            for n in range(N):
        
                TPF, UPF     =    self.fine.propagate(parareal_state(k, n)) 
                TPG, UPG     =    self.coarse.propagate(parareal_state(k, n))
                
                TPG_s, UPG_s =    self.coarse.propagate(parareal_state(k+1, n))
                TT[k+1, n+1] =    TPF - TPG + TPG_s 
                U[k+1, n+1]  =    UPF - UPG + UPG_s   
        print("Parareal iterations completed.")
        return TT, U
    
def PararealOld(F, G, t0, u0, N, K):
    
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