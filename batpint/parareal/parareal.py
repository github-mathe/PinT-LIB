import copy

import numpy as np


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

    def __init__(self, fine, coarse, coarse_s, first_guess=None):
        self.fine = fine
        self.coarse = coarse
        self.coarse_s = coarse_s
        self.first_guess = "coarse" if first_guess is None else first_guess

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
        TT = np.zeros((K + 1, N + 1), dtype=float)
        TT[:, 0] = t0

        U = np.empty((K + 1, N + 1), dtype=object)

        for k in range(K + 1):
            U[k, 0] = copy.deepcopy(u0)

        print("Initialization completed.")

        # First coarse propagation
        print(f"Starting first guess with {self.first_guess} propagation...")

        for n in range(N):
            if self.first_guess == "coarse":
                TPG, UPG = self.coarse.propagate(TT[0, n], U[0, n], cycle=n)
            if self.first_guess == "fine":
                TPG, UPG = self.fine.propagate(TT[0, n], U[0, n], cycle=n)
            TT[0, n + 1] = TPG
            U[0, n + 1] = copy.deepcopy(UPG)

        print("First coarse propagation completed.")
        print("Starting Parareal iterations...")

        # Parareal iterations
        for k in range(K):
            for n in range(N):
                TPF, UPF = self.fine.propagate(TT[k, n], U[k, n], cycle=n)
                TPG, UPG = self.coarse.propagate(TT[k, n], U[k, n], cycle=n)
                TPG_s, UPG_s = self.coarse_s.propagate(TT[k + 1, n], U[k + 1, n], cycle=n)

                TT[k + 1, n + 1] = TPF + (TPG_s - TPG)
                U[k + 1, n + 1] = np.sum([UPF, UPG_s, -UPG], axis=0, dtype=np.double)

        print("Parareal iterations completed.")
        return TT, U
