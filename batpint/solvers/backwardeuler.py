import numpy as np
try:
    from .newton import Newton
except Exception:
    from batpint.solvers.newton import Newton

def BackwardEuler(
        f,
        df,
        t0: float,
        y0: float | complex | np.ndarray,
        dt: float,
        T:  float):
    """
    Backward Euler method for solving ODEs of the form dy/dt = f(t,y)
    
    Parameters:
    f : function
        The function f(t,y) defining the ODE.
    t0 : float
        The initial time.
    y0 : float
        The initial value of y at time t0.
    dt : float
        The time step size.
    T : float
        The final time up to which the solution is computed.
    Returns:
    t : numpy array
        Array of time points where the solution is computed.
    y : numpy array
        Array of solution values corresponding to each time point in t.
    """
    y0 = np.atleast_1d(y0)
    if T<t0: raise ValueError("T must be bigger than t0")
    if dt <= 0: raise ValueError("dt must be positive")
    
    num_steps = int(round((T - t0) / dt))

    if num_steps <= 0: raise ValueError("(T - t0) must be an integer multiple of dt")
    
    Newton_iterations = []
    t   =    np.linspace(t0, T, num_steps+1, endpoint=True)
    y   =    np.zeros((num_steps + 1, len(y0)), dtype = y0.dtype)
    y[0,:]  =    y0
    for i in range(num_steps):
        F   =   lambda w: w - y[i] - dt * f(t[i+1], w)
        dF  =   lambda w: np.eye(len(y0)) - dt * df(t[i+1], w)
        y[i+1], n, _   =    Newton(f=F, df=dF, x=y[i].copy(), max_iter = 20)
        Newton_iterations.append(n)
        if n>=100: print(f"Warning: Newton's method did not converge at time {t[i] + dt}.")

    y = np.squeeze(y[1:])
    t = np.squeeze(t[1:])
    return t, y, Newton_iterations

    
    