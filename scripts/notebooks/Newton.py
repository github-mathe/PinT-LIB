import numpy as np
from collections.abc import Callable
def Newton(
    f: Callable,
    x: float | complex | np.ndarray,
    df: Callable,
    tol: float = 1e-7,
    max_iter: int = 100,
    store_history: bool = False
):
    """
    Newton's method for solving f(x) = 0.
    """  
    
    is_scalar = np.ndim(x) == 0
    if is_scalar:
        x = np.asarray(x).item()
    else:
        x = np.asarray(x).copy()
        
        
    f_value = f(x)
    n = 0
    if store_history:
        history = [(x, f_value)]
    while np.linalg.norm(f_value) > tol and n < max_iter:
        dfx = df(x)
        if np.linalg.norm(dfx) < 1e-14:
            raise ValueError("Derivative is too small, Newton's method may not converge")
        correction  =    f_value / dfx if is_scalar else np.linalg.solve(dfx, f_value)
        x = x -  correction
        n   +=   1
        f_value     =    f(x)
        
        if store_history: history.append((x, f_value))
    if store_history:
        return np.squeeze(x), history
    else:
        return np.squeeze(x), n, f_value

if __name__ == "__main__":
    # Example usage skalar function
    def f(x):
        return np.exp(-0.1*x**2)*np.sin(np.pi/2*x)
    
    def df(x):
        return -0.2*x*np.exp(-0.1*x**2)*np.sin(np.pi/2*x) + np.pi/2*np.exp(-0.1*x**2)*np.cos(np.pi/2*x)
    x0 = 3.1
    x, history = Newton(f, x0, df, store_history=True)
    print(f"Root found: {x}")
    for i in range(len(history)):
        print(f"Iteration {i}: x = {history[i][0]}, f(x) = {history[i][1]}") 
        
    # example usage vector function
    def f_vec(x):
        return np.array([x[0]**2 + x[1]**2 - 4, x[0] - x[1]])
    
    def df_vec(x):
        return np.array([[2*x[0], 2*x[1]], [1, -1]])
    
    x0_vec = np.array([2.0, 3.5])
    x_vec, history_vec = Newton(f_vec, x0_vec, df_vec, store_history=True)
    print(f"Root found: {x_vec}")
    for i in range(len(history_vec)):
        print(f"Iteration {i}: x = {history_vec[i][0]}, f(x) = {history_vec[i][1]}")