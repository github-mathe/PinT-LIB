import numpy as np
from collections.abc import Callable

def BackwardEulerEvent(
        f,
        df,
        t0: float,
        y0: float | complex | np.ndarray,
        dt: float,
        event_func: Callable, 
        num_steps = 1000,
):
    """
        Backward Euler method for solving ODEs of the form dy/dt = f(t,y) with event detection.
        
        Parameters:
            f: Callable
                The function f(t, y) defining the ODE.
            df: Callable
                The derivative of f with respect to y, used for Newton's method.
            t0: float
                Initial time.
            y0: float | complex | np.ndarray
                Initial value of the solution.
            dt: float
                Time step size.
            event_func: Callable
                A function that defines the event condition. It should return a value that changes sign when the event occurs.
            num_steps: int
                Number of steps to take in each Backward Euler call before checking for the event.
        Returns:
            t_values: list
                List of time values at each step until the event is detected.
            y_values: list
                List of solution values at each step until the event is detected.
        """
    t_values = [np.array([t0])]
    y_values = [np.asarray([y0])]

    t = t0
    y = y0
    
    while True:
        # Perform a single Backward Euler step
        T = t + num_steps * dt
        t_next,y_next,_ = BackwardEuler(f, df, t, y, dt, T)
        is_event, t_event, y_event = event_func(t_next, y_next)

        # Check for event detection
        if is_event:
            y_values.append(y_next[t_next<t_event])  # Store the last value before the event
            t_values.append(t_next[t_next<t_event])  # Store the
            break

        # Store the new values
        t_values.append(t_next[1:])
        y_values.append(y_next[1:])

        # Update time and solution
        y = y_next[-1]
        t = t_next[-1]    

    t_values = np.concatenate(t_values)
    y_values = np.concatenate(y_values)
    return t_values, y_values, t_event, y_event   


if __name__ == "__main__":
    from modified_dahlquist import *
    from BackwardEuler import *
    # Dahlquist test problem
    u0 = 1.0 + 0.0j
    t0 = 0.0
    alpha = 6.001
    lam_f = lambda n: 1j*(1+0.01*n)  # Example lambda function
    lam = lam_f(1)  # Example lambda value for the test
    Period_start = 1
    dt = 0.001
    f = lambda t, u: lam * u + np.sin(alpha * t)
    df = lambda t, u: lam  # Derivative of f with respect to u
    event_func = lambda t, u: check_event(t0, u0, t, u, alpha, lam)  # Example event function
    t_values, y_values, t_event, y_event = BackwardEulerEvent(f, df, t0, u0, dt, event_func, num_steps=1000)

    import matplotlib.pyplot as plt
    plt.plot(t_values, y_values.real, label='Real part of solution')
    plt.plot(t_values, y_values.imag, label='Imaginary part of solution')
    plt.axvline(x=t_event, color='r', linestyle='--', label=' Event detected')
    plt.title('Backward Euler Method with Event Detection')
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.legend()
    plt.show()