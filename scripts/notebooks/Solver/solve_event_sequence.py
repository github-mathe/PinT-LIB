import numpy as np
from collections.abc import Callable
import time
def solve_event_sequence(t0,
                        y0,
                        dt,
                        reset_parameters: Callable,
                        Event_Solver: Callable,
                        event_func: Callable,
                        final_time = None,
                        num_events = 1,
                        num_steps = 1000,
                        ):
    """
        Solve an ODE using the specified Event_Solver with event detection.
        
        Parameters:
            Event_Solver: Callable
                The event solver function to use (e.g., BackwardEulerEvent).
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
                Number of steps to take in each Event_Solver call before checking for the event.
            final_time: float
                The final time up to which to integrate.
            num_events: int
                The number of events to detect.
        Returns:
            t_values: list
                List of time values at each step until the event is detected.
            y_values: list
                List of solution values at each step until the event is detected.
            t_event: list
                List of times at which events were detected.
            y_event: list
                List of solution values at which events were detected.
        """
    t_values = []
    y_values = []
    t_event = [t0] 
    y_event = [y0.copy() if isinstance(y0, np.ndarray) else y0] 

    if final_time is not None and t0 >= final_time:
        raise ValueError("Initial time t0 must be less than final_time.")
    
    current_event_count = 0    
    while num_events > current_event_count or (final_time is not None and t_event[-1] < final_time):
        t_start = t_event[-1]
        y_start = y_event[-1]
        f,df = reset_parameters( t_start, y_start, current_event_count)
        local_event_func = lambda t_sample, y_sample: event_func(
                            t_start,
                            y_start,
                            t_sample,
                            y_sample,
                            event_count=current_event_count)
        
        # Solve
        t_next, y_next, t_evt, y_evt = Event_Solver(
                            t0=t_start,
                            y0=y_start,
                            f=f,
                            df=df,
                            dt=dt,
                            num_steps=num_steps,
                            event_func=local_event_func)
        if len(t_next) == 0 or len(y_next) == 0:
            print("Warning: No steps were taken in the Event_Solver. Check your parameters.")
            break
        if final_time is not None and t_next[-1] > final_time:
            # If the next time step exceeds final_time, truncate the results
            is_final = t_next <= final_time
            t_next = t_next[is_final]
            y_next = y_next[is_final]
            num_events = 0  # Stop after this iteration since we reached final_time
        # Append results to lists
        t_values.append(t_next)  
        y_values.append(y_next)
        t_event.append(t_evt)
        y_event.append(y_evt)
        current_event_count += 1
    return t_values,y_values, t_event, y_event, current_event_count


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    _DAHLQUIST_DIR = _THIS_DIR.parent
    _MODIFIED_DIR = _DAHLQUIST_DIR / "ModifiedProblem"
    if str(_MODIFIED_DIR) not in sys.path:
        sys.path.insert(0, str(_MODIFIED_DIR))
    check_event = __import__("modified_dahlquist").check_event
    from scripts.notebooks.Solver.BackwardEulerEvent import BackwardEulerEvent
    
    # Dahlquist test problem
    u0 = 1.0 + 0.0j
    t0 = 0.0
    alpha = 6.001
    Period_start = 1
    lam_f = lambda n: 1j*(1+0.01*(n+Period_start))  # Example lambda function

    dt = 1e-5

    event_func = lambda t0, u0, t, u, event_count: check_event(t0, u0, t, u, alpha, lam_f(event_count))  # Example event function
    def reset_parameters(t0, y0, event_count):
        f = lambda t, u: lam_f(event_count) * u + np.sin(alpha * t)
        df = lambda t, u: lam_f(event_count)
        return f, df
    t_values, y_values, t_event, y_event, current_event_count = solve_event_sequence(
    t0=t0,
    y0=u0,
    dt=dt,
    reset_parameters=reset_parameters,
    Event_Solver=BackwardEulerEvent,
    event_func=event_func,
    final_time=11.0,
    num_steps=1_000,)

    import matplotlib.pyplot as plt
    for i in range(current_event_count):
        plt.plot(t_values[i], y_values[i].real, label=f'Real part of solution (Event {i+1})')
        plt.plot(t_values[i], y_values[i].imag, label=f'Imaginary part of solution (Event {i+1})')
    plt.scatter(t_event, [y.imag for y in y_event], color='blue', label='Event Detection (Imaginary)')
    plt.show()