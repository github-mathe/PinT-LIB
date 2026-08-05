import numpy as np
import matplotlib.pyplot as plt
from modified_dahlquist import (exact_local_solution,
                                exact_global_solution,)
# Parameters
u0 = 1.0 + 0.0j
t0 = 0.0
alpha = 6.001
Period_start = 1
Period_end = 3
num_points = 1000
def epsilon(n: int) -> float:
    return 0.01 * n
def lambda_n(n: int) -> complex:
    return 1j * (1.0 + epsilon(n))

def plot_complex_solution(t, u, event_times=None, event_values=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax1, ax2 = axes

    ax1.plot(t, u.real, label="Re(u)")
    ax1.plot(t, u.imag, label="Im(u)")

    ax1.set_xlabel("t")
    ax1.set_ylabel("u(t)")
    ax1.set_title("Solution over time")
    ax1.grid()

    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
    )

    ax2.plot(
        u.real,
        u.imag,
        label="u(t)",
    )
    if event_times is not None and event_values is not None:
        ax1.scatter(
            event_times[1:],
            event_values[1:].imag,
            marker="x",
            label="Events: Im(u)",
        )
        ax2.scatter(
            event_values[1:].real,
            event_values[1:].imag,
            marker="o",
            label="Events",
            color="red"
        )

    ax2.set_xlabel(r"$\mathrm{Re}(u)$")
    ax2.set_ylabel(r"$\mathrm{Im}(u)$")
    ax2.set_title("Solution in the complex plane")
    ax2.grid()
    ax2.axis("equal")

    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
    )

    fig.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def main():
    t, u, event_times, event_values = exact_global_solution(
        u_start=u0,
        t_start=t0,
        alpha=alpha,
        lam=lambda_n,
        Period_start=Period_start,
        Period_end=Period_end,
        num_points=num_points,
    )
    print(f"Number of pseudo-windows: {event_times.size - 1}")
    print(f"Final time: {t[-1]}")
    print(f"Final value: {u[-1]}")

    plot_complex_solution(t, u, event_times, event_values)

    t_local = np.linspace(t0, event_times[1], num_points)
    u_local = exact_local_solution(
        u_start=u0,
        t_start=t0,
        t = t_local,
        alpha=alpha,
        lam=lambda_n(1),
    )
    plot_complex_solution(t_local, u_local)

if __name__ == "__main__":
    main()
