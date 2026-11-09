# -*- coding: utf-8 -*-

def plot_error_per_iteration(ax_left, ax_right, errTT_max, errU_diff_max, K, N, left_style=None, right_style=None, **kwargs):
    left_kwargs = {**kwargs, **(left_style or {})}
    right_kwargs = {**kwargs, **(right_style or {})}
    ax_left.semilogy(range(K+1), errTT_max, **left_kwargs)
    ax_left.set_yscale('symlog', linthresh=1e-14)
    ax_left.set_xlabel("Parareal iteration k"), ax_left.set_ylabel(r'$\|t_{Parareal} - t_{Fine}\|_\infty$'), ax_left.grid(True);
    ax_left.set_xticks(range(K+1))
    
    
    ax_right.semilogy(range(K+1), errU_diff_max, **right_kwargs)
    ax_right.set_yscale('symlog', linthresh=1e-14)
    ax_right.set_xlabel("Parareal iteration k"), ax_right.set_ylabel(r'$\|u_{Parareal} - u_{Fine}\|_\infty$'), ax_right.grid(True);

    return ax_left, ax_right

def plot_error_per_cycle(ax_left, ax_right, errTT, errU_diff, K, N, left_style=None, right_style=None, **kwargs):
    left_kwargs = {**kwargs, **(left_style or {})}
    right_kwargs = {**kwargs, **(right_style or {})}
    for k in range(K+1):
        ax_left.semilogy(range(N+1), errTT[k,:], label=f"k={k}", **left_kwargs)
        ax_right.semilogy(range(N+1), errU_diff[k,:], label=f"k={k}", **right_kwargs)
    ax_left.set_xticks(range(N+1))
    ax_left.set_yscale('symlog', linthresh=1e-14)
    ax_left.set_xlabel("Cycles"), ax_left.set_ylabel(r'$\|t_{Parareal} - t_{Fine}\|_\infty$'), ax_left.grid(True);

    ax_right.set_yscale('symlog', linthresh=1e-14)
    ax_right.set_xlabel("Cycles"), ax_right.set_ylabel(r'$\|u_{Parareal} - u_{Fine}\|_\infty$'), ax_right.grid(True);
    return ax_left, ax_right
