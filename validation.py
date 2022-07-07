#!/usr/bin/env python3

import numpy as np
import itertools as it
import multiprocessing as mp
import time
import itertools as it
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Import custom functions for optimization
from src.optimization import get_best_fischer_results, get_new_combinations_from_best
from src.solving import factorize_reduced, convert_S_matrix_to_determinant, convert_S_matrix_to_sumeigenval, convert_S_matrix_to_mineigenval, calculate_Fischer_observable
from pool_model_plots import make_nice_plot, make_convergence_plot, make_plots, make_plots_mean
from src.database import convert_fischer_results, generate_new_collection, insert_fischer_dataclasses, drop_all_collections


# System of equation for pool-model and sensitivities
def pool_model_sensitivity(y, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    (n, sa, sb, sc) = y
    return [
        (a*Temp + c) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max),
        # Derivative dlog(n)/dlog(p)
        # a / n * ( (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa ),
        # b / n * ( (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb ),
        # c / n * ( (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc )
        # Derivative dn/dp
        (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc
        # Derivative normalized dn/dp * (p/n)
        # a / n_max * (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        # b / n_max * (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        # c / n_max * (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc
    ]


def exp_growth(y, t, Q, P, Const):
    '''Implement the ode n' = a*c*(1-b*Temp*n/nmax)'''
    (a, b, c) = P
    a = a*2e6
    (Temp,) = Q
    (n0, nmax) = Const
    (n, sa, sb, sc) = y
    dfdn = - a*c*Temp*b/nmax
    return [
        a*c*Temp*(1-b*n/nmax),
        a / n_max * (  c*Temp*(1-b*n/nmax) + dfdn * sa),
        b / n_max * (a*c*Temp*(    n/nmax) + dfdn * sb),
        c / n_max * (a  *Temp*(1-b*n/nmax) + dfdn * sc)
    ]


def exp_growth_jacobi(y, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, nmax) = Const
    (n, sa, sb, sc) = y
    


def jacobi(y, t, Q, P, Const):
    (n, sa, sb, sc) = y
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    dfdn = (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t))
    return np.array([
        [dfdn, 0, 0, 0],
        [(  Temp    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sa, dfdn, 0, 0],
        [(a*Temp + c) * (  -  n0/n_max * t * Temp * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sb, 0, dfdn, 0],
        [(     1    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sc, 0, 0, dfdn]
    ])


def sorting_key(x):
    '''Contents of x are typically results of calculate_Fischer_determinant (see above)
    Thus x = (obs, times, P, Q_arr, Const, Y0)'''
    #norm = max(x[1].size, 1.0)**0.5
    norm = len(x[2]) * x[1].size
    # norm = 1.0
    seperate_times = 1.0
    for t in x[1]:
        if len(np.unique(t)) != len(t) or len(np.unique(x[3][0])) != len(x[3][0]):
            seperate_times = 0.0
    return x[0] * seperate_times /norm


def get_x(x):
    return x[1].shape[0] * x[3][0].size


def get_y(x):
    return x[0]


def plot_odes(ODE_func, y0_t0, times_plot, Q_arr, P, Const, times_mark=None, jacobian=None, filename="plots/validation.png"):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    (y0, t0) = y0_t0

    # Check if additional t_mark was supplied. If not assume it is same as t_plot
    if type(times_mark) == type(None):
        times_mark = times_plot
    elif times_mark.shape[0] != times_plot.shape[0]:
        raise ValueError("times_mark and times_plot need to have same x-dimension!")

    res_mark = []
    res_plot = []
    # Iterate over all combinations of Q-Values
    for index in it.product(*[range(len(q)) for q in Q_arr]):
        # Store the results of the respective ODE solution
        Q = [Q_arr[i][j] for i, j in enumerate(index)]
        t_plot = times_plot[index]
        t_mark = times_mark[index]

        # Actually solve the ODE for the selected parameter values
        #r = solve_ivp(ODE_func, [t0, t.max()], y0, method='Radau', t_eval=t,  args=(Q, P, Const), jac=jacobian).y.T[1:,:]
        r_plot = odeint(ODE_func, y0, np.insert(t_plot, 0, t0), args=(Q, P, Const), Dfun=jacobian)
        r_mark = odeint(ODE_func, y0, np.insert(t_mark, 0, t0), args=(Q, P, Const), Dfun=jacobian)
        res_plot.append((t_plot, r_plot, Q))
        res_mark.append((t_mark, r_mark, Q))

    fig, ax = plt.subplots((r_plot.shape[1]), figsize=(12, 8))
    cols = ["red", "blue", "green", "black", "orange", "yellow"]
    linestyles = ["-", ":", "--", "-."]
    for i, (t_plot, r_plot, Q) in enumerate(res_plot):
        ax[0].plot(t_plot, r_plot[1:,0], color=cols[i])
        for j in range(1,r_plot.shape[1]):
            ax[j].plot(t_plot, r_plot[1:,j], color=cols[i], linestyle=linestyles[j])
    
    for i, (t_mark, r_mark, Q) in enumerate(res_mark):
        ax[0].plot(t_mark, r_mark[1:,0], marker=".", label='$n(t)$ for Temperature: {:3.1f}'.format(Q[0]), color=cols[i])
        ax[0].legend()
        for j in range(1,r_plot.shape[1]):
            ax[j].plot(t_mark, r_mark[1:,j], marker=".", label='$\partial n/\partial p_{}$'.format(j) + ' Temperature: {:3.1f}'.format(Q[0]), color=cols[i], linestyle="")
            ax[j].legend()
    fig.tight_layout()
    fig.savefig(filename)


def get_S_matrix(ODE_func, y0_t0, times, Q_arr, P, Const, jacobian=None):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    (y0, t0) = y0_t0
    S = np.zeros((len(P),) + (times.shape[-1],) + tuple(len(x) for x in Q_arr))

    # Iterate over all combinations of Q-Values
    for index in it.product(*[range(len(q)) for q in Q_arr]):
        # Store the results of the respective ODE solution
        Q = [Q_arr[i][j] for i, j in enumerate(index)]
        t = times[index]

        # Actually solve the ODE for the selected parameter values
        #r = solve_ivp(ODE_func, [t0, t.max()], y0, method='Radau', t_eval=t,  args=(Q, P, Const), jac=jacobian).y.T[1:,:]
        r = odeint(ODE_func, y0, np.insert(t, 0, t0), args=(Q, P, Const), Dfun=jacobian)
        q = r.T[1:,1:]

        # Calculate the S-Matrix with the supplied jacobian
        S[(slice(None), slice(None)) + index] = q
    
    # Reshape to 2D Form (len(P),:)
    S = S.reshape((len(P),np.prod(S.shape[1:])))
    return S


if __name__ == "__main__":
    # Define constants for the simulation duration
    n0 = 0.25
    n_max = 2e4
    effort_low = 2
    effort = 11
    effort_max = 20
    Const = (n0, n_max)

    # Define initial parameter guesses
    a = 0.065
    b = 0.01
    c = 1.31

    #2nd choice of parameters:
    #a = 0.0673
    #b = 0.01
    #c = 1.314

    P = (a, b, c)

    # Initial values for complete ODE (with S-Terms)
    y0 = [n0, 0, 0, 0]

    # Define bounds for sampling
    temp_low = 2.0
    temp_high = 16.0
    dtemp = 1.0
    n_temp_max = int((temp_high - temp_low) / dtemp + 1) # effort+1
    temp_total = np.linspace(temp_low, temp_low + dtemp * (n_temp_max - 1) , n_temp_max)

    times_low = 0.0
    times_high = 15.0
    dtimes = 1.0
    n_times_max = int((times_high-times_low) / dtimes + 1) # effort+1
    times_total = np.linspace(times_low, times_low + dtimes * (n_times_max - 1), n_times_max)

    # Initial conditions with initial time
    y0_t0 = (y0, times_low)

    # How often should we choose a sample with same number of temperatures and times
    N_mult = 1000
    # How many optimization runs should we do
    N_opt = 100
    # How many best results should be propagated forward?
    N_best = 20
    # How many new combinations should an old result spawn?
    N_spawn = 10
    # How many processes will be run in parallel
    N_parallel = 46

    # Begin sampling of time and temperature values
    combinations = []

    

    n_times = 75
    n_temps = 4
    temperatures = np.linspace(temp_low, temp_high, n_temps)
    times = np.array([np.linspace(times_low, times_high, n_times) for _ in range(len(temperatures))])

    ode = pool_model_sensitivity
    plot_odes(ode, y0_t0, times, [temperatures], P, Const, jacobian=jacobi)
    S = get_S_matrix(ode, y0_t0, times, [temperatures], P, Const, jacobian=jacobi)

    # Print the fischer matrix and its eigenvalues and determinant
    F = S.dot(S.T)
    print(F)
    print(np.linalg.eigvals(F))
    print(np.linalg.det(F))

    # Intelligently choose parameters for good and bad runs
    n_times_good = 10
    t0 = 6
    t1 = 11
    td = 2
    times_inflections = np.linspace(t1, t0, n_temps)
    times_good = np.array([
        np.linspace(times_inflections[i] - td, times_inflections[i] + td, n_times_good) for i in range(n_temps)
    ])
    times_worse = np.array([np.linspace(times_low + td, times_high - td, n_times_good) for _ in range(len(temperatures))])
    S_good = get_S_matrix(ode, y0_t0, times_good, [temperatures], P, Const, jacobian=jacobi)
    S_worse = get_S_matrix(ode, y0_t0, times_worse, [temperatures], P, Const, jacobian=jacobi)

    times_plot = np.array([
        np.linspace(times_low, times_high) for _ in range(len(temperatures))
    ])
    plot_odes(ode, y0_t0, times_plot, [temperatures], P, Const, jacobian=jacobi, times_mark=times_good, filename="plots/validation_times_good.png")
    plot_odes(ode, y0_t0, times_plot, [temperatures], P, Const, jacobian=jacobi, times_mark=times_worse, filename="plots/validation_times_worse.png")

    # Plot good and bad times
    print("Determinant good: {}".format(np.linalg.det(S_good.dot(S_good.T))))
    print("Determinant worse: {}".format(np.linalg.det(S_worse.dot(S_worse.T))))
    
    # for n in range(2,201):
    #     temperatures = np.linspace(temp_low, temp_high, n)
    #     times = np.array([np.linspace(times_low, times_high, n) for _ in range(len(temperatures))])
    #     combinations.append((times, [temperatures], P, Const))
    # 
    # p = mp.Pool(N_parallel)
    # fischer_results = p.starmap(
    #     calculate_Fischer_observable,
    #     zip(
    #         combinations,
    #         it.repeat(pool_model_sensitivity),
    #         it.repeat(y0_t0),
    #         it.repeat(jacobi),
    #         it.repeat(convert_S_matrix_to_determinant)
    #     )
    # )
    # 
    # x = p.map(get_x, fischer_results)
    # y = p.map(get_y, fischer_results)
    # 
    # import matplotlib.pyplot as plt
    # 
    # plt.plot(x, y)
    # # plt.ylim(bottom=1e33)
    # # plt.yscale('log')
    # plt.savefig("validation.png")
    # plt.show()