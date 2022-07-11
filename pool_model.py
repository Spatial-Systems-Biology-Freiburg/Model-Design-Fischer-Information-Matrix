#!/usr/bin/env python3

import numpy as np
import itertools as it


# Import custom functions for optimization
from src.optimization import get_best_fischer_results, get_new_combinations_from_best
from src.solving import factorize_reduced, convert_S_matrix_to_determinant, convert_S_matrix_to_sumeigenval, convert_S_matrix_to_mineigenval, calculate_Fischer_observable
from pool_model_plots import make_nice_plot, make_convergence_plot, make_plots, make_plots_mean
from src.database import convert_fischer_results, generate_new_collection, insert_fischer_dataclasses, drop_all_collections
from src.optimization import RaDi


# System of equation for pool-model and sensitivities
def pool_model_sensitivity(y, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    (n, sa, sb, sc) = y
    return [
        (a*Temp + c) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc
    ]


def jacobi(y, t, Q, P, Const):
    (n, sa, sb, sc) = y
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    dfdn = (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t))
    return np.array([
        [   dfdn,                                                                                             0,    0,    0   ],
        [(  Temp    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sa, dfdn, 0,    0   ],
        [(a*Temp + c) * (  -  n0/n_max * t * Temp * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sb, 0,    dfdn, 0   ],
        [(     1    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sc, 0,    0,    dfdn]
    ])


def observable(times, Q_arr, P, Const, S):
    '''Contents of x are typically results of calculate_Fischer_determinant (see above)
    Thus x = (obs, times, P, Q_arr, Const, Y0)'''
    d = convert_S_matrix_to_determinant(S)
    norm = len(Q_arr) * times.size
    seperate_times = 1.0
    for t in times:
        if len(np.unique(t)) != len(t) or len(np.unique(Q_arr[0])) != len(Q_arr[0]):
            seperate_times = 0.0
    return d * seperate_times /norm


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

    # Iterate over every combination for total effort eg. for effort=16 we get combinations (2,8), (4,4), (8,2)
    # We exclude the "boundary" cases of (1,16) and (16,1)
    # Generate pairs of temperature and time and put everything in a large list
    for _ in range(N_mult):
        # Sample only over combinatins of both
        # for (n_temp, n_times) in factorize_reduced(effort):
        for (n_times, n_temp) in it.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))):
            temperatures = np.random.choice(temp_total, n_temp, replace=False)
            #temperatures = np.linspace(temp_low, temp_low + dtemp * (n_temp - 1) , n_temp)
            times = np.array([np.sort(np.random.choice(times_total, n_times, replace=False)) for _ in range(len(temperatures))])
            combinations.append((times, [temperatures], P, Const))

    fisses = RaDi(N_parallel, N_opt, N_spawn, N_best, n_times_max, n_temp_max, effort_low, effort_high, temp_low, temp_high, dtemp, times_low, times_high, dtimes, combinations, pool_model_sensitivity, y0_t0, jacobi, observable)
    print(fisses)
    # Database part
    fischer_dataclasses = convert_fischer_results(fisses)
    coll = generate_new_collection("pool_model_random_grid_determinant_div_m")
    insert_fischer_dataclasses(fischer_dataclasses, coll)
