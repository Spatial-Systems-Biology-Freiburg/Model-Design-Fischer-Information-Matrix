#!/usr/bin/env python3

import numpy as np
import itertools as iter
import multiprocessing as mp
import time
from functools import partial

# Import custom functions for optimization
from src.optimization import get_best_fischer_results, get_new_combinations_from_best, set_multistart_combinations, get_best_fischer_results2, gradient_descent, discrete_random
from src.solving import factorize_reduced, convert_S_matrix_to_determinant, convert_S_matrix_to_sumeigenval, convert_S_matrix_to_mineigenval, calculate_Fischer_observable, convert_S_matrix_to_eigval_ratio
from src.database import convert_fischer_results, generate_new_collection, insert_fischer_dataclasses, drop_all_collections


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


def sorting_key(x):
    '''Contents of x are typically results of calculate_Fischer_determinant (see above)
    Thus x = (obs, times, P, Q_arr, Const, Y0)'''
    #norm = x[1].size**(2)
    norm = 1.0
    seperate_times = 1.0
    for t in x[1]:
        if len(np.unique(t)) != len(t) or len(np.unique(x[3][0])) != len(x[3][0]):
            seperate_times = 0.0
    return x[0] * seperate_times / norm


def optimization_run(combination, ODE_func, Y0, jacobian, observable, N_opt, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes, method='discrete_random', method_cov='wo_error', err=None): 
    func_FIM_calc = partial(calculate_Fischer_observable, ODE_func= ODE_func, Y0=Y0, jacobian=jacobian, observable=observable, method=method_cov, err=err)

    if method == 'discrete_random':
        optim_func = discrete_random
    elif method == 'gradient_descent':
        optim_func = gradient_descent

    combination_best = [combination]  
    for opt_run in range (N_opt):
        fisses = optim_func(combination_best, func_FIM_calc, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes)
        if opt_run != N_opt - 1:
            combination_best = [(times, P, Q_arr, Const) for (obs, times, P, Q_arr, Const, Y0) in fisses]
    return fisses


if __name__ == "__main__":
    # Define constants for the simulation duration
    n0 = 0.25
    n_max = 2e4
    effort_low = 2
    effort = 11
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
    temp_bnds = (temp_low, temp_high)
    dtemp = 1.0
    n_temp_max = int((temp_high - temp_low) / dtemp + 1) # effort+1
    temp_total = np.linspace(temp_low, temp_low + dtemp * (n_temp_max - 1) , n_temp_max)

    times_low = 0.0
    times_high = 15.0
    times_bnds = (times_low, times_high)
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
        for (n_times, n_temp) in iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))):
            combinations.append(set_multistart_combinations(n_times, n_temp, times_total, temp_total, P, Const, 'cont'))

    # Create pool we will later use
    p = mp.Pool(N_parallel)

    # Begin optimization scheme
    start_time = time.time()
    print_line = "[Time: {:> 8.3f} Run: {:> " +str(len(str(N_opt))) + "}] Optimizing"

    fisses = p.starmap(optimization_run, zip(
            combinations,
            iter.repeat(pool_model_sensitivity),
            iter.repeat(y0_t0),
            iter.repeat(jacobi),
            iter.repeat(convert_S_matrix_to_determinant),
            iter.repeat(N_opt),
            iter.repeat(N_spawn),
            iter.repeat(N_best),
            iter.repeat(temp_bnds),
            iter.repeat(0.1),
            iter.repeat(times_bnds),
            iter.repeat(0.1),
            iter.repeat('gradient_descent'),
            iter.repeat('relative_error'),
            iter.repeat(0.25) # relative error
        ), chunksize=100)

    print(print_line.format(time.time()-start_time, N_opt), "done")

    # Database part
    fischer_dataclasses = convert_fischer_results(fisses)
    coll = generate_new_collection("pool_model_random_grid_determinant_div_m")
    insert_fischer_dataclasses(fischer_dataclasses, coll)
