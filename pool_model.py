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
    (Temp, measurement_type) = Q
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


def optimization_run(combination, ODE_func, Y0, jacobian, observable, N_opt, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes, num_temp_fixed, method='discrete_random', method_cov='wo_error', err=None): 
    func_FIM_calc = partial(calculate_Fischer_observable, ODE_func= ODE_func, Y0=Y0, jacobian=jacobian, observable=observable, method=method_cov, err=err)

    if method == 'discrete_random':
        optim_func = discrete_random
    elif method == 'gradient_descent':
        optim_func = gradient_descent

    combination_best = [combination]  
    for opt_run in range (N_opt):
        fisses = optim_func(combination_best, func_FIM_calc, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes, times_fixed, temp_fixed)
        if opt_run != N_opt - 1:
            combination_best = [(times, P, Q_arr, Const) for (obs, times, P, Q_arr, Const, Y0) in fisses]
    return fisses


if __name__ == "__main__":
    # Define constants for the simulation duration
    measured_types = ['PC_O2', 'PC_N2', 'MRS_O2', 'MRS_N2']
    n0_val = [4.0, 4.0, 1.0, 1.0]
    n_max_val = [55333.333, 4060.0, 61691.75, 8760.0]
    Const = tuple([(n0_val[i], n_max_val[i]) for i in range (len(measured_types))])

    # Define initial parameter guesses
    P_PC_O2 = (0.19997174, 0.0043666, 0.26221441)
    P_PC_N2 = (0.11509934, 0.00598643, 0.39997619)
    P_MRS_O2 = (0.13543291, 0.00499995, 0.81373839)
    P_MRS_N2 = (0.13562779, 0.00384184, 0.76810113)
    P = (P_PC_O2, P_PC_N2, P_MRS_O2, P_MRS_N2)

    # Initial values for complete ODE (with S-Terms)
    y0 = [[n0, 0, 0, 0] for n0 in n0_val]

    # Define bounds for sampling
    temp_low = 2.0
    temp_high = 16.0
    temp_bnds = (temp_low, temp_high)
    dtemp = 1.0
    n_temp_max = int((temp_high - temp_low) / dtemp + 1) # effort+1
    temp_total = np.linspace(temp_low, temp_low + dtemp * (n_temp_max - 1) , n_temp_max)
    temp_fixed = [2.0, 10.0]

    times_low = 0.0
    times_high = [25.0, 15.0] # set 2 time boundaries (1st for small temperatures (2, 3, 4 grad) 2nd for others)
    times_bnds = [(times_low, t_high) for t_high in times_high]
    dtimes = 1.0
    n_times_max = [int((t_high-times_low) / dtimes + 1) for t_high in times_high] # effort+1
    times_total = [np.linspace(times_low, times_low + dtimes * (n - 1), n)  for n in n_times_max]


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

    # Determine optimization methods:
    covariance_method = 'combined_error'
    measurement_err = [0.5*n0_val[0], 1.0]

    # Iterate over every combination for total effort eg. for effort=16 we get combinations (2,8), (4,4), (8,2)
    # We exclude the "boundary" cases of (1,16) and (16,1)
    # Generate pairs of temperature and time and put everything in a large list
    for _ in range(N_mult):
        # Sample only over combinatins of both
        # for (n_temp, n_times) in factorize_reduced(effort):
        for (n_times, n_temp) in iter.product(range(effort_low, min(effort_times, min(n_times_max) - 2)), range(effort_low, min(effort_temp, n_temp_max - 2))):
            combinations.append(set_multistart_combinations(n_times, n_temp, times_total, temp_total, P, Const, measured_types, times_fixed, temp_fixed, 'discr'))
    #print(combinations[0])
    # Create pool we will later use
    p = mp.Pool(N_parallel)

    # Begin optimization scheme
    start_time = time.time()
    print_line = "[Time: {:> 8.3f} Run: {:> " +str(len(str(N_opt))) + "}] Optimizing"

    # Choose from N_mult_prior cobinations N_mult best for further optimization
    #fischer_results = p.starmap(calculate_Fischer_observable, zip(
    #        combinations,
    #        iter.repeat(pool_model_sensitivity),
    #        iter.repeat(y0_t0),
    #        iter.repeat(jacobi),
    #        iter.repeat(convert_S_matrix_to_determinant),
    #        iter.repeat(covariance_method), # True if use covariance error matrix, False if not
    #        iter.repeat(measurement_err) 
    #))
    #fisses = p.starmap(get_best_fischer_results, zip(
    #        iter.product(range(effort_low, min(effort_times, min(n_times_max) - 2)), range(effort_low, min(effort_temp, n_temp_max - 2))),
    #        iter.repeat(fischer_results),
    #        iter.repeat(sorting_key),
    #        iter.repeat(N_mult)
    #), chunksize=100)
    #combinations = []
    #for ff in fisses:
    #    for (obs, times, P, Q_arr, Const, Y0) in ff:
    #        combinations.append((times, P, Q_arr, Const))
    #print(print_line.format(time.time()-start_time, N_opt), "done")

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
