#!/usr/bin/env python3

import numpy as np
import itertools as iter
import multiprocessing as mp
import time
from functools import partial

# Import custom functions for optimization
from src.optimization import set_multistart_combinations, gradient_descent, discrete_random
from src.solving import factorize_reduced, calculate_Fischer_observable, convert_S_matrix_to_determinant, convert_S_matrix_to_sumeigenval, convert_S_matrix_to_mineigenval, convert_S_matrix_to_eigval_ratio
from src.database import convert_fischer_results, generate_new_collection, insert_fischer_dataclasses, drop_all_collections
#from models.pool_model import pool_model_sensitivity, jacobi
from models.pool_model_with_nmax import pool_model_sensitivity, jacobi


def optimization_run(combination, ODE_func, Y0, jacobian, observable, N_opt, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes, times_fixed=[], temp_fixed=[], method='discrete_random', method_cov='wo_error', err=None): 
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
    Const = tuple([() for i in range (len(measured_types))])

    effort_low = 2
    effort_temp = 3
    effort_times = 3
    
    # Define initial parameter guesses
    P_PC_O2  = (0.19997174, 0.0043666,  0.26221441, 55333.333, 4.0)
    P_PC_N2  = (0.11509934, 0.00598643, 0.39997619, 4060.0,    4.0)
    P_MRS_O2 = (0.13543291, 0.00499995, 0.81373839, 61691.75,  1.0)
    P_MRS_N2 = (0.13562779, 0.00384184, 0.76810113, 8760.0,    1.0)
    P = (P_PC_O2, P_PC_N2, P_MRS_O2, P_MRS_N2)

    # Initial values for complete ODE (with S-Terms)
    y0 = [[n0, 0, 0, 0, 0, 0] for n0 in n0_val]

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
    times_fixed = [np.array([0.0, 1.0, 3.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0])]


    # Initial conditions with initial time
    y0_t0 = (y0, times_low)

    #N_mult_prior = 1000
    # How often should we choose a sample with same number of temperatures and times
    N_mult = 2
    # How many optimization runs should we do
    N_opt = 2
    # How many best results should be propagated forward?
    N_best = 1
    # How many new combinations should an old result spawn?
    N_spawn = 2
    # How many processes will be run in parallel
    N_parallel = 1
    # Begin sampling of time and temperature values
    combinations = []

    # Determine optimization methods:
    covariance_method = 'combined_error'
    measurement_err = [0.5*n0_val[0], 1.0]
    optimization_method = 'discrete_random'

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
            iter.repeat(dtemp),
            iter.repeat(times_bnds),
            iter.repeat(dtimes),
            iter.repeat(times_fixed),
            iter.repeat(temp_fixed),
            iter.repeat(optimization_method),
            iter.repeat(covariance_method),
            iter.repeat(measurement_err)
        ), chunksize=1)
    
    p.close()
    p.join()
    print(print_line.format(time.time()-start_time, 1), "done")

    # Database part
    fischer_dataclasses = convert_fischer_results(fisses)
    coll = generate_new_collection(f"pool_model_{optimization_method}_{covariance_method}_det_for_10_grad")
    insert_fischer_dataclasses(fischer_dataclasses, coll)
