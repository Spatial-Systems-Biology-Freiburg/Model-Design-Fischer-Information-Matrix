#!/usr/bin/env python3

from multiprocessing import pool
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import itertools as iter
import multiprocessing as mp
import time
import json
import heapq


def pool_model(n, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    return (a*Temp + c)*(n-n0*np.exp(-b*Temp*t))*(1-n/n_max)


def jacobi(n, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    return np.array([
        (  Temp) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (a*Temp) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max),
        (     1) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max)
    ])


def factorize_reduced(M):
    res = []
    for i in range(2, M):
        if (M % i == 0):
            res.append((i, round(M/i)))
    return res


def get_S_matrix(ODE_func, n0, times, Q_arr, P, Const, jacobian):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    # res = np.zeros((len(P),) + tuple(len(x) for x in Q_arr) + (t.size,))
    # res = np.zeros(tuple(len(x) for x in Q_arr) + (t.size,))
    S = np.zeros((len(P),) + (times.shape[-1],) + tuple(len(x) for x in Q_arr))

    # Iterate over all combinations of Q-Values
    for index in iter.product(*[range(len(q)) for q in Q_arr]):
        # Store the results of the respective ODE solution
        Q = [Q_arr[i][j] for i, j in enumerate(index)]
        t = times[index]

        # Actually solve the ODE for the selected parameter values
        r = odeint(ODE_func, n0, t, args=(Q, P, Const)).reshape(t.size)

        # Calculate the S-Matrix with the supplied jacobian
        S[(slice(None), slice(None)) + index] = jacobian(r, t, Q, P, Const)

    # Reshape to 2D Form (len(P),:)
    S = S.reshape((len(P),np.prod(S.shape[1:])))
    return S


def convert_S_matrix_to_determinant(S):
    # Calculate Fisher Matrix
    F = S.dot(S.T)

    # Calculate Determinant
    det = np.linalg.det(F)
    return det


def calculate_Fischer_determinant(combinations, ODE_func, Y0, jacobian, observable):
    times, Q_arr, P, Const = combinations
    S = get_S_matrix(ODE_func, Y0, times, Q_arr, P, Const, jacobian)
    obs = observable(S)
    return obs, times, P, Q_arr, Const, Y0


def sorting_key(x):
    '''Contents of x are typically results of calculate_Fischer_determinant (see above)
    Thus x = (obs, times, P, Q_arr, Const, Y0)'''
    norm = max(len(x[2]) * x[1].size * np.prod([len(x) for x in x[3]]), 1.0)
    return x[0]/norm


def make_nice_plot(fischer_results, sorting_key):
    # Remember that entries in the fischer matrix have the form
    # fischer_results[0] = (obs, times, P, Q_arr, Const, Y0)
    fig, ax = plt.subplots()

    x = [f[1].shape[-1] for f in fischer_results]
    y = [len(f[3][0]) for f in fischer_results]
    weights = [sorting_key(f) for f in fischer_results]

    b = (
        np.arange(min(x)-0.5, max(x)+1.5, 1.0),
        np.arange(min(y)-0.5, max(y)+1.5, 1.0)
    )
    ax.hist2d(x, y, bins=b, weights=weights, cmap="viridis")
    ax.set_title("Weighted Final Results")
    ax.set_xlabel("#Time Steps")
    ax.set_ylabel("#Temperature Values")
    fig.savefig("plots/pool_model-Time-Temperature-2D-Hist.png")
    fig.clf()


def make_convergence_plot(fischer_results, effort):
    fig, ax = plt.subplots()

    fig.clf()


def make_plots(fisses, sorting_key):
                                                            # sorting_key(f[0])
    new_comb = sorted([(f[0][1].shape[-1] * len(f[0][3][0]), f[0][0]) for f in fisses], key=lambda l:l[0])
    final_comb = []
    for i in range (0, len(new_comb)):
        if i == 0 or new_comb[i][0] != new_comb[i - 1][0]:
            final_comb.append(new_comb[i])
        else:
            final_comb[-1] = (new_comb[i][0], max(new_comb[i][1], new_comb[i - 1][1]))
            

    x = [f[0] for f in final_comb]
    y = [f[1] for f in final_comb]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_yscale('log')
    ax.set_xlabel('# of measurements')
    ax.set_ylabel('det(F)')
    #ax.set_xlim(2, 40)
    fig.savefig("plots/determinant_FIM_vs_num_measurements.png")
    plt.show()



def get_best_fischer_results(n_time_temp, fischer_results, sorting_key, N_best, chunksize=400):
    (n_times, n_temp) = n_time_temp
    ret = []
    for x in fischer_results:
        if x[1].shape[-1]==n_times and len(x[3][0])==n_temp:
            if len(ret) > 0:
                if sorting_key(x) > sorting_key(max(ret, key=sorting_key)):
                    heapq.heapreplace(ret, x)
            else:
                heapq.heappush(ret, x)
    return ret


def get_new_combinations_from_best(best, N_spawn, temp_low, temp_high, dtemp, times_low, times_high, dtimes):
    combinations = []
    for (det, times, P, Q_arr, Const, Y0) in best:
        # Also depend old result in case its better
        combinations.append((times, Q_arr, P, Const))
        # Now spawn new results via next neighbors of current results
        for _ in range(0, N_spawn):
            temps_new = Q_arr[0]
            #temps_new = np.array(
             #   [np.random.choice([max(temp_low, T-dtemp), T, min(temp_high, T+dtemp)]) for T in Q_arr[0]]
            #)
            times_new = np.array(
                [
                    np.sort(np.array([np.random.choice(
                        [max(times_low, t-dtimes), t, min(times_high, t+dtimes)]
                    ) for t in times[i]]))
                    for i in range(len(Q_arr[0]))
                ]
            )
            combinations.append((times_new, [temps_new], P, Const))
    return combinations


if __name__ == "__main__":
    # Define constants for the simulation duration
    n0 = 0.25
    n_max = 2e4
    effort_low = 2
    effort = 2**4
    Const = (n0, n_max)

    # Define initial parameter guesses
    a = 0.065
    b = 0.01
    c = 1.31
    P = (a, b, c)

    # Define bounds for sampling
    temp_low = 2.0
    temp_high = 16.0
    dtemp = 2.0
    n_temp_max = int((temp_high - temp_low) / dtemp + 1) # effort+1
    temp_total = np.linspace(temp_low, temp_low + dtemp * (n_temp_max - 1) , n_temp_max)

    times_low = 0.0
    times_high = 20.0
    dtimes = 1.0
    n_times_max = int((times_high-times_low) / dtimes + 1) # effort+1
    times_total = np.linspace(times_low, times_low + dtimes * (n_times_max - 1), n_times_max)

    # How often should we choose a sample with same number of temperatures and times
    N_mult = 50
    # How many optimization runs should we do
    N_opt = 20
    # How many best results should be propagated forward?
    N_best = 6
    # How many new combinations should an old result spawn?
    N_spawn = 20
    # How many processes will be run in parallel
    N_parallel = 44

    # Begin sampling of time and temperature values
    combinations = []

    # Iterate over every combination for total effort eg. for effort=16 we get combinations (2,8), (4,4), (8,2)
    # We exclude the "boundary" cases of (1,16) and (16,1)
    # Generate pairs of temperature and time and put everything in a large list
    for _ in range(N_mult):
        # Sample only over combinatins of both
        # for (n_temp, n_times) in factorize_reduced(effort):
        for (n_times, n_temp) in iter.product(range(effort_low, min(effort, n_times_max)), range(effort_low, min(effort, n_temp_max))):
            #temperatures = np.random.choice(temp_total, n_temp, replace=False)
            temperatures = np.linspace(temp_low, temp_low + dtemp * (n_temp - 1) , n_temp)
            times = np.array([np.sort(np.random.choice(times_total, n_times, replace=False)) for _ in range(len(temperatures))])
            combinations.append((times, [temperatures], P, Const))

    # Create pool we will later use
    p = mp.Pool(N_parallel)

    # Begin optimization scheme
    start_time = time.time()
    print_line = "[Time: {:> 8.3f} Run: {:> " +str(len(str(N_opt))) + "}] Optimizing"
    for opt_run in range(0, N_opt):
        print(print_line.format(time.time()-start_time, opt_run+1), end="\r")
        # Calculate new results
        # fischer_results will have entries of the form
        # (obs, times, P, Q_arr, Const, Y0)
        fischer_results = p.starmap(calculate_Fischer_determinant, zip(
            combinations,
            iter.repeat(pool_model),
            iter.repeat(n0),
            iter.repeat(jacobi),
            iter.repeat(convert_S_matrix_to_determinant)
        ))

        # Do not optimize further if we are in the last run
        if opt_run != N_opt-1:
            # Delete old combinations
            combinations.clear()
            fisses = p.starmap(get_best_fischer_results, zip(
                    iter.product(range(effort_low, min(effort, n_times_max)), range(effort_low, min(effort, n_temp_max))),
                    iter.repeat(fischer_results),
                    iter.repeat(sorting_key),
                    iter.repeat(N_best)
            ), chunksize=100)
            # Calculate new combinations parallelized
            combinations = p.starmap(get_new_combinations_from_best, zip(
                fisses,
                iter.repeat(N_spawn),
                iter.repeat(temp_low),
                iter.repeat(temp_high),
                iter.repeat(dtemp),
                iter.repeat(times_low),
                iter.repeat(times_high),
                iter.repeat(dtimes)
            ))
            combinations = [x for comb_list in combinations for x in comb_list]

    fisses = p.starmap(get_best_fischer_results, zip(
        iter.product(range(effort_low, min(effort, n_times_max)), range(effort_low, min(effort, n_temp_max))),
        iter.repeat(fischer_results),
        iter.repeat(sorting_key),
        iter.repeat(1)
    ), chunksize=100)

    print(print_line.format(time.time()-start_time, opt_run+1), "done")

    make_nice_plot(fischer_results, sorting_key)

    make_convergence_plot(fischer_results, effort)
    
    make_plots(fisses, sorting_key)
