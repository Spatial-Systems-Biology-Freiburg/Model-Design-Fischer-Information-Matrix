#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools as iter
import multiprocessing as mp
import time
import json
from scipy import stats
from functools import partial


def pool_model(n, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    return (a*Temp + c)*(n-n0*np.exp(-b*Temp*t))*(1-n/n_max)

def pool_model_sensitivity(t, n_s, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    (n, sa, sb, sc) = n_s
    return [
        (a*Temp + c) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max),
        (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max),
        (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max)
    ]

def jacobi(Q, P, Const, t, n, sa, sb, sc):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    #(n, sa, sb, sc) = n_s
    return np.array([
        [(a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)), 0, 0, 0],
        [(  Temp    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)), 0, 0, 0],
        [(a*Temp + c) * (  -  n0/n_max * t * Temp * np.exp(-b*Temp*t)), 0, 0, 0],
        [(     1    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)), 0, 0, 0]
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
        #print(Q)
       # print(t)

        # Actually solve the ODE for the selected parameter values
        #r = odeint(ODE_func, n0, t, args=(Q, P, Const)).reshape(t.size)
        #jac_matrix = partial(jacobian, Q, P, Const)
        #sparsity = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        r = solve_ivp(ODE_func, [0, 20], [n0, 0.0, 0.0, 0.0], method='Radau', t_eval=t, args=(Q, P, Const))#, jac_sparsity=sparsity).y # sol for n, sa, sb, sc
        #print(r[0])
        # Calculate the S-Matrix with the supplied jacobian
        S[(slice(None), slice(None)) + index] = r[1:]

    # Reshape to 2D Form (len(P),:)
    S = S.reshape((len(P),np.prod(S.shape[1:])))
    return S


def convert_S_matrix_to_determinant(S):
    # Calculate Fisher Matrix
    F = S.dot(S.T)

    # Calculate Determinant
    det = np.linalg.det(F)
    return det

def convert_S_matrix_to_sumeigenval(S):
    # Calculate Fisher Matrix
    F = S.dot(S.T)

    # Calculate sum eigenvals
    sumeigval = np.sum(np.linalg.eigvals(F))
    return sumeigval

def convert_S_matrix_to_mineigenval(S):
    # Calculate Fisher Matrix
    F = S.dot(S.T)

    # Calculate sum eigenvals
    mineigval = np.max(np.linalg.eigvals(F))
    return mineigval


def calculate_Fischer_determinant(combinations, ODE_func, Y0, jacobian, observable):
    times, Q_arr, P, Const = combinations
    S = get_S_matrix(ODE_func, Y0, times, Q_arr, P, Const, jacobian)
    obs = observable(S)
    return obs, times, P, Q_arr, Const, Y0


def sorting_key(x):
    '''Contents of x are typically results of calculate_Fischer_determinant (see above)
    Thus x = (obs, times, P, Q_arr, Const, Y0)'''
    #norm = max(x[1].size, 1.0)**0.5
    #norm = len(x[2]) * x[1].size
    norm = 1.0
    seperate_times = 1.0
    for t in x[1]:
        if len(np.unique(t)) != len(t) or len(np.unique(x[3][0])) != len(x[3][0]):
            seperate_times = 0.0
    return x[0] * seperate_times /norm


def make_nice_plot(fischer_results, sorting_key):
    # Remember that entries in the fischer matrix have the form
    # fischer_results[0] = (obs, times, P, Q_arr, Const, Y0)
    fig, ax = plt.subplots()

    x = [f[0][1].shape[-1] for f in fischer_results]
    y = [len(f[0][3][0]) for f in fischer_results]
    weights = [sorting_key(f[0]) for f in fischer_results]

    b = (
        np.arange(min(x)-0.5, max(x)+1.5, 1.0),
        np.arange(min(y)-0.5, max(y)+1.5, 1.0)
    )

    n_measurenents = [20, 40, 60, 80, 100, 120]
    x2 = np.linspace(1, 21, 21)
    y_of_eff = [[eff/xx for xx in x2] for eff in n_measurenents]

    ax.hist2d(x, y, bins=b, weights=weights, cmap="viridis")
    for y2 in y_of_eff:
        ax.plot(x2, y2, linewidth=2, color='r')
    ax.text(4.1, 5, 'M = 20', fontsize=13, color='r')
    ax.text(6, 7, '40', fontsize=13, color='r')
    ax.text(7.8, 8, '60', fontsize=13, color='r')
    ax.text(9.3, 8.9, '80', fontsize=13, color='r')
    ax.text(10.8, 9.5, '100', fontsize=13, color='r')
    ax.text(12, 10.3, '120', fontsize=13, color='r')
    ax.set_title("Weighted Final Results", fontsize=13)
    ax.set_xlabel("#Time Steps", fontsize=13)
    ax.set_ylabel("#Temperature Values", fontsize=13)
    fig.savefig("plots/pool_model-Time-Temperature-2D-Hist.png")
    fig.clf()


def make_convergence_plot(fischer_results, effort_low, effort_high, sorting_key, N_best):
    # Intermediate step to calcualte values of grid points
    best_grid = np.zeros(shape=(effort_high-effort_low+1, effort_high-effort_low+1))
    for n, m in iter.product(range(effort_high-effort_low+1), range(effort_high-effort_low+1)):
        fisses = get_best_fischer_results((effort_low + n, effort_low + m), fischer_results, sorting_key, N_best)
        # Reminder:
        # (obs, times, P, Q_arr, Const, Y0) = fisses[0]
        if len(fisses) > 0:
            best_grid[n,m] = np.average(np.array([f[0] for f in fisses]))
            # best_grid[n,m] = fisses[0][0]
    color_value = lambda n, k: best_grid[max(0, min(effort_high-effort_low, round(n-effort_low))), max(0, min(effort_high-effort_low, round(k/n)))]
    # Now plot lines for efforts
    fig, ax = plt.subplots()
    for k in range(effort_low, effort_high**2+1):
        x = np.array([f[0] for f in factorize_reduced(k)])
        x = x[x<=effort_high]
        x = x[k/x<=effort_high]
        if x.size >= 5:
            x_smooth = np.linspace(x.min(), x.max())
            y = k/x
            y_smooth = k/x_smooth
            cv = np.array([color_value(n, k) for n in x])
            if cv.max()-cv.min() > 0.0:
                size_values = 2 * (cv-cv.min())/(cv.max()-cv.min()) * mpl.rcParams['lines.markersize'] ** 2
                ax.scatter(x, y, marker="o", s=size_values, c=cv, cmap="viridis")
                ax.plot(x_smooth, y_smooth, c="k", linestyle=":", alpha=0.7)
    ax.set_title("Effort lines")
    ax.set_xlabel("#Time Measurements")
    ax.set_ylabel("#Temp Measurements")
    fig.savefig("plots/Effort_lines.png")
    fig.clf()


def make_plots(fisses, sorting_key):
    new_comb = sorted([(f[0][1].shape[-1] * len(f[0][3][0]), sorting_key(f[0])) for f in fisses], key=lambda l:l[0])
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
    ax.set_xlabel('# of measurements', fontsize=15)
    ax.set_ylabel('det(F)', fontsize=15)
    # ax.tick_params(fontsize=13)
    fig.savefig("plots/determinant_FIM_vs_num_measurements.png")
    fig.clf()

def make_plots_mean(fisses, sorting_key):
    new_comb = sorted([(f[0][1].shape[-1] * len(f[0][3][0]), sorting_key(f[0])) for f in fisses], key=lambda l:l[0])
    final_comb = [] # effort, mean_det, std_err_det
    print(new_comb[-1][0])
    effort_list = set([c[0] for c in new_comb])
    for eff in effort_list:
        #print(eff)
        same_eff_comb = list(filter(lambda x: x[0]==eff, new_comb))
    #    print(same_eff_comb)
        final_comb.append([eff, np.mean(same_eff_comb, axis=0)[1], stats.sem(same_eff_comb, axis=0)[1]])

    x = [f[0] for f in final_comb]
    y = [f[1] for f in final_comb]
    y_std = [f[2] for f in final_comb]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.errorbar(x, y, yerr=y_std, fmt = 'o')
    ax.set_yscale('log')
    ax.set_xlabel('# of measurements', fontsize=15)
    ax.set_ylabel('det(F)', fontsize=15)
        # ax.tick_params(fontsize=13)
    fig.savefig("plots/determinant_FIM_vs_num_measurements2_mean.png")
    fig.clf()


def write_in_file(fisses, num_iter, crit_name, effort_max, sorting_key):
    P = fisses[0][0][2]
    Const = fisses[0][0][4]
    filename = f"Experimental_design_iter_{num_iter}_crit_{crit_name}_a_{P[0]:.3f}_b_{P[1]:.3f}_c_{P[2]:.3f}_n0_{Const[0]}_nmax_{Const[1]}"#_effmax_{effort_max}"
    path = 'results'
    filenamepath ='./' + path + '/' + filename + '.json'
    new_comb = sorted([(f[0][1].shape[-1] * len(f[0][3][0]), sorting_key(f[0]), f[0][1].shape[-1], len(f[0][3][0]), [list(ff) for ff in (f[0][1])], list(f[0][3][0])) for f in fisses], key=lambda l:l[0])
    #new_comb = [(f[0][1].shape[-1] * len(f[0][3][0]), f[0][0], f[0][1].shape[-1], len(f[0][3][0]), [list(ff) for ff in (f[0][1])], list(f[0][3][0])) for f in fisses]
    #new_comb = sorted(filter(lambda x: x[0] <= effort_max, new_comb), key=lambda l: l[1], reverse=True)[:10]
    with open(filenamepath, "w") as file:
        for c in new_comb:
            opt_design_dict = {'eff': c[0], 'obs': c[1], 'n_times': c[2], 'n_temp': c[3], 'times': c[4], 'temp': c[5]}
            json.dump(opt_design_dict, file, indent=1)
    file.close()


def get_best_fischer_results(n_time_temp, fischer_results, sorting_key, N_best):
    (n_times, n_temp) = n_time_temp
    # TODO use partial sort or some other efficient alrogithm to obtain O(n) scaling behvaiour
    # for best result retrieval
    return sorted(filter(lambda x: x[1].shape[-1]==n_times and len(x[3][0])==n_temp, fischer_results), key=sorting_key, reverse=True)[:N_best]


def get_new_combinations_from_best(best, N_spawn, temp_low, temp_high, dtemp, times_low, times_high, dtimes):
    combinations = []
    for (det, times, P, Q_arr, Const, Y0) in best:
        # Also depend old result in case its better
        combinations.append((times, Q_arr, P, Const))
        # Now spawn new results via next neighbors of current results
        for _ in range(0, N_spawn):
            #temps_new = Q_arr[0]
            temps_new = np.array(
                [np.random.choice([max(temp_low, T-dtemp), T, min(temp_high, T+dtemp)]) for T in Q_arr[0]]
            )
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

    # How often should we choose a sample with same number of temperatures and times
    N_mult = 50
    # How many optimization runs should we do
    N_opt = 20
    # How many best results should be propagated forward?
    N_best = 6
    # How many new combinations should an old result spawn?
    N_spawn = 20
    # How many processes will be run in parallel
    N_parallel = 2

    # Begin sampling of time and temperature values
    combinations = []

    # Iterate over every combination for total effort eg. for effort=16 we get combinations (2,8), (4,4), (8,2)
    # We exclude the "boundary" cases of (1,16) and (16,1)
    # Generate pairs of temperature and time and put everything in a large list
    for _ in range(N_mult):
        # Sample only over combinatins of both
        # for (n_temp, n_times) in factorize_reduced(effort):
        for (n_times, n_temp) in iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))):
            temperatures = np.random.choice(temp_total, n_temp, replace=False)
            #temperatures = np.linspace(temp_low, temp_low + dtemp * (n_temp - 1) , n_temp)
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
                    iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))),
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
        iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))),
        iter.repeat(fischer_results),
        iter.repeat(sorting_key),
        iter.repeat(1)
    ), chunksize=100)

    print(print_line.format(time.time()-start_time, opt_run+1), "done")

    make_nice_plot(fisses, sorting_key)

    make_convergence_plot(fischer_results, effort_low, effort, sorting_key, N_best)

    make_plots(fisses, sorting_key)
    write_in_file(fisses, 1, 'D', effort_max, sorting_key)
    make_plots_mean(fisses, sorting_key)
