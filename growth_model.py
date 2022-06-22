#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools as iter
from multiprocessing import Pool
import time


def ODE_RHS(n, t, a, b, n_max, temp):
    return (a*temp + b) * n * (1 - n/n_max)


def generate_result(ODE_func, n0, t, params, vars):
    (a, b, n_max) = params
    (temp,) = vars
    return odeint(ODE_func, n0, t, args=(a, b, n_max, temp))


def obtain_results(sample_times, params, variables, ODE_func):
    results = np.zeros(list(x.size for x in params + variables) + [sample_times.size])

    for index_param in iter.product(*[range(x.size) for x in params]):
        for index_var in iter.product(*[range(y.size) for y in variables]):
            p = list(params[i][index_param[i]] for i in range(len(params)))
            v = list(variables[j][index_var[j]] for j in range(len(variables)))
            
            results[index_param][index_var] = generate_result(ODE_func, n0, sample_times, p, v).reshape(sample_times.size)
    return results


def observable(sample_times, params, variables, ode_results):
    return ode_results


def get_large_S_matrix(sample_times, params, variables, results):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form 
    O   -->  observable
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    S = np.zeros((len(params),) + tuple(x.size for x in variables) + (sample_times.size,))
    for i in range(len(params)):
        S[i] = (results[tuple(1+(i==j) for j in range(len(params)))] - results[tuple(1-(i==j) for j in range(len(params)))])/(2*(params[i][2]-params[i][0]))
    return S


def Large_S_matrix_to_determinant(S, sample_times, params, variables):
    S_trans = S.reshape(len(params), sum(y.size for y in variables) * sample_times.size)
    
    # Calculate Fisher Matrix
    F = S_trans.dot(S_trans.T)
    
    # Calculate Determinant
    det = np.linalg.det(F)
    return det


def Large_S_matrix_to_eigenvalue(S, sample_times, params, variables):
    S_trans = S.reshape(len(params), sum(y.size for y in variables) * sample_times.size)
    
    # Calculate Fisher Matrix
    F = S_trans.dot(S_trans.T)
    
    # Calculate Eigenvalues and determinant
    w, v = np.linalg.eig(F)
    return w


def calculate_Fischer_determinant(sample_times, params, variables, ODE_func):
    results = obtain_results(sample_times, params, variables, ODE_func)
    S = get_large_S_matrix(sample_times, params, variables, results)
    det = Large_S_matrix_to_determinant(S, sample_times, params, variables)
    return det, sample_times, params, variables


def make_nice_plots(fischer_results, N_best=5):
    # Sort the results with respect to the value calculated (determinant in this example) and print first few ones
    fischer_results = sorted(fischer_results, key=sorting_key, reverse=True)
    
    print("The first 4 winners are:")
    for det, times, param, var in fischer_results[0:5]:
        print("det:", det)
        print("Times: ", times)
        print("Parameter Values: ", param)
        print("Variable Values: ", var)

    # Now make a plot for results
    fig1, ax1 = plt.subplots(N_best, sharex='all')
    for i, (det, times, param, var) in enumerate(fischer_results[0:N_best]):
        ax1[i].plot(times, times*0.0, marker="o", linestyle="", label="Det: {:e}\nN_t: {}".format(det, times.size))
        box = ax1[i].get_position()
        ax1[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1[i].set_xlabel("Time")
        ax1[i].set_yticklabels([])
    ax1[0].set_title("First "+ str(N_best) + " Winners")
    fig1.tight_layout()
    fig1.subplots_adjust(hspace=.0)
    fig1.savefig("Winners_current.png")
    fig1.clf()

    fig2, ax2 = plt.subplots(1)
    ax2.set_title("Determinant vs N_t")
    x = [f[1].size for f in fischer_results]
    y = [sorting_key(f) for f in fischer_results]
    ax2.plot(x, y, marker="x", linestyle="", color="k", label="Observable Value")
    ax2.set_xticks(x)
    ax2.set_xlabel("Number of time steps sampled")
    ax2.set_ylabel("Determinant Value")
    ax2.legend()
    fig2.savefig("Success_N_times.png")


def save_to_files(fischer_results, N_best=5):
    for i, (det, times, param, var) in enumerate(fischer_results[0:N_best]):
        np.save("result_{}_times".format(i), times)
        for j, p in enumerate(param):
            np.save("result_{}_param_{}".format(i, j), p)
        for j, v in enumerate(var):
            np.save("result_{}_var_{}".format(i, j), v)


def sorting_key(x):
    return x[0]


if __name__ == "__main__":
    # Choose literature values
    a_0 = 1.0
    b_0 = 2.0
    n_max_0 = 1000

    # This is just to calculate the differential at the point
    diff_low = 0.9
    diff_high = 1.1
    diff_delta = diff_high - diff_low
    d = np.array([diff_low, 1.0, diff_high])
    sample_params_a = d*a_0
    sample_params_b = d*b_0
    sample_n_max = d*n_max_0

    # Parameters which will later be derived in the Fischer Matrix
    # We assume that all these lists are of the form sample_params_a = [1, 2, 3]
    # With only 3 entries to calculate the derivative and the value itself
    params = [
        sample_params_a,
        sample_params_b,
        sample_n_max,
    ]

    # Define t-values
    t0 = 0.0
    tmax = 0.4

    # Just try for single temperature and initial value for n
    n0 = 20

    # Define sample space
    sample_temps = np.arange(20, 30, 1)

    # # Additional Variables (like Temperature) over which we want to iterate
    variables = [
        sample_temps
    ]

    # How many points of time should be sampled at maximum? (Starting at 1)
    N_t_min = 2
    N_t_max = 11
    N_t_step = 1
    # How many times should the random sampling take a sample of (eg. 3) time points
    N_t_mult = 20
    # Final result should look like this:
    # many_sample_times = [
    #   random_time_samples(len=1), random_time_samples(len=1), ...,        ==> N_t_mult times
    #   random_time_samples(len=2), random_time_samples(len=2), ...,        ==> N_t_mult times
    #   ...
    # ]
    many_sample_times = [np.sort(np.random.uniform(low=t0, high=tmax, size=n)) for n in range(N_t_min, N_t_max, N_t_step) for j in range(N_t_mult)]
    
    # How many optimization runs are we doing?
    N_opt = 30
    # How many best results are we picking
    N_best = 5
    # How many new results are we generating from the chosen ones
    N_new = 5
    # How much do we decrease the interval?
    diff_mod = 0.9

    # Do multiple runs to extract the best result and optimize it
    start_time = time.time()
    for n in range(N_opt):
        # Use multithreading to solve the equations and obtain results
        fischer_results = []
        with Pool(14) as p:
            fischer_results = p.starmap(calculate_Fischer_determinant, zip(many_sample_times, iter.repeat(params), iter.repeat(variables), iter.repeat(ODE_RHS)))
        
        # Delete old entries
        many_sample_times = []
        # We want to filter the list of results for each number of time-steps that we are solving
        for nt in range(N_t_min, N_t_max, N_t_step):
            for det, times, param, var in sorted(filter(lambda x: len(x[1]) == nt, fischer_results), key=sorting_key, reverse=True)[0:N_best]:
                # We create new time values by perturbing old ones
                times_new = np.zeros((len(times), N_new))
                
                # We calculate the lower and upper bound of the next iteration
                for i, t in enumerate(times):
                    # In this case we are in between
                    if i>0 and i<len(times)-1:
                        t_low=t - (t-times[i-1])*diff_mod**(n+1)
                        t_high=t + (times[i+1]-t)*diff_mod**(n+1)
                    # We hit the bottom --> calculate with t0
                    elif i==0:
                        t_low=t - (t-t0)*diff_mod**(n+1)
                        t_high=t + (times[i+1]-t)*diff_mod**(n+1)
                    # We hit the top --> calculate with tmax
                    elif i==len(times)-1:
                        t_low=t - (t-times[i-1])*diff_mod**(n+1)
                        t_high=t - (tmax-t)*diff_mod**(n+1)
                    # Now we sample the first time-step of the next series N_new times
                    # If we repeat this procedure, every time-step will be sampled
                    times_new[i,:] = np.random.uniform(low=t_low, high=t_high, size=N_new)

                # Now we have to slice the correct times from the newly created array
                many_sample_times += [np.sort(times_new[:,k]) for k in range(N_new)]
                # Also add the original one in case that this is better
                many_sample_times += [times]
        # Make nice output
        line = "[{: " + str(len(str(N_opt))) + "d}/{:d}, {: 8.3f}s] Optimization run done"
        print(line.format(n+1, N_opt, time.time()-start_time), end="\r")
    print(line.format(n+1, N_opt, time.time()-start_time), end="\r")

    make_nice_plots(fischer_results)
    save_to_files(fischer_results)
