#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools as iter


def ODE_RHS(n, t, a, b, n_max, temp):
    return (a*temp + b) * n * (1 - n/n_max)


def generate_result(ODE_func, n0, t, params, vars):
    (a, b, n_max) = params
    (temp,) = vars
    return odeint(ODE_func, n0, t, args=(a, b, n_max, temp))


def obtain_results(ODE_func, params, variables, sample_times):
    results = np.zeros(list(x.size for x in params + variables) + [sample_times.size])

    for index_param in iter.product(*[range(x.size) for x in params]):
        for index_var in iter.product(*[range(y.size) for y in variables]):
            p = list(params[i][index_param[i]] for i in range(len(params)))
            v = list(variables[j][index_var[j]] for j in range(len(variables)))
            
            results[index_param][index_var] = generate_result(ODE_func, n0, sample_times, p, v).reshape(sample_times.size)
    return results


def observable(sample_times, ode_results):
    return ode_results


def get_large_S_matrix(params, variables, results):
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


if __name__ == "__main__":
    # Choose literature values
    a_0 = 1.0
    b_0 = 2.0
    n_max_0 = 1000

    # Define t-values
    t0 = 0.0
    tmax = 0.4
    dt = 0.05

    # Just try for single temperature and initial value for n
    n0 = 20

    # Define sample space
    sample_temps = np.arange(20, 30, 2.5)
    sample_times = np.arange(t0, tmax, dt)

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

    # # Additional Variables (like Temperature) over which we want to iterate
    variables = [
        sample_temps
    ]

    results = obtain_results(ODE_RHS, params, variables, sample_times)

    S = get_large_S_matrix(params, variables, results)
    
    S_trans = S.reshape(len(params), sum(y.size for y in variables) * sample_times.size)
    
    # Calculate Fisher Matrix
    F = S_trans.dot(S_trans.T)
    
    # Calculate Eigenvalues and determinant
    w, v = np.linalg.eig(F)
    det = np.linalg.det(F)

    print(det)