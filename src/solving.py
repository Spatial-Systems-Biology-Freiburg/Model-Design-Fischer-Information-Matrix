import numpy as np
from scipy.integrate import odeint
import itertools as iter
import multiprocessing as mp
import time

def factorize_reduced(M):
    res = []
    for i in range(2, M):
        if (M % i == 0):
            res.append((i, round(M/i)))
    return res

                                                                        #  err = relat_err/ abs_err/ cov_matrix or (abs_err, relat_err) for diff methods
def get_S_matrix(ODE_func, y0_t0, times, Q_arr, P_arr, Const_arr, jacobian=None, method='wo_error', err=None):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    (y0_arr, t0) = y0_t0

    S = []
    n_solution = []

    # Iterate over all combinations of Q-Values
    for index in iter.product(*[range(len(q)) for q in Q_arr]):
        # Store the results of the respective ODE solution
        (ind_temp, ind_meas_type, *ind_other) = index
        Q = [Q_arr[i][j] for i, j in enumerate(index)]
        t = times[ind_temp]
        P = P_arr[ind_meas_type]
        Const = Const_arr[ind_meas_type]
        y0 = y0_arr[ind_meas_type]

        # Actually solve the ODE for the selected parameter values
        r = odeint(ODE_func, y0, np.sort(np.insert(t, 0, t0)), args=(Q, P, Const), Dfun=jacobian).T[:, 1:]
       
        # Calculate the S-Matrix with the supplied jacobian
        S.append(r[1:])
        n_solution.append(r[0])

    # Reshape to 2D Form (len(P),:)
    S = np.concatenate(S, axis=1)
    n_solution = np.concatenate(n_solution, axis=0)
    # Calculate covariance matrix
    if method == 'manual':
        cov_matrix = err
    else:
        cov_matrix = np.eye(len(n_solution), len(n_solution))
        if method == 'relative_error':
            error_n =  n_solution * err
        elif method == 'absolute_error':
            error_n =  err
        elif method == 'combined_error':
            error_n =  err[0] + n_solution * err[1]
    cov_matrix = np.eye(len(error_n), len(error_n)) * error_n**2
    C = np.linalg.inv(cov_matrix)
    return S, C



def convert_S_matrix_to_determinant(times, Q_arr, P, Const, S, C):
    # Calculate Fisher Matrix
    F = (S.dot(C)).dot(S.T)

    # Calculate Determinant
    det = np.linalg.det(F)
    return det


def convert_S_matrix_to_sumeigenval(times, Q_arr, P, Const, S, C):
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    sumeigval = np.sum(np.linalg.eigvals(F))
    return sumeigval

def convert_S_matrix_to_mineigenval(times, Q_arr, P, Const, S, C):
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    mineigval = np.min(np.linalg.eigvals(F))
    return mineigval


def convert_S_matrix_to_eigval_ratio(times, Q_arr, P, Const, S, C):
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    ratio_eigval = np.min(np.linalg.eigvals(F)) / np.max(np.linalg.eigvals(F))
    return ratio_eigval


def calculate_Fischer_observable(combinations, ODE_func, Y0, jacobian, observable, method='wo_error', err=None):
    # methods = 'wo_error', 'relative_error', 'absolute_error', 'combined_error'
    times, P, Q_arr, Const = combinations
    if method != 'wo_error' and err == None:
        print('Error: Argument "err" should be provided')     
    S, C = get_S_matrix(ODE_func, Y0, times, Q_arr, P, Const, jacobian, method, err)
    obs = observable(times, Q_arr, P, Const, S, C)

    (temps, *other_variables) = Q_arr
    # Penalty function (not allowing the same choice of measurement=(time, temp)) and norm:
    norm = 1.0
    seperate_measurements = 1.0
    for t in times:
        for t1, t2 in iter.combinations(t, 2):
            if np.abs(t1 - t2) <= 0.05:
                seperate_measurements = 0.0
    for T1, T2 in iter.combinations(temps, 2):    
        if np.abs(T1 - T2) <= 0.05:
            seperate_measurements = 0.0   
    obs = obs * seperate_measurements / norm
    
    return obs, times, P, Q_arr, Const, Y0
