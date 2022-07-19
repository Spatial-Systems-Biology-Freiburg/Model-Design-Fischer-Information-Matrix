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
def get_S_matrix(ODE_func, y0_t0, times, Q_arr, P, Const, jacobian=None, method='wo_error', err=None):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    (y0, t0) = y0_t0
    S = np.zeros((len(P),) + (times.shape[-1],) + tuple(len(x) for x in Q_arr))
    n_solution = np.zeros((times.shape[-1],) + tuple(len(x) for x in Q_arr))

    # Iterate over all combinations of Q-Values
    for index in iter.product(*[range(len(q)) for q in Q_arr]):
        # Store the results of the respective ODE solution
        Q = [Q_arr[i][j] for i, j in enumerate(index)]
        t = times[index]

        # Actually solve the ODE for the selected parameter values
        #r = solve_ivp(ODE_func, [t0, t.max()], y0, method='Radau', t_eval=t,  args=(Q, P, Const), jac=jacobian).y.T[1:,:]
        r = odeint(ODE_func, y0, np.insert(t, 0, t0), args=(Q, P, Const), Dfun=jacobian).T[:, 1:]

        # Calculate the S-Matrix with the supplied jacobian
        S[(slice(None), slice(None)) + index] = r[1:]
        n_solution[:, index] = r[0].reshape(times.shape[-1], 1)

    # Reshape to 2D Form (len(P),:)
    S = S.reshape((len(P),np.prod(S.shape[1:])))
    # Calculate covariance matrix
    if method == 'manual':
        cov_matrix = err
    else:
        cov_matrix = np.eye(times.size, times.size)
        error_n = np.zeros((times.shape[-1],) + tuple(len(x) for x in Q_arr))
        for index in iter.product(*[range(len(q)) for q in Q_arr]):
            if method == 'relative_error':
                error_n[:, index] =  n_solution[:, index] * err
            elif method == 'absolute_error':
                error_n[:, index] =  err
            elif method == 'combined_error':
                error_n[:, index] =  err[0] + n_solution[:, index] * err[1]
        error_n = error_n.reshape(np.prod(error_n.shape))
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


def calculate_Fischer_observable(combinations, ODE_func, Y0, jacobian, observable, method='wo_error', err=None):
    # methods = 'wo_error', 'relative_error', 'absolute_error', 'combined_error'
    times, P, Q_arr, Const = combinations
    if method != 'wo_error' and err == None:
        print('Error: Argument "err" should be provided')
    S, C = get_S_matrix(ODE_func, Y0, times, Q_arr, P, Const, jacobian, method, err)
    obs = observable(times, Q_arr, P, Const, S, C)
    return obs, times, P, Q_arr, Const, Y0
