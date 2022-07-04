import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import itertools as iter
import multiprocessing as mp
import time
from functools import partial


def factorize_reduced(M):
    res = []
    for i in range(2, M):
        if (M % i == 0):
            res.append((i, round(M/i)))
    return res


def get_S_matrix(ODE_func, y0, times, Q_arr, P, Const, jacobian):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    S = np.zeros((len(P),) + (times.shape[-1],) + tuple(len(x) for x in Q_arr))

    # Iterate over all combinations of Q-Values
    for index in iter.product(*[range(len(q)) for q in Q_arr]):
        # Store the results of the respective ODE solution
        Q = [Q_arr[i][j] for i, j in enumerate(index)]
        t = times[index]

        # Actually solve the ODE for the selected parameter values
        r = odeint(ODE_func, y0, t, args=(Q, P, Const), Dfun=jacobian).T[1:,:]

        # Calculate the S-Matrix with the supplied jacobian
        S[(slice(None), slice(None)) + index] = r

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


def calculate_Fischer_observable(combinations, ODE_func, Y0, jacobian, observable):
    times, Q_arr, P, Const = combinations
    S = get_S_matrix(ODE_func, Y0, times, Q_arr, P, Const, jacobian)
    obs = observable(S)
    return obs, times, P, Q_arr, Const, Y0

