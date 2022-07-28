#!/usr/bin/env python3

import numpy as np

# System of equation for pool-model and sensitivities
def pool_model_sensitivity(y, t, Q, P, Const):
    (a, b, c, n_max, n0) = P
    (Temp, measurement_type) = Q
    #(n0,) = Const
    (n, sa, sb, sc, snmax, sn0) = y
    return [
        (a*Temp + c) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max)  + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max)  + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max)  + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc,
        (a*Temp + c) * (n -        n0 * np.exp(-b*Temp*t))*(n/n_max**2) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * snmax,
        (a*Temp + c) * (  -             np.exp(-b*Temp*t))*(n/n_max**2) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sn0 
    ]


def jacobi(y, t, Q, P, Const):
    (a, b, c, n_max, n0) = P
    (Temp, measurement_type) = Q
    #(n0,) = Const
    (n, sa, sb, sc, snmax, sn0) = y
    dfdn = (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t))
    return np.array([
        [   dfdn,                                                                                              0,    0,    0,    0,    0   ],
        [(  Temp    ) * (1  - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sa, dfdn, 0,    0,    0,    0   ],
        [(a*Temp + c) * (   -  n0/n_max * t * Temp * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sb, 0,    dfdn, 0,    0,    0   ],
        [(     1    ) * (1  - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sc, 0,    0,    dfdn, 0,    0   ],
        [(a*Temp + c) * (-n + 2*n0*np.exp(-b*Temp*t))/n_max + (a*Temp + c) * (1 - 2 / n_max) * snmax,          0,    0,    0,    dfdn, 0   ],
        [(a*Temp + c) * (np.exp(-b*Temp*t)          )/n_max + (a*Temp + c) * (1 - 2 / n_max) * sn0,            0,    0,    0,    0,    dfdn] 
    ])