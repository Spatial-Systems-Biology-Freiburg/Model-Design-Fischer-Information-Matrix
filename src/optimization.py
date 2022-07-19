import numpy as np
import itertools as iter



def get_best_fischer_results(n_time_temp, fischer_results, sorting_key, N_best):
    (n_times, n_temp) = n_time_temp
    # TODO use partial sort or some other efficient alrogithm to obtain O(n) scaling behvaiour
    # for best result retrieval
    return sorted(filter(lambda x: x[1].shape[-1]==n_times and len(x[3][0])==n_temp, fischer_results), key=sorting_key, reverse=True)[:N_best]


def get_best_fischer_results2(fischer_results, sorting_key, N_best):
    return sorted(fischer_results, key=sorting_key, reverse=True)[:N_best]


def get_new_combinations_from_best(comb_old, N_spawn, temp_bnds, dtemp, times_bnds, dtimes):
    (temp_low, temp_high) = temp_bnds
    (times_low, times_high) = times_bnds
    combinations = []
    for (times, P, Q_arr, Const) in comb_old:
        # Also depend old result in case its better
        combinations.append((times, P, Q_arr, Const))
        # Now spawn new results via next neighbors of current results
        for _ in range(0, N_spawn):
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
            combinations.append((times_new, P, [temps_new], Const))
    return combinations

# Generate the initial combinations of times and temperatures
def set_multistart_combinations(n_times, n_temp, times_total, temp_total, P, Const, method='discr'):
    if method == 'discr':
        temperatures = np.random.choice(temp_total, int(n_temp), replace=False)
        times = np.array([np.sort(np.random.choice(times_total, int(n_times), replace=False)) for _ in range(n_temp)])
    if method == 'cont':
        times = np.array([np.sort(np.random.uniform(np.min(times_total), np.max(times_total), size=(n_times))) for _ in range(n_temp)])
        temperatures = np.random.uniform(np.min(temp_total), np.max(temp_total), size=(n_temp))
    return (times, P, [temperatures], Const)

# One optimization iteration using discrete random search
def discrete_random(combination, func_FIM_calc, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes):
    # From 1 combination given choose N_spawn new ones 
    combinations = get_new_combinations_from_best(combination, N_spawn, temp_bnds, dtemp, times_bnds, dtimes)
    # calc fisher result for all new combinations 
    fischer_results = []
    for comb in combinations:
        fischer_results.append(func_FIM_calc(combinations=comb))
    # Choose best solutions according to optimization_method 
    fisses = get_best_fischer_results2(fischer_results, sorting_key, N_best) # e.g. get_best_fischer_results    
    return fisses

# Calculate the gradient of the Fisher observable needed for gradient descent method                                      
def gradient(combination_low, combination_high, dval, func_FIM_calc):
    fisher_result_T_low = func_FIM_calc(combinations=combination_low)
    fisher_result_T_high = func_FIM_calc(combinations=combination_high)
    if dval != 0:
        grad = (fisher_result_T_high[0] - fisher_result_T_low[0]) / 2 * dval
    else:
        grad = 0
    return grad

# One optimization iteration using continuous gradient descent approach
def gradient_descent(combination, func_FIM_calc, N_spawn, N_best, temp_bnds, dtemp, times_bnds, dtimes):
    fisses = []
    for comb in combination:
        fisher_result_old = func_FIM_calc(combinations=comb)
        (obs, times, P, Q_arr, Const, Y0) = fisher_result_old
        learn_rate = 1 / obs # ??? mb change this
        (n_temp, n_times) = (len(Q_arr[0]), times.shape[-1])
        (temps, *variables_other) = Q_arr
        times_new = times
        temps_new = temps
        # if we want to calc grad along every temperature
        # for (ind_temp, *index_other) in iter.product(*[range(len(q)) for q in Q_arr]): 
        # randomly choose N_spawn temperatures along which the gradient is calculated
        for (ind_temp, *ind_other) in iter.product(np.random.choice(np.linspace(0, n_temp-1, n_temp, dtype=int), min(N_spawn, len(temps)), replace=False), *[range(len(q)) for q in variables_other]):
            dT = min(dtemp, 0.5*(temps[ind_temp] - temp_bnds[0]), 0.5*(temp_bnds[1] - temps[ind_temp]))
            grad_T = gradient(
                (times, P, [np.array([temps[i] - dT if i==ind_temp else temps[i] for i in range (len(temps))])], Const), 
                (times, P, [np.array([temps[i] + dT if i==ind_temp else temps[i] for i in range (len(temps))])], Const), 
                dT, 
                func_FIM_calc
            )
            temps_new[ind_temp] = min(max(temps[ind_temp] + learn_rate*grad_T, temp_bnds[0]), temp_bnds[1])
            #for ind_time in range (times.shape[-1]):  
            # randomly choose N_spawn times along which the gradient is calculated
            for ind_time in np.random.choice(np.linspace(0, n_times-1, n_times, dtype=int), min(N_spawn, times.shape[-1]), replace=False):        
                if ind_time == 0:
                    dt = min(dtimes, 0.5*(times[ind_temp][ind_time] - times_bnds[0]), 0.5*(times[ind_temp][ind_time+1] - times[ind_temp][ind_time]))
                elif ind_time == times.shape[-1] - 1:
                    dt = min(dtimes, 0.5*(times_bnds[1] - times[ind_temp][ind_time]), 0.5*(times[ind_temp][ind_time] - times[ind_temp][ind_time-1]))
                else:
                    dt = min(dtimes,  0.5*(times[ind_temp][ind_time+1] - times[ind_temp][ind_time]), 0.5*(times[ind_temp][ind_time] - times[ind_temp][ind_time-1]))
                grad_t = gradient(
                    (np.array([np.sort(np.array([times[i][j] - dt if i==ind_temp and j==ind_time else times[i][j] for j in range (times.shape[-1])])) for i in range (times.shape[0])]), P, Q_arr, Const),
                    (np.array([np.sort(np.array([times[i][j] + dt if i==ind_temp and j==ind_time else times[i][j] for j in range (times.shape[-1])])) for i in range (times.shape[0])]), P, Q_arr, Const), 
                    dt, 
                    func_FIM_calc
                )
                times_new[ind_temp][ind_time] = min(max(times[ind_temp][ind_time] + learn_rate*grad_t, times_bnds[0]), times_bnds[1])
        Q_arr_new = [temps_new, *variables_other]
        times_new = np.array([np.sort(time) for time in times_new])
        fisses.append(func_FIM_calc(combinations=(times_new, P, Q_arr_new, Const)))
    return fisses



    


