import numpy as np


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