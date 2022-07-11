import numpy as np
import multiprocessing as mp
import time
import itertools as it

from src.solving import calculate_Fischer_observable


def get_best_fischer_results(n_time_temp, fischer_results, observable, N_best):
    (n_times, n_temp) = n_time_temp
    # TODO use partial sort or some other efficient alrogithm to obtain O(n) scaling behvaiour
    # for best result retrieval
    return sorted(filter(lambda x: x[1].shape[-1]==n_times and len(x[3][0])==n_temp, fischer_results), key=lambda x: x[0], reverse=True)[:N_best]


def get_new_combinations_from_best(best, N_spawn, temp_low, temp_high, dtemp, times_low, times_high, dtimes):
    combinations = []
    for (det, times, P, Q_arr, Const, Y0) in best:
        # Also depend old result in case its better
        combinations.append((times, Q_arr, P, Const))
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
            combinations.append((times_new, [temps_new], P, Const))
    return combinations


def RaDi(
    N_parallel,
    N_opt,
    N_spawn,
    N_best,
    n_times_max,
    n_temp_max,
    effort_low,
    effort_high,
    temp_low,
    temp_high,
    dtemp,
    times_low,
    times_high,
    dtimes,
    combinations,
    ode,
    y0_t0,
    jacobi,
    observable
):
    '''Random Discrete Optimization.
    '''
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
        fischer_results = p.starmap(calculate_Fischer_observable, zip(
            combinations,
            it.repeat(ode),
            it.repeat(y0_t0),
            it.repeat(jacobi),
            it.repeat(observable)
        ))

        # Do not optimize further if we are in the last run
        if opt_run != N_opt-1:
            # Delete old combinations
            combinations.clear()
            # Choose the N_best results with the largest objective function value from fischer_results
            fisses = p.starmap(get_best_fischer_results, zip(
                    it.product(range(effort_low, effort_high), range(effort_low, effort_high)),
                    it.repeat(fischer_results),
                    it.repeat(observable),
                    it.repeat(N_best)
            ), chunksize=100)
            # Calculate new combinations parallelized
            combinations = p.starmap(get_new_combinations_from_best, zip(
                fisses,
                it.repeat(N_spawn),
                it.repeat(temp_low),
                it.repeat(temp_high),
                it.repeat(dtemp),
                it.repeat(times_low),
                it.repeat(times_high),
                it.repeat(dtimes)
            ))
            combinations = [x for comb_list in combinations for x in comb_list]
    # Choose 1 best result for each (n_times, n_temp) combination
    fisses = p.starmap(get_best_fischer_results, zip(
        it.product(range(effort_low, effort_high), range(effort_low, effort_high)),
        it.repeat(fischer_results),
        it.repeat(observable),
        it.repeat(1)
    ), chunksize=100)
    print(print_line.format(time.time()-start_time, opt_run+1), "done")

    return fisses