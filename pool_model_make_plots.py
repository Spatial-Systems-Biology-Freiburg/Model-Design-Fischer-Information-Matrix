#!/usr/bin/env python3

from pool_model_plots import make_nice_plot, make_convergence_plot, make_plots, make_plots_mean
from src.database import get_fischer_results_from_collection
from pool_model import sorting_key


if __name__ == "__main__":
    collection = "2022/07/04-19:05:22_pool_model_random_grid_determinant_div_m"
    
    fischer_results = get_fischer_results_from_collection(collection)

    make_nice_plot(fischer_results, sorting_key)
    # TODO currently not working due to casting from fischer_results
    # make_convergence_plot(fischer_results, effort_low=2, effort_high=11, sorting_key=sorting_key, N_best=5)
    make_plots(fischer_results, sorting_key)
    # write_in_file(fisses, 1, 'D', effort_max, sorting_key)
    make_plots_mean(fischer_results, sorting_key)