#!/usr/bin/env python3

#from pool_model_plots import make_nice_plot, make_convergence_plot, make_plots, make_plots_mean, plot_solution_with_exp_design_choice, plot_solution_with_exp_design_choice2, find_optimal_design_O2N2
from pool_model_plots2 import make_plots_mean, plot_solution_with_exp_design_choice, plot_solution_with_exp_design_choice2
from src.database import get_fischer_results_from_collection
#from models.pool_model import sorting_key, pool_model_sensitivity
from models.pool_model_with_nmax import pool_model_sensitivity


if __name__ == "__main__":
    
# collection w/o covariance error matrix 
    #collection = "2022/07/06-15:18:06_pool_model_random_grid_determinant_div_m"

    fischer_results = get_fischer_results_from_collection(collection)

    make_nice_plot(fischer_results, sorting_key)
    
    # make_convergence_plot(fischer_results, effort_low=2, effort_high=11, sorting_key=sorting_key, N_best=5)
    make_plots(fischer_results, sorting_key)
    # write_in_file(fisses, 1, 'D', effort_max, sorting_key)
    make_plots_mean(fischer_results, sorting_key)
    # Plot N_best experimental designs
    N_best = 3
    plot_solution_with_exp_design_choice([5, 3], fischer_results, sorting_key, N_best, pool_model_sensitivity)
    fischer_results = get_fischer_results_from_collection(collection)
    caption = ['PC', 'O2']
    bacteria = f' Aerobe Keimzahl ({caption[0]}) ({caption[1]})'
    N_best = 2
    
    n_temp_with_fixed_times = 1
    make_plots_mean(fischer_results, bacteria, caption, n_temp_with_fixed_times)

    plot_solution_with_exp_design_choice([2, 2], fischer_results, N_best, pool_model_sensitivity, [0.1, 1.0], bacteria, caption, n_temp_with_fixed_times)
