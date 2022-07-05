import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import itertools as iter

# Import custom functions
from src.solving import factorize_reduced
from src.optimization import get_best_fischer_results


def make_nice_plot(fischer_results, sorting_key):
    # Remember that entries in the fischer matrix have the form
    # fischer_results[0] = (obs, times, P, Q_arr, Const, Y0)
    fig, ax = plt.subplots()

    x = [np.array(f[0][1]).shape[-1] for f in fischer_results]
    y = [len(f[0][3][0]) for f in fischer_results]
    weights = [sorting_key([np.array(fi) for fi in f[0]]) for f in fischer_results]

    b = (
        np.arange(min(x)-0.5, max(x)+1.5, 1.0),
        np.arange(min(y)-0.5, max(y)+1.5, 1.0)
    )

    n_measurenents = [20, 40, 60, 80, 100, 120]
    x2 = np.linspace(1, 21, 21)
    y_of_eff = [[eff/xx for xx in x2] for eff in n_measurenents]

    ax.hist2d(x, y, bins=b, weights=weights, cmap="viridis")
    for y2 in y_of_eff:
        ax.plot(x2, y2, linewidth=2, color='r')
    ax.text(4.1, 5, 'M = 20', fontsize=13, color='r')
    ax.text(6, 7, '40', fontsize=13, color='r')
    ax.text(7.8, 8, '60', fontsize=13, color='r')
    ax.text(9.3, 8.9, '80', fontsize=13, color='r')
    ax.text(10.8, 9.5, '100', fontsize=13, color='r')
    ax.text(12, 10.3, '120', fontsize=13, color='r')
    ax.set_title("Weighted Final Results", fontsize=13)
    ax.set_xlabel("#Time Steps", fontsize=13)
    ax.set_ylabel("#Temperature Values", fontsize=13)
    fig.savefig("plots/pool_model-Time-Temperature-2D-Hist.png")
    fig.clf()


def make_convergence_plot(fischer_results, effort_low, effort_high, sorting_key, N_best):
    # Intermediate step to calcualte values of grid points
    best_grid = np.zeros(shape=(effort_high-effort_low+1, effort_high-effort_low+1))
    for n, m in iter.product(range(effort_high-effort_low+1), range(effort_high-effort_low+1)):
        fisses = get_best_fischer_results((effort_low + n, effort_low + m), fischer_results, sorting_key, N_best)
        # Reminder:
        # (obs, times, P, Q_arr, Const, Y0) = fisses[0]
        if len(fisses) > 0:
            best_grid[n,m] = np.average(np.array([f[0] for f in fisses]))
            # best_grid[n,m] = fisses[0][0]
    color_value = lambda n, k: best_grid[max(0, min(effort_high-effort_low, round(n-effort_low))), max(0, min(effort_high-effort_low, round(k/n)))]
    # Now plot lines for efforts
    fig, ax = plt.subplots()
    for k in range(effort_low, effort_high**2+1):
        x = np.array([f[0] for f in factorize_reduced(k)])
        x = x[x<=effort_high]
        x = x[k/x<=effort_high]
        if np.array(x).size >= 5:
            x_smooth = np.linspace(x.min(), x.max())
            y = k/x
            y_smooth = k/x_smooth
            cv = np.array([color_value(n, k) for n in x])
            if cv.max()-cv.min() > 0.0:
                size_values = 2 * (cv-cv.min())/(cv.max()-cv.min()) * mpl.rcParams['lines.markersize'] ** 2
                ax.scatter(x, y, marker="o", s=size_values, c=cv, cmap="viridis")
                ax.plot(x_smooth, y_smooth, c="k", linestyle=":", alpha=0.7)
    ax.set_title("Effort lines")
    ax.set_xlabel("#Time Measurements")
    ax.set_ylabel("#Temp Measurements")
    fig.savefig("plots/Effort_lines.png")
    fig.clf()


def make_plots(fisses, sorting_key):
    new_comb = sorted([(np.array(f[0][1]).shape[-1] * len(f[0][3][0]), sorting_key([np.array(fi) for fi in f[0]])) for f in fisses], key=lambda l:l[0])
    final_comb = []
    for i in range (0, len(new_comb)):
        if i == 0 or new_comb[i][0] != new_comb[i - 1][0]:
            final_comb.append(new_comb[i])
        else:
            final_comb[-1] = (new_comb[i][0], max(new_comb[i][1], new_comb[i - 1][1]))

    x = [f[0] for f in final_comb]
    y = [f[1] for f in final_comb]

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="X")
    # ax.set_yscale('log')
    ax.set_xlabel('# of measurements', fontsize=15)
    ax.set_ylabel('det(F)', fontsize=15)
    # ax.tick_params(fontsize=13)
    fig.savefig("plots/determinant_FIM_vs_num_measurements.png")
    fig.clf()

def make_plots_mean(fisses, sorting_key):
    new_comb = sorted([(np.array(f[0][1]).shape[-1] * len(f[0][3][0]), sorting_key([np.array(fi) for fi in f[0]])) for f in fisses], key=lambda l:l[0])
    final_comb = [] # effort, mean_det, std_err_det
    print(new_comb[-1][0])
    effort_list = set([c[0] for c in new_comb])
    for eff in effort_list:
        same_eff_comb = list(filter(lambda x: x[0]==eff, new_comb))
        final_comb.append([eff, np.mean(same_eff_comb, axis=0)[1], stats.sem(same_eff_comb, axis=0)[1]])

    x = [f[0] for f in final_comb]
    y = [f[1] for f in final_comb]
    y_std = [f[2] for f in final_comb]

    fig, ax = plt.subplots()
    # ax.scatter(x, y, marker="X")
    ax.errorbar(x, y, yerr=y_std, fmt = 'X')
    # ax.set_yscale('log')
    ax.set_xlabel('# of measurements', fontsize=15)
    ax.set_ylabel('det(F)', fontsize=15)
    # ax.tick_params(fontsize=13)
    fig.savefig("plots/determinant_FIM_vs_num_measurements2_mean.png")
    fig.clf()


def write_in_file(fisses, num_iter, crit_name, effort_max, sorting_key):
    P = fisses[0][0][2]
    Const = fisses[0][0][4]
    filename = f"Experimental_design_iter_{num_iter}_crit_{crit_name}_a_{P[0]:.3f}_b_{P[1]:.3f}_c_{P[2]:.3f}_n0_{Const[0]}_nmax_{Const[1]}"#_effmax_{effort_max}"
    path = 'results'
    filenamepath ='./' + path + '/' + filename + '.json'
    new_comb = sorted([(np.array(f[0][1]).shape[-1] * len(f[0][3][0]), sorting_key([np.array(fi) for fi in f[0]]), np.array(f[0][1]).shape[-1], len(f[0][3][0]), [list(ff) for ff in (f[0][1])], list(f[0][3][0])) for f in fisses], key=lambda l:l[0])
    with open(filenamepath, "w") as file:
        for c in new_comb:
            opt_design_dict = {'eff': c[0], 'obs': c[1], 'n_times': c[2], 'n_temp': c[3], 'times': c[4], 'temp': c[5]}
            json.dump(opt_design_dict, file, indent=1)
    file.close()