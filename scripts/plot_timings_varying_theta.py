from pathlib import Path
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt


def main():
    n = 10
    eps = 0
    show_plots = True
    save = False
    a = 0.07842896999259844  # Hernquist scaling length. See fit_mass_distribution.py.
    R_hm = 0.18934428303908363  # Half mass radius. See fit_mass_distribution.py.

    in_file = Path(f'../output/timings_py_n={n}_{eps=}_theta_max=[0.1,1.3]_.txt')
    pd.set_option('max_columns', None)
    data = pd.read_csv(in_file)
    mean = data.groupby('theta_max').mean()
    err = data.groupby('theta_max').sem()
    mean_mono, mean_quad, mean_direct = mean['mono'], mean['quad'], mean['direct']
    err_mono, err_quad, err_direct = err['mono'], err['quad'], err['direct']

    plt.errorbar(mean_mono.index, mean_mono, yerr=err_mono, fmt='.', label='mono')
    plt.errorbar(mean_quad.index, mean_quad, yerr=err_quad, fmt='.', label='quad')
    plt.errorbar(mean_direct.index, mean_direct, yerr=err_direct, fmt='.', label='direct')
    plt.xlabel(r'$\theta_{max}$ [rad]')
    plt.ylabel(r'runtime [s]')
    plt.legend()
    repetitions = len(data)/len(mean)
    plt.title(f'{n = }, $\\epsilon = {eps}, rep={repetitions}$')

    if save:
        out_file = Path(f'../output/timings_plot_py_varying_theta_n={n}_eps={eps}_.png')
        plt.savefig(out_file, dpi=300)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
