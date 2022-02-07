from pathlib import Path
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt


def main():
    n = 1001
    eps = 0
    show_plots = True
    save = True

    in_file = Path(f'../output/timings_py_n={n}_{eps=}_theta_max=[0.1,1.3]_.txt')
    pd.set_option('max_columns', None)
    data = pd.read_csv(in_file)
    mean = data.groupby('theta_max').mean()
    err = data.groupby('theta_max').sem()
    mean_mono, mean_quad, mean_direct = mean['mono'], mean['quad'], data['direct'].mean()
    err_mono, err_quad, err_direct = err['mono'], err['quad'], data['direct'].sem()

    plt.axhline(mean_direct, ls='--', color='black', label=f'direct summation $\pm$ {err_direct:.2f}s')
    plt.errorbar(mean_mono.index, mean_mono, yerr=err_mono, capsize=2, fmt='.', label='monopole')
    plt.errorbar(mean_quad.index, mean_quad, yerr=err_quad, capsize=2, fmt='.', label='quadrupole')
    plt.xlabel(r'$\theta_{max}$ [rad]')
    plt.ylabel(r'runtime [s]')
    plt.legend()
    repetitions = int(len(data)/len(mean))
    plt.title(f'{n = }, $\\epsilon = {eps}, rep={repetitions}$')

    if save:
        out_file = Path(f'../output/timings_plot_py_varying_theta_n={n}_eps={eps}_.png')
        plt.savefig(out_file, dpi=300)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
