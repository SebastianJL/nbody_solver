from pathlib import Path
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt


def main():
    show_plots = True
    save = True
    eps = 0

    in_file = Path(f'../output/timings_py_n=[10, 4168]_eps={eps}_theta_max=0.5+0.9_.txt')
    pd.set_option('max_columns', None)
    data = pd.read_csv(in_file)
    mean = data.groupby('n').mean()
    err = data.groupby('n').sem()
    mean_mono, mean_quad, mean_direct = mean['mono'], mean['quad'], mean['direct']
    err_mono, err_quad, err_direct = err['mono'], err['quad'], mean['direct']

    plt.errorbar(mean_direct.index, mean_direct, yerr=err_direct, capsize=2, fmt='.', label=r'direct summation')
    plt.errorbar(mean_mono.index, mean_mono, yerr=err_mono, capsize=2, fmt='.', label=r'monopole, $\theta_{max} = 0.5$')
    plt.errorbar(mean_quad.index, mean_quad, yerr=err_quad, capsize=2, fmt='.', label=r'quadrupole, $\theta_{max} = 0.9$')
    plt.xlabel('N')
    plt.ylabel(r'runtime [s]')
    plt.legend()
    repetitions = int(len(data)/len(mean))
    plt.title(f'$\\epsilon = {eps}, rep={repetitions}$')

    if save:
        out_file = Path(f'../output/timings_plot_py_varying_n_eps={eps}_.png')
        plt.savefig(out_file, dpi=300)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
