from pathlib import Path
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt


def read_binary(path: Path):
    data = np.fromfile(path, dtype=np.float64)
    cols = ['m', 'x', 'y', 'z', 'ax', 'ay', 'az']
    n_cols = len(cols)
    data = data.reshape(data.size // n_cols, n_cols)
    df = pd.DataFrame(data, columns=cols)
    return df


def main():
    n = 1001
    eps = 0
    theta_max_list = np.arange(0.1, 1.31, 0.1)
    show_plots = True
    save = True

    delta_acc_mono = []
    delta_acc_err_mono = []
    delta_acc_quad = []
    delta_acc_err_quad = []
    for theta_max in theta_max_list:
        kind = 'direct_rs'
        in_file = Path(f'../output/acc_{kind}_n={n}_eps={eps}_.dat')
        data_direct = read_binary(in_file)
        a_direct = data_direct[['ax', 'ay', 'az']].apply(np.linalg.norm, axis=1)

        kind = 'tree_mono_py'
        in_file = Path(f'../output/acc_{kind}_n={n}_eps={eps}_theta_max={theta_max:.1f}_.dat')
        data_mono = read_binary(in_file)
        a_mono = data_mono[['ax', 'ay', 'az']].apply(np.linalg.norm, axis=1)
        a_rel_mono = np.abs((a_mono-a_direct) / a_direct).mean()
        a_rel_error_mono = np.abs((a_mono-a_direct) / a_direct).sem()
        delta_acc_mono.append(a_rel_mono)
        delta_acc_err_mono.append(a_rel_error_mono)

        kind = 'tree_quad_py'
        in_file = Path(f'../output/acc_{kind}_n={n}_eps={eps}_theta_max={theta_max:.1f}_.dat')
        data_quad = read_binary(in_file)
        a_quad = data_quad[['ax', 'ay', 'az']].apply(np.linalg.norm, axis=1)
        a_rel_quad = np.abs((a_quad-a_direct) / a_direct).mean()
        a_rel_error_quad = np.abs((a_quad-a_direct) / a_direct).sem()
        delta_acc_quad.append(a_rel_quad)
        delta_acc_err_quad.append(a_rel_error_quad)

    plt.axhline(delta_acc_mono[4], ls='--', color='black', alpha=0.5)
    plt.errorbar(theta_max_list, delta_acc_mono, yerr=delta_acc_err_mono, capsize=2, fmt='.', label='monopole')
    plt.errorbar(theta_max_list, delta_acc_quad, yerr=delta_acc_err_quad, capsize=2, fmt='.', label='quadrupole')
    plt.xlabel(r'$\theta_{max}$ [rad]')
    plt.ylabel(r'$\Delta a / a$')
    plt.title(f'{n = }, $\\epsilon = {eps}$')

    plt.legend()

    if save:
        out_file = Path(f'../output/accuracy_plot_py_varying_theta_n={n}_eps={eps}_.png')
        plt.savefig(out_file, dpi=300)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
