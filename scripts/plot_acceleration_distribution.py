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


def acc_hernquist(r, M, a):
    """

    Args:
        r: radius.
        M: total mass of the system.
        a: scaling length according to hernquist model.
    """
    return M / (r + a) ** 2


def plot(path: Path, n, eps, save=False):
    in_file = path

    data = read_binary(in_file)
    assert (data.shape[0] == n)

    a = 0.07842896999259844  # Hernquist scaling length. See fit_mass_distribution.py.
    R_hm = 0.18934428303908363  # Half mass radius. See fit_mass_distribution.py.
    r = data[['x', 'y', 'z']].apply(np.square).sum(axis=1).apply(np.sqrt)
    acc = data[['ax', 'ay', 'az']].apply(np.square).sum(axis=1).apply(np.sqrt)
    M = data['m'].sum()
    r_min, r_max = 0, r.max()
    acc_min, acc_max = 0, acc_hernquist(r_min, M, a)
    r_model = np.linspace(r_min, r_max, int(1e5))
    acc_model = acc_hernquist(r_model, M, a)

    fig = plt.figure()
    plt.loglog(r, acc, '.', label='direct summation')
    plt.loglog(r_model, acc_model, label='M / (r+a)**2,')
    plt.title(f'{n = }, $\\epsilon = {eps}$, {a = :g}')
    plt.xlabel('radius $[L_0]$')
    plt.ylabel(r'acceleration $[L_0/T_0^2]$')
    plt.legend()

    if save:
        out_file = Path(f'../output/accelerations_plot_n={n}_eps={eps}_.png')
        plt.savefig(out_file, dpi=300)


def main():
    n = 1001
    eps = 0  # no softening.
    show_plots = False
    save = True
    plot(Path(f'../output/accelerations_n={n}_eps={eps}_.dat'), n, eps, save=save)

    eps = 0.027106022  # mean inter-particle separation (for 50_010 particle dataset).
    plot(Path(f'../output/accelerations_n={n}_eps={eps}_.dat'), n, eps, save)

    eps = 0.013553011
    plot(Path(f'../output/accelerations_n={n}_eps={eps}_.dat'), n, eps, save)

    eps = 0.0027106022
    plot(Path(f'../output/accelerations_n={n}_eps={eps}_.dat'), n, eps, save)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
