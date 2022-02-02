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


def main():
    n = 50_010
    kind = 'direct_rs'
    show_plots = False
    save = True
    a = 0.07842896999259844  # Hernquist scaling length. See fit_mass_distribution.py.
    R_hm = 0.18934428303908363  # Half mass radius. See fit_mass_distribution.py.

    eps_values = [
        0,              # 0*r_mean
        0.0027106022,   # 0.1*r_mean
        0.013553011,    # 0.5*r_mean
        0.027106022,    # 1*r_mean
        0.040659033,    # 1.5*r_mean
        0.054212045,    # 2*r_mean
    ]
    multipliers = [0, 0.1, 0.5, 1, 1.5, 2]

    fig = plt.figure()
    plt.axvline(R_hm, 0, 1, color='grey')
    ax = plt.gca()
    plt.text(0.45, 0.9, '$R_{hm}$', transform=ax.transAxes, color='grey')

    for eps, mul in zip(eps_values, multipliers):
        in_file = Path(f'../output/acc_{kind}_n={n}_eps={eps}_.dat')
        data = read_binary(in_file)
        assert (data.shape[0] == n)

        r = data[['x', 'y', 'z']].apply(np.square).sum(axis=1).apply(np.sqrt)
        acc = data[['ax', 'ay', 'az']].apply(np.square).sum(axis=1).apply(np.sqrt)
        M = data['m'].sum()
        r_min, r_max = 0, r.max()

        plt.loglog(r, acc, '.', label=f'$\\epsilon = {eps:0.4f} = = {mul} \cdot r_{{mean}}$')

    r_model = np.linspace(r_min, r_max, int(1e5))
    acc_model = acc_hernquist(r_model, M, a)
    plt.loglog(r_model, acc_model, label='M / (r+a)^2,')


    plt.title(f'{n = }, {a = :g}')
    plt.xlabel('radius $[L_0]$')
    plt.ylabel(r'acceleration $[L_0/T_0^2]$')
    plt.legend()
    if save:
        out_file = Path(f'../output/acc_{kind}_plot_n={n}_eps=variable_.png')
        plt.savefig(out_file, dpi=300)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
