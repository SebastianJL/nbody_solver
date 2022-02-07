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
    n = 1001
    eps = 0
    theta_max = 0.5
    show_plots = False
    save = True
    a = 0.07842896999259844  # Hernquist scaling length. See fit_mass_distribution.py.
    R_hm = 0.18934428303908363  # Half mass radius. See fit_mass_distribution.py.

    kinds = ['direct_rs', 'tree_mono_py', 'tree_quad_py']

    fig = plt.figure()
    plt.axvline(R_hm, 0, 1, color='grey')
    ax = plt.gca()
    plt.text(0.45, 0.9, '$R_{hm}$', transform=ax.transAxes, color='grey')

    in_file = Path(f'../output/acc_{kinds[0]}_n={n}_eps={eps}_.dat')
    ref_data = read_binary(in_file)
    ref_acc = ref_data[['ax', 'ay', 'az']].apply(np.square).sum(axis=1).apply(np.sqrt)
    print(f'{(ref_acc < 0).any()}')
    for kind in kinds[1:]:
        in_file = Path(f'../output/acc_{kind}_n={n}_eps={eps}_theta_max={theta_max:.1f}_.dat')
        data = read_binary(in_file)
        assert (data.shape[0] == n)

        r = data[['x', 'y', 'z']].apply(np.square).sum(axis=1).apply(np.sqrt)
        acc = data[['ax', 'ay', 'az']].apply(np.square).sum(axis=1).apply(np.sqrt)
        print(f'{(acc < 0).any()}')
        print(f'{(acc/ref_acc < 0).any()}')
        plt.semilogx(r, acc/ref_acc, '.', label=f'{kind}')

    plt.title(f'{n = }, {theta_max = :.1f}')
    plt.xlabel('radius $[L_0]$')
    plt.ylabel(r'acc / acc_direct')
    plt.legend()
    if save:
        out_file = Path(f'../output/relative_acc_plot_n={n}_eps={eps}_theta_max={theta_max}_.png')
        plt.savefig(out_file, dpi=300)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()
