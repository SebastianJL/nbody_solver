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
    return M / (r+a)**2


def main():
    path = Path('../output/accelerations.dat')

    data = read_binary(path)
    assert (data.shape[0] == 50010)

    a = 0.07842896999259844  # Hernquist scaling length. See fit_mass_distribution.py.
    R_hm = 0.18934428303908363  # Half mass radius. See fit_mass_distribution.py.
    r = data[['x', 'y', 'z']].apply(np.square).sum(axis=1).apply(np.sqrt)
    acc = data[['ax', 'ay', 'az']].apply(np.square).sum(axis=1).apply(np.sqrt)
    M = data['m'].sum()
    r_model = np.linspace(0, R_hm, int(1e3))
    acc_model = acc_hernquist(r_model, M, a)

    print(data.iloc[1])

    plt.plot(r, acc, '.', label='direct summation')
    plt.plot(r_model, acc_model, label='Hernquist')
    plt.xlabel('radius $[L_0]$')
    plt.ylabel(r'acceleration $[L_0/T_0^2]$')
    plt.xlim(0, R_hm)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
