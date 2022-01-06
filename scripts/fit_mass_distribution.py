from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def log_bins(min: 'numeric', max: 'numeric', n: int) -> np.array:
    """Creates logarithmic bin edges for n bins including the rightmost edge.

    The edges (e_0, e_1, ..., e_n) are scaled such that they have an equal distance in log space, i.e.
        ln(e_{k+1}) - ln(e_k) = d
    where d*n = ln(e_n) - ln(e_0).

    Args:
        min: Smallest bin edge e_0.
        max: Largest bin edge e_n.
        n: Number of bins.
    Returns: Bin edges.
    """
    scale = np.power(max / min, 1 / n)
    bin_edges = np.zeros(n + 1)
    bin_edges[0] = min
    for i in range(1, n + 1):
        bin_edges[i] = scale * bin_edges[i - 1]
    return bin_edges


def number_of_particles_in_shell(rmin: float, rmax: float, a: float, N: int) -> float:
    """Number of particles in a spherical shell from rmin to rmax according to the Hernquist model.

    The cumulative mass distribution is
    M(r) = M * r**2 / (r+a)**2
    where M is the total mass of the system and a is a scaling parameter.

    All particles are assumed to have the same mass m, therefore M = N*m, and it follows that
    N(r) = N * r**2 / (r+a)**2 .

    Args:
        rmin: Minimal radius.
        rmax: Maximal radius.
        a: Hernquist scaling parameter.
        N: Total number of particles of the system.

    Returns: N(rmax) - N(rmin)
    """
    return N * (rmax ** 2 / (rmax + a) ** 2 - rmin ** 2 / (rmin + a) ** 2)


def residual(N: int, bin_edges: np.array, a: np.array) -> np.array:
    """Residual to be optimized with scipy.optimize.curvefit.

    Args:
        bin_edges: Right and left inclusive edges of the bins.
        a: Hernquist scaling parameter.
        N: Total number of particles of the system.

    Returns: Estimated number of particles per bin.
    """
    rmin, rmax = bin_edges[:-1], bin_edges[1:]
    n_per_bin = number_of_particles_in_shell(rmin, rmax, a, N)
    return n_per_bin


if __name__ == '__main__':
    # Read data.
    header = 'ID, Masses, x, y, z, Vx, Vy, Vz, softening, potential'.split(', ')
    data = pd.read_csv('data/data.txt', delimiter='\t', header=None, names=header, index_col='ID')
    print(f'{data.isna().any().any() = }')

    # Bin radius.
    r = data[['x', 'y', 'z']].apply(np.square).sum(axis=1).apply(np.sqrt)
    print(f'{r.describe() = }')
    n_bins = 50
    bin_edges = log_bins(r.min(), r.max() + 1, n_bins)  # Make upper limit a little bigger to include last particle.
    hist, bin_edges = np.histogram(r, bins=bin_edges)

    # Fit histogram data to Hernquist distribution.
    N = np.sum(hist)
    func = partial(residual, N)
    xdata = bin_edges  # (bin_edges[:-1] + bin_edges[1:]) / 2
    ydata = hist
    sigma = np.sqrt(ydata) + 1
    a0 = [1e-1]
    bounds = ([0], [np.inf])
    popt, pcov = curve_fit(func, xdata, ydata, a0, sigma, bounds=bounds)

    # Plot.
    fig, ax = plt.subplots()
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    widths = bin_edges[1:] - bin_edges[:-1]
    ax.bar(midpoints, hist, widths, yerr=np.sqrt(hist), color='orange', capsize=3, label=f'binned data, {n_bins = }')
    ax.plot(midpoints, func(bin_edges, popt), color='blue', label='least-squares fit')
    ax.set_title(f"Hernquist particle distribution\n {N = }, a_opt = {popt[0]:.3g}, a_err = {pcov[0,0]**2: .3g}")
    ax.set_xlabel('r')
    ax.set_ylabel('# particles')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend()

    plt.show()
