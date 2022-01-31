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
    See `Hernquist AN ANALYTICAL MODEL FOR SPHERICAL GALAXIES AND BULGES for reference`.

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


def main():
    # Read data.
    header = 'ID, Masses, x, y, z, Vx, Vy, Vz, softening, potential'.split(', ')
    data = pd.read_csv('data/data_small_1001.txt', delimiter='\t', header=None, names=header, index_col='ID')
    print(f'{data.isna().any().any() = }\n')

    # Bin radius.
    r = data[['x', 'y', 'z']].apply(np.square).sum(axis=1).apply(np.sqrt)
    print(f'{r.describe() = }\n')
    n_bins = 70
    bin_edges = log_bins(r.min(), r.max() + 1, n_bins)  # Make upper limit a little bigger to include last particle.
    hist, bin_edges = np.histogram(r, bins=bin_edges)

    # Fit histogram data to Hernquist distribution.
    N = np.sum(hist)
    func = partial(residual, N)
    xdata = bin_edges
    ydata = hist
    sigma = np.sqrt(ydata) + 1
    p0 = [0.08]
    bounds = ([0], [np.inf])
    popt, pcov = curve_fit(func, xdata, ydata, p0, sigma, bounds=bounds)
    a_opt = popt[0]
    a_err = np.sqrt(pcov[0, 0])
    a_rerr = a_err / a_opt
    chi2 = np.sum(np.square((func(xdata, a_opt) - ydata) / sigma)) / (len(xdata) - len(p0))
    print(f'{a_opt = }')
    print(f'{a_err = }')
    print(f'{a_rerr = : .1%}')
    print(f'{chi2 = : .2}')

    # Estimate mean interparticle separation
    R_hm = (1 + np.sqrt(2)) * a_opt  # See `Hernquist, AN ANALYTICAL MODEL FOR SPHERICAL GALAXIES AND BULGES` for formula.
    print(f'half mass radius {R_hm = }')
    N_hm = r[r < R_hm].count()
    print(f'number of particles contained in R_hm {N_hm = }')
    print(f'number of particles contained in R_hm {r.count() // 2 = }')
    r_mean = 4 / 3 * np.pi * R_hm / np.power(N_hm, 1 / 3)  # https://en.wikipedia.org/wiki/Mean_inter-particle_distance
    print(f'mean inter-particle separation {r_mean = }')

    # Plot.
    fig, ax = plt.subplots()
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    widths = bin_edges[1:] - bin_edges[:-1]
    ax.bar(midpoints, hist, widths, yerr=np.sqrt(hist), color='orange', capsize=3, label=f'binned data, {n_bins = }')
    ax.plot(midpoints, func(bin_edges, popt), color='blue', label='least-squares fit')
    ax.set_title(f"Hernquist particle distribution\n {N = }, {a_opt = :.3g}, {a_rerr = : .1%}, {chi2 = : .1f}")
    ax.set_xlabel('r $[L_0]$')
    ax.set_ylabel('# particles')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
