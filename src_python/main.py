from pathlib import Path
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src_python.tree import OctTree


def read_particles(path: Path) -> (np.ndarray, np.ndarray, np.ndarray):
    cols = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'eps', 'pot']
    data = pd.read_csv(path, header=None, names=cols, sep='\t')

    return data['mass'].to_numpy(), data[['x', 'y', 'z']].to_numpy(), data[['vx', 'vy', 'vz']].to_numpy()


def write_particles(path: Path, masses, positions, accelerations):
    """Write data into binary file.

    File has order
        m, x, y, z, ax, ay, az, ...repeating.
    """
    data = np.concatenate([masses[:, None], positions, accelerations], axis=1)
    data.tofile(path)


def main():
    # Read file.
    in_path = Path(f'../data/data_small_1001.txt')
    masses, positions, velocities = read_particles(in_path)
    oct_tree = OctTree(masses, positions, velocities)

    # Build tree.
    start = time.perf_counter()
    oct_tree.build()
    tree_build_duration = time.perf_counter() - start
    print(f'{tree_build_duration = :g}s')

    # Validate tree
    start = time.perf_counter()
    oct_tree.validate()
    validate_duration = time.perf_counter() - start
    print(f'{validate_duration = :g}s')

    # Plot tree.
    # start = time.perf_counter()
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # oct_tree.plot_2d(ax, 0, 1, level=-1)
    # plt.axis('equal')
    # plot_duration = time.perf_counter() - start
    # print(f'{plot_duration = :g}s')
    # plt.show()

    # Calculate accelerations.
    eps = 0
    eps2 = eps*2
    start = time.perf_counter()
    accelerations = oct_tree.calculate_accelerations(eps2)
    acc_calc_duration = time.perf_counter() - start
    print(f'{acc_calc_duration = :g}s')

    out_path = Path(f'../output/acc_tree_n={len(masses)}_eps={eps}_.dat')
    start = time.perf_counter()
    write_particles(out_path, masses, positions, accelerations)
    write_duration = time.perf_counter() - start
    print(f'{write_duration = :g}s')



if __name__ == '__main__':
    main()
