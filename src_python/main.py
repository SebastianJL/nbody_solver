from pathlib import Path
import time
import pandas as pd
from matplotlib import pyplot as plt

from src_python.tree import OctTree


def read_particles(path: Path):
    cols = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'eps', 'pot']
    data = pd.read_csv(path, header=None, names=cols, sep='\t')

    return data['mass'].to_numpy(), data[['x', 'y', 'z']].to_numpy(), data[['vx', 'vy', 'vz']].to_numpy()


def main():
    # Read file.
    ipath = Path(f'../data/data_small_10.txt')
    masses, positions, velocities = read_particles(ipath)
    oct_tree = OctTree(masses, positions, velocities)

    # Build tree.
    start = time.perf_counter()
    oct_tree.build()
    tree_build_duration = time.perf_counter() - start
    print(f'{tree_build_duration = :g}s')

    start = time.perf_counter()
    oct_tree.validate()
    validate_duration = time.perf_counter() - start
    print(f'{validate_duration = :g}s')

    # Plot tree.
    fig = plt.figure()
    ax = fig.add_subplot()
    oct_tree.plot_2d(ax, 0, 1)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
