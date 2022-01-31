from pathlib import Path
import time
import pandas as pd

from src_python.tree import OctTree


def read_particles(path: Path):
    cols = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'eps', 'pot']
    data = pd.read_csv(path, header=None, names=cols, sep='\t')

    return data['mass'].to_numpy(), data[['x', 'y', 'z']].to_numpy(), data[['vx', 'vy', 'vz']].to_numpy()


def main():
    ipath = Path(f'../data/data.txt')

    masses, positions, velocities = read_particles(ipath)
    oct_tree = OctTree(masses, positions, velocities)

    start = time.perf_counter()
    oct_tree.build()
    duration = time.perf_counter() - start
    print(duration)


if __name__ == '__main__':
    main()
