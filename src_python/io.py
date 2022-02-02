from pathlib import Path

import numpy as np
import pandas as pd


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