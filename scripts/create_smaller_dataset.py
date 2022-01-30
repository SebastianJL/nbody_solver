import math
from pathlib import Path

import pandas as pd


def main():
    ipath = Path(f'../data/data.txt')

    cols = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'eps', 'pot']
    data = pd.read_csv(ipath, header=None, names=cols, sep='\t')

    target_lines = 1000
    step = math.floor(data.shape[0] / target_lines)
    data_small = data.iloc[::step]
    data_small.loc[:, 'mass'] *= step

    opath = Path(f'../data/data_small_{data_small.shape[0]}.txt')
    data_small.to_csv(opath, header=False, index=False, sep='\t')


if __name__ == '__main__':
    main()