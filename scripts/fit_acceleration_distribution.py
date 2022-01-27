from pathlib import Path

import numpy as np


def read_binary(path):
    data = np.fromfile(path, dtype=np.float64)
    data = data.reshape(data.size // 3, 3)
    return data

    # # (num_planets, num_params)
    # x = x.reshape((x.shape[0] // 9, 9))
    #
    # d = {"x": x[:, 0],
    #      "y": x[:, 1],
    #      "z": x[:, 2],
    #      "vx": x[:, 3],
    #      "vy": x[:, 4],
    #      "vz": x[:, 5],
    #      }
    # df = pd.DataFrame(data=d)
    # return df


if __name__ == '__main__':
    path = Path('../output/accelerations.dat')

    data = read_binary(path)
    print(data.shape)
    print(data)
