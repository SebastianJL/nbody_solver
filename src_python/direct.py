import numpy as np


def calculate_accelerations_direct(
        masses: np.ndarray[('N', 3), np.float32],
        positions: np.ndarray[('N', 3), np.float32],
        eps2: np.float32) -> np.ndarray[('N', 3), np.float32]:
    acc = np.zeros_like(positions)
    for (i, p1) in enumerate(positions):
        for (j, p2, m2) in enumerate(zip(positions, masses)):
            r = p2 - p1
            r2 = r.dot(r)
            acc[i] += m2 * r / (r2 + eps2)**(3/2)

    return acc
