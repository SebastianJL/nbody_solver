from pathlib import Path
import time

from src_python.direct import calculate_accelerations_direct
from src_python.io import read_particles, write_particles
from src_python.tree import OctTree


def main():
    # Read file.
    in_path = Path(f'../data/data_small_10.txt')
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

    # With tree ...
    start = time.perf_counter()
    accelerations_tree = oct_tree.calculate_accelerations(eps2)
    acc_calc_tree_duration = time.perf_counter() - start
    print(f'{acc_calc_tree_duration = :g}s')

    out_path = Path(f'../output/acc_tree_py_n={len(masses)}_eps={eps}_.dat')
    start = time.perf_counter()
    write_particles(out_path, masses, positions, accelerations_tree)
    write_duration = time.perf_counter() - start
    print(f'{write_duration = :g}s')

    # ... or direct
    start = time.perf_counter()
    accelerations_direct= calculate_accelerations_direct(masses, positions, eps2)
    acc_calc_direct_duration = time.perf_counter() - start
    print(f'{acc_calc_direct_duration = :g}s')

    out_path = Path(f'../output/acc_direct_py_n={len(masses)}_eps={eps}_.dat')
    start = time.perf_counter()
    write_particles(out_path, masses, positions, accelerations_tree)
    write_duration = time.perf_counter() - start
    print(f'{write_duration = :g}s')


if __name__ == '__main__':
    main()
