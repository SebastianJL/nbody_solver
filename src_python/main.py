from pathlib import Path
import time

from src_python.direct import calculate_accelerations_direct
from src_python.io import read_particles, write_particles
from src_python.tree import OctTree


def main():
    # Read file.
    in_path = Path(f'../data/data_small_1001.txt')
    masses, positions, velocities = read_particles(in_path)
    n = len(masses)
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

    # With tree and monopole ...
    start = time.perf_counter()
    accelerations_tree = oct_tree.calculate_accelerations(eps2, quadrupole=False)
    acc_calc_tree_mono_duration = time.perf_counter() - start
    print(f'{acc_calc_tree_mono_duration = :g}s')

    out_path = Path(f'../output/acc_tree_mono_py_n={n}_eps={eps}_.dat')
    start = time.perf_counter()
    write_particles(out_path, masses, positions, accelerations_tree)
    write_duration = time.perf_counter() - start
    print(f'{write_duration = :g}s')

    # ... plus quadrupole ...
    start = time.perf_counter()
    accelerations_tree = oct_tree.calculate_accelerations(eps2, quadrupole=True)
    acc_calc_tree_quad_duration = time.perf_counter() - start
    print(f'{acc_calc_tree_quad_duration = :g}s')

    out_path = Path(f'../output/acc_tree_quad_py_n={n}_eps={eps}_.dat')
    start = time.perf_counter()
    write_particles(out_path, masses, positions, accelerations_tree)
    write_duration = time.perf_counter() - start
    print(f'{write_duration = :g}s')

    # ... or just direct
    start = time.perf_counter()
    accelerations_direct = calculate_accelerations_direct(masses, positions, eps2)
    acc_calc_direct_duration = time.perf_counter() - start
    print(f'{acc_calc_direct_duration = :g}s')

    out_path = Path(f'../output/acc_direct_py_n={n}_eps={eps}_.dat')
    start = time.perf_counter()
    write_particles(out_path, masses, positions, accelerations_direct)
    write_duration = time.perf_counter() - start
    print(f'{write_duration = :g}s')

    with open(f'../output/timings_py_n={n}.txt', 'a') as ofile:
        ofile.write(f'{acc_calc_tree_mono_duration = :g}s\n')
        ofile.write(f'{acc_calc_tree_quad_duration = :g}s\n')
        ofile.write(f'{acc_calc_direct_duration = :g}s\n')


if __name__ == '__main__':
    main()
