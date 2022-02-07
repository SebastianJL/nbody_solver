from pathlib import Path

from src_python.direct import calculate_accelerations_direct
from src_python.io import read_particles, write_particles
from src_python.timer import Timed
from src_python.tree import OctTree


def main():

    # Read file.
    in_path = Path(f'../data/data_small_10.txt')
    with Timed('reading file'):
        masses, positions, velocities = read_particles(in_path)

    n = len(masses)
    eps = 0
    eps2 = eps*2
    theta_max = 0.1

    print(f'{n = }')
    print(f'{eps = }')
    print(f'{theta_max = }')
    print()

    # Build tree.
    with Timed('building tree'):
        oct_tree = OctTree(masses, positions, velocities)

        oct_tree.build()

    # Validate tree
    with Timed('validating tree'):
        oct_tree.validate()

    # Calculate accelerations.
    # With tree and monopole ...
    with Timed('calculating monopole') as t_mono:
        accelerations_tree = oct_tree.calculate_accelerations(eps2, theta_max, quadrupole=False)

    out_path = Path(f'../output/acc_tree_mono_py_n={n}_eps={eps}_{theta_max=:.1f}_.dat')
    with Timed('writing monopole'):
        write_particles(out_path, masses, positions, accelerations_tree)

    # ... plus quadrupole ...
    with Timed('calculating quadrupole') as t_quad:
        accelerations_tree = oct_tree.calculate_accelerations(eps2, theta_max, quadrupole=True)

    out_path = Path(f'../output/acc_tree_quad_py_n={n}_eps={eps}_{theta_max=:.1f}_.dat')
    with Timed('writing monopole'):
        write_particles(out_path, masses, positions, accelerations_tree)

    # ... or just direct
    with Timed('calculating direct') as t_direct:
        accelerations_direct = calculate_accelerations_direct(masses, positions, eps2)

    out_path = Path(f'../output/acc_direct_py_n={n}_eps={eps}_.dat')
    with Timed('writing direct'):
        write_particles(out_path, masses, positions, accelerations_direct)


if __name__ == '__main__':
    main()
