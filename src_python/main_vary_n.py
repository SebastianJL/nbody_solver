from pathlib import Path

import numpy as np

from src_python.timer import Timed
from src_python.direct import calculate_accelerations_direct
from src_python.io import read_particles, write_particles
from src_python.tree import OctTree


def main():

    n_list = [10, 101, 2001, 3126, 4168] #, 10_002]
    # n_list = [4168]
    eps = 0
    eps2 = eps*2
    repetitions: int = 10
    out_timings_path = f'../output/timings_py_n=[{n_list[0]}, {n_list[-1]}]_{eps=}_theta_max=0.5+0.9_.txt'
    with open(out_timings_path, 'w') as out_file:
        out_file.write('n,')
        out_file.write('mono,')
        out_file.write('quad,')
        out_file.write('direct\n')

    print()
    for i in range(repetitions):
        print(f'{i = }')
        print()
        for n in n_list:
            print(f'{n = }')

            # Read file.
            in_path = Path(f'../data/data_small_{n}.txt')
            with Timed('reading file'):
                masses, positions, velocities = read_particles(in_path)
                assert(n == len(masses))

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
                accelerations_tree = oct_tree.calculate_accelerations(eps2, theta_max=0.5, quadrupole=False)

            out_path = Path(f'../output/acc_tree_mono_py_n={n}_eps={eps}_theta_max={0.5:.1f}_.dat')
            with Timed('writing monopole'):
                write_particles(out_path, masses, positions, accelerations_tree)

            # ... plus quadrupole ...
            with Timed('calculating quadrupole') as t_quad:
                accelerations_tree = oct_tree.calculate_accelerations(eps2, theta_max=0.9, quadrupole=True)

            out_path = Path(f'../output/acc_tree_quad_py_n={n}_eps={eps}_theta_max={0.9:.1f}_.dat')
            with Timed('writing quadrupole'):
                write_particles(out_path, masses, positions, accelerations_tree)

            # ... or just direct
            with Timed('calculating direct') as t_direct:
                accelerations_direct = calculate_accelerations_direct(masses, positions, eps2)

            out_path = Path(f'../output/acc_direct_py_n={n}_eps={eps}_.dat')
            with Timed('writing direct'):
                write_particles(out_path, masses, positions, accelerations_direct)

            with open(out_timings_path, 'a') as out_file:
                out_file.write(f'{n},')
                out_file.write(f'{t_mono.duration},')
                out_file.write(f'{t_quad.duration},')
                out_file.write(f'{t_direct.duration}\n')

            print('')


if __name__ == '__main__':
    main()
