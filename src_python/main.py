from pathlib import Path
import time

from src_python.direct import calculate_accelerations_direct
from src_python.io import read_particles, write_particles
from src_python.tree import OctTree


class Timed(object):
    """Context manager for printing runtime of enclosed code."""

    def __init__(self, msg):
        self.msg = msg
        self._start = time.perf_counter()
        self._duration = None

    @property
    def duration(self):
        if self._duration is None:
            raise "Cannot read duration before context manager is left."
        else:
            return self._duration

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._duration = time.perf_counter() - self._start
        print(f'{self.msg}: {self._duration:g}s')


def main():

    # Read file.
    in_path = Path(f'../data/data_small_1001.txt')
    with Timed('reading file'):
        masses, positions, velocities = read_particles(in_path)
        n = len(masses)

    eps = 0
    eps2 = eps*2
    theta_max = 1.8

    out_path = Path(f'../output/acc_tree_mono_py_n={n}_eps={eps}_{theta_max=}_.dat')

    # Build tree.
    with Timed('building tree'):
        oct_tree = OctTree(masses, positions, velocities)

        oct_tree.build()

    # Validate tree
    with Timed('validating tree'):
        oct_tree.validate()

    # # Plot tree.
    # with Timed('plotting tree'):
    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    #     oct_tree.plot_2d(ax, 0, 1, level=-1)
    #     plt.axis('equal')
    #     plt.show()

    # Calculate accelerations.

    # With tree and monopole ...
    with Timed('calculating monopole') as t_mono:
        accelerations_tree = oct_tree.calculate_accelerations(eps2, theta_max, quadrupole=False)

    with Timed('writing monopole'):
        write_particles(out_path, masses, positions, accelerations_tree)

    # ... plus quadrupole ...
    with Timed('calculating quadrupole') as t_quad:
        accelerations_tree = oct_tree.calculate_accelerations(eps2, theta_max, quadrupole=True)

    out_path = Path(f'../output/acc_tree_quad_py_n={n}_eps={eps}_.dat')
    with Timed('writing monopole'):
        write_particles(out_path, masses, positions, accelerations_tree)

    # ... or just direct
    with Timed('calculating direct') as t_direct:
        accelerations_direct = calculate_accelerations_direct(masses, positions, eps2)

    out_path = Path(f'../output/acc_direct_py_n={n}_eps={eps}_.dat')
    with Timed('writing direct'):
        write_particles(out_path, masses, positions, accelerations_direct)

    with open(f'../output/timings_py_{n=}_{theta_max=}.txt', 'a') as ofile:
        ofile.write(f'{t_mono.duration:g}s, ')
        ofile.write(f'{t_quad.duration:g}s, ')
        ofile.write(f'{t_direct.duration:g}s\n')


if __name__ == '__main__':
    main()
