from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes


@dataclass
class OctTreeNode:
    masses: npt.NDArray[np.float32] = field(repr=False)  # shape (N, 3)
    positions: npt.NDArray[np.float32] = field(repr=False)  # shape (N, 3)
    min: np.ndarray[(3,), np.float32]
    max: np.ndarray[(3,), np.float32]
    is_leaf: bool = field(default=False)
    com: np.ndarray[(3,), np.float32] = field(init=False, default=None)
    nodes: np.ndarray[(2, 2, 2), Optional["OctTreeNode"]] = \
        field(init=False, default_factory=lambda: np.empty((2, 2, 2), dtype=object))

    def build(self):
        intersections = (self.min + self.max) / 2

        x_lower = self.positions[:, 0] <= intersections[0]
        y_lower = self.positions[:, 1] <= intersections[1]
        z_lower = self.positions[:, 2] <= intersections[2]

        # Initialize child nodes.
        for i in range(2):
            for j in range(2):
                for k in range(2):

                    match (i, j, k):
                        case (0, 0, 0):
                            mask = (x_lower & y_lower & z_lower)
                        case (0, 0, 1):
                            mask = (x_lower & y_lower & ~z_lower)
                        case (0, 1, 0):
                            mask = (x_lower & ~y_lower & z_lower)
                        case (0, 1, 1):
                            mask = (x_lower & ~y_lower & ~z_lower)
                        case (1, 0, 0):
                            mask = (~x_lower & y_lower & z_lower)
                        case (1, 0, 1):
                            mask = (~x_lower & y_lower & ~z_lower)
                        case (1, 1, 0):
                            mask = (~x_lower & ~y_lower & z_lower)
                        case (1, 1, 1):
                            mask = (~x_lower & ~y_lower & ~z_lower)
                        case _:
                            raise "This should not have happened."

                    child_masses = self.masses[mask]
                    child_positions = self.positions[mask]

                    if child_masses.shape[0] == 0:
                        self.nodes[i, j, k] = None
                    else:
                        child_min = np.array((
                            self.min[0] if i == 0 else intersections[0],
                            self.min[1] if j == 0 else intersections[1],
                            self.min[2] if k == 0 else intersections[2],
                        ))
                        child_max = np.array((
                            self.max[0] if i == 1 else intersections[0],
                            self.max[1] if j == 1 else intersections[1],
                            self.max[2] if k == 1 else intersections[2],
                        ))
                        is_leaf = child_masses.shape[0] == 1
                        self.nodes[i, j, k] = OctTreeNode(child_masses, child_positions, child_min, child_max, is_leaf)

        # Build child nodes.
        for node in self.nodes.flatten():
            if node is not None and not node.is_leaf:
                node.build()

        self.com = center_of_mass(self.masses, self.positions)

    def validate(self, leaf_counter: List[int]):
        if self.is_leaf:
            leaf_counter[0] += 1
            assert len(self.positions) == 1
            assert (self.nodes == None).all()
        else:
            assert len(self.positions) > 1

        assert (self.min < self.positions).all()
        assert (self.max > self.positions).all()
        assert len(self.nodes.flatten()) == 8

        for node in self.nodes.flatten():
            if node is not None:
                assert (self.min <= node.min).all()
                assert (self.max >= node.max).all()
                node.validate(leaf_counter)

    def plot_2d(self, ax, dim1, dim2):
        xmin, xmax = self.min[dim1], self.max[dim2]
        ymin, ymax = self.min[dim1], self.max[dim2]
        plot_box(xmin, xmax, ymin, ymax, ax)

        for node in self.nodes.flatten():
            if node is not None:
                node.plot_2d(ax, dim1, dim2)


@dataclass
class OctTree:
    masses: npt.NDArray[np.float32]  # shape (N, 3)
    positions: npt.NDArray[np.float32]  # shape (N, 3)
    velocities: npt.NDArray[np.float32]  # shape (N, 3)
    root: OctTreeNode = field(init=False, default=None)

    def build(self):
        # Construct min, max cube around sphere with radius r and center COM.
        com = center_of_mass(self.masses, self.positions)
        r = np.abs(self.positions - com).max()
        r = 1.1 * r  # Enlarge radius, such that all particles are inside the cube.
        _min = com - r * np.ones(3)
        _max = com + r * np.ones(3)
        assert ((_min < self.positions).all())
        assert ((_max > self.positions).all())

        self.root = OctTreeNode(self.masses, self.positions, _min, _max)
        self.root.build()

    def validate(self):
        """Walk the tree and do some sanity checks."""
        leaf_counter = [0]
        self.root.validate(leaf_counter)
        assert leaf_counter[0] == len(self.positions)

    def plot_2d(self, ax: Axes, dim1: int, dim2: int):
        """Plot a 2 dimensional projection of the OctTree.

        Args:
            ax: Axes to plot onto.
            dim1: First dimension to plot. Must be in (0, 1, 2)
            dim2: Second dimension to plot. Must be in (0, 1, 2)
        """
        xs, ys = self.positions[:, dim1], self.positions[:, dim2]
        ax.scatter(xs, ys)
        xmin, xmax = self.root.min[dim1], self.root.max[dim2]
        ymin, ymax = self.root.min[dim1], self.root.max[dim2]
        ax.set_xlim(xmin - 10, xmax + 10)
        ax.set_ylim(ymin - 10, ymax + 10)

        self.root.plot_2d(ax, dim1, dim2)


def center_of_mass(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Calculate the center of mass.

    Args:
        masses: shape=(N,)
        positions: shape=(N, 3)

    Returns:
        Center of mass
    """
    # Todo: Optimize by simplifying calculation, due to all masses being the same.
    return (masses[:, None] * positions).sum(axis=0) / masses.sum()


def plot_box(xmin, xmax, ymin, ymax, ax):
    """Below code does the same as the following in a more succinct version.

    ax.plot([xmin, xmin], [ymin, ymax])  # bottom left to top left
    ax.plot([xmin, xmax], [ymax, ymax])  # top left to top right
    ax.plot([xmax, xmax], [ymax, ymin])  # top right to bottom right
    ax.plot([xmax, xmin], [ymin, ymin])  # bottom right to bottom left
    """
    xs = [xmin, xmin, xmin, xmax, xmax, xmax, xmax, xmin]
    ys = [ymin, ymax, ymax, ymax, ymax, ymin, ymin, ymin]
    ax.plot(xs, ys)
