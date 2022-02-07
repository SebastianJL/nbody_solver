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
    is_leaf: bool = field(init=False, default=None)
    com: np.ndarray[(3,), np.float32] = field(init=False, default=None)
    n_particles: int = field(init=False, default=None)
    size: int = field(init=False, default=None)
    _monopole: np.float32 = field(init=False, default=None)
    _quadrupole: np.ndarray[(3, 3), np.float32] = field(init=False, default=None)
    nodes: np.ndarray[(2, 2, 2), Optional["OctTreeNode"]] = \
        field(init=False, default_factory=lambda: np.empty((2, 2, 2), dtype=object))

    def monopole(self) -> np.float32:
        """Calculate the monopole.

        Returns:
            M = sum_i m_i
        """
        if self._monopole is None:
            self._monopole = self.masses.sum()
        return self._monopole

    def quadrupole(self) -> np.ndarray[(3, 3), np.float32]:
        """Calculate the quadrupole.

        Returns:
            Quadrupole matrix Q where
            Q_ij = sum_k m_k ( 3(r_k)[i] (r_k)[j] - delta_ij (r_k)^2 )
            where r_k = s - x_k
            and s = center of mass.
        """
        if self._quadrupole is None:
            s = self.com
            Q = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    for (m_k, x_k) in zip(self.masses, self.positions):
                        r_k = s - x_k
                        Q[i, j] += 3 * m_k * r_k[i] * r_k[j]
                        if i == j:
                            Q[i, j] -= m_k * r_k.dot(r_k)


            self._quadrupole = Q
        return self._quadrupole

    def build(self):
        # Calculate properties of node.
        self.com = center_of_mass(self.masses, self.positions)
        self.n_particles = len(self.masses)
        self.is_leaf = (self.n_particles == 1)
        self.size = np.sqrt(np.sum((self.max - self.min) ** 2))  # Diameter of sphere enclosing the node box.

        if self.is_leaf:
            return

        # Build Rest of the tree.
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

                    n_particles = child_masses.shape[0]
                    if n_particles == 0:
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
                        self.nodes[i, j, k] = OctTreeNode(child_masses, child_positions, child_min, child_max)

        # Build child nodes.
        for node in self.nodes.flatten():
            if node is not None:
                node.build()

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

    def calculate_acceleration(self, position: np.ndarray[(1, 3), np.float32], eps2: np.float32, quadrupole: bool) -> \
    np.ndarray[(3,), np.float32]:
        """ Calculate the acceleration that the tree causes at `position`.

        G=1 is assumed.

        Formulas used:
        * For leaf nodes:
            acc(r) = M / (y^2 + eps^2)^(3/2) * y
            where y = r - com is the distance vector from COM of the node to `position`,
            and M is the total mass of the node.
        * For nodes that fulfill the angle criterion:
            acc = M / abs(y)^3 * y - 1/2 (y.Q + Q.y) / abs(y)^5 + 5/2 y.Q.y / abs(y)^7 * y
            where Q is the quadrupole.

        Args:
            eps2: Softening factor squared.
            position: Position at which acceleration is to be calculated.
            quadrupole: Calculate the Monopole + Quadrupole in non leaf tree nodes.

        Returns:
            The accelerations for each position.
        """

        r = position
        com = self.com
        y = r - com
        y2 = y.dot(y)

        if self.is_leaf:
            if np.all(self.positions[0] == position):
                # Don't compute self interaction.
                return np.zeros(3)
            else:
                # Compute softened acceleration for leaf.
                M = self.monopole()
                acc = M * y / (y2 + eps2) ** (3 / 2)
        else:
            y_abs = np.sqrt(y2)
            opening_angle = self.size / y_abs
            if opening_angle < 1.8:
                M = self.monopole()
                acc = M * y / y_abs ** 3
                if quadrupole:
                    Q = self.quadrupole()
                    acc += - 1 / 2 * (y.dot(Q) + Q.dot(y)) / y_abs ** 5 + 5 / 2 * y.dot(Q).dot(y) / y_abs ** 7 * y
            else:
                # Compute accelerations for children.
                acc = sum(
                    node.calculate_acceleration(position, eps2, quadrupole) for node in self.nodes.flatten() if node is not None)

        return acc

    def plot_2d(self, ax, dim1, dim2, level):
        if level == 0:
            return
        xmin, xmax = self.min[dim1], self.max[dim1]
        ymin, ymax = self.min[dim2], self.max[dim2]
        plot_box(xmin, xmax, ymin, ymax, ax)

        for node in self.nodes.flatten():
            if node is not None:
                node.plot_2d(ax, dim1, dim2, level - 1)


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

    def calculate_accelerations(self, eps2, quadrupole=False) -> np.ndarray[('N', 3), np.float32]:
        """Calculate all accelerations caused on self.positions.

        Args:
            eps2: Softening factor squared.
            quadrupole: Calculate the Monopole + Quadrupole in non leaf tree nodes.
        """
        acc = np.zeros((self.root.n_particles, 3))
        for (i, pos) in enumerate(self.positions):
            acc[i] = self.root.calculate_acceleration(pos, eps2, quadrupole)
        return acc

    def plot_2d(self, ax: Axes, dim1: int, dim2: int, level: int = -1):
        """Plot a 2 dimensional projection of the OctTree.

        Args:
            ax: Axes to plot onto.
            dim1: First dimension to plot. Must be in (0, 1, 2)
            dim2: Second dimension to plot. Must be in (0, 1, 2)
            level: Plot boundaries n levels deep. -1 means plot all levels.
        """
        xs, ys = self.positions[:, dim1], self.positions[:, dim2]
        ax.scatter(xs, ys)

        self.root.plot_2d(ax, dim1, dim2, level)


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
