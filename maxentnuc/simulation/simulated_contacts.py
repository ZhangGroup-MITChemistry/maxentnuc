from typing import Callable
from MDAnalysis import Universe
from MDAnalysis.analysis.distances import self_distance_array
from .upper_triangular import full_to_triu, triu_to_full
import numpy as np
import math


def contact_indicator(distances: np.ndarray, tanh_sigma: float, tanh_r_c: float, use_rc4: bool = False) -> np.ndarray:
    p = 0.5 * (1.0 + np.tanh(tanh_sigma * (tanh_r_c - distances)))
    if use_rc4:
        mask = distances > tanh_r_c
        p[mask] = 0.5 * tanh_r_c ** 4 / distances[mask] ** 4
    return p


def neighbor_mask(n_atoms: int, n_neighbors: int) -> np.ndarray:
    """
    Returns a mask indicating neighbors up to a distance of n_neighbors.
    The mask is in an upper-triangular representation, not including
    self-self entries.
    """
    mask = np.ones((n_atoms, n_atoms), dtype=bool)
    mask = np.triu(mask, 0)
    mask = np.tril(mask, n_neighbors)
    return full_to_triu(mask, k=1)


def get_distance_traces(psf_file: str, dcd_file: str, burnin: int = 0, skip: int = 1,
                        atom_selection: str = 'all') -> np.ndarray:
    """
    Computes the distance between pairs of atoms in each frame of the simulation.

    If atom_selection is provided, only the indicated atoms are included.

    Atom pairs with in num_excluded_neighbors bonds are not included in the output.

    The output is a flat matrix containing entries for the upper triangle of the full distance matrix. To create
    a full matrix, you should use triu_to_full(x, k=num_excluded_neighbors).
    """
    u = Universe(psf_file, dcd_file)
    atoms = u.select_atoms(atom_selection)

    n_atoms = len(atoms)
    n_pairs = int((n_atoms * (n_atoms - 1)) / 2)
    n_frames = math.ceil((len(u.trajectory) - burnin) / skip)

    distances = np.zeros((n_frames, n_pairs))
    for i, frame in enumerate(u.trajectory[burnin::skip]):
        self_distance_array(atoms, result=distances[i], backend='OpenMP')

    distances /= 10  # Å -> nm
    return distances


def get_contacts(psf_file: str, dcd_file: str, indicator: Callable[[np.ndarray], np.ndarray],
                 skip=1, burnin=0, end=None, atom_selection='all', competition=False) -> np.ndarray:
    """
    Compute the mean contact probability between all non-bonded beads in a trajectory.

    The output is a flat matrix containing entries for the upper triangle of the full distance matrix. To create
    a full matrix, you should use triu_to_full(x, k=1).
    """
    u = Universe(psf_file, dcd_file)
    atoms = u.select_atoms(atom_selection)

    n_atoms = len(atoms)
    n_pairs = int((n_atoms * (n_atoms - 1)) / 2)
    distances = np.zeros(n_pairs)
    contacts = np.zeros(n_pairs)
    n_frames = 0
    for frame in u.trajectory[burnin:end:skip]:
        self_distance_array(atoms, result=distances, backend='OpenMP')
        distances /= 10  # Å -> nm
        _contacts = indicator(distances)
        if competition:
            # Divide each entry by the number of (raw) contacts formed by either bead.
            _contacts = triu_to_full(_contacts, k=1)
            crowding = _contacts.sum(axis=0, keepdims=True)
            _contacts /= crowding + crowding.T - _contacts  # Subtraction prevents double counting.
            _contacts = full_to_triu(_contacts, k=1)
        contacts += _contacts
        n_frames += 1
    contacts /= n_frames
    return contacts
