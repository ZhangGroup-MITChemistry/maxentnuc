import numpy as np
from math import isqrt


def triu_to_full(triu: np.ndarray, k=1, fill=1.0) -> np.ndarray:
    """
    Convert a 1D representation of only the upper-triangular entries
    to a full 2D array.

    The value of `k` should match the value given to `full_to_triu`.

    :param triu: 1-D upper triangular representation of matrix.
    :param k: Number of ignored diagonals.
    :return:
    """
    _n = 8 * triu.shape[0] + 1  # Define so we can check that it has an integer square root.
    assert len(triu.shape) == 1, f'Invalid triu array. Shape is {triu.shape}'
    assert isqrt(_n)**2 == _n, f'Invalid triu array. Shape is {triu.shape}'
    n = int(0.5 * (isqrt(_n) + 1)) + k - 1

    full = fill*np.ones((n, n))
    index = np.triu_indices(n, k)
    full[index] = triu
    full.T[index] = triu
    return full


def full_to_triu(full: np.ndarray, k=1) -> np.ndarray:
    """
    Convert the 2D array `full` to a 1D representation containing only the
    upper-triangular entries.

    `k` diagonals are not included. Ex: if k=1, the main diagonal is not included.

    It is assumed that full is symmetric about the diagonal. Only
    the upper-triangular entries are considered.

    :param full: a symmetric 2D array
    :param k: Number of diagonals to ignore.
    :return: a 1D array containing the upper-triangular entries
    """
    assert len(full.shape) == 2, full.shape
    assert full.shape[0] == full.shape[1]
    n = full.shape[0]
    mask = np.tri(n, k=-k, dtype=bool).T
    return full[mask]


if __name__ == '__main__':
    full = np.array(list(range(36))).reshape(6, 6)
    triu = full_to_triu(full, k=0)
    print(full)
    print(triu)
    print(triu_to_full(triu, k=0))
