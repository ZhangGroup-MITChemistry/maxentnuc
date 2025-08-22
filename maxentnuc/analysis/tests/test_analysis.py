import pytest
from maxentnuc.analysis.analysis import *
from maxentnuc.analysis.domain_analyzer import *
import numpy as np


def test_cg_trajectory():
    trajectory = np.array([[[0, 0, 0],
                            [1, 1, 1],
                            [2, 2, 2],
                            [3, 3, 3]]])
    cg = cg_trajectory(trajectory, 2)

    exp = np.array([[[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]]])
    np.testing.assert_allclose(cg, exp)


def test_get_local_rg():
    positions = np.array([[[0, 0, 0],
                           [1, 0, 0],
                           [2, 0, 0],
                           [3, 0, 0],
                           [4, 0, 0],
                           [5, 0, 0],
                           [6, 0, 0]]])
    window = 5
    result = get_local_rg(positions, window)

    exp = np.sqrt(10/5)
    assert result == pytest.approx([0, 0, exp, exp, exp, 0, 0], rel=1e-2)


def test_get_loops():
    loops = get_loops(np.array([-1, 0, 0, 0]))
    assert len(loops) == 1
    assert loops[0][0] == 0
    assert loops[0][1] == 1

    loops = get_loops(np.array([0, 0, 0, -1]))
    assert len(loops) == 1
    assert loops[0][0] == 3
    assert loops[0][1] == 4

    loops = get_loops(np.array([0, 0, -1, 0]))
    assert len(loops) == 1
    assert loops[0][0] == 2
    assert loops[0][1] == 3

    loops = get_loops(np.array([0, -1, -1, 0]))
    assert len(loops) == 1
    assert loops[0][0] == 1
    assert loops[0][1] == 3

    loops = get_loops(np.array([0, 0, 0, 0]))
    assert len(loops) == 0

    loops = get_loops(np.array([0, -1, 0, -1, 0]))
    assert len(loops) == 2

def test_remove_short_linkers():
    da = DomainAnalyzer(200, 30, min_loop_size=2)

    labels = np.array([-1, 0, 0, 0])
    labels = da.remove_short_linkers(labels)
    assert np.array_equal(labels, np.array([0, 0, 0, 0]))

    labels = np.array([0, 0, 0, -1])
    labels = da.remove_short_linkers(labels)
    assert np.array_equal(labels, np.array([0, 0, 0, 0]))

    labels = np.array([0, -1, 0, 0])
    labels = da.remove_short_linkers(labels)
    assert np.array_equal(labels, np.array([0, 0, 0, 0]))

    labels = np.array([0, -1, 1, 1])
    labels = da.remove_short_linkers(labels)
    assert np.array_equal(labels, np.array([0, -1, 1, 1]))

    labels = np.array([0, -1, -1, 0])
    labels = da.remove_short_linkers(labels)
    assert np.array_equal(labels, np.array([0, -1, -1, 0]))
