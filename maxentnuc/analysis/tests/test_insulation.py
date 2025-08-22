import pytest
import numpy as np
from na_genes.analysis.insulation import get_partition
from numpy.testing import assert_allclose


def test_get_partition_nobreaks():
    track = [0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=5)
    assert_allclose(partition, np.array([0, 0, 0]))


def test_get_partition_window0():
    track = [0, 0, 0, 1, 1, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track)
    assert_allclose(partition, np.array([0, 0, 0, -1, -1, 1, 1, 1]))


def test_get_partition_window1_small_even():
    track = [0, 0, 0, 1, 1, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=1)
    assert_allclose(partition, np.array([0, 0, 0, 0, -1, 1, 1, 1]))


def test_get_partition_window1_small_odd():
    track = [0, 0, 0, 1, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=1)
    assert_allclose(partition, np.array([0, 0, 0, -1, 1, 1, 1]))


def test_get_partition_window1_big():
    track = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=1)
    assert_allclose(partition, np.array([0, 0, 0, 0, -1, 1, 1, 1, 1]))


def test_get_partition_window2_big():
    track = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=2)
    assert_allclose(partition, np.array([0, 0, 0, 0, 0,  -1, -1, 1, 1, 1, 1, 1]))


def test_get_partition_window2_peak():
    track = [0, 0, 0, 1, 1, 1, 1, 2, 1, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=2)
    assert_allclose(partition, np.array([0, 0, 0, 0, 0,  -1, -1, -1, 1, 1, 1, 1]))


def test_get_partition_window5_peak():
    track = [0, 0, 0, 1, 1, 2, 0, 0, 0]
    track = np.array(track)
    partition = get_partition(track, threshold=0.5, window=5)
    assert_allclose(partition, np.array([0, 0, 0, 0, 0, -1, 1, 1, 1]))
