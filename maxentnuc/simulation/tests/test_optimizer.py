import pytest
from maxentnuc.simulation.optimizer import *
import numpy as np


def test_gd_mechanics():
    target = np.array([0.1, 0.5, 0.9])
    beta = 0.9
    optimizer = GradientDescentOptimizer(len(target), learning_rate=0.01, beta=beta)
    simulation0 = np.array([0.2, 0.1, 0.6])
    optimizer.update(target, simulation0)
    alpha = optimizer.get_alpha()
    assert alpha[0] > 0
    assert alpha[1] < 0
    assert alpha[2] < 0
    assert np.allclose(optimizer.contact_map, simulation0, atol=1e-2)

    simulation1 = np.array([0.1, 0.5, 0.9])
    optimizer.update(target, simulation1)
    expect = (simulation0*beta*(1-beta) + simulation1*(1-beta)) / (1 - beta**2)
    assert np.allclose(optimizer.contact_map, expect, atol=1e-2)


def test_gd_mechanics_warmup():
    target = np.array([0.1, 0.5, 0.9])
    beta = 0.9
    optimizer = GradientDescentOptimizer(len(target), learning_rate=0.01, beta=beta, warmup_t=1)

    simulation = np.array([0.99, 0, 0.3])
    optimizer.update(target, simulation)

    simulation0 = np.array([0.2, 0.1, 0.6])
    optimizer.update(target, simulation0)
    alpha = optimizer.get_alpha()
    assert np.allclose(optimizer.contact_map, simulation0, atol=1e-2)

    simulation1 = np.array([0.1, 0.5, 0.9])
    optimizer.update(target, simulation1)
    expect = (simulation0*beta*(1-beta) + simulation1*(1-beta)) / (1 - beta**2)
    assert np.allclose(optimizer.contact_map, expect, atol=1e-2)


def test_gd_noise():
    target = np.array([0.1, 0.5, 0.9])
    optimizer = GradientDescentOptimizer(len(target), learning_rate=0.01, verbose=False)
    for i in range(1000):
        alpha = optimizer.get_alpha()
        noise = np.random.normal(0, 0.3, alpha.shape)
        true_simulation = np.exp(-alpha) / (1 + np.exp(-alpha))
        alpha = alpha + noise
        simulation = np.exp(-alpha) / (1 + np.exp(-alpha))
        optimizer.update(target, simulation)
    assert np.allclose(true_simulation, target, atol=1e-1)


def test_gd_noise_momentum():
    target = np.array([0.1, 0.5, 0.9])
    optimizer = GradientDescentOptimizer(len(target), learning_rate=0.01, beta=0.9, verbose=False)
    for i in range(1000):
        alpha = optimizer.get_alpha()
        noise = np.random.normal(0, 0.3, alpha.shape)
        true_simulation = np.exp(-alpha) / (1 + np.exp(-alpha))
        alpha = alpha + noise
        simulation = np.exp(-alpha) / (1 + np.exp(-alpha))
        optimizer.update(target, simulation)
    assert np.allclose(true_simulation, target, atol=1e-1)
