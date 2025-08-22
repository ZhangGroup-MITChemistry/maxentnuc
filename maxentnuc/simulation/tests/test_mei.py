import pytest
import tempfile
from maxentnuc.simulation.mei import *
from maxentnuc.simulation.optimizer import get_optimizer


def test_restart_scratch():
    config = {'optimizer_state': '{i}_optimizer_state.npz', 'optimizer': 'uncoupled_newton', 'optimizer_params': {}}
    optimizer = get_optimizer(config, 10)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        i = restart(config, optimizer)
        assert i == 0
        assert np.all(optimizer.alpha == 0.0)
        assert np.all(optimizer.contact_map == 0.0)
        assert optimizer.t == 1
    os.chdir(cwd)


def test_restart_from_state():
    last_i = 5
    true_t = 6
    true_alpha = np.linspace(0, 1, 10)
    true_contact_map = np.linspace(-1, 0, 10)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        config = {'optimizer_state': '{i}_optimizer_state.npz', 'optimizer': 'uncoupled_newton', 'optimizer_params': {}}
        optimizer = get_optimizer(config, 10)
        optimizer.alpha = true_alpha.copy()
        optimizer.contact_map = true_contact_map.copy()
        optimizer.t = true_t
        optimizer.save_state(config['optimizer_state'].format(i=format_round(last_i)))

        i = restart(config, optimizer)
        assert i == last_i + 1
        assert np.all(optimizer.alpha == true_alpha)
        assert np.all(optimizer.contact_map == true_contact_map)
        assert optimizer.t == true_t
    os.chdir(cwd)
