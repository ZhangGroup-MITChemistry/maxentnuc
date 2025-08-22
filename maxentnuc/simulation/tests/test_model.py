import pytest
from maxentnuc.simulation.model import *


def test_initial_positions():
    model = PolymerModel(1_000_000 // 200)
    positions = model.generate_initial_positions()
