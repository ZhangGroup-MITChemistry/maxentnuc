from na_genes.maximum_entropy_inversion.model import PolymerModel, simulate
import openmm
import time
import os

model_params = {
    'angle_constant': 1.0,
    'nuc_nuc_attraction': -1.0,
    'n_dna': 5,
    'n_cap': 50,
    'dna_diameter': 5 / 21,
    'terminal_nucleosome_diameter': 300 / 21,
}

model = PolymerModel(1000, **model_params)
model.build()

positions = model.generate_initial_positions()

if os.path.exists('test.dcd'):
    os.remove('test.dcd')
if os.path.exists('test.log'):
    os.remove('test.log')

start = time.time()
success = simulate(topology=model.topology,
                   system=model.system,
                   positions=positions,
                   dt=0.005 * openmm.unit.picoseconds,
                   friction=1 / openmm.unit.picoseconds,
                   platform='CPU',
                   n_steps=10000,
                   dcd='test.dcd',
                   log='test.log',
                   report_interval=10000)

print('Simulation took', time.time() - start, 'seconds')
