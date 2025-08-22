from na_genes.maximum_entropy_inversion.hbond_model import HBondModel
from na_genes.maximum_entropy_inversion.model import simulate, write_psf
from na_genes.maximum_entropy_inversion.upper_triangular import full_to_triu
import numpy as np

settings = {
    'nucleosome_diameter': 10.0,
    'bond_length': 21.0,
    'bead_size': 200,
    'nucleosome_mass': 232000.0,
    'k_n12_bond': 0.20,
    'n13_angle': 27.0,
    'k_n13_angle': 1.0,
    'linker_angle': 90.0,
    'k_linker_angle': 10.0,
    'linker_dihedral': 0.0,
    'k_linker_dihedral': 6.0,
    'r_hb_0': 0.0,
    'k_hb_r': -0.05,
    'theta_hb_0': 180.0,
    'k_hb_a': -1.0,
    'k_lj':  0.35,
    'k_hb_p_mean': -5.0,
    'k_hb_p_stdv': 0.75,
}
N = 100
model = HBondModel(N, settings)
model.build()

alpha = np.random.normal(0.0, 0.1, size=(N, N))
alpha = full_to_triu(alpha, k=2)
model.add_tanh_potential(alpha, 2/21, 42)
positions = model.generate_initial_positions()

write_psf(model.topology, 'test.psf')

simulate(topology=model.topology, system=model.system, positions=positions,
         dt=10.0, T=300.0, friction=1.0, n_steps=100000, report_interval=10000,
         dcd='test.dcd', log='test.log', platform='CPU')
