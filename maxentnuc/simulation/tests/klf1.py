import os
from maxentnuc.simulation.mei import MaximumEntropyInversion
from maxentnuc import data_root


data_root = '/home/joepaggi/share'


assert os.path.exists('klf1')
os.chdir('klf1')
if __name__ == '__main__':
    config = {
        'region': 'chr6:122455000-122465000',
        'max_iter': 10,
        'error_tolerance': 0.0,

        'contact_map_path': f'{data_root}/rcmc/WT_BR1/select_corrected_minus_inward.mcool',
        'normalization': 'average_neighbor',
        'normalization_params': {'max_prob': 0.95, 'neighbor_prob': 0.5},
        'smoothing': 'interpolate_diagonals',
        'smoothing_params': {'sigma': 5},

        #'reweighting_steps': 1000,
        'optimizer': 'gradient_descent',
        'optimizer_params': {'learning_rate': 1e-2,
                             'initial_alpha': 0.0,
                             },

        'model': 'polymer',
        'model_params': {'nucleosome_diameter': 11,
                         'bond_length': 21,
                         'n_dna': 5,
                         'dna_diameter': 5,
                         'n_cap': 50,
                         'nuc_nuc_attraction': -0.1,
                         'terminal_nucleosome_diameter': 300,
                         'angle_constant': 1.0,
                         },
        'dt': 0.2,
        'friction': 0.01,

        # 'model': 'hbond',
        # 'model_params': {
        #     'nucleosome_diameter': 10.0,
        #     'bond_length': 21.0,
        #     'bead_size': 200,
        #     'nucleosome_mass': 232000.0,
        #     'k_n12_bond': 0.20,
        #     'n13_angle': 27.0,
        #     'k_n13_angle': 1.0,
        #     'linker_angle': 90.0,
        #     'k_linker_angle': 10.0,
        #     'linker_dihedral': 0.0,
        #     'k_linker_dihedral': 6.0,
        #     'r_hb_0': 0.0,
        #     'k_hb_r': -0.05,
        #     'theta_hb_0': 180.0,
        #     'k_hb_a': -1.0,
        #     'k_lj':  0.35,
        #     'k_hb_p_mean': -3.0,
        #     'k_hb_p_stdv': 0.75,
        # },
        # 'atom_selection': 'name A*',
        # 'dt': 10.0,
        # 'friction': 3e-4,

        'n_trajectories': 2,
        'n_steps': 50_000,
        'report_interval': 1_000,
        'platform': 'CPU',
        'n_steps_burnin': 0,
    }

    MaximumEntropyInversion(config).run_main()
