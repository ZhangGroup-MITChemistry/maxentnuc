import numpy as np
import os
import yaml
import openmm
from .model import PolymerModel, simulate, write_psf
from .optimizer import get_optimizer, Optimizer
from . import simulated_contacts as scm
from .upper_triangular import triu_to_full, full_to_triu
from .experimental_contacts import load_and_process_contact_map
from neighbor_balance.plotting import ContactMap, parse_region
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from MDAnalysis import Universe
from functools import partial
from glob import glob
import click
from mpi4py import MPI
import logging


def yaml_save(fname: str, data: dict):
    with open(fname, 'w') as fp:
        yaml.dump(data, fp)


def yaml_load(fname: str) -> dict:
    with open(fname) as fp:
        data = yaml.safe_load(fp)
    return data


def get_dist_env() -> tuple[int, int]:
    world_size = MPI.COMM_WORLD.Get_size()
    global_rank = MPI.COMM_WORLD.Get_rank()
    return global_rank, world_size


def percentage_error(contacts_sim: np.ndarray, contacts_exp: np.ndarray) -> float:
    """
    Compute the fraction of experimental contacts recovered in simulation.
    This can be greater than 1 if there are many loci with low
    experimental contact probabilities, but high simulated contact probability.

    :param contacts_sim: Simulated contact frequency.
    :param contacts_exp: Experimentally-determined contact frequency.
    :return: Percentage error between simulated and experimental contact frequencies
    """
    return np.nansum(np.abs(contacts_sim - contacts_exp)) / np.nansum(contacts_exp)


class MaximumEntropyInversion:
    """
    See the optimizer, normalization, and model modules for more information on the parameters.
    """
    def __init__(self, config, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

        required = ['region', 'contact_map_path']
        defaults = {
            # General parameters.
            'resolution': 200,
            'max_iter': 100,
            'error_tolerance': 0.01,
            'paths': {},

            # Simulated contact map parameters.
            'contact_scale': 2/21,
            'contact_r_c': 42.0,
            'bias_contact_scale': None,
            'bias_contact_r_c': None,
            'normalize_neighbors': True,
            'competition': False,
            'atom_selection': 'name NUC',

            # RCMC data processing parameters.
            'map_region': None,
            'normalization': 'neighbor',
            'normalization_params': {},
            'smoothing': None,
            'smoothing_params': {},
            'capture_probes_path': None,

            # Optimization parameters.
            'optimizer': 'uncoupled_newton',
            'optimizer_params': {},

            # Model parameters.
            'model': 'polymer',
            'model_params': {},

            # Simulation parameters.
            'n_trajectories': 4,
            'dt': 0.1,
            'friction': 1.0,
            'platform': 'CPU',
            'n_steps': 11_000_000,
            'n_steps_burnin': 100,
            'report_interval': 10_000,
        }
        for key in config:
            assert key in required or key in defaults, f"Invalid key {key}."

        self.config = {}
        for key in required:
            self.config[key] = config[key]
        for key, value in defaults.items():
            self.config[key] = config.get(key, value)

        chrom, start, end = parse_region(self.config['region'])
        assert (end - start) % self.config['resolution'] == 0, \
            f"Region {self.config['region']} not divisible by resolution {self.config['resolution']}."
        self.N = (end - start) // self.config['resolution']

        if self.config['map_region'] is None:
            self.config['map_region'] = self.config['region']

        if self.config['bias_contact_scale'] is None:
            self.config['bias_contact_scale'] = self.config['contact_scale']
        if self.config['bias_contact_r_c'] is None:
            self.config['bias_contact_r_c'] = self.config['contact_r_c']

        # File paths (that will be created by the program).
        # `i` is the iteration number (padded by self.format_round), `j` is the rank of the process.
        paths = config['paths'] if 'paths' in config else {}
        self.paths = {
            'processed_contact_map': paths.get('processed_contact_map', 'processed_contact_map.npy'),
            'alpha': paths.get('alpha', '{i}_alpha.npy'),
            'trajectory_dcd': paths.get('trajectory_dcd', '{i}_trajectory.{j}.dcd'),
            'trajectory_log': paths.get('trajectory_log', '{i}_trajectory.{j}.log'),
            'topology_psf': paths.get('topology_psf', '{i}_topology.{j}.psf'),
            'system_xml': paths.get('system_xml', '{i}_system.{j}.xml'),
            'optimizer_state': paths.get('optimizer_state', '{i}_optimizer_state.npz'),
            'simulated_contact_map': paths.get('simulated_contact_map', '{i}_simulated_contact_map.npy'),
            'contact_map_plot': paths.get('contact_map_plot', '{i}_contact_map.png')
        }

    @staticmethod
    def format_round(i: int):
        """
        Return a string representation of an integer with leading zeros.
        """
        return str(i).rjust(3, '0')

    def get_path(self, key, iteration=None, rank=None):
        """
        Only specify the rank during post-processing.
        """
        if type(iteration) == int:
            iteration = self.format_round(iteration)
        rank = self.rank if rank is None else rank
        return self.paths[key].format(i=iteration, j=rank)

    def run_main(self):
        """
        Compute contact energies (alpha) required to recapitulate experimental contact maps.

        As a convention, 2D matrices are stored as 1D `np.ndarray`s containing
        the entries of the upper triangular (triu). The utility functions in upper_triangular.py
        can be used to convert between full and triu representations.
        """
        contacts_exp_full = self.load_experimental_contact_map()
        assert self.N == contacts_exp_full.shape[0], contacts_exp_full.shape[0]
        contacts_exp = full_to_triu(contacts_exp_full, k=1)
        neighbors = scm.neighbor_mask(self.N, 1)
        optimizer = get_optimizer(self.config['optimizer'], self.config['optimizer_params'],
                                  contacts_exp[~neighbors].shape[0])
        assert not optimizer.uses_cov()  # Not implemented

        i = self.restart(optimizer)
        while i < self.config['max_iter']:
            np.save(self.get_path('alpha', i), optimizer.get_alpha())

            # Run simulation(s).
            if self.world_size > 1:
                iteration = MPI.COMM_WORLD.scatter([i] * self.world_size, root=0)
                self.run_simulation(iteration)
                result = self.compute_contact_map(iteration)
                contacts_sim_split = MPI.COMM_WORLD.gather(result, root=0)
            else:
                self.run_simulation(i)
                contacts_sim_split = [self.compute_contact_map(i)]

            contacts_sim_split = [x for xx in contacts_sim_split for x in xx]  # Flatten list.
            contacts_sim_split = np.vstack(contacts_sim_split)
            contacts_sim = contacts_sim_split.mean(axis=0)

            if self.config['normalize_neighbors']:
                sim_neighbors = np.nanmean(contacts_sim[neighbors])
                exp_neighbors = np.nanmean(contacts_exp[neighbors])
                factor = exp_neighbors / sim_neighbors
                print(f'Neighbor factor: {factor:.2f} =  {exp_neighbors:.2f} / {sim_neighbors:.2f}', flush=True)
                contacts_sim *= factor
                contacts_sim_split *= factor

            # Update alpha.
            optimizer.update(contacts_exp[~neighbors], contacts_sim[~neighbors])
            optimizer.save_state(self.get_path('optimizer_state', i))
            np.save(self.get_path('simulated_contact_map', i), contacts_sim)

            # Report status of optimization.
            self.plot_status(i, contacts_sim, contacts_sim_split, contacts_exp_full)
            error = percentage_error(contacts_sim[~neighbors], contacts_exp[~neighbors])
            print(f'Error: {error:.4f}', flush=True)
            if error < self.config['error_tolerance']:
                print(f'Converged to an error of {error} on iteration {i}.')
                return

            i += 1
        else:
            print('Not converged.')

    def run_worker(self):
        """
        Worker processes get sent an iteration number, run a simulation, and return the simulated contact map.
        """
        while True:
            iteration = MPI.COMM_WORLD.scatter(None, root=0)
            self.run_simulation(iteration)
            result = self.compute_contact_map(iteration)
            MPI.COMM_WORLD.gather(result, root=0)

    def load_experimental_contact_map(self) -> np.ndarray:
        if os.path.exists(self.get_path('processed_contact_map')):
            contacts = np.load(self.get_path('processed_contact_map'))
            return contacts
        contacts = load_and_process_contact_map(self.config['contact_map_path'], self.config['region'], self.config['resolution'],
                                                self.config['normalization'], self.config['normalization_params'],
                                                smoothing=self.config['smoothing'],
                                                smoothing_params=self.config['smoothing_params'],
                                                capture_probes=self.config['capture_probes_path'],
                                                map_region=self.config['map_region'])

        np.save(self.get_path('processed_contact_map'), contacts)
        chrom, start, end = parse_region(self.config['region'])
        contact_map = ContactMap(contact_map=contacts, resolution=self.config['resolution'], chrom=chrom, start=start, end=end)
        contact_map.plot_contact_map_and_marginal(vmin=1e-4)
        plt.savefig(self.get_path('processed_contact_map').replace('.npy', '.png'))
        return contacts

    def run_simulation(self, iteration):
        """
        Run a simulation with the given configuration.
        """
        if self.config['model'] == 'polymer':
            model = PolymerModel(self.N, **self.config['model_params'])
            T = None  # uses reduced energy units
        else:
            assert False, f"Invalid simulation model {self.config['model']}"

        model.build()
        model.add_tanh_potential(np.load(self.get_path('alpha', iteration)),
                                 self.config['bias_contact_scale'], self.config['bias_contact_r_c'])

        write_psf(model.topology, self.get_path('topology_psf', iteration))

        if iteration > 0:
            u = Universe(self.get_path('topology_psf', iteration), self.get_path('trajectory_dcd', iteration-1))
            frames_per_trajectory = len(u.trajectory) // self.config['n_trajectories']
            assert self.config['n_trajectories'] * frames_per_trajectory == len(u.trajectory), \
                f"Number of frames {len(u.trajectory)} not divisible by {self.config['n_trajectories']}."
            positions = [u.trajectory[i*frames_per_trajectory - 1].positions / 10.0
                         for i in range(self.config['n_trajectories'])]
        else:
            positions = model.generate_initial_positions()
            positions = [positions.copy() for _ in range(self.config['n_trajectories'])]

        dcd = self.get_path('trajectory_dcd', iteration)
        log = self.get_path('trajectory_log', iteration)
        device_index = str(self.rank) if self.config['platform'] == 'CUDA' else None
        for rep, position in enumerate(positions):
            success = simulate(topology=model.topology,
                               system=model.system,
                               positions=position,
                               dt=self.config['dt'] * openmm.unit.picoseconds,
                               friction=self.config['friction'] / openmm.unit.picoseconds,
                               platform=self.config['platform'],
                               n_steps=self.config['n_steps'],
                               dcd=dcd,
                               log=log,
                               device_index=device_index,
                               report_interval=self.config['report_interval'],
                               T=T,
                               append=rep > 0)

            if success:
                print(f'Completed simulation {dcd} {rep+1}/{self.config["n_trajectories"]}.', flush=True)
            else:
                print(f'Simulation {dcd} {rep}/{self.config["n_trajectories"]} failed. Exiting.', flush=True)
                exit()

    def compute_contact_map(self, iteration):
        indicator = partial(scm.contact_indicator,
                            tanh_sigma=self.config['contact_scale'], tanh_r_c=self.config['contact_r_c'])

        frames_per_trajectory = self.config['n_steps'] // self.config['report_interval']
        contacts_sim = []
        for rep in range(self.config['n_trajectories']):
            s = rep * frames_per_trajectory + self.config['n_steps_burnin']
            e = (rep+1) * frames_per_trajectory
            contacts_sim.append(scm.get_contacts(self.get_path('topology_psf', iteration),
                                                 self.get_path('trajectory_dcd', iteration),
                                                 indicator, burnin=s, end=e,
                                                 atom_selection=self.config['atom_selection'],
                                                 competition=self.config['competition']))
        return contacts_sim

    def restart(self, optimizer: Optimizer):
        """
        Restart optimization from an existing alpha file if it exists. Return the next iteration number.
        """
        optimizer_states = glob(self.get_path('optimizer_state', '*'))
        if optimizer_states:
            assert not os.path.dirname(optimizer_states[0]), 'Optimizer state files should be in the current directory.'
            optimizer_states = sorted([int(a.split('_')[0]) for a in optimizer_states])
            prev = optimizer_states[-1]
            i = prev + 1

            optimizer_state = self.get_path('optimizer_state', prev)
            print(f'Restarting using optimizer state from {optimizer_state}.\n', flush=True)
            optimizer.load_state(optimizer_state)
        else:
            print('Starting from scratch.', flush=True)
            i = 0

        incomplete = list(glob(self.format_round(i) + '*'))
        if incomplete:
            logging.warning(f'Incomplete files found for iteration {i}: {incomplete}. Moving to "incomplete" directory.')
            if not os.path.exists('incomplete'):
                os.mkdir('incomplete')
            for f in incomplete:
                os.rename(f, os.path.join('incomplete', f))

        return i

    def plot_status(self, iteration, contacts_sim, contacts_sim_split, contacts_exp_full, vmin=1e-4):
        def make_contact_map(contacts):
            chrom, start, end = parse_region(self.config['region'])
            return ContactMap(contact_map=contacts,
                              resolution=self.config['resolution'],
                              chrom=chrom, start=start, end=end)

        gs = GridSpec(2, 3, figure=plt.figure(figsize=(20, 8)), width_ratios=[2, 1, 1])

        # Contact map.
        ax = plt.subplot(gs[:, :1])
        f_sim_mu_full = triu_to_full(contacts_sim, k=1)
        m = np.triu(f_sim_mu_full, k=1) + np.triu(contacts_exp_full, k=1).T
        m = make_contact_map(m)
        m.plot_contact_map(ax=ax, vmin=1e-4)
        ax.set_ylabel('Experiment', size=15)
        ax.set_title('Simulation', size=15)

        # Distance average.
        ax = plt.subplot(gs[0, 1])
        m = make_contact_map(contacts_exp_full)
        m.plot_distance_average(label='Experiment', ax=ax, color='black')
        m = make_contact_map(f_sim_mu_full)
        m.plot_distance_average(label='Simulation', ax=ax, color='green')
        ax.set_ylim(vmin, 1e0)
        ax.legend()

        # Sim v. exp scatter.
        ax = plt.subplot(gs[0, 2])
        # Scatter non-neighbor contacts.
        mask = np.triu_indices_from(contacts_exp_full, k=2)
        x = contacts_exp_full[mask]
        y = f_sim_mu_full[mask]
        assert not np.any(np.isnan(y))
        x, y = x[~np.isnan(x)], y[~np.isnan(x)]  # simulated values should have no nans.
        non_neighbor_corr = spearmanr(x, y)
        ax.scatter(x, y, s=5, alpha=0.5, c='blue')
        ax.set_xlim(vmin, 1.1)

        # Scatter zero contacts.
        zeros = x == 0
        ax.scatter(np.ones(sum(zeros)), y[zeros], s=5, c='red')
        zeros = y == 0
        ax.scatter(x[zeros], np.ones(sum(zeros)), s=5, c='red')

        # Scatter neighbor contacts.
        x = np.diag(contacts_exp_full, k=1)
        y = np.diag(f_sim_mu_full, k=1)
        assert (not np.any(y == 0)) and (not np.any(np.isnan(y)))
        x, y = x[~np.isnan(x)], y[~np.isnan(x)]  # simulated values should have no nans.
        neighbor_corr = spearmanr(x, y)
        ax.scatter(x, y, s=5, c='gray')

        ax.text(0.05, 0.95, f'Non-neighbor corr: {non_neighbor_corr[0]:.2f}\nNeighbor corr: {neighbor_corr[0]:.2f}',
                transform=ax.transAxes, verticalalignment='top')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Simulation')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Sim v. sim scatter.
        if len(contacts_sim_split) > 1:
            ax = plt.subplot(gs[1, 1])
            mid = len(contacts_sim_split) // 2
            x = contacts_sim_split[:mid].mean(axis=0)
            y = contacts_sim_split[mid:].mean(axis=0)
            ax.scatter(x, y, s=5, alpha=0.5, c='blue')

            # Scatter zero contacts.
            zeros = x == 0
            ax.scatter(np.ones(sum(zeros)), y[zeros], s=5, c='red')
            zeros = y == 0
            ax.scatter(x[zeros], np.ones(sum(zeros)), s=5, c='red')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('Simulation group 0')
            ax.set_ylabel('Simulation group 1')
            ax.set_xscale('log')
            ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.get_path('contact_map_plot', iteration))
        plt.close()


@click.command()
@click.argument('config')
@click.option('--process-contact-map', is_flag=True, help='Process the contact map and exit.')
def main(config, process_contact_map):
    rank, world_size = get_dist_env()

    config = yaml_load(config)
    mei = MaximumEntropyInversion(config, rank=rank, world_size=world_size)

    if mei.rank == 0:
        print(mei.config, flush=True)

    if process_contact_map:
        mei.load_experimental_contact_map()
        return

    if rank == 0:
        mei.run_main()
    else:
        mei.run_worker()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Rank {MPI.COMM_WORLD.Get_rank()} encountered an error: {e}", flush=True)
        MPI.COMM_WORLD.Abort(1)
