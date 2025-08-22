import matplotlib.pyplot as plt
import numpy as np
from .convergence_analyzer import ConvergenceAnalyzer
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from glob import glob
import os
import math
from maxentnuc.simulation.mei import MaximumEntropyInversion, yaml_load
from maxentnuc.simulation.upper_triangular import triu_to_full, full_to_triu
from .analysis import load_polymer
from neighbor_balance.plotting import ContactMap
from scipy.stats import spearmanr, pearsonr


def show(fname=None):
    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def coarsen_contact_map(contact_map, factor):
    contact_map = contact_map.copy()
    for i in range(0, contact_map.shape[0], factor):
        for j in range(i, contact_map.shape[1], factor):
            contact_map[i:i + factor, j:j + factor] = np.mean(contact_map[i:i + factor, j:j + factor])
            contact_map[j:j + factor, i:i + factor] = np.mean(contact_map[j:j + factor, i:i + factor])
    return contact_map


def subtract_diagonals(cmap):
    cmap = cmap.copy()
    for offset in range(1, cmap.shape[0] - 100):
        diag = np.diag(cmap, k=offset)
        cmap[np.arange(len(diag)), np.arange(len(diag)) + offset] -= np.nanmean(diag)
        cmap[np.arange(len(diag)), np.arange(len(diag)) + offset] /= np.nanstd(diag)
    return cmap


def compare_maps(cmap1, cmap2):
    smooth_cmap1 = coarsen_contact_map(cmap1, 100)
    smooth_cmap2 = coarsen_contact_map(cmap2, 100)

    metrics = {}
    for name, _cmap1, _cmap2 in [('', cmap1, cmap2), ('smooth', smooth_cmap1, smooth_cmap2)]:
        _cmap1_triu = full_to_triu(_cmap1)
        _cmap2_triu = full_to_triu(_cmap2)
        metrics.update({
            f'{name} rmse': np.sqrt(np.mean((_cmap2_triu - _cmap1_triu) ** 2)),
            f'{name} spearman': spearmanr(_cmap1_triu, _cmap2_triu)[0],
            f'{name} spearman diag3': spearmanr(np.diag(_cmap1, 3), np.diag(_cmap2, 3))[0],
            f'{name} log2 ratio stddev': np.nanstd(np.log2(_cmap2_triu/_cmap1_triu)),
            f'{name} r': pearsonr(full_to_triu(subtract_diagonals(_cmap1)),
                                  full_to_triu(subtract_diagonals(_cmap2)))[0],
            f'{name} total contact log2 ratio': np.log2(np.nansum(_cmap2_triu) / np.nansum(_cmap1_triu)),
            f'{name} marginal rmse': np.sqrt(np.mean((_cmap1.sum(axis=1) - _cmap2.sum(axis=1)) ** 2)),
        })
    return metrics


def compare_maps_plot(region, cmap1, cmap2, name1='map1', name2='map2'):
    smooth_cmap1 = coarsen_contact_map(cmap1, 100)
    smooth_cmap2 = coarsen_contact_map(cmap2, 100)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 1], hspace=0.3)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_hist1 = fig.add_subplot(gs[0, 1])
    ax_hist2 = fig.add_subplot(gs[1, 1])

    ratio = cmap2 / cmap1
    smooth_ratio = smooth_cmap2 / smooth_cmap1
    combined = np.triu(ratio, k=0) + np.tril(smooth_ratio, k=-1)
    plot_pairwise(combined, region, log_norm=True, cmap='coolwarm', vmin=1 / 10, vmax=10,
                  ax=ax_img)
    ax_img.set_title(f'{name2} / {name1}')

    ax_hist1.hist(full_to_triu(cmap2) / full_to_triu(cmap1), bins=np.logspace(-1.5, 1.5, 1000), density=True)
    ax_hist1.set_xscale('log')
    ax_hist1.axvline(1, c='k')
    ax_hist1.axvline(2, c='gray', ls='--')
    ax_hist1.axvline(0.5, c='gray', ls='--')
    ax_hist1.set_title(f'{name2} / {name1}')

    ax_hist2.hist(full_to_triu(smooth_cmap2) / full_to_triu(smooth_cmap1), bins=np.logspace(-1.5, 1.5, 100),
                  density=True)
    ax_hist2.set_xscale('log')
    ax_hist2.axvline(1, c='k')
    ax_hist2.axvline(2, c='gray', ls='--')
    ax_hist2.axvline(0.5, c='gray', ls='--')
    ax_hist2.set_title(f'{name2} / {name1} smooth')


class MEIAnalyzer:
    def __init__(self, config_fname, scale=0.1):
        self.mei = MaximumEntropyInversion(yaml_load(config_fname))
        self.root = os.path.dirname(config_fname)
        self.scale = scale

    def get_path(self, key, iteration=None, rank=None):
        return f'{self.root}/{self.mei.get_path(key, iteration=iteration, rank=rank)}'

    def get_iterations(self):
        iterations = []
        iteration = 0
        while os.path.exists(self.get_path('simulated_contact_map', iteration)):
            iterations += [iteration]
            iteration += 1
        return iterations

    def get_reference_contact_map(self):
        cmap = np.load(self.get_path('processed_contact_map'))
        np.fill_diagonal(cmap, 1)
        return cmap

    def get_precomputed_contact_map(self, iteration):
        fname = self.get_path('simulated_contact_map', iteration)
        return triu_to_full(np.load(fname), fill=1)

    def dcds(self, iteration):
        return sorted(glob(self.get_path('trajectory_dcd', iteration, rank='*')))

    def psf(self, iteration):
        return self.get_path('topology_psf', iteration, rank=0)

    def get_positions(self, iteration, skip=1, burnin=0):
        data = []
        for dcd in self.dcds(iteration):
            positions = load_polymer(self.psf(iteration), dcd,
                                     selection=self.mei.config['atom_selection'],
                                     scale=self.scale, skip=skip,
                                     burnin=burnin)
            if positions.shape[0] % self.mei.config['n_trajectories'] != 0:
                print('Incompatible shape', positions.shape[0], self.mei.config['n_trajectories'])
                return None
            positions = positions.reshape(self.mei.config['n_trajectories'], -1, positions.shape[1], 3)
            data += [positions]
        return np.vstack(data)

    def get_end_to_end_distances(self, iteration, start=0, end=-1, skip=1):
        positions = self.get_positions(iteration, skip=skip)
        if positions is None:
            return None
        return np.linalg.norm(positions[:, :, end] - positions[:, :, start], axis=-1)

    def energy_convergence(self):
        for r in [range(5), range(5, 10)]:
            f, ax = plt.subplots(1, 5, figsize=(20, 3))
            for k, i in enumerate(r):
                energy = []
                for j in range(4):
                    log = pd.read_csv(f'{self.root}/00{i}_trajectory.{j}.log')
                    _energy = log['Potential Energy (kJ/mole)'].to_numpy()
                    _energy = _energy.reshape(4, -1).T
                    energy += [_energy]
                energy = np.hstack(energy)

                y = np.mean(energy, axis=1)
                ax[k].set_title(i)
                ax[k].plot(y, alpha=0.2)
                ax[k].plot(gaussian_filter1d(y, 100))
                ax[k].axhline(y.mean(), c='k')
            plt.show()

    def end_to_end_distance_convergence(self, plot_freq=5, start=0, end=-1, skip=1):
        means = []
        for iteration in self.get_iterations():
            data = self.get_end_to_end_distances(iteration, start=start, end=end, skip=skip)
            if data is None:
                break

            if plot_freq > 0 and not iteration % plot_freq:
                print(f'Iteration: {iteration}')
                analyzer = ConvergenceAnalyzer(data)
                print(analyzer.report())
                analyzer.plot()

            if means and data.shape[0] != means[-1].shape[0]:
                print('Incompatible shapes', iteration, data.shape[0], means[-1].shape[0])
                break

            means += [data.mean(axis=1)]
        means = np.stack(means)

        sem = np.std(means, axis=1) / np.sqrt(means.shape[1])
        plt.plot(means, label=range(means.shape[1]))
        plt.fill_between(np.arange(means.shape[0]), means.mean(axis=1) - sem, means.mean(axis=1) + sem, color='k',
                         alpha=0.2)
        plt.plot(means.mean(axis=1), c='k', lw=2, label='mean')
        plt.axhline(means[-1].mean(), c='grey', ls='--')
        plt.ylabel('End-to-end distance (nm)')
        plt.xlabel('Iteration')
        show(f'{self.root}/convergence_end_to_end.png')

    def compare_maps_convergence(self):
        metrics_round, metrics_ref = [], []
        iterations = self.get_iterations()
        ref = self.get_reference_contact_map()
        for iteration in iterations[:-1]:
            cmap1 = self.get_precomputed_contact_map(iteration)
            cmap2 = self.get_precomputed_contact_map(iteration + 1)

            metrics_round += [compare_maps(cmap1, cmap2)]
            metrics_ref += [compare_maps(cmap1, ref)]

        iteration = iterations[-1]
        cmap1 = self.get_precomputed_contact_map(iteration)
        metrics_ref += [compare_maps(cmap1, ref)]

        keys = list(metrics_round[0].keys())
        cols = 4
        rows = math.ceil(len(keys) / cols)
        f, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axs = axs.flatten()
        for i, k in enumerate(keys):
            axs[i].plot([m[k] for m in metrics_ref], c='green', label='to ref')
            axs[i].plot([m[k] for m in metrics_round], c='magenta', label='to next round')
            axs[i].set_ylabel(k)
            axs[i].axhline(0, c='k')
        axs[len(keys)-1].legend()
        plt.tight_layout()
        show(f'{self.root}/convergence_metrics.png')

        compare_maps_plot(self.mei.config['region'], cmap1, ref, f'Iteration {iteration}', 'Reference')
        show(f'{self.root}/convergence_map.png')

    def marginal_convergence(self, stride=5):
        region = self.mei.config['region']
        chrom, start, end = parse_region(region)

        f, ax = plt.subplots(2, 2, figsize=(15, 20), sharex='col',
                             gridspec_kw={'height_ratios': [5, 1], 'width_ratios': [1, 0.05], 'wspace': 0.1,
                                          'hspace': 0.1})

        ref_contact_map = self.get_reference_contact_map()
        ax[1, 0].plot(range(start, end, 200), get_marginal(ref_contact_map), label='rcmc', c='grey')

        # sample at rate stride, adding the last iteration if not in phase.
        _iters = self.get_iterations()
        iters = _iters[::stride]
        if _iters[-1] not in iters:
            iters += [_iters[-1]]

        for i in iters:
            simulated_contact_map = self.get_precomputed_contact_map(i)
            ax[1, 0].plot(range(start, end, 200), get_marginal(simulated_contact_map), label=i)

        im = plot_pairwise(simulated_contact_map - ref_contact_map, region, ax=ax[0, 0],
                           vmin=-.1, vmax=.1, colorbar=False, cmap='coolwarm')
        plt.colorbar(im, cax=ax[0, 1], shrink=0.5)
        ax[0, 0].set_title('Simulated - Experimental')

        # put legend to the right of the plot
        ax[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[1, 0].set_xlim(start, end)
        ax[1, 1].axis('off')
        show(f'{self.root}/convergence_marginal.png')

    def visualize(self, script_base_name, iteration, colors, burnin=0, skip=110, sele='all', load=True):
        color_fname = f'{script_base_name}_colors.dat'
        script_fname = f'{script_base_name}.tcl'

        script = """set line ""
        for {set i 0} {$i < $nf} {incr i} {
          gets $fp line
          $sel frame $i
          $sel set user $line
        }
        close $fp
        $sel delete

        mol color User
        mol material Diffuse
        mol representation Licorice 10.0 12.0 50.0
        mol selection "name NUC"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol scaleminmax top $lastRep 0.000000 10.000000
        mol colupdate $lastRep top  on

        mol color User
        mol material Diffuse
        mol representation vdw 1.0 50.0
        mol selection "name NUC"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol scaleminmax top $lastRep 0.000000 10.000000
        mol colupdate $lastRep top on

        mol color ColorID 2
        mol material Diffuse
        mol representation Licorice 10.0 12.0 50.0
        mol selection "user < 0"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol colupdate $lastRep top on
        mol selupdate $lastRep top on

        mol color ColorID 2
        mol material Diffuse
        mol representation vdw 1.0 50.0
        mol selection "user < 0"
        mol addrep top
        set numReps [molinfo top get numreps]
        set lastRep [expr {$numReps - 1}]
        mol colupdate $lastRep top on
        mol selupdate $lastRep top on

        color scale method Turbo
        """

        # In angstroms.
        nucleosome_radius = 55.0
        dna_radius = 25.0
        # cap_radius = 5.5 Use to visualize the big cap residue.

        # Save the colors to a file.
        with open(color_fname, 'w') as fp:
            for color in colors:
                fp.write(f"{' '.join(map(str, color))}\n")

        # Write a VMD TCL script.

        with open(script_fname, 'w') as fp:
            if load:
                fp.write(f'mol new {self.psf(iteration)} waitfor all\n')
                for dcd in self.dcds(iteration):
                    fp.write(f'mol addfile {dcd} step {skip} first {burnin} last -1  waitfor all\n')

            fp.write(f'set sel [atomselect top "{sele}"]\n')
            fp.write('set nf [molinfo top get numframes]\n')
            fp.write(f'set fp [open {color_fname} r]\n')
            fp.write(script)

            fp.write('set sel [atomselect top "name NUC CAP"]\n'
                     f'$sel set radius {nucleosome_radius}\n'
                     'set sel [atomselect top "name DNA"]\n'
                     f'$sel set radius {dna_radius}\n'
                     # f'set sel [atomselect top "index 0 {2*shift + len(contact_map) - 1}"]\n'
                     # f'$sel set radius {cap_radius}\n'
                     )

        print(f'source {script_fname}')
