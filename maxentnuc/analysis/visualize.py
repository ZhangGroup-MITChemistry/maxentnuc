from na_genes.maximum_entropy_inversion.maximum_entropy_inversion import yaml_load, MaximumEntropyInversion
from neighbor_balance.plotting import plot_contact_map, parse_region
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from na_genes.epigenetics.visualization import load_bw, bin_track
import click
import os


def get_insulation_score(contact_map, window_size):
    insulation_score = []
    for i in range(len(contact_map)):
        left_start = max(0, i - window_size)
        left_end = i
        right_start = i
        right_end = min(len(contact_map), i + window_size)
        intra_score = (np.nanmean(contact_map[left_start:left_end, left_start:left_end])
                       + np.nanmean(contact_map[right_start:right_end, right_start:right_end])) / 2
        inter_score = np.nanmean(contact_map[left_start:left_end, right_start:right_end])
        insulation_score.append(np.log10(intra_score / inter_score))
    return np.array(insulation_score)


def color_region(color, start, end, shift):
    return (
        'mol color ColorID {}\n'
        'mol representation licorice 10.0\n'
        'mol selection index {} to {}\n'
        'mol material Opaque\n'
        'mol addrep top\n'.format(color, start + shift, end + shift))


def color_positions(color, positions, shift):
    positions = [x + shift for x in positions]
    positions = ' '.join(map(str, positions))
    return (
        'mol color ColorID {}\n'
        'mol representation vdw\n'
        'mol selection index {}\n'
        'mol material Opaque\n'
        'mol addrep top\n'.format(color, positions))


def color_domains(contact_map, window_size, shift, colors=(1, 3, 4, 19, 21, 27, 25)):
    insulation_score = get_insulation_score(contact_map, window_size)
    domain_boundaries = find_peaks(insulation_score, distance=window_size, prominence=0.1)[0]

    borders = [0] + [boundary for boundary in domain_boundaries] + [len(contact_map)]

    out = ''
    for i, (start, end) in enumerate(zip(borders[:-1], borders[1:])):
        color = colors[i % len(colors)]
        out += color_region(color, start, end, shift)

    return out, domain_boundaries, insulation_score


@click.command()
@click.argument('base')
@click.argument('iteration')
@click.option('--script-path', default='load.tcl')
@click.option('--window-size', default=30)
@click.option('--step', default=1)
@click.option('--rank', default=0)
@click.option('--track', default=None)
@click.option('--track-cutoff', default=0.5)
def main(base, iteration, window_size, step, script_path, rank, track, track_cutoff):
    """
    Example usage:
    python visualize.py ~/supercloud/mei_runs/sox2/v1 016 --step 100 --track ~/supercloud/genomics_data/GSE98390_E14_ATAC_MERGED.DANPOS.mm39.bw --track-cutoff 200 --window-size 300
    """
    config_fname = f'{base}/config.yaml'
    mei = MaximumEntropyInversion(yaml_load(config_fname))
    mei.paths = {k: f'{base}/{v}' for k, v in mei.paths.items()}

    contact_map = np.load(mei.get_path('processed_contact_map'))

    f, ax = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True, gridspec_kw={'height_ratios': [10, 0.2, 1, 1], 'hspace': 0.01,
                                                                           'top': 0.95, 'bottom': 0.05, 'left': 0.125})
    plot_contact_map(contact_map, mei.config['region'], ax=ax[0], vmin=1e-3, colorbar=False)

    print(f'source {os.path.abspath(script_path)}')
    with open(script_path, 'w') as f:
        f.write('mol new {} waitfor all\n'.format(mei.get_path('topology_psf', iteration)))
        f.write('mol addfile {} step {}\n'.format(mei.get_path('trajectory_dcd', iteration, rank=rank), step))
        if 'n_cap' in mei.config['model_params']:
            shift = mei.config['model_params']['n_cap']
        else:
            shift = 0

        nucleosome_radius = mei.config['model_params']['nucleosome_diameter'] if 'nucleosome_diameter' in mei.config['model_params'] else 11/21
        dna_radius = mei.config['model_params']['dna_diameter'] if 'dna_diameter' in mei.config['model_params'] else 2.5/21
        cap_radius = mei.config['model_params']['terminal_nucleosome_diameter'] if 'terminal_nucleosome_diameter' in mei.config['model_params'] else nucleosome_radius

        nucleosome_radius, dna_radius, cap_radius = nucleosome_radius * 10 / 2, dna_radius * 10 / 2, cap_radius * 10 / 2

        f.write('set sel [atomselect top "name NUC CAP"]\n'
                f'$sel set radius {nucleosome_radius}\n'
                'set sel [atomselect top "name DNA"]\n'
                f'$sel set radius {dna_radius}\n'
                f'set sel [atomselect top "index 0 {2*shift + len(contact_map) - 1}"]\n'
                f'$sel set radius {cap_radius}\n')

        colors = [1, 3, 4, 19, 21, 27, 25]
        colors_mpl = ['red', 'orange', 'yellow', 'green', 'cyan', 'magenta', 'purple']
        out, domain_boundaries, insulation_score = color_domains(contact_map, window_size, shift, colors=colors)
        f.write(out)
        chrom, start, end = parse_region(mei.config['region'])

        _domain_boundaries = [0, *domain_boundaries, len(contact_map)]
        for i in range(len(_domain_boundaries)-1):
            ax[1].axvspan(start + _domain_boundaries[i] * mei.config['resolution'],
                          start + _domain_boundaries[i+1] * mei.config['resolution'],
                          color=colors_mpl[i % len(colors_mpl)], alpha=0.5)
        ax[1].set_yticks([])
        ax[2].plot(range(start, end, mei.config['resolution']), insulation_score)
        for boundary in domain_boundaries:
            ax[2].axvline(start + boundary * mei.config['resolution'], color='r', linestyle='--')

        if track is not None:
            chrom, start, end = parse_region(mei.config['region'])
            vals = load_bw(track, chrom, start, end)
            bin_vals = bin_track(vals, step=mei.config['resolution'])

            positions = np.where(bin_vals > track_cutoff)[0]
            f.write(color_positions(16, positions, shift))

            ax[3].plot(range(start, end, mei.config['resolution']), bin_vals, c='g')
            ax[3].axhline(track_cutoff, c='grey', ls='--')

    plt.show()


if __name__ == '__main__':
    main()
