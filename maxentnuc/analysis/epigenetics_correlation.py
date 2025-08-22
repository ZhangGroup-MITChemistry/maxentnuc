"""


for i in "v1 015" "v2 015" "v3 016" "v4 031" "v6 026"; do
    i=( $i )
    cd ~/mei_runs/nanog/${i[0]}
    echo sbatch --mem=16G --wrap="python ~/na_genes/analysis/polymer_model/epigenetics_correlation.py ${i[1]}"
done;


"""

from .analysis import *
from scipy.stats import spearmanr
from .mei_analyzer import MEIAnalyzer
from na_genes.microc.plotting import parse_region, format_ticks, plot_contact_map
from na_genes.epigenetics.visualization import load_bw, bin_track
import click
from os.path import expanduser
import numpy as np
import pickle


def diagonal_to_track(diagonal, sep):
    track = np.zeros(len(diagonal) + sep)
    track[sep:] += diagonal
    track[:-sep] += diagonal
    track[sep:-sep] /= 2
    return track


def collect_metrics(trajectory):
    metrics = {}

    # parameters = get_single_particle_diffusion_parameters(positions, max_lag=10)
    #
    # metrics['Diffusion coefficient'] = parameters[:, 0]
    # metrics['Diffusion exponent'] = parameters[:, 1]
    # metrics['Confinement radius'] = parameters[:, 2]
    # lag = 10
    # for sep in [10, 50]:
    #     name = f'Local diffusion at sep={sep}, lag={lag}'
    #     msds = get_two_particle_msds(positions, sep=sep, lag=lag)
    #     metrics[name] = diagonal_to_track(msds, sep)

    metrics['Local Rg (window=5)'] = get_local_rg(trajectory, window=5)
    metrics['Local Rg (window=11)'] = get_local_rg(trajectory, window=11)
    metrics['Local Rg (window=21)'] = get_local_rg(trajectory, window=21)
    metrics['Local Rg (window=51)'] = get_local_rg(trajectory, window=51)

    dists = distances_from_surface(trajectory, radii=5.5, probe_radius=20.0)
    metrics['Median distance to surface'] = np.median(dists, axis=0)
    metrics['Time on surface'] = np.mean(dists < 5.51, axis=0)

    metrics['Local density'] = get_local_concentration(trajectory)
    #metrics['Accessibility'] = get_accessibility_centers(trajectory, distance_thresh=30, count_thresh=5)
    metrics['Accessibility 20 nm'] = get_accessibility_surface(trajectory, probe_radius=20.0)

    return metrics


def get_ylim(data, pad, buffer=0.1):
    low = data[pad:-pad].min()
    high = data[pad:-pad].max()
    return low - buffer * (high - low), high + buffer * (high - low)


def plot_metrics_v_tracks(contact_map, metrics, tracks, chrom, start, end, step=200, pad=50):
    f = plt.figure(figsize=(22, 30))
    m = len(metrics) + len(tracks)
    gs = plt.GridSpec(1 + m, 2, height_ratios = [10]+[1]*m, width_ratios=[1, 0.5], figure=f, hspace=0, wspace=0)

    axs = [plt.subplot(gs[0, 0])]
    axs += [plt.subplot(gs[i, 0], sharex=axs[0]) for i in range(1, 1 + m)]

    plot_contact_map(contact_map, f'{chrom}:{start}-{end}', ax=axs[0], colorbar=False)

    for i, (metric_name, metric) in enumerate(metrics.items()):
        i += 1
        print(metric_name)
        for _metric in metric:
            axs[i].plot(range(start, end, step), _metric, alpha=0.5)
        axs[i].plot(range(start, end, step), np.mean(metric, axis=0), c='black', lw=2)
        metric = np.mean(metric, axis=0)
        axs[i].axhline(np.median(metric[pad:-pad]), c='grey', ls='--')
        axs[i].set_ylabel(metric_name, ha='right', rotation=0)
        axs[i].set_ylim(*get_ylim(metric, pad))

    for i, (track_name, track) in enumerate(tracks.items()):
        i += len(metrics) + 1
        vals = load_bw(track, chrom, start, end)
        bin_vals = bin_track(vals, step=200)
        axs[i].plot(range(start, end), vals, c='green', alpha=0.5)
        axs[i].plot(range(start, end, step), bin_vals, c='green')
        axs[i].set_ylabel(track_name, ha='right', rotation=0)
        axs[i].set_ylim(*get_ylim(vals, pad))

    for ax in axs:
        ax.axvline(start + pad * step, c='grey', ls='--')
        ax.axvline(end - pad * step, c='grey', ls='--')

    corr = np.zeros((len(metrics), len(tracks)))
    for i, (name1, metric) in enumerate(metrics.items()):
        metric = np.mean(metric, axis=0)
        for j, (name2, track) in enumerate(tracks.items()):
            vals = load_bw(track, chrom, start, end)
            bin_vals = bin_track(vals, step=200)
            corr[i, j] = spearmanr(bin_vals[pad:-pad], metric[pad:-pad])[0]
    cmap = plt.get_cmap('coolwarm')
    ax = plt.subplot(gs[1:, 1])
    ax.table(cellText=np.round(corr, 3), colLabels=list(tracks.keys()), rowLabels=list(metrics.keys()),
             bbox=[0.5, .25, 0.5, .5], cellColours=cmap(corr + 1 / 2), fontsize=100)
    ax.text(0.5, 0.8, 'Spearman correlation', ha='center', va='top', fontsize=16, transform=ax.transAxes)
    ax.axis('off')


@click.group()
def main():
    pass


@main.command('collect')
@click.argument('iteration', type=int)
@click.option('--scale', default=2.1)
@click.option('--skip', default=1)
@click.option('--config', default='config.yaml')
def collect(iteration, scale, skip, config):
    mei = MEIAnalyzer(config, scale=scale)

    metrics = {}
    trajectories = mei.get_positions(iteration, skip=skip)
    for trajectory in trajectories:
        _metrics = collect_metrics(trajectory)
        for k, v in _metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    with open(f'{mei.mei.format_round(iteration)}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)


@main.command('plot')
@click.argument('iteration', type=int)
@click.option('--config', default='config.yaml')
def plot(iteration, config):
    mei = MEIAnalyzer(config)
    region = mei.mei.config['region']
    chrom, start, end = parse_region(region)

    tracks = {'ATAC': '~/genomics_data/GSE98390_E14_ATAC_MERGED.DANPOS.mm39.bw',
              'H3K27ac': '~/genomics_data/GSM2417096_ESC_H3K27ac.mm39.bw',
              'H3K27me3': '~/genomics_data/GSM2417100_ESC_H3K27me3.mm39.bw'}
    tracks = {k: expanduser(v) for k, v in tracks.items()}
    base = expanduser(mei.mei.format_round(iteration))

    with open(f'{base}_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    simulated_map = mei.get_precomputed_contact_map(iteration)
    experimental_map = mei.get_reference_contact_map()
    merged_map = np.triu(simulated_map, k=1) + np.tril(experimental_map)

    _tracks = {k: tracks[k] for k in ['ATAC', 'H3K27ac', 'H3K27me3']}
    _metrics = {k: metrics[k] for k in
                ['Local density', 'Time spent on surface', 'Local Rg (window=11)', 'Local Rg (window=51)',
                 'Accessibility', 'Accessibility surface']}
    plot_metrics_v_tracks(merged_map, _metrics, _tracks, chrom, start, end)
    plt.tight_layout()
    plt.savefig(f'{base}_epigenetics.pdf')
    plt.close()

    ################################################################################################
    pad = 50
    track_name = 'ATAC'
    atac = bin_track(load_bw(tracks[track_name], chrom, start, end))

    f, ax = plt.subplots(figsize=(15, 5))
    ax.plot(range(start, end, 200), np.mean(metrics['Accessibility'], axis=0), label='Simulated Accessibility')
    tax = ax.twinx()
    tax.fill_between(range(start, end, 200), 0, atac, label=track_name, color='green', alpha=0.5)
    plt.xlim(start + pad * 200, end - pad * 200)
    tax.legend(loc='upper right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.6)
    tax.set_ylim(0)
    format_ticks(ax, y=False)
    plt.savefig(f'{base}_accessibility.pdf')
    plt.close()

    ################################################################################################
    pad = 50
    track_name = 'H3K27ac'
    atac = bin_track(load_bw(tracks[track_name], chrom, start, end))

    f, ax = plt.subplots(figsize=(15, 5))
    ax.plot(range(start, end, 200), np.mean(metrics['Time spent on surface'], axis=0), label='Time spent on surface')
    tax = ax.twinx()
    tax.fill_between(range(start, end, 200), 0, atac, label=track_name, color='red', alpha=0.5)
    plt.xlim(start + pad * 200, end - pad * 200)
    tax.legend(loc='upper right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.6)
    tax.set_ylim(0)
    format_ticks(ax, y=False)
    plt.savefig(f'{base}_surface.pdf')
    plt.close()

################################################################################################

conditions = [
    ('klf1', 'v3', '011'),
    ('nanog', 'v36', '018'),
    ('sox2', 'v11', '013'),
    ('ppm1g', 'v2', '010'),
    ('fbn2', 'v2', '012'),
]


@main.command('collect-all')
def collect_all():
    for system, version, iteration in conditions:
        print(f'cd ~/mei_runs/{system}/{version}')
        print(f'sbatch --mem=16G --time=48:00:00 -J {system}_{version}_{iteration} --wrap="python -m na_genes.analysis.epigenetics_correlation collect {iteration} --scale 0.1 --skip 11"')


@main.command('plot-all')
def plot_all():
    for system, version, iteration in conditions:
        print(f'cd ~/mei_runs/{system}/{version}')
        print(f'sbatch --mem=16G -J {system}_{version}_{iteration} --dependency=SINGLETON --wrap="python -m na_genes.analysis.epigenetics_correlation plot {iteration}"')


if __name__ == '__main__':
    main()
