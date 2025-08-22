from maxentnuc.simulation.simulated_contacts import get_contacts, contact_indicator
from maxentnuc.simulation.upper_triangular import triu_to_full, full_to_triu
from maxentnuc.simulation.experimental_contacts import *
from neighbor_balance.plotting import *
import cooler
import pandas as pd
import click
from maxentnuc.simulation.model import PolymerModel, simulate, write_psf
import numpy as np
import matplotlib.pyplot as plt


def make_synthetic_cooler(contact_map, name, noise=False):
    """
    Create a cooler file with the given contact map.
    """
    if noise:
        a = np.random.lognormal(0, 1, contact_map.shape[0])
        contact_map = contact_map * a.reshape(-1, 1) * a.reshape(1, -1)

    bins = pd.DataFrame([('chr1', i, i + 200) for i in range(0, 200 * contact_map.shape[0] + 1, 200)],
                        columns=['chrom', 'start', 'end'])

    n = 10_000_000
    pixels = []
    for i in range(contact_map.shape[0]):
        for j in range(i, contact_map.shape[0]):
            pixels.append((i, j, (n * contact_map[i, j]).astype(int)))
    pixels = pd.DataFrame(pixels, columns=['bin1_id', 'bin2_id', 'count'])

    cooler.create_cooler(f'{name}.cool', bins, pixels)
    cool = cooler.Cooler(f'{name}.cool')
    weights, info = cooler.balance_cooler(cool, ignore_diags=1, mad_max=0)
    bins['weight'] = weights
    cooler.create_cooler(f'{name}.mcool', bins, pixels)


def normalize(x):
    return normalize_contact_map_average(x, neighbor_prob=1.0, max_prob=10.0)


def assess(contact_map, name):
    step = 200
    chrom = 'chr1'
    start = 0
    end = step * contact_map.shape[0]
    region = f'{chrom}:{start}-{end}'

    cool = cooler.Cooler(f'{name}.mcool')
    balance = cool.matrix(balance=True).fetch(region)
    no_balance = cool.matrix(balance=False).fetch(region)

    cmaps = {
        'True': normalize(contact_map),
        'Corrupted': normalize(no_balance.copy()),
        'ICE': normalize(balance.copy()),
        'Neighbor': normalize(normalize_contact_map_neighbor(normalize(balance.copy()), neighbor_prob=1.0, max_prob=10.0, bw=0, eps=0)),
    }

    f, ax = plt.subplots(3, 3, sharex=True, sharey='row', figsize=(6, 2.7),
                         gridspec_kw={'height_ratios': [3, 0.5, 0.5], 'hspace': 0.15})

    ax[1, 0].set_ylabel('CD')
    ax[2, 0].set_ylabel('NC')

    for _ax in f.axes:
        _ax.tick_params(axis='x', labelsize=8)
        _ax.tick_params(axis='y', labelsize=8)

    for i, name in enumerate(['True', 'ICE', 'Neighbor']):
        cmap = cmaps[name]
        cmap = ContactMap(cmap, 'chr1', 0, cmap.shape[0] * 200, 200)

        ax[0, i].set_title(name)

        if name == 'True':
            corrupted = cmaps['Corrupted']
            corrupted = ContactMap(corrupted, 'chr1', 0, corrupted.shape[0] * 200, 200)
            merged = cmap.get_merged_map(corrupted)
            merged.plot_contact_map(ax=ax[0, i], colorbar=False)
            ax[0, i].set_ylabel('True + noise', fontproperties=ax[0, i].title.get_fontproperties())
        else:
            cmap.plot_contact_map(ax=ax[0, i], colorbar=False)

        ax[1, i].plot(cmap.x(), cmap.get_marginal(), c='maroon')
        diagonal = np.diagonal(cmap.contact_map, 1)
        neighbors = (np.concatenate((diagonal, [0])) + np.concatenate(([0], diagonal))) / 2
        ax[2, i].plot(cmap.x(), neighbors, c='navy')

    ax[0, 0].set_yticks(range(0, step*len(cmaps['True']), 20_000))
    ax[0, 0].set_xticks(range(0, step*len(cmaps['True']), 20_000))
    ax[1, 0].set_ylim(0, 60)
    ax[2, 0].set_ylim(0, 3)
    ax[1, 0].set_yticks([0, 40])
    ax[2, 0].set_yticks([1])
    
    for row in range(3):
        for col in range(3):
            ax[row, col].yaxis.tick_right()

    ax[0, 0].yaxis.set_tick_params(labelright=False)
    for row in [1, 2]:
        ax[row, 0].yaxis.set_tick_params(labelright=False)
        ax[row, -1].yaxis.set_tick_params(labelright=True)

    return f, ax


def plot(name):
    def indicator(x):
        return contact_indicator(x, 2/21, 42)
    contact_map_thresh = get_contacts(f'{name}.psf', f'{name}.dcd', indicator, burnin=0)
    contact_map_thresh = triu_to_full(contact_map_thresh)
    make_synthetic_cooler(contact_map_thresh.copy(), noise=True, name=name + '_thresh')
    f, ax = assess(contact_map_thresh, name + '_thresh')
    return f, ax


def run_simulation(alpha, scale, r_c, name):
    n = alpha.shape[0]
    model = PolymerModel(n, bond_length=21.0, nucleosome_diameter=11.0)
    model.build()
    model.add_tanh_potential(full_to_triu(alpha, k=2), scale, r_c)

    write_psf(model.topology, f'{name}.psf')

    simulate(topology=model.topology,
             system=model.system,
             positions=model.generate_initial_positions(),
             dt=0.125, friction=0.01, platform='CPU',
             report_interval=1000, n_steps=10_000_000,
             dcd=f'{name}.dcd', log=f'{name}.log')


@click.group()
def cli():
    pass


@cli.command()
def domains():
    pad = 100
    first = 100
    gap = 10
    second = 50
    n = first + gap + second + 2 * pad

    alpha = np.zeros((n, n)) - 0.4
    alpha[pad:pad + first, pad:pad + first] = -1.0
    alpha[pad + first:pad + first + gap, :] = 0.0
    alpha[:, pad + first:pad + first + gap] = 0.0
    alpha[pad + first + gap:pad + first + gap + second, pad + first + gap:pad + first + gap + second] = -1.0

    scale = 4 / 21
    r_c = 15.0
    run_simulation(alpha, scale, r_c, 'test')


@cli.command()
def loops():
    pad = 100
    positions = [0, 50, 100, 125]
    gap = 10
    n = 2 * pad + len(positions) * gap + positions[-1]

    alpha = np.zeros((n, n)) - 0.5
    for position in positions:
        alpha[pad + position:pad + position + gap, :] = 0.0
        alpha[:, pad + position:pad + position + gap] = 0.0

    scale = 4 / 21
    r_c = 15.0
    run_simulation(alpha, scale, r_c, 'data/loops')


@cli.command()
def analyze():
    f, ax = plot('data/domains')
    plt.savefig('img/synthetic_domains.pdf')
    plt.close(f)

    f, ax = plot('data/loops')
    plt.savefig('img/synthetic_loops.pdf')
    plt.close(f)


if __name__ == '__main__':
    cli()
